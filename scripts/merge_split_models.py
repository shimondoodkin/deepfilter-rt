"""Merge 4 split streaming ONNX models into a single combined_streaming.onnx.

Merges enc_conv_streaming.onnx + enc_gru_streaming.onnx + erb_dec_streaming.onnx + df_dec_streaming.onnx (produced by
split_encoder_and_patch_decoders.py) into a single model that preserves all GRU
state I/O and quality.

This is fundamentally different from patch_onnx_streaming.py which operates on the
original combined.onnx and achieves lower quality. The merged model from this script
matches the split models exactly.

Install dependencies:
    pip install onnx onnxsim

Usage:
    python scripts/merge_split_models.py [model_dir]

Defaults to models/dfn3. Produces combined_streaming.onnx in the same directory.
"""

import sys
from pathlib import Path

import onnx
from onnx import helper


def prefix_names(model: onnx.ModelProto, prefix: str, shared_dim_params: set[str] | None = None) -> dict[str, str]:
    """Prefix ALL names in a graph in-place. Returns the rename map.

    shared_dim_params: dim_param strings to keep as-is (e.g. {"S"} for the
    sequence dimension that must match across wired subgraphs).
    """
    graph = model.graph
    shared = shared_dim_params or set()
    rename: dict[str, str] = {}

    def pname(name: str) -> str:
        if not name:
            return name
        if name not in rename:
            rename[name] = f"{prefix}{name}"
        return rename[name]

    def prefix_dim_params(type_proto) -> None:
        """Prefix dim_param strings in a TensorType shape."""
        if not type_proto.tensor_type.shape:
            return
        for dim in type_proto.tensor_type.shape.dim:
            if dim.dim_param and dim.dim_param not in shared:
                dim.dim_param = f"{prefix}{dim.dim_param}"

    # Initializers
    for init in graph.initializer:
        init.name = pname(init.name)

    # Graph inputs (both real inputs and weight-inputs)
    for inp in graph.input:
        inp.name = pname(inp.name)
        prefix_dim_params(inp.type)

    # Graph outputs
    for out in graph.output:
        out.name = pname(out.name)
        prefix_dim_params(out.type)

    # All nodes: inputs, outputs, name
    for node in graph.node:
        node.name = f"{prefix}{node.name}" if node.name else ""
        for i in range(len(node.output)):
            node.output[i] = pname(node.output[i])
        for i in range(len(node.input)):
            node.input[i] = pname(node.input[i])

    # value_info (intermediate tensor shapes)
    for vi in graph.value_info:
        vi.name = pname(vi.name)
        prefix_dim_params(vi.type)

    return rename


def topological_sort(nodes, _external_inputs=None):
    """Topologically sort ONNX nodes based on tensor dependencies."""
    # Build output->node index map
    out_to_idx = {}
    for i, node in enumerate(nodes):
        for out in node.output:
            if out:
                out_to_idx[out] = i

    # Build adjacency: for each node, which nodes must come before it
    n = len(nodes)
    deps = [set() for _ in range(n)]
    for i, node in enumerate(nodes):
        for inp in node.input:
            if inp and inp in out_to_idx:
                deps[i].add(out_to_idx[inp])

    # Kahn's algorithm
    in_degree = [len(d) for d in deps]
    queue = [i for i in range(n) if in_degree[i] == 0]
    result = []
    while queue:
        idx = queue.pop(0)
        result.append(nodes[idx])
        for i in range(n):
            if idx in deps[i]:
                deps[i].discard(idx)
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)

    if len(result) != n:
        print(f"WARNING: Could not topologically sort all nodes ({len(result)}/{n})")
        # Append remaining nodes in original order
        sorted_set = set(id(r) for r in result)
        for node in nodes:
            if id(node) not in sorted_set:
                result.append(node)

    return result


def merge_split_models(model_dir: Path) -> onnx.ModelProto:
    """Merge enc_conv, enc_gru, erb_dec, df_dec into a single combined_streaming model."""
    enc_conv = onnx.load(str(model_dir / "enc_conv_streaming.onnx"))
    enc_gru = onnx.load(str(model_dir / "enc_gru_streaming.onnx"))
    erb_dec = onnx.load(str(model_dir / "erb_dec_streaming.onnx"))
    df_dec = onnx.load(str(model_dir / "df_dec_streaming.onnx"))

    models = {
        "conv": enc_conv,
        "gru": enc_gru,
        "erb": erb_dec,
        "df": df_dec,
    }

    # Record original I/O names before prefixing
    orig_io: dict[str, dict] = {}
    for key, model in models.items():
        init_names = {init.name for init in model.graph.initializer}
        orig_io[key] = {
            "inputs": [inp.name for inp in model.graph.input if inp.name not in init_names],
            "outputs": [out.name for out in model.graph.output],
        }

    for key, io in orig_io.items():
        print(f"{key} inputs:  {io['inputs']}")
        print(f"{key} outputs: {io['outputs']}")

    # Prefix all names in each subgraph (keep "S" and "B" shared)
    shared = {"S", "B"}
    renames: dict[str, dict[str, str]] = {}
    for key, model in models.items():
        renames[key] = prefix_names(model, f"{key}/", shared)

    # Build wiring: consumer prefixed input -> producer prefixed output
    # enc_conv.emb -> enc_gru.emb
    # enc_gru.emb_out -> erb_dec.emb, df_dec.emb
    # enc_conv.e0..e3 -> erb_dec.e0..e3
    # enc_conv.c0 -> df_dec.c0
    wire_map: dict[str, str] = {}

    # Wire enc_conv outputs to enc_gru inputs
    for gru_inp in orig_io["gru"]["inputs"]:
        if gru_inp in orig_io["conv"]["outputs"]:
            wire_map[renames["gru"][gru_inp]] = renames["conv"][gru_inp]

    # Wire enc_conv skip connections to erb_dec (e0..e3)
    # Do this BEFORE gru→decoder wiring so emb_out overrides emb
    for erb_inp in orig_io["erb"]["inputs"]:
        if erb_inp in orig_io["conv"]["outputs"]:
            wire_map[renames["erb"][erb_inp]] = renames["conv"][erb_inp]

    # Wire enc_conv outputs to df_dec (c0)
    for df_inp in orig_io["df"]["inputs"]:
        if df_inp in orig_io["conv"]["outputs"]:
            wire_map[renames["df"][df_inp]] = renames["conv"][df_inp]

    # Wire enc_gru outputs to erb_dec inputs (overrides conv wiring for emb)
    # enc_gru output "emb_out" feeds erb_dec input "emb"
    gru_to_erb = {"emb_out": "emb"}
    for gru_out, erb_inp in gru_to_erb.items():
        if gru_out in orig_io["gru"]["outputs"] and erb_inp in orig_io["erb"]["inputs"]:
            wire_map[renames["erb"][erb_inp]] = renames["gru"][gru_out]

    # Wire enc_gru outputs to df_dec inputs (overrides conv wiring for emb)
    gru_to_df = {"emb_out": "emb"}
    for gru_out, df_inp in gru_to_df.items():
        if gru_out in orig_io["gru"]["outputs"] and df_inp in orig_io["df"]["inputs"]:
            wire_map[renames["df"][df_inp]] = renames["gru"][gru_out]

    print(f"\nWire map:")
    for consumer, producer in wire_map.items():
        print(f"  {consumer} <- {producer}")

    # Apply wiring to all nodes in wired models
    for key in ["gru", "erb", "df"]:
        for node in models[key].graph.node:
            for i in range(len(node.input)):
                if node.input[i] in wire_map:
                    node.input[i] = wire_map[node.input[i]]

    # Determine external inputs (not wired from another subgraph)
    wired_inputs = set(wire_map.keys())

    # Map from prefixed name -> user-facing name for external inputs
    external_input_remap: dict[str, str] = {}

    # enc_conv external inputs: feat_erb, feat_spec
    for name in orig_io["conv"]["inputs"]:
        prefixed = renames["conv"][name]
        external_input_remap[prefixed] = name

    # enc_gru external inputs (not wired from conv): h0
    for name in orig_io["gru"]["inputs"]:
        prefixed = renames["gru"][name]
        if prefixed not in wired_inputs:
            external_input_remap[prefixed] = name

    # erb_dec external inputs (not wired from conv/gru): erb_h0
    for name in orig_io["erb"]["inputs"]:
        prefixed = renames["erb"][name]
        if prefixed not in wired_inputs:
            external_input_remap[prefixed] = name

    # df_dec external inputs (not wired from conv/gru): df_h0
    for name in orig_io["df"]["inputs"]:
        prefixed = renames["df"][name]
        if prefixed not in wired_inputs:
            external_input_remap[prefixed] = name

    print(f"\nExternal input remap:")
    for prefixed, final in external_input_remap.items():
        print(f"  {prefixed} -> {final}")

    # Rename external inputs in nodes back to user-facing names
    for key in models:
        for node in models[key].graph.node:
            for i in range(len(node.input)):
                if node.input[i] in external_input_remap:
                    node.input[i] = external_input_remap[node.input[i]]

    # Build combined inputs
    combined_inputs = []
    seen_input_names = set()
    for key in ["conv", "gru", "erb", "df"]:
        for inp in models[key].graph.input:
            # Check if this is an external input (by checking original name)
            for orig_prefixed, final_name in external_input_remap.items():
                if inp.name == orig_prefixed and final_name not in seen_input_names:
                    seen_input_names.add(final_name)
                    combined_inputs.append(
                        helper.make_tensor_value_info(
                            final_name,
                            inp.type.tensor_type.elem_type,
                            [d.dim_value if d.dim_value > 0 else d.dim_param
                             for d in inp.type.tensor_type.shape.dim],
                        )
                    )

    # Collect all nodes
    all_nodes = []
    for key in ["conv", "gru", "erb", "df"]:
        all_nodes.extend(list(models[key].graph.node))

    # Build combined outputs with Identity rename nodes
    # Only include the 6 known outputs (internal tensors like emb_out are wired, not exposed)
    known_outputs = {
        "lsnr": ("gru", "lsnr"),
        "m": ("erb", "m"),
        "coefs": ("df", "coefs"),
        "h1": ("gru", "h1"),
        "erb_h1": ("erb", "erb_h1"),
        "df_h1": ("df", "df_h1"),
    }

    all_graph_outputs = {}
    for key in models:
        for out in models[key].graph.output:
            all_graph_outputs[out.name] = out

    combined_outputs = []
    for final_name, (src_key, src_name) in known_outputs.items():
        if src_name not in renames[src_key]:
            continue
        prefixed_name = renames[src_key][src_name]
        if prefixed_name not in all_graph_outputs:
            continue

        orig_out = all_graph_outputs[prefixed_name]
        combined_outputs.append(
            helper.make_tensor_value_info(
                final_name,
                orig_out.type.tensor_type.elem_type,
                [d.dim_value if d.dim_value > 0 else d.dim_param
                 for d in orig_out.type.tensor_type.shape.dim]
                if orig_out.type.tensor_type.shape
                else None,
            )
        )
        # Identity node to rename internal prefixed name to final output name
        all_nodes.append(helper.make_node(
            "Identity",
            inputs=[prefixed_name],
            outputs=[final_name],
            name=f"rename_{final_name}",
        ))

    print(f"\nCombined inputs:  {[i.name for i in combined_inputs]}")
    print(f"Combined outputs: {[o.name for o in combined_outputs]}")

    # Collect initializers and value_info from all subgraphs
    all_initializers = []
    all_value_info = []
    for key in ["conv", "gru", "erb", "df"]:
        all_initializers.extend(list(models[key].graph.initializer))
        all_value_info.extend(list(models[key].graph.value_info))

    # Internal wired outputs become value_info (not model inputs/outputs)
    for key in models:
        for out in models[key].graph.output:
            all_value_info.append(out)

    # Topological sort to ensure ONNX validation passes
    external_names = set(i.name for i in combined_inputs)
    external_names.update(i.name for i in all_initializers)
    all_nodes = topological_sort(all_nodes, external_names)

    combined_graph = helper.make_graph(
        all_nodes,
        "combined_streaming_deepfilter",
        combined_inputs,
        combined_outputs,
        initializer=all_initializers,
        value_info=all_value_info,
    )

    combined_model = helper.make_model(combined_graph, opset_imports=enc_conv.opset_import)
    combined_model.ir_version = enc_conv.ir_version

    onnx.checker.check_model(combined_model)
    print("\nModel validation passed")

    return combined_model


def main():
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    if positional:
        model_dirs = [Path(a) for a in positional]
    else:
        model_dirs = [project_dir / "models" / "dfn3"]

    for model_dir in model_dirs:
        if not model_dir.exists():
            print(f"Model directory not found: {model_dir}")
            continue

        # Check that split models exist
        required = ["enc_conv_streaming.onnx", "enc_gru_streaming.onnx", "erb_dec_streaming.onnx", "df_dec_streaming.onnx"]
        missing = [f for f in required if not (model_dir / f).exists()]
        if missing:
            print(f"Missing split models in {model_dir}: {missing}")
            print(f"Run split_encoder_and_patch_decoders.py first")
            continue

        print(f"\n{'='*60}")
        print(f"Merging split models from {model_dir}")
        print(f"{'='*60}")

        combined = merge_split_models(model_dir)

        # Optionally simplify with onnxsim (skip with --no-simplify)
        if "--no-simplify" not in sys.argv:
            try:
                from onnxsim import simplify
                print("Running onnxsim...")
                combined, check = simplify(combined)
                if not check:
                    print("WARNING: onnxsim simplification check failed, using unsimplified model")
            except ImportError:
                print("onnxsim not installed, skipping simplification (pip install onnxsim)")
        else:
            print("Skipping onnxsim (--no-simplify)")

        out_path = model_dir / "combined_streaming.onnx"
        onnx.save(combined, str(out_path))
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"Saved merged model to {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
