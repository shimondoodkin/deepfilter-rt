"""Merge 3 DeepFilter ONNX models (enc, erb_dec, df_dec) into a single model.

Install dependencies:
    pip install onnx onnxsim

If onnxsim fails to install (e.g. on Python 3.12+/Windows), run:
    python scripts/build_onnxsim.py

Usage:
    python scripts/merge_onnx.py [model_dir]

Defaults to models/dfn3_ll. Produces combined.onnx in the same directory.
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


def merge_models(model_dir: Path) -> onnx.ModelProto:
    """Merge enc.onnx, erb_dec.onnx, df_dec.onnx into a single model."""
    enc = onnx.load(str(model_dir / "enc.onnx"))
    erb_dec = onnx.load(str(model_dir / "erb_dec.onnx"))
    df_dec = onnx.load(str(model_dir / "df_dec.onnx"))

    # Record original I/O names before prefixing
    enc_init_names = {init.name for init in enc.graph.initializer}
    enc_input_names = [inp.name for inp in enc.graph.input if inp.name not in enc_init_names]
    enc_output_names = [out.name for out in enc.graph.output]

    erb_init_names = {init.name for init in erb_dec.graph.initializer}
    erb_input_names = [inp.name for inp in erb_dec.graph.input if inp.name not in erb_init_names]
    erb_output_names = [out.name for out in erb_dec.graph.output]

    df_init_names = {init.name for init in df_dec.graph.initializer}
    df_input_names = [inp.name for inp in df_dec.graph.input if inp.name not in df_init_names]
    df_output_names = [out.name for out in df_dec.graph.output]

    print(f"Encoder inputs:  {enc_input_names}")
    print(f"Encoder outputs: {enc_output_names}")
    print(f"ERB dec inputs:  {erb_input_names}")
    print(f"ERB dec outputs: {erb_output_names}")
    print(f"DF dec inputs:   {df_input_names}")
    print(f"DF dec outputs:  {df_output_names}")

    # Prefix all names in each subgraph (keep "S" shared — it's the sequence dim
    # that flows from encoder to decoders and must match at runtime)
    shared = {"S"}
    enc_rename = prefix_names(enc, "enc/", shared)
    erb_rename = prefix_names(erb_dec, "erb/", shared)
    df_rename = prefix_names(df_dec, "df/", shared)

    # Build wiring: decoder prefixed inputs -> encoder prefixed outputs
    wire_map: dict[str, str] = {}
    for erb_inp in erb_input_names:
        if erb_inp in enc_output_names:
            wire_map[erb_rename[erb_inp]] = enc_rename[erb_inp]
    for df_inp in df_input_names:
        if df_inp in enc_output_names:
            wire_map[df_rename[df_inp]] = enc_rename[df_inp]

    print(f"Wire map: {wire_map}")

    # Apply wiring to decoder nodes
    for node in list(erb_dec.graph.node) + list(df_dec.graph.node):
        for i in range(len(node.input)):
            if node.input[i] in wire_map:
                node.input[i] = wire_map[node.input[i]]

    # Rename encoder's real inputs back to user-facing names (feat_erb, feat_spec, h0)
    enc_input_remap: dict[str, str] = {}
    for name in enc_input_names:
        prefixed = enc_rename[name]
        enc_input_remap[prefixed] = name
    for node in enc.graph.node:
        for i in range(len(node.input)):
            if node.input[i] in enc_input_remap:
                node.input[i] = enc_input_remap[node.input[i]]

    # Build combined inputs from encoder's real inputs
    combined_inputs = []
    for inp in enc.graph.input:
        orig_name = inp.name.removeprefix("enc/")
        if orig_name in enc_input_names:
            combined_inputs.append(
                helper.make_tensor_value_info(
                    orig_name,
                    inp.type.tensor_type.elem_type,
                    [d.dim_value if d.dim_value > 0 else d.dim_param
                     for d in inp.type.tensor_type.shape.dim],
                )
            )

    # Collect all nodes
    all_nodes = list(enc.graph.node) + list(erb_dec.graph.node) + list(df_dec.graph.node)

    # Combined outputs: lsnr (enc), m (erb_dec), coefs (df_dec), optionally h1 (enc)
    output_map = {
        "lsnr": enc_rename["lsnr"],
        "m": erb_rename["m"],
        "coefs": df_rename["coefs"],
    }
    if "h1" in enc_output_names:
        output_map["h1"] = enc_rename["h1"]

    all_graph_outputs = {out.name: out for out in
                         list(enc.graph.output) + list(erb_dec.graph.output) + list(df_dec.graph.output)}

    combined_outputs = []
    for final_name, prefixed_name in output_map.items():
        if prefixed_name in all_graph_outputs:
            orig_out = all_graph_outputs[prefixed_name]
            combined_outputs.append(
                helper.make_tensor_value_info(
                    final_name,
                    orig_out.type.tensor_type.elem_type,
                    [d.dim_value if d.dim_value > 0 else d.dim_param
                     for d in orig_out.type.tensor_type.shape.dim],
                )
            )
            # Identity node to rename internal prefixed name to final output name
            all_nodes.append(helper.make_node(
                "Identity",
                inputs=[prefixed_name],
                outputs=[final_name],
                name=f"rename_{final_name}",
            ))

    # Collect initializers and value_info from all subgraphs
    all_initializers = (list(enc.graph.initializer) +
                        list(erb_dec.graph.initializer) +
                        list(df_dec.graph.initializer))

    all_value_info = (list(enc.graph.value_info) +
                      list(erb_dec.graph.value_info) +
                      list(df_dec.graph.value_info))
    # Encoder outputs are now internal tensors — add as value_info
    for out in enc.graph.output:
        all_value_info.append(out)

    combined_graph = helper.make_graph(
        all_nodes,
        "combined_deepfilter",
        combined_inputs,
        combined_outputs,
        initializer=all_initializers,
        value_info=all_value_info,
    )

    combined_model = helper.make_model(combined_graph, opset_imports=enc.opset_import)
    combined_model.ir_version = enc.ir_version

    onnx.checker.check_model(combined_model)
    print("Model validation passed")

    return combined_model


def main():
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    if positional:
        model_dir = Path(positional[0])
    else:
        model_dir = project_dir / "models" / "dfn3_ll"

    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        sys.exit(1)

    print(f"Merging models from {model_dir}")
    combined = merge_models(model_dir)

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

    out_path = model_dir / "combined.onnx"
    onnx.save(combined, str(out_path))
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved combined model to {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
