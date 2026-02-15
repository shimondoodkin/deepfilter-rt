"""Split enc.onnx into enc_conv_streaming.onnx + enc_gru_streaming.onnx and patch decoders for stateful streaming.

This converts a standard DeepFilterNet export (enc.onnx, erb_dec.onnx, df_dec.onnx)
into the split-encoder format used by dfn3_h0, enabling true frame-by-frame streaming
with persistent GRU states across all modules.

Creates (original files are NOT modified):
  - enc_conv_streaming.onnx: Encoder convolutions (no GRU), outputs sliced to T[-1:]
  - enc_gru_streaming.onnx: Encoder GRU with h0/h1 state
  - erb_dec_streaming.onnx: ERB decoder with erb_h0/erb_h1 state I/O
  - df_dec_streaming.onnx: DF decoder with df_h0/df_h1 state I/O

Usage:
    python scripts/split_encoder_and_patch_decoders.py models/dfn3_ll
"""

import sys
from pathlib import Path
from collections import defaultdict

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np


def get_node_by_output(graph, output_name):
    for node in graph.node:
        for out in node.output:
            if out == output_name:
                return node
    return None


def get_all_predecessors(graph, target_names):
    """Get all node names that (transitively) produce the given tensor names."""
    out_to_node = {}
    for node in graph.node:
        for out in node.output:
            out_to_node[out] = node

    visited = set()
    result = set()
    queue = list(target_names)
    while queue:
        name = queue.pop()
        if name in visited:
            continue
        visited.add(name)
        if name in out_to_node:
            node = out_to_node[name]
            result.add(node.name)
            for inp in node.input:
                queue.append(inp)
    return result


def get_initializer_names(graph):
    return {init.name for init in graph.initializer}


def get_nodes_initializers(graph, node_names):
    init_names = get_initializer_names(graph)
    needed = set()
    for node in graph.node:
        if node.name in node_names:
            for inp in node.input:
                if inp in init_names:
                    needed.add(inp)
    return needed


def split_encoder(model_dir):
    """Split enc.onnx into enc_conv_streaming.onnx + enc_gru_streaming.onnx."""
    enc_path = model_dir / "enc.onnx"
    if not enc_path.exists():
        print(f"  No enc.onnx found in {model_dir}")
        return False

    model = onnx.load(str(enc_path))
    graph = model.graph

    # Find the GRU node
    gru_node = None
    for node in graph.node:
        if node.op_type == 'GRU':
            gru_node = node
            break
    if gru_node is None:
        print("  No GRU found in enc.onnx")
        return False

    hidden_size = None
    for attr in gru_node.attribute:
        if attr.name == 'hidden_size':
            hidden_size = attr.i
    print(f"  Encoder GRU: {gru_node.name}, hidden_size={hidden_size}")

    # The GRU's X input comes from a Transpose node
    gru_x_tensor = gru_node.input[0]
    transpose_node = get_node_by_output(graph, gru_x_tensor)
    pre_gru_emb_tensor = transpose_node.input[0]
    print(f"  Pre-GRU embedding: {pre_gru_emb_tensor}")

    # Find all nodes that feed into the pre-GRU embedding
    pre_gru_names = get_all_predecessors(graph, [pre_gru_emb_tensor])
    print(f"  Pre-GRU nodes: {len(pre_gru_names)}")

    # Build maps
    out_to_node = {}
    for node in graph.node:
        for out in node.output:
            out_to_node[out] = node

    in_to_consumers = defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            in_to_consumers[inp].append(node)

    # Post-GRU nodes
    post_gru_names = set()
    queue = list(gru_node.output)
    visited = set()
    while queue:
        tensor = queue.pop()
        if tensor in visited:
            continue
        visited.add(tensor)
        for consumer in in_to_consumers.get(tensor, []):
            if consumer.name not in pre_gru_names and consumer.name != gru_node.name:
                post_gru_names.add(consumer.name)
                for out in consumer.output:
                    queue.append(out)
    print(f"  Post-GRU nodes: {len(post_gru_names)}")

    # GRU constant/initializer nodes
    init_names = get_initializer_names(graph)
    gru_const_names = set()
    for inp in gru_node.input:
        if inp and inp not in init_names:
            producer = out_to_node.get(inp)
            if producer and producer.name not in pre_gru_names:
                gru_const_names.add(producer.name)
                preds = get_all_predecessors(graph, [inp])
                for p in preds:
                    if p not in pre_gru_names:
                        gru_const_names.add(p)

    # ================================================================
    # Create enc_conv_streaming.onnx
    # ================================================================
    # We work on a fresh copy to avoid mutating the shared graph
    conv_model = onnx.load(str(enc_path))
    conv_graph = conv_model.graph

    # Rebuild pre_gru_names for this copy
    conv_pre_gru = get_all_predecessors(conv_graph, [pre_gru_emb_tensor])

    # Keep only pre-GRU nodes
    nodes_to_remove = []
    for i, node in enumerate(conv_graph.node):
        if node.name not in conv_pre_gru:
            nodes_to_remove.append(i)
    for i in reversed(nodes_to_remove):
        del conv_graph.node[i]

    # Add time-slice constants
    conv_graph.initializer.append(numpy_helper.from_array(
        np.array([-1], dtype=np.int64), name='_slice_starts'))
    conv_graph.initializer.append(numpy_helper.from_array(
        np.array([np.iinfo(np.int64).max], dtype=np.int64), name='_slice_ends'))
    conv_graph.initializer.append(numpy_helper.from_array(
        np.array([2], dtype=np.int64), name='_slice_axes_2'))  # For [B,C,T,F]
    conv_graph.initializer.append(numpy_helper.from_array(
        np.array([1], dtype=np.int64), name='_slice_axes_1'))  # For [B,T,H]

    # Add Slice nodes for e0, e1, e2, e3, c0 (dim 2) and emb (dim 1)
    # First rename original outputs to _raw versions, then slice
    orig_output_names = {}
    for o in conv_graph.output:
        orig_output_names[o.name] = o

    # Clear outputs, we'll rebuild them
    while len(conv_graph.output) > 0:
        conv_graph.output.pop()

    # Slice e0, e1, e2, e3, c0 on dimension 2 (time axis in [B,C,T,F])
    for name in ['e0', 'e1', 'e2', 'e3', 'c0']:
        if name in orig_output_names:
            raw_name = f"_raw_{name}"
            # Rename the producing node's output
            for node in conv_graph.node:
                for i, out in enumerate(node.output):
                    if out == name:
                        node.output[i] = raw_name
            # Also rename in any internal consumers
            for node in conv_graph.node:
                for i, inp in enumerate(node.input):
                    if inp == name:
                        node.input[i] = raw_name

            # Add Slice node
            slice_node = helper.make_node(
                'Slice',
                inputs=[raw_name, '_slice_starts', '_slice_ends', '_slice_axes_2'],
                outputs=[name],
                name=f'_slice_{name}',
            )
            conv_graph.node.append(slice_node)

            out_info = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            conv_graph.output.append(out_info)

    # Slice emb on dimension 1 (time axis in [B,T,H])
    raw_emb = f"_raw_emb"
    # Rename the pre-GRU embedding output
    for node in conv_graph.node:
        for i, out in enumerate(node.output):
            if out == pre_gru_emb_tensor:
                node.output[i] = raw_emb
    for node in conv_graph.node:
        for i, inp in enumerate(node.input):
            if inp == pre_gru_emb_tensor:
                node.input[i] = raw_emb

    slice_emb = helper.make_node(
        'Slice',
        inputs=[raw_emb, '_slice_starts', '_slice_ends', '_slice_axes_1'],
        outputs=['emb'],
        name='_slice_emb',
    )
    conv_graph.node.append(slice_emb)
    conv_graph.output.append(helper.make_tensor_value_info('emb', TensorProto.FLOAT, None))

    # Remove unused initializers
    used_names = set()
    for node in conv_graph.node:
        for inp in node.input:
            used_names.add(inp)
    inits_to_remove = []
    for i, init in enumerate(conv_graph.initializer):
        if init.name not in used_names:
            inits_to_remove.append(i)
    for i in reversed(inits_to_remove):
        del conv_graph.initializer[i]

    conv_path = model_dir / "enc_conv_streaming.onnx"
    onnx.save(conv_model, str(conv_path))
    print(f"  Saved {conv_path} ({conv_path.stat().st_size / 1024:.0f} KB)")

    # ================================================================
    # Create enc_gru_streaming.onnx
    # ================================================================
    gru_model = onnx.load(str(enc_path))
    gru_graph = gru_model.graph

    # Keep: transpose, GRU, GRU constants, post-GRU nodes, AND all their dependencies
    gru_keep = set()
    # Find transpose node in this copy
    for node in gru_graph.node:
        for out in node.output:
            if out == gru_x_tensor:
                gru_keep.add(node.name)
    # Find GRU node
    for node in gru_graph.node:
        if node.op_type == 'GRU':
            gru_keep.add(node.name)
            break
    gru_keep.update(gru_const_names)
    gru_keep.update(post_gru_names)

    # Also include all transitive predecessors of kept nodes that aren't pre-GRU conv nodes
    # (these are constant/reshape nodes needed by the post-GRU path)
    gru_out_to_node = {}
    for node in gru_graph.node:
        for out in node.output:
            gru_out_to_node[out] = node
    gru_init_set = {i.name for i in gru_graph.initializer}

    # Collect all tensor inputs of kept nodes
    needed_tensors = set()
    for node in gru_graph.node:
        if node.name in gru_keep:
            for inp in node.input:
                needed_tensors.add(inp)

    # Find producer nodes for those tensors (that aren't already kept or pre-GRU)
    pre_gru_copy = get_all_predecessors(gru_graph, [pre_gru_emb_tensor])
    changed = True
    while changed:
        changed = False
        for tensor in list(needed_tensors):
            if tensor in gru_out_to_node:
                producer = gru_out_to_node[tensor]
                if producer.name not in gru_keep and producer.name not in pre_gru_copy:
                    gru_keep.add(producer.name)
                    changed = True
                    for inp in producer.input:
                        needed_tensors.add(inp)

    nodes_to_remove = []
    for i, node in enumerate(gru_graph.node):
        if node.name not in gru_keep:
            nodes_to_remove.append(i)
    for i in reversed(nodes_to_remove):
        del gru_graph.node[i]

    # Find the GRU node in this copy
    gru_n = None
    for node in gru_graph.node:
        if node.op_type == 'GRU':
            gru_n = node
            break

    # Replace GRU's initial_h with h0 input
    old_h_init = gru_n.input[5] if len(gru_n.input) > 5 else ""
    while len(gru_n.input) < 6:
        gru_n.input.append('')
    gru_n.input[5] = "h0"

    # Remove old h_init producer nodes
    if old_h_init:
        old_producers = get_all_predecessors(gru_graph, [old_h_init])
        nodes_to_remove = []
        for i, node in enumerate(gru_graph.node):
            if node.name in old_producers:
                nodes_to_remove.append(i)
        for i in reversed(nodes_to_remove):
            del gru_graph.node[i]

    # In the original graph, the model output "emb" is the POST-GRU embedding.
    # We need to:
    # 1. Rename the pre-GRU tensor to "_pre_gru_emb" (internal)
    # 2. Rename the model input to "emb" (same name as enc_conv output)
    # 3. Rename the post-GRU "emb" output to "emb_out" (including all consumers)

    # Step 1: First rename the post-GRU "emb" to "emb_out" in all nodes
    # (both as producer and consumer, since lsnr_fc consumes the old "emb")
    for node in gru_graph.node:
        for i, out in enumerate(node.output):
            if out == 'emb':
                node.output[i] = 'emb_out'
        for i, inp in enumerate(node.input):
            if inp == 'emb':
                node.input[i] = 'emb_out'
    for o in gru_graph.output:
        if o.name == 'emb':
            o.name = 'emb_out'

    # Step 2: Rename pre-GRU embedding tensor to "emb" (the model input name)
    for node in gru_graph.node:
        for i, inp in enumerate(node.input):
            if inp == pre_gru_emb_tensor:
                node.input[i] = "emb"

    # Ensure GRU has Y_h output, add h1 identity if needed
    if len(gru_n.output) < 2 or not gru_n.output[1]:
        while len(gru_n.output) < 2:
            gru_n.output.append('')
        gru_n.output[1] = '_gru_h1_raw'

    gru_h_out = gru_n.output[1]
    # Check if any existing node already produces "h1" (e.g. H0 variants)
    existing_h1 = any(
        out == 'h1' for node in gru_graph.node for out in node.output
    )
    if gru_h_out != 'h1' and not existing_h1:
        h1_node = helper.make_node(
            'Identity', inputs=[gru_h_out], outputs=['h1'],
            name='_patch_h1_identity',
        )
        gru_graph.node.append(h1_node)
    elif gru_h_out != 'h1' and existing_h1:
        # Rename existing h1 producer to use our GRU output
        pass  # h1 already produced by another node
    # else: gru_h_out is already 'h1', no rename needed

    # Set inputs and outputs
    while len(gru_graph.input) > 0:
        gru_graph.input.pop()
    # Determine the pre-GRU embedding size from the GRU's W matrix
    # W shape is [num_directions, 3*hidden_size, input_size]
    w_name = gru_n.input[1]
    emb_size = hidden_size  # fallback
    for init in gru_graph.initializer:
        if init.name == w_name:
            emb_size = init.dims[2]
            break
    print(f"  Pre-GRU embedding size: {emb_size}")
    gru_graph.input.append(helper.make_tensor_value_info("emb", TensorProto.FLOAT, [1, 'S', emb_size]))
    gru_graph.input.append(helper.make_tensor_value_info("h0", TensorProto.FLOAT, [1, 'B', hidden_size]))

    while len(gru_graph.output) > 0:
        gru_graph.output.pop()
    gru_graph.output.append(helper.make_tensor_value_info("emb_out", TensorProto.FLOAT, None))
    gru_graph.output.append(helper.make_tensor_value_info("lsnr", TensorProto.FLOAT, None))
    gru_graph.output.append(helper.make_tensor_value_info("h1", TensorProto.FLOAT, [1, 'B', hidden_size]))

    # Remove unused initializers
    used_names = set()
    for node in gru_graph.node:
        for inp in node.input:
            used_names.add(inp)
    inits_to_remove = []
    for i, init in enumerate(gru_graph.initializer):
        if init.name not in used_names:
            inits_to_remove.append(i)
    for i in reversed(inits_to_remove):
        del gru_graph.initializer[i]

    gru_path = model_dir / "enc_gru_streaming.onnx"
    onnx.save(gru_model, str(gru_path))
    print(f"  Saved {gru_path} ({gru_path.stat().st_size / 1024:.0f} KB)")

    return True


def patch_decoder(model_dir, filename, h0_name, h1_name, module_prefix):
    """Patch a decoder ONNX to expose GRU hidden states as h0/h1.

    Reads the original `filename` and writes to `*_streaming.onnx`.
    The original file is NOT modified.
    """
    path = model_dir / filename
    if not path.exists():
        print(f"  No {filename} found, skipping")
        return False

    # Output goes to *_streaming.onnx (e.g. erb_dec.onnx -> erb_dec_streaming.onnx)
    stem = Path(filename).stem  # e.g. "erb_dec"
    out_filename = f"{stem}_streaming.onnx"
    out_path = model_dir / out_filename

    # Check if output already exists and has the h0 input
    if out_path.exists():
        existing = onnx.load(str(out_path))
        for inp in existing.graph.input:
            if inp.name == h0_name:
                print(f"  {out_filename} already has {h0_name} input, skipping")
                return True

    model = onnx.load(str(path))
    graph = model.graph

    # Find all GRU nodes
    gru_nodes = []
    for node in graph.node:
        if node.op_type == 'GRU':
            hs = None
            for attr in node.attribute:
                if attr.name == 'hidden_size':
                    hs = attr.i
            gru_nodes.append({'node': node, 'hidden_size': hs})

    if not gru_nodes:
        print(f"  No GRU nodes in {filename}")
        return False

    num_layers = len(gru_nodes)
    hidden_size = gru_nodes[0]['hidden_size']
    print(f"  {filename}: {num_layers} GRU layers, hidden_size={hidden_size}")

    nodes_to_add = []

    # Create h0 input: [num_layers, 1, hidden_size]
    h0_input = helper.make_tensor_value_info(
        h0_name, TensorProto.FLOAT, [num_layers, 1, hidden_size]
    )

    final_h_names = []

    for i, gru_info in enumerate(gru_nodes):
        gru = gru_info['node']

        # Slice h0[i:i+1] for per-layer initial_h
        slice_output = f"_patch_{module_prefix}_h0_layer{i}"
        starts_name = f"_patch_{module_prefix}_starts_{i}"
        ends_name = f"_patch_{module_prefix}_ends_{i}"
        axes_name = f"_patch_{module_prefix}_axes_{i}"

        graph.initializer.append(numpy_helper.from_array(
            np.array([i], dtype=np.int64), name=starts_name))
        graph.initializer.append(numpy_helper.from_array(
            np.array([i + 1], dtype=np.int64), name=ends_name))
        graph.initializer.append(numpy_helper.from_array(
            np.array([0], dtype=np.int64), name=axes_name))

        nodes_to_add.append(helper.make_node(
            'Slice',
            inputs=[h0_name, starts_name, ends_name, axes_name],
            outputs=[slice_output],
            name=f'_patch_{module_prefix}_slice_h0_{i}',
        ))

        while len(gru.input) < 6:
            gru.input.append('')
        gru.input[5] = slice_output

        # Ensure Y_h output exists
        if len(gru.output) < 2 or not gru.output[1]:
            while len(gru.output) < 2:
                gru.output.append('')
            gru.output[1] = f'_patch_{module_prefix}_h1_layer{i}'
        final_h_names.append(gru.output[1])

    # Concat or identity for h1
    if len(final_h_names) == 1:
        nodes_to_add.append(helper.make_node(
            'Identity', inputs=[final_h_names[0]], outputs=[h1_name],
            name=f'_patch_{module_prefix}_h1_identity',
        ))
    else:
        nodes_to_add.append(helper.make_node(
            'Concat', inputs=final_h_names, outputs=[h1_name],
            name=f'_patch_{module_prefix}_h1_concat', axis=0,
        ))

    h1_output = helper.make_tensor_value_info(
        h1_name, TensorProto.FLOAT, [num_layers, 1, hidden_size]
    )

    for node in nodes_to_add:
        graph.node.append(node)
    graph.input.append(h0_input)
    graph.output.append(h1_output)

    # Save to *_streaming.onnx (original file unchanged)
    onnx.save(model, str(out_path))
    print(f"  Saved {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    return True


def verify_models(model_dir):
    """Quick verification with onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not available, skipping")
        return

    print("\n  Verification:")

    for fname, T in [("enc_conv_streaming.onnx", 3), ("enc_gru_streaming.onnx", 1),
                      ("erb_dec_streaming.onnx", 1), ("df_dec_streaming.onnx", 1)]:
        path = model_dir / fname
        if not path.exists():
            continue
        try:
            sess = ort.InferenceSession(str(path))
            feed = {}
            for i in sess.get_inputs():
                shape = []
                for d in i.shape:
                    if isinstance(d, int):
                        shape.append(d)
                    elif 'h0' in i.name or 'B' in str(d):
                        shape.append(1)
                    else:
                        shape.append(T)
                feed[i.name] = np.zeros(shape, dtype=np.float32)
            result = sess.run(None, feed)
            out_shapes = [(o.name, r.shape) for o, r in zip(sess.get_outputs(), result)]
            print(f"    {fname} (T={T}): {out_shapes}")
        except Exception as e:
            print(f"    {fname}: ERROR - {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/split_encoder_and_patch_decoders.py <model_dir> [...]")
        sys.exit(1)

    for model_dir_str in sys.argv[1:]:
        model_dir = Path(model_dir_str)
        print(f"\n{'='*60}")
        print(f"Processing {model_dir}")
        print(f"{'='*60}")

        if (model_dir / "enc_conv_streaming.onnx").exists():
            print("  Already has enc_conv_streaming.onnx, skipping encoder split")
        else:
            split_encoder(model_dir)

        patch_decoder(model_dir, "erb_dec.onnx", "erb_h0", "erb_h1", "erb")
        patch_decoder(model_dir, "df_dec.onnx", "df_h0", "df_h1", "df")

        verify_models(model_dir)


if __name__ == "__main__":
    main()
