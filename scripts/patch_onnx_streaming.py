"""Patch a DeepFilterNet combined.onnx to expose GRU hidden states as I/O.

This converts any stateless combined.onnx into a streaming-capable model by:
1. Replacing zero-initialized GRU hidden states with external model inputs
2. Exposing final GRU hidden states as model outputs
3. Multi-layer GRU states are concatenated into single h0/h1 tensors per module
4. Inserting time-slice nodes before GRUs and decoder skip connections so that
   only the LAST time frame enters the GRU/decoder path (convolutions still
   see the full T-frame window for temporal context)

After patching, the model processes T frames through convolutions for context,
but GRUs only see 1 frame per call with persistent state — matching the behavior
of Tract's PulsedModel and the split encoder approach.

Usage:
    python scripts/patch_onnx_streaming.py models/dfn3
    python scripts/patch_onnx_streaming.py models/dfn2_ll

Creates combined_streaming.onnx in the model directory.
"""

import sys
from pathlib import Path
from collections import defaultdict

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np


def find_gru_groups(graph):
    """Find and group GRU nodes by module (enc/erb/df)."""
    groups = defaultdict(list)
    for node in graph.node:
        if node.op_type != 'GRU':
            continue
        name = node.name
        if 'enc/' in name:
            module = 'enc'
        elif 'erb/' in name:
            module = 'erb'
        elif 'df/' in name:
            module = 'df'
        else:
            module = 'unknown'

        hidden_size = None
        for attr in node.attribute:
            if attr.name == 'hidden_size':
                hidden_size = attr.i

        groups[module].append({
            'node': node,
            'hidden_size': hidden_size,
            'initial_h': node.input[5] if len(node.input) > 5 else None,
            'final_h': node.output[1] if len(node.output) > 1 else None,
        })

    return dict(groups)


def find_encoder_conv_outputs(graph):
    """Find the encoder conv outputs (e0, e1, e2, e3, c0) tensor names.

    These are [B, C, T, F] tensors from encoder convolutions that feed both
    the next conv layer AND the decoder skip connections. We need to slice
    them to T[-1:] on the decoder path.
    """
    outputs = {}
    for node in graph.node:
        for out in node.output:
            if out in ('enc/e0', 'enc/e1', 'enc/e2', 'enc/e3', 'enc/c0'):
                outputs[out.split('/')[-1]] = out
    return outputs


def find_decoder_consumers(graph, enc_outputs):
    """Find decoder nodes that consume encoder outputs directly.

    Returns list of (node, input_index, tensor_name, tensor_key) tuples
    where tensor_key is 'e0', 'e1', etc.
    """
    consumers = []
    for node in graph.node:
        # Skip encoder nodes — they need the full T dimension
        if node.name.startswith('enc/'):
            continue
        for idx, inp in enumerate(node.input):
            for key, tensor_name in enc_outputs.items():
                if inp == tensor_name:
                    consumers.append((node, idx, tensor_name, key))
    return consumers


def insert_time_slice_nodes(graph, enc_outputs, decoder_consumers, nodes_to_add):
    """Insert Slice nodes to extract T[-1:] from encoder outputs going to decoders.

    For each encoder output consumed by a decoder node, we:
    1. Create a Slice node that extracts the last time step: tensor[:, :, -1:, :]
    2. Rewire the decoder node to use the sliced tensor instead

    The encoder conv chain keeps using the full T-frame tensors.
    """
    # Shared constants for slicing: axis=2 (time dim), starts=-1, ends=MAX
    slice_starts = numpy_helper.from_array(
        np.array([-1], dtype=np.int64), name='_patch_tslice_starts')
    slice_ends = numpy_helper.from_array(
        np.array([np.iinfo(np.int64).max], dtype=np.int64), name='_patch_tslice_ends')
    slice_axes = numpy_helper.from_array(
        np.array([2], dtype=np.int64), name='_patch_tslice_axes')

    graph.initializer.append(slice_starts)
    graph.initializer.append(slice_ends)
    graph.initializer.append(slice_axes)

    # Create one slice node per unique encoder output tensor
    sliced_names = {}  # tensor_name -> sliced_tensor_name
    for key, tensor_name in enc_outputs.items():
        sliced_name = f"_patch_tslice_{key}"
        sliced_names[tensor_name] = sliced_name

        slice_node = helper.make_node(
            'Slice',
            inputs=[tensor_name, '_patch_tslice_starts', '_patch_tslice_ends', '_patch_tslice_axes'],
            outputs=[sliced_name],
            name=f'_patch_tslice_{key}',
        )
        nodes_to_add.append(slice_node)

    # Rewire decoder consumers to use sliced tensors
    rewired = 0
    for node, idx, tensor_name, key in decoder_consumers:
        node.input[idx] = sliced_names[tensor_name]
        rewired += 1

    print(f"  Time-slicing: {len(sliced_names)} encoder outputs, {rewired} decoder edges rewired")
    return sliced_names


def insert_gru_input_slice(graph, gru_node, nodes_to_add):
    """Insert a Slice node before a GRU to extract T[-1:] from its input.

    The GRU input X has shape [T, B, H] (after Transpose).
    We slice to [1, B, H] so the GRU only processes the latest frame.
    """
    x_name = gru_node.input[0]
    sliced_name = f"_patch_gruin_{gru_node.name.replace('/', '_')}"

    # Slice on axis 0 (T dimension): [-1:]
    starts = numpy_helper.from_array(
        np.array([-1], dtype=np.int64), name=f"{sliced_name}_starts")
    ends = numpy_helper.from_array(
        np.array([np.iinfo(np.int64).max], dtype=np.int64), name=f"{sliced_name}_ends")
    axes = numpy_helper.from_array(
        np.array([0], dtype=np.int64), name=f"{sliced_name}_axes")

    graph.initializer.append(starts)
    graph.initializer.append(ends)
    graph.initializer.append(axes)

    slice_node = helper.make_node(
        'Slice',
        inputs=[x_name, f"{sliced_name}_starts", f"{sliced_name}_ends", f"{sliced_name}_axes"],
        outputs=[sliced_name],
        name=f'_patch_gruin_slice_{gru_node.name.replace("/", "_")}',
    )
    nodes_to_add.append(slice_node)

    # Rewire GRU input
    gru_node.input[0] = sliced_name


def patch_model(model_path, output_path=None):
    """Patch a combined.onnx to expose GRU states and add time-slicing."""
    model = onnx.load(str(model_path))
    graph = model.graph

    groups = find_gru_groups(graph)

    print(f"Found GRU groups:")
    for module, layers in groups.items():
        sizes = [l['hidden_size'] for l in layers]
        print(f"  {module}: {len(layers)} layers, hidden_sizes={sizes}")

    nodes_to_add = []

    # --- Step 1: Time-slice encoder outputs going to decoders ---
    enc_outputs = find_encoder_conv_outputs(graph)
    if enc_outputs:
        decoder_consumers = find_decoder_consumers(graph, enc_outputs)
        insert_time_slice_nodes(graph, enc_outputs, decoder_consumers, nodes_to_add)

    # --- Step 2: Time-slice GRU inputs (first layer of each module) ---
    # The first GRU layer in each module receives input from the encoder
    # with T frames. We slice to T=1 so the GRU only processes the new frame.
    # Subsequent GRU layers already get T=1 from the previous layer's output.
    for module, layers in groups.items():
        first_gru = layers[0]['node']
        insert_gru_input_slice(graph, first_gru, nodes_to_add)
        print(f"  GRU input sliced: {first_gru.name}")

    # --- Step 3: Expose GRU hidden states as model I/O ---
    new_inputs = []
    new_outputs = []

    module_state_names = {
        'enc': ('h0', 'h1'),
        'erb': ('erb_h0', 'erb_h1'),
        'df': ('df_h0', 'df_h1'),
    }

    for module, layers in groups.items():
        if module not in module_state_names:
            print(f"  Skipping unknown module: {module}")
            continue

        h0_name, h1_name = module_state_names[module]
        num_layers = len(layers)
        hidden_size = layers[0]['hidden_size']

        for l in layers:
            assert l['hidden_size'] == hidden_size, \
                f"Mixed hidden sizes in {module}: {[l['hidden_size'] for l in layers]}"

        # Create h0 input: [num_layers, 1, hidden_size]
        h0_input = helper.make_tensor_value_info(
            h0_name, TensorProto.FLOAT, [num_layers, 1, hidden_size]
        )
        new_inputs.append(h0_input)

        final_h_names = []

        for i, layer in enumerate(layers):
            # Slice h0[i:i+1] -> per-layer initial_h [1, 1, H]
            slice_output = f"_patch_{module}_h0_layer{i}"

            starts_name = f"_patch_{module}_starts_{i}"
            ends_name = f"_patch_{module}_ends_{i}"
            axes_name = f"_patch_{module}_axes_{i}"

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
                name=f'_patch_{module}_slice_h0_{i}',
            ))

            # Replace GRU's initial_h input
            gru_node = layer['node']
            while len(gru_node.input) < 6:
                gru_node.input.append('')
            gru_node.input[5] = slice_output

            if layer['final_h']:
                final_h_names.append(layer['final_h'])

        # Create h1 output
        if len(final_h_names) == 1:
            nodes_to_add.append(helper.make_node(
                'Identity', inputs=[final_h_names[0]], outputs=[h1_name],
                name=f'_patch_{module}_h1_identity',
            ))
        else:
            nodes_to_add.append(helper.make_node(
                'Concat', inputs=final_h_names, outputs=[h1_name],
                name=f'_patch_{module}_h1_concat', axis=0,
            ))

        h1_output = helper.make_tensor_value_info(
            h1_name, TensorProto.FLOAT, [num_layers, 1, hidden_size]
        )
        new_outputs.append(h1_output)

        print(f"  State: {h0_name} [{num_layers}, 1, {hidden_size}] -> "
              f"{h1_name} [{num_layers}, 1, {hidden_size}]")

    # Add new nodes, inputs, outputs
    for node in nodes_to_add:
        graph.node.append(node)
    for inp in new_inputs:
        graph.input.append(inp)
    for out in new_outputs:
        graph.output.append(out)

    # Validate
    try:
        onnx.checker.check_model(model)
        print("Model validation passed")
    except Exception as e:
        print(f"WARNING: Model validation: {e}")

    # Save
    if output_path is None:
        output_path = model_path.parent / "combined_streaming.onnx"
    onnx.save(model, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved streaming model to {output_path} ({size_mb:.1f} MB)")

    return model


def verify_model(model_path):
    """Quick verification: run the patched model with onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not available, skipping verification")
        return

    sess = ort.InferenceSession(str(model_path))
    inputs = {i.name: i for i in sess.get_inputs()}
    outputs = {o.name: o for o in sess.get_outputs()}

    print(f"\nVerification - Model I/O:")
    print(f"  Inputs:  {list(inputs.keys())}")
    print(f"  Outputs: {list(outputs.keys())}")

    # Create dummy inputs with T=3 (typical enc_window)
    feed = {}
    for name, inp in inputs.items():
        shape = [d if isinstance(d, int) else 3 for d in inp.shape]
        # h0 inputs have fixed dims, not 'S'
        if 'h0' in name:
            shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        feed[name] = np.zeros(shape, dtype=np.float32)

    result = sess.run(None, feed)
    print(f"  Ran with T=3 input, output shapes: {[r.shape for r in result]}")
    # Verify outputs are T=1 (time-sliced)
    for i, o in enumerate(sess.get_outputs()):
        shape = result[i].shape
        if 'h' not in o.name and len(shape) >= 3:
            t_dim = shape[2] if len(shape) == 4 else shape[1]
            print(f"    {o.name}: T={t_dim} {'(sliced to 1)' if t_dim == 1 else '(WARN: not sliced!)'}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/patch_onnx_streaming.py <model_dir> [model_dir2 ...]")
        print("Example: python scripts/patch_onnx_streaming.py models/dfn3 models/dfn2_ll")
        sys.exit(1)

    for model_dir_str in sys.argv[1:]:
        model_dir = Path(model_dir_str)
        combined_path = model_dir / "combined.onnx"

        if not combined_path.exists():
            print(f"ERROR: {combined_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Patching {combined_path}")
        print(f"{'='*60}")

        patch_model(combined_path)
        verify_model(model_dir / "combined_streaming.onnx")


if __name__ == "__main__":
    main()
