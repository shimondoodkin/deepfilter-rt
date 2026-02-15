"""Export stateful streaming DeepFilterNet ONNX models with all GRU hidden states.

Creates *_streaming.onnx files that coexist with original stateless models.
Supports both DFN2 (with alpha output) and DFN3.

Exports:
  - enc_conv_streaming.onnx:   encoder convolutions only (stateless, T[-1:] sliced)
  - enc_gru_streaming.onnx:    encoder GRU with h0/h1 state
  - erb_dec_streaming.onnx:    ERB decoder with erb_h0/erb_h1 state I/O
  - df_dec_streaming.onnx:     DF decoder with df_h0/df_h1 state I/O (+alpha for DFN2)
  - enc.onnx:                  full stateful encoder (h0/h1)

Optionally also exports combined_streaming.onnx (single file, ~3x faster inference)
via --combined (direct PyTorch export) or --merge (ONNX graph surgery).

Original enc.onnx, erb_dec.onnx, df_dec.onnx are NOT modified.

Install dependencies:
    pip install torch deepfilternet onnx

Usage:
    python scripts/export_onnx_stateful.py --to models/dfn3_h0
    python scripts/export_onnx_stateful.py --to models/dfn3_h0 --combined
    python scripts/export_onnx_stateful.py --to models/dfn3_h0 --merge
"""

import argparse
import os
import shutil

import torch
import torch.nn.functional as F

from df.enhance import init_df, df_features
from df.model import ModelParams


class EncoderWithState(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, feat_erb, feat_spec, h0):
        # feat_erb: [B,1,T,E], feat_spec: [B,2,T,F]
        e0 = self.enc.erb_conv0(feat_erb)
        e1 = self.enc.erb_conv1(e0)
        e2 = self.enc.erb_conv2(e1)
        e3 = self.enc.erb_conv3(e2)
        c0 = self.enc.df_conv0(feat_spec)
        c1 = self.enc.df_conv1(c0)
        cemb = c1.permute(0, 2, 3, 1).flatten(2)
        cemb = self.enc.df_fc_emb(cemb)
        emb = e3.permute(0, 2, 3, 1).flatten(2)
        emb = self.enc.combine(emb, cemb)
        # Use GRU with state
        emb, h1 = self.enc.emb_gru(emb, h0)
        lsnr = self.enc.lsnr_fc(emb) * self.enc.lsnr_scale + self.enc.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr, h1


class EncoderConvOnly(torch.nn.Module):
    """Encoder conv layers + combine + df_fc_emb, WITHOUT GRU.

    For streaming: run this with T=kernel_t frames to get conv outputs with
    full temporal context. Returns only the LAST time frame (T=1) from each
    output — the convolutions process all T frames internally for context,
    but only the last frame has full receptive field coverage.
    """

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, feat_erb, feat_spec):
        # feat_erb: [B,1,T,E], feat_spec: [B,2,T,F]
        e0 = self.enc.erb_conv0(feat_erb)
        e1 = self.enc.erb_conv1(e0)
        e2 = self.enc.erb_conv2(e1)
        e3 = self.enc.erb_conv3(e2)
        c0 = self.enc.df_conv0(feat_spec)
        c1 = self.enc.df_conv1(c0)
        cemb = c1.permute(0, 2, 3, 1).flatten(2)
        cemb = self.enc.df_fc_emb(cemb)
        emb = e3.permute(0, 2, 3, 1).flatten(2)
        emb = self.enc.combine(emb, cemb)
        # Return only last time frame — convs already have full context
        return (e0[:, :, -1:, :], e1[:, :, -1:, :], e2[:, :, -1:, :],
                e3[:, :, -1:, :], emb[:, -1:, :], c0[:, :, -1:, :])


class EncoderGruOnly(torch.nn.Module):
    """Encoder GRU + lsnr_fc only.

    For streaming: receives 1 frame of pre-GRU embedding, runs GRU with
    persistent hidden state, produces post-GRU embedding + lsnr.
    """

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, emb, h0):
        # emb: [B,T,H] (T=1 for streaming), h0: [L,B,D]
        emb, h1 = self.enc.emb_gru(emb, h0)
        lsnr = self.enc.lsnr_fc(emb) * self.enc.lsnr_scale + self.enc.lsnr_offset
        return emb, lsnr, h1


class ErbDecoderWithState(torch.nn.Module):
    """ERB decoder wrapper that exposes the emb_gru hidden state as I/O.

    Matches DFN3 ErbDecoder structure:
      emb_gru (SqueezedGRU_S) -> conv3p+convt3 -> conv2p+convt2 -> conv1p+convt1 -> conv0p+conv0_out
    """

    def __init__(self, erb_dec):
        super().__init__()
        self.erb_dec = erb_dec

    def forward(self, emb, e3, e2, e1, e0, erb_h0):
        # emb: [B,T,H], e0..e3: encoder skip connections, erb_h0: GRU hidden state
        b, _, t, f8 = e3.shape

        # Run the emb_gru with explicit hidden state
        emb, erb_h1 = self.erb_dec.emb_gru(emb, erb_h0)
        emb = emb.view(b, t, f8, -1).permute(0, 3, 1, 2)  # [B, C*8, T, F/8]

        # Decoder upsampling path (matches DFN3 ErbDecoder.forward)
        e3 = self.erb_dec.convt3(self.erb_dec.conv3p(e3) + emb)
        e2 = self.erb_dec.convt2(self.erb_dec.conv2p(e2) + e3)
        e1 = self.erb_dec.convt1(self.erb_dec.conv1p(e1) + e2)
        m = self.erb_dec.conv0_out(self.erb_dec.conv0p(e0) + e1)
        return m, erb_h1


def _df_dec_has_alpha(df_dec):
    """Detect whether DfDecoder actually uses alpha by checking its forward return type.

    DFN2's forward returns (coefs, alpha), DFN3's returns just coefs.
    Both may have df_fc_a attribute, but only DFN2 uses it in forward().
    """
    try:
        import inspect
        src = inspect.getsource(df_dec.forward)
        # DFN2 returns "return c, alpha" — check for alpha in return statements
        return "alpha" in src and "df_fc_a" in src
    except (OSError, TypeError):
        return False


class DfDecoderWithState(torch.nn.Module):
    """DF decoder wrapper that exposes the df_gru hidden state as I/O.

    Works with both DFN3 (returns coefs only) and DFN2 (returns coefs + alpha).
    Auto-detects DFN2 by inspecting the original forward method.
    """

    def __init__(self, df_dec):
        super().__init__()
        self.df_dec = df_dec
        self.has_alpha = _df_dec_has_alpha(df_dec)

    def forward(self, emb, c0, df_h0):
        # emb: [B,T,H], c0: [B,C,T,F], df_h0: GRU hidden state
        b, t, _ = emb.shape

        # Run the df_gru with explicit hidden state
        c, df_h1 = self.df_dec.df_gru(emb, df_h0)
        if self.df_dec.df_skip is not None:
            c = c + self.df_dec.df_skip(emb)

        c0 = self.df_dec.df_convp(c0).permute(0, 2, 3, 1)  # [B, T, F, O*2]

        if self.has_alpha:
            # DFN2: also generate alpha (importance weight)
            alpha = self.df_dec.df_fc_a(c)

        c = self.df_dec.df_out(c)  # [B, T, F*O*2]
        c = c.view(b, t, self.df_dec.df_bins, self.df_dec.df_out_ch) + c0

        if self.has_alpha:
            return c, alpha, df_h1
        return c, df_h1


class CombinedStreamingModel(torch.nn.Module):
    """Full streaming pipeline in a single model: enc_conv + enc_gru + erb_dec + df_dec.

    Produces combined_streaming.onnx — a single ONNX file with all GRU states as I/O.
    ~3x faster than 4 separate sessions since ONNX runtime only runs 1 session per frame.
    """

    def __init__(self, enc, erb_dec, df_dec):
        super().__init__()
        self.enc_conv = EncoderConvOnly(enc)
        self.enc_gru = EncoderGruOnly(enc)
        self.erb_dec = ErbDecoderWithState(erb_dec)
        self.df_dec = DfDecoderWithState(df_dec)
        self.has_alpha = self.df_dec.has_alpha

    def forward(self, feat_erb, feat_spec, h0, erb_h0, df_h0):
        e0, e1, e2, e3, emb, c0 = self.enc_conv(feat_erb, feat_spec)
        emb, lsnr, h1 = self.enc_gru(emb, h0)
        m, erb_h1 = self.erb_dec(emb, e3, e2, e1, e0, erb_h0)
        df_out = self.df_dec(emb, c0, df_h0)
        if self.has_alpha:
            coefs, alpha, df_h1 = df_out
            return m, coefs, alpha, lsnr, h1, erb_h1, df_h1
        else:
            coefs, df_h1 = df_out
            return m, coefs, lsnr, h1, erb_h1, df_h1


def main():
    parser = argparse.ArgumentParser(
        description="Export fully-stateful DeepFilterNet ONNX models with all GRU hidden states."
    )
    parser.add_argument(
        "--from", dest="source", default=None,
        help="Source model/checkpoint directory (default: uses DeepFilterNet cached model)"
    )
    parser.add_argument(
        "--to", required=True,
        help="Output directory for enc.onnx, erb_dec.onnx, df_dec.onnx, config.ini"
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="Also export combined_streaming.onnx directly (single ONNX file, ~3x faster)"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Create combined_streaming.onnx via ONNX graph surgery (alternative to --combined)"
    )
    args = parser.parse_args()

    export_dir = args.to
    os.makedirs(export_dir, exist_ok=True)

    if args.source:
        print(f"Loading model from: {args.source}")
        model, df_state, _ = init_df(model_base_dir=args.source)
    else:
        print("Loading model from DeepFilterNet cache")
        model, df_state, _ = init_df(model_base_dir=None)
    model.eval()

    p = ModelParams()

    # Synthetic silence is sufficient for ONNX tracing (shapes are the same for any audio)
    audio = torch.zeros(1, 48000)

    audio = F.pad(audio, (0, df_state.fft_size()))

    _spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device="cpu")

    # Encoder export expects feat_spec with re/im in channel axis
    feat_spec_enc = feat_spec.transpose(1, 4).squeeze(4)

    # --- Encoder (stateful: h0/h1) ---
    enc_stateful = EncoderWithState(model.enc)
    h0 = torch.zeros((1, feat_erb.shape[0], p.emb_hidden_dim))

    enc_path = os.path.join(export_dir, "enc.onnx")
    print(f"Exporting encoder to {enc_path}")
    torch.onnx.export(
        enc_stateful,
        (feat_erb, feat_spec_enc, h0),
        enc_path,
        input_names=["feat_erb", "feat_spec", "h0"],
        output_names=["e0", "e1", "e2", "e3", "emb", "c0", "lsnr", "h1"],
        dynamic_axes={
            "feat_erb": {2: "S"},
            "feat_spec": {2: "S"},
            "e0": {2: "S"},
            "e1": {2: "S"},
            "e2": {2: "S"},
            "e3": {2: "S"},
            "emb": {1: "S"},
            "c0": {2: "S"},
            "lsnr": {1: "S"},
            "h0": {1: "B"},
            "h1": {1: "B"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # --- Split encoder: conv-only (stateless) ---
    enc_conv = EncoderConvOnly(model.enc)
    enc_conv_path = os.path.join(export_dir, "enc_conv_streaming.onnx")
    print(f"Exporting encoder conv-only to {enc_conv_path}")
    torch.onnx.export(
        enc_conv,
        (feat_erb, feat_spec_enc),
        enc_conv_path,
        input_names=["feat_erb", "feat_spec"],
        output_names=["e0", "e1", "e2", "e3", "emb", "c0"],
        dynamic_axes={
            # Inputs have dynamic T for temporal context
            "feat_erb": {2: "S"},
            "feat_spec": {2: "S"},
            # Outputs are always T=1 (last frame only) — no dynamic axes
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # --- Split encoder: GRU-only (stateful h0/h1) ---
    enc_gru_mod = EncoderGruOnly(model.enc)
    # Run conv to get the pre-GRU embedding shape
    with torch.no_grad():
        _, _, _, _, pre_gru_emb, _ = enc_conv(feat_erb, feat_spec_enc)
    # For streaming: T=1 frame
    pre_gru_emb_1 = pre_gru_emb[:, :1, :]  # [B, 1, H]
    enc_gru_path = os.path.join(export_dir, "enc_gru_streaming.onnx")
    print(f"Exporting encoder GRU-only to {enc_gru_path}")
    torch.onnx.export(
        enc_gru_mod,
        (pre_gru_emb_1, h0),
        enc_gru_path,
        input_names=["emb", "h0"],
        output_names=["emb_out", "lsnr", "h1"],
        dynamic_axes={
            "emb": {1: "S"},
            "emb_out": {1: "S"},
            "lsnr": {1: "S"},
            "h0": {1: "B"},
            "h1": {1: "B"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Run enc to get inputs for decoders
    with torch.no_grad():
        e0, e1, e2, e3, emb, c0, _, _ = enc_stateful(feat_erb, feat_spec_enc, h0)

    # --- ERB decoder (stateful: erb_h0/erb_h1) ---
    # Determine GRU dimensions
    erb_gru = model.erb_dec.emb_gru
    # For SqueezedGRU/SqueezedGRU_S, the inner nn.GRU is at .gru
    if hasattr(erb_gru, 'gru'):
        erb_num_layers = erb_gru.gru.num_layers
        erb_hidden_dim = erb_gru.gru.hidden_size
    else:
        # GroupedGRU
        erb_num_layers = erb_gru.num_layers
        erb_hidden_dim = erb_gru.hidden_size
    erb_h0 = torch.zeros(erb_num_layers, emb.shape[0], erb_hidden_dim)

    erb_stateful = ErbDecoderWithState(model.erb_dec)
    erb_path = os.path.join(export_dir, "erb_dec_streaming.onnx")
    print(f"Exporting ERB decoder to {erb_path} (GRU: {erb_num_layers} layers, {erb_hidden_dim} hidden)")
    torch.onnx.export(
        erb_stateful,
        (emb, e3, e2, e1, e0, erb_h0),
        erb_path,
        input_names=["emb", "e3", "e2", "e1", "e0", "erb_h0"],
        output_names=["m", "erb_h1"],
        dynamic_axes={
            "emb": {1: "S"},
            "e3": {2: "S"},
            "e2": {2: "S"},
            "e1": {2: "S"},
            "e0": {2: "S"},
            "m": {2: "S"},
            "erb_h0": {1: "B"},
            "erb_h1": {1: "B"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # --- DF decoder (stateful: df_h0/df_h1) ---
    df_gru = model.df_dec.df_gru
    if hasattr(df_gru, 'gru'):
        df_num_layers = df_gru.gru.num_layers
        df_hidden_dim = df_gru.gru.hidden_size
    else:
        df_num_layers = df_gru.num_layers
        df_hidden_dim = df_gru.hidden_size
    df_h0 = torch.zeros(df_num_layers, emb.shape[0], df_hidden_dim)

    df_stateful = DfDecoderWithState(model.df_dec)
    has_alpha = df_stateful.has_alpha
    df_path = os.path.join(export_dir, "df_dec_streaming.onnx")
    alpha_info = ", +alpha" if has_alpha else ""
    print(f"Exporting DF decoder to {df_path} (GRU: {df_num_layers} layers, {df_hidden_dim} hidden{alpha_info})")

    output_names = ["coefs", "alpha", "df_h1"] if has_alpha else ["coefs", "df_h1"]
    dynamic_axes = {
        "emb": {1: "S"},
        "c0": {2: "S"},
        "coefs": {1: "S"},
        "df_h0": {1: "B"},
        "df_h1": {1: "B"},
    }
    if has_alpha:
        dynamic_axes["alpha"] = {1: "S"}

    torch.onnx.export(
        df_stateful,
        (emb, c0, df_h0),
        df_path,
        input_names=["emb", "c0", "df_h0"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    # Copy config.ini — prefer source dir, fall back to DeepFilterNet cache
    cfg_dst = os.path.join(export_dir, "config.ini")
    cfg_copied = False
    if args.source:
        cfg_src = os.path.join(args.source, "config.ini")
        if os.path.isfile(cfg_src):
            shutil.copyfile(cfg_src, cfg_dst)
            print(f"Copied config.ini from {cfg_src}")
            cfg_copied = True

    if not cfg_copied:
        cache_dir = os.path.join(
            os.path.expanduser("~"),
            "AppData",
            "Local",
            "DeepFilterNet",
            "DeepFilterNet",
            "Cache",
            "DeepFilterNet3",
        )
        cfg_src = os.path.join(cache_dir, "config.ini")
        if os.path.isfile(cfg_src):
            shutil.copyfile(cfg_src, cfg_dst)
            print(f"Copied config.ini from {cfg_src}")
        else:
            print(f"Warning: config.ini not found at {cfg_src}")

    print(f"\nExported fully-stateful ONNX models to {export_dir}")
    print(f"  enc.onnx:                  h0/h1  ({1} layer, {p.emb_hidden_dim} hidden)")
    print(f"  enc_conv_streaming.onnx:   stateless conv-only")
    print(f"  enc_gru_streaming.onnx:    h0/h1  ({1} layer, {p.emb_hidden_dim} hidden)")
    print(f"  erb_dec_streaming.onnx:    erb_h0/erb_h1  ({erb_num_layers} layers, {erb_hidden_dim} hidden)")
    alpha_str = " + alpha" if has_alpha else ""
    print(f"  df_dec_streaming.onnx:     df_h0/df_h1  ({df_num_layers} layers, {df_hidden_dim} hidden{alpha_str})")
    print(f"\nOriginal enc.onnx, erb_dec.onnx, df_dec.onnx are NOT modified.")

    # Optionally export combined_streaming.onnx
    if args.combined:
        combined_mod = CombinedStreamingModel(model.enc, model.erb_dec, model.df_dec)
        combined_mod.eval()

        # Get kernel_t for conv context window
        kernel_t = 2
        try:
            w = model.enc.erb_conv0
            if hasattr(w, 'conv'):
                kernel_t = w.conv.kernel_size[0]
        except Exception:
            pass

        erb_trace = feat_erb[:, :, :kernel_t, :]
        spec_trace = feat_spec_enc[:, :, :kernel_t, :]

        if has_alpha:
            comb_outputs = ["m", "coefs", "alpha", "lsnr", "h1", "erb_h1", "df_h1"]
        else:
            comb_outputs = ["m", "coefs", "lsnr", "h1", "erb_h1", "df_h1"]

        comb_path = os.path.join(export_dir, "combined_streaming.onnx")
        print(f"\nExporting combined model to {comb_path}")
        torch.onnx.export(
            combined_mod,
            (erb_trace, spec_trace, h0, erb_h0, df_h0),
            comb_path,
            input_names=["feat_erb", "feat_spec", "h0", "erb_h0", "df_h0"],
            output_names=comb_outputs,
            dynamic_axes={
                "feat_erb": {2: "S"},
                "feat_spec": {2: "S"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        size_mb = os.path.getsize(comb_path) / (1024 * 1024)
        print(f"  Saved {comb_path} ({size_mb:.1f} MB)")

    elif args.merge:
        try:
            from pathlib import Path
            from merge_split_models import merge_split_models
            import onnx

            print(f"\nMerging split models into combined_streaming.onnx...")
            combined_graph = merge_split_models(Path(export_dir))
            out_path = os.path.join(export_dir, "combined_streaming.onnx")
            onnx.save(combined_graph, out_path)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"  Saved {out_path} ({size_mb:.1f} MB)")
        except ImportError as e:
            print(f"\nCould not merge: {e}")
            print(f"Run manually: python scripts/merge_split_models.py {export_dir}")

    if not args.combined and not args.merge:
        print(f"\nFor combined: --combined (direct) or --merge (ONNX surgery)")


if __name__ == "__main__":
    main()
