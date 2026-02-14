"""Export fully-stateful DeepFilterNet ONNX models with all GRU hidden states.

Exports 3 GRU hidden states (not just the encoder's):
  - enc h0/h1:  encoder emb_gru  (1 layer, emb_hidden_dim)
  - erb h0/h1:  ERB decoder emb_gru  (emb_num_layers-1 layers, emb_hidden_dim)
  - df  h0/h1:  DF decoder df_gru  (df_num_layers layers, df_hidden_dim)

Install dependencies:
    pip install torch deepfilternet onnx

Usage:
    python scripts/export_onnx_stateful.py --to models/dfn3_h0
    python scripts/export_onnx_stateful.py --from path/to/checkpoint_dir --to models/dfn3_h0

After export, merge into combined.onnx:
    python scripts/merge_onnx.py models/dfn3_h0
"""

import argparse
import os
import shutil
from typing import Optional, Tuple

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


class DfDecoderWithState(torch.nn.Module):
    """DF decoder wrapper that exposes the df_gru hidden state as I/O.

    Matches DFN3 DfDecoder structure:
      df_gru (SqueezedGRU_S) + df_skip -> df_out + df_convp -> coefs
    """

    def __init__(self, df_dec):
        super().__init__()
        self.df_dec = df_dec

    def forward(self, emb, c0, df_h0):
        # emb: [B,T,H], c0: [B,C,T,F], df_h0: GRU hidden state
        b, t, _ = emb.shape

        # Run the df_gru with explicit hidden state
        c, df_h1 = self.df_dec.df_gru(emb, df_h0)
        if self.df_dec.df_skip is not None:
            c = c + self.df_dec.df_skip(emb)

        c0 = self.df_dec.df_convp(c0).permute(0, 2, 3, 1)  # [B, T, F, O*2]
        c = self.df_dec.df_out(c)  # [B, T, F*O*2]
        c = c.view(b, t, self.df_dec.df_bins, self.df_dec.df_out_ch) + c0
        return c, df_h1


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

    spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device="cpu")

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
    enc_conv_path = os.path.join(export_dir, "enc_conv.onnx")
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
    enc_gru_path = os.path.join(export_dir, "enc_gru.onnx")
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
    erb_path = os.path.join(export_dir, "erb_dec.onnx")
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
    df_path = os.path.join(export_dir, "df_dec.onnx")
    print(f"Exporting DF decoder to {df_path} (GRU: {df_num_layers} layers, {df_hidden_dim} hidden)")
    torch.onnx.export(
        df_stateful,
        (emb, c0, df_h0),
        df_path,
        input_names=["emb", "c0", "df_h0"],
        output_names=["coefs", "df_h1"],
        dynamic_axes={
            "emb": {1: "S"},
            "c0": {2: "S"},
            "coefs": {1: "S"},
            "df_h0": {1: "B"},
            "df_h1": {1: "B"},
        },
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
    print(f"  enc:      h0/h1  ({1} layer, {p.emb_hidden_dim} hidden)")
    print(f"  enc_conv: stateless conv-only (for streaming)")
    print(f"  enc_gru:  h0/h1  ({1} layer, {p.emb_hidden_dim} hidden, for streaming)")
    print(f"  erb_dec:  erb_h0/erb_h1  ({erb_num_layers} layers, {erb_hidden_dim} hidden)")
    print(f"  df_dec:   df_h0/df_h1  ({df_num_layers} layers, {df_hidden_dim} hidden)")
    print(f"\nFor combined model: python scripts/merge_onnx.py {export_dir}")
    print(f"For streaming: use enc_conv.onnx + enc_gru.onnx + erb_dec.onnx + df_dec.onnx directly")


if __name__ == "__main__":
    main()
