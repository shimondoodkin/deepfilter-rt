"""Export stateful (GRU) DeepFilterNet ONNX models with h0/h1 hidden state.

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


def main():
    parser = argparse.ArgumentParser(
        description="Export stateful (GRU) DeepFilterNet ONNX models with h0/h1 hidden state."
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
        model, df_state, _, _ = init_df(model_base_dir=args.source)
    else:
        print("Loading model from DeepFilterNet cache")
        model, df_state, _, _ = init_df(model_base_dir=None)
    model.eval()

    p = ModelParams()

    # Synthetic silence is sufficient for ONNX tracing (shapes are the same for any audio)
    audio = torch.zeros(1, 48000)

    audio = F.pad(audio, (0, df_state.fft_size()))

    spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device="cpu")

    # Encoder export expects feat_spec with re/im in channel axis
    feat_spec_enc = feat_spec.transpose(1, 4).squeeze(4)

    # Stateful encoder
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

    # Run enc to get inputs for decoders
    with torch.no_grad():
        e0, e1, e2, e3, emb, c0, _, _ = enc_stateful(feat_erb, feat_spec_enc, h0)

    # ERB decoder
    erb_path = os.path.join(export_dir, "erb_dec.onnx")
    print(f"Exporting ERB decoder to {erb_path}")
    torch.onnx.export(
        model.erb_dec,
        (emb, e3, e2, e1, e0),
        erb_path,
        input_names=["emb", "e3", "e2", "e1", "e0"],
        output_names=["m"],
        dynamic_axes={
            "emb": {1: "S"},
            "e3": {2: "S"},
            "e2": {2: "S"},
            "e1": {2: "S"},
            "e0": {2: "S"},
            "m": {2: "S"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # DF decoder
    df_path = os.path.join(export_dir, "df_dec.onnx")
    print(f"Exporting DF decoder to {df_path}")
    torch.onnx.export(
        model.df_dec,
        (emb, c0),
        df_path,
        input_names=["emb", "c0"],
        output_names=["coefs"],
        dynamic_axes={
            "emb": {1: "S"},
            "c0": {2: "S"},
            "coefs": {1: "S"},
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

    print(f"\nExported stateful ONNX models to {export_dir}")
    print(f"Next step: python scripts/merge_onnx.py {export_dir}")


if __name__ == "__main__":
    main()
