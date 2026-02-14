● Search(pattern: "rolling", path: "/c/Users/user/Documents/projects/aiphone/deepfilter\_rt/src/lib.rs")

&nbsp; ⎿  Found 28 lines (ctrl+o to expand)



&nbsp;                                                                                       04:20 PM claude-opus-4-6

● No. rolling.rs exists and is declared as a module, but it's behind #\[cfg(feature = "rolling")] and nothing in the main

&nbsp;  code uses it. The main inference path uses ONNX Runtime for everything (GRUs included). The rolling\_spec\_buf\_x/y

&nbsp; fields in lib.rs are just VecDeque buffers for the deep filtering step — not from the rolling module.



● Bash(wc -l /c/Users/user/Documents/projects/aiphone/deepfilter\_rt/src/rolling.rs)

&nbsp; ⎿  747 /c/Users/user/Documents/projects/aiphone/deepfilter\_rt/src/rolling.rs



&nbsp;                                                                                       04:21 PM claude-opus-4-6

● 747 lines of code that's never used. With the ONNX graph patching approach working well, the rolling module (manual

&nbsp; GRU/STFT/ISTFT in Rust) isn't needed — all the state management is handled by the patched ONNX model directly.



&nbsp; Want me to remove it, or keep it around as a potential future option (e.g., for platforms without ONNX Runtime)?



❯ add comment that it is not used and mention the patched models



● Read(~\\Documents\\projects\\aiphone\\deepfilter\_rt\\src\\rolling.rs)

&nbsp; ⎿  Read 15 lines



● Read(~\\Documents\\projects\\aiphone\\deepfilter\_rt\\src\\lib.rs)

&nbsp; ⎿  Read 8 lines



● Update(C:\\Users\\user\\Documents\\projects\\aiphone\\deepfilter\_rt\\src\\rolling.rs)

&nbsp; ⎿  Added 9 lines

&nbsp;      1  //! Rolling (streaming) implementations for STFT, ISTFT, GRU, and normalization.

&nbsp;      2  //!

&nbsp;      3 +//! \*\*NOTE:\*\* This module is currently unused. The main inference path uses ONNX

&nbsp;      4 +//! Runtime with patched streaming models (`combined\_streaming.onnx`) that expose

&nbsp;      5 +//! GRU hidden states as model I/O and time-slice the decoder path to T=1. See

&nbsp;      6 +//! `scripts/patch\_onnx\_streaming.py` and `MODES.md` for details.

&nbsp;      7 +//!

&nbsp;      8 +//! This module remains available behind `#\[cfg(feature = "rolling")]` as a

&nbsp;      9 +//! potential alternative for platforms without ONNX Runtime or for fully

&nbsp;     10 +//! custom inference pipelines.

&nbsp;     11 +//!

&nbsp;     12  //! All types in this module are designed for zero heap allocation per frame:

&nbsp;     13  //! all memory is pre-allocated at construction time. This makes them suitable

&nbsp;     14  //! for real-time audio processing where allocation jitter is unacceptable.

&nbsp; ⎿  1 PostToolUse hook ran



● Update(C:\\Users\\user\\Documents\\projects\\aiphone\\deepfilter\_rt\\src\\lib.rs)

&nbsp; ⎿  Added 2 lines

&nbsp;     115  //! Each processor instance is independent and `Send`. Create separate instances

&nbsp;     116  //! for parallel processing - they do not share state.

&nbsp;     117

&nbsp;     118 +/// Currently unused — streaming is handled by patched ONNX models

&nbsp;     119 +/// (`combined\_streaming.onnx`). See `scripts/patch\_onnx\_streaming.py`.

&nbsp;     120  #\[cfg(feature = "rolling")]

&nbsp;     121  pub mod rolling;

&nbsp;     122

&nbsp; ⎿  1 PostToolUse hook ran



&nbsp;                                                                                       04:22 PM claude-opus-4-6

● Done. Added comments in both places explaining that rolling.rs is unused and pointing to the patched ONNX approach

&nbsp; instead.



❯ write file document in details how python model works what you have learned



∴ Thinking…



&nbsp; The user wants me to write a detailed document about how the Python DeepFilterNet model works, based on everything

&nbsp; I've learned during this project. Let me compile all my knowledge about the architecture, forward pass, padding, GRU

&nbsp; states, etc.



● Explore(Explore DFN3 Python model details)

&nbsp; ⎿  Prompt:

&nbsp;      In C:\\Users\\user\\Documents\\projects\\aiphone\\deepfilter\_rt\\DeepFilterNet, thoroughly read and report on the

&nbsp;      DeepFilterNet3 Python model implementation. I need detailed information for documentation. Check these

&nbsp;      files:



&nbsp;      1. DeepFilterNet/df/deepfilternet3.py — Read the FULL file. Report:

&nbsp;        - DfNet class: forward() method step by step

&nbsp;        - Encoder class: all conv layers, combine, emb\_gru, lsnr\_fc

&nbsp;        - ErbDecoder class: emb\_gru, conv3p/convt3, conv2p/convt2, conv1p/convt1, conv0p/conv0\_out

&nbsp;        - DfDecoder class: df\_gru, df\_skip, df\_convp, df\_out

&nbsp;        - pad\_feat and pad\_spec — what they do exactly

&nbsp;        - All parameter shapes and dimensions

&nbsp;      2. DeepFilterNet/df/model.py — ModelParams class, all config parameters and defaults

&nbsp;      3. DeepFilterNet/df/modules.py — SqueezedGRU\_S class, how it wraps nn.GRU with linear\_in/linear\_out/skip

&nbsp;      4. libDF/src/lib.rs — The Rust DFState: frame\_analysis(), frame\_synthesis(), feat\_erb(), feat\_cplx(),

&nbsp;      apply\_erb\_gains(), deep\_filter()

&nbsp;        - How STFT/ISTFT works

&nbsp;        - How ERB bands are computed

&nbsp;        - How deep filtering coefficients are applied



&nbsp;      Report all shapes, dimensions, and the complete data flow from raw audio to enhanced audio.

&nbsp; ⎿  Read(C:\\Users\\user\\Documents\\projects\\aiphone\\deepfilter\_rt\\DeepFilterNet\\DeepFilterNet\\df\\deepfilternet3.py)



&nbsp; ⎿  Read(C:\\Users\\user\\Documents\\projects\\aiphone\\deepfilter\_rt\\DeepFilterNet\\DeepFilterNet\\df\\model.py)



&nbsp; ⎿  Read(C:\\Users\\user\\Documents\\projects\\aiphone\\deepfilter\_rt\\DeepFilterNet\\DeepFilterNet\\df\\modules.py)



&nbsp; ⎿  Read(C:\\Users\\user\\Documents\\projects\\aiphone\\deepfilter\_rt\\libDF\\src\\lib.rs)



&nbsp; ⎿  Error: File does not exist.



&nbsp; ⎿  Error: Sibling tool call errored



&nbsp; ⎿  Bash(find "C:\\Users\\user\\Documents\\projects\\aiphone\\deepfilter\_rt" -name "modules.py" -o -name "lib.rs"

&nbsp;        2>/dev/null | head -20)

&nbsp; ⎿  Read(C:\\Users\\user\\Documents\\projects\\aiphone\\deepfilter\_rt\\DeepFilterNet\\DeepFilterNet\\df\\modules.py)



&nbsp; ⎿  Read(C:\\Users\\user\\Documents\\projects\\aiphone\\deepfilter\_rt\\DeepFilterNet\\libDF\\src\\lib.rs)



&nbsp; ⎿  Search(pattern: "nb\_erb|nb\_df|fft\_size|hop\_size|df\_order|lsnr\_min|lsnr\_max")



&nbsp; ⎿  Read(C:\\Users\\user\\Documents\\projects\\aiphone\\deepfilter\_rt\\DeepFilterNet\\DeepFilterNet\\df\\config.py · lines

&nbsp;        1-150)

&nbsp; ⎿  Read(C:\\Users\\user\\Documents\\projects\\aiphone\\deepfilter\_rt\\DeepFilterNet\\DeepFilterNet\\df\\multiframe.py ·

&nbsp;        lines 1-100)

&nbsp; ⎿  Response:

&nbsp;      Excellent! Now I have all the detailed information. Let me compile a comprehensive report on the

&nbsp;      DeepFilterNet3 implementation.



&nbsp;      Comprehensive DeepFilterNet3 Implementation Report



&nbsp;      I have thoroughly analyzed the DeepFilterNet3 implementation across Python and Rust components. Here is the

&nbsp;      complete technical documentation:



&nbsp;      ---

&nbsp;      1. DfNet Class - Forward Method Data Flow



&nbsp;      File: /DeepFilterNet/df/deepfilternet3.py (Lines 334-456)



&nbsp;      Forward Method Overview



&nbsp;      Input Shapes:

&nbsp;        spec: \[B, 1, T, F, 2]         # Enhanced spectrum (real valued, real/imag parts)

&nbsp;        feat\_erb: \[B, 1, T, E]        # ERB features (dB scale, normalized)

&nbsp;        feat\_spec: \[B, 1, T, F', 2]   # Complex spectrogram features



&nbsp;      Output Shapes:

&nbsp;        spec\_e: \[B, 1, T, F, 2]       # Enhanced spectrum

&nbsp;        m: \[B, 1, T, E]               # ERB mask estimate

&nbsp;        lsnr: \[B, T, 1]               # Local SNR estimate (dB)

&nbsp;        df\_coefs: \[B, T, F\_df, O\*2]   # Deep filter coefficients



&nbsp;      Complete Data Flow (Step-by-Step):



&nbsp;      1. Input Processing:

&nbsp;        - feat\_spec squeezed and permuted: \[B, 1, T, F', 2] → \[B, F', T, C]

&nbsp;        - Lookahead padding applied if conv\_lookahead > 0:

&nbsp;            - pad\_feat: ConstantPad2d padding temporal dimension with -conv\_lookahead to +conv\_lookahead

&nbsp;          - pad\_spec: ConstantPad3d for DF lookahead (if df\_lookahead > 0)

&nbsp;      2. Encoder Stage:

&nbsp;      e0, e1, e2, e3, emb, c0, lsnr = enc(feat\_erb, feat\_spec)

&nbsp;      2. See detailed Encoder section below.

&nbsp;      3. Optional LSNR Dropout Stage:

&nbsp;        - If lsnr\_dropout=True and SNR < -10dB:

&nbsp;            - Zeroed outputs for low-SNR frames

&nbsp;          - Tensors sliced to only process high-SNR frames

&nbsp;      4. ERB Decoder Stage (if run\_erb=True):

&nbsp;      m = erb\_dec(emb, e3, e2, e1, e0)  # Shape: \[B, 1, T, E]

&nbsp;      spec\_m = mask(spec, m)             # Apply ERB mask to spectrum

&nbsp;      4. Outputs ERB mask in \[0, 1] range via sigmoid activation.

&nbsp;      5. Deep Filter (DF) Stage (if run\_df=True):

&nbsp;        - df\_coefs = df\_dec(emb, c0) → \[B, T, F\_df, O\*2]

&nbsp;        - df\_coefs = df\_out\_transform(df\_coefs) → \[B, O, T, F\_df, 2]

&nbsp;        - Complex multiplication applied: spec\_e = df\_op(spec, df\_coefs)

&nbsp;            - Only first nb\_df frequency bins processed

&nbsp;          - Rest filled from ERB masked spectrum: spec\_e\[..., nb\_df:, :] = spec\_m\[..., nb\_df:, :]

&nbsp;      6. Post-Filter Stage (optional, if post\_filter=True):

&nbsp;        - Applies Valin et al. perceptual post-filter

&nbsp;        - Gain computation: pf = (1 + β) / (1 + β \* (mask / mask\_sin)²)

&nbsp;        - Applied: spec\_e = spec\_e \* pf.unsqueeze(-1)

&nbsp;        - Default pf\_beta=0.02



&nbsp;      ---

&nbsp;      2. Encoder Class - Architecture



&nbsp;      File: /DeepFilterNet/df/deepfilternet3.py (Lines 100-185)



&nbsp;      Convolutional Layers



&nbsp;      The Encoder processes two parallel streams:



&nbsp;      ERB Stream (feat\_erb: \[B, 1, T, E]):

&nbsp;      erb\_conv0: Conv2dNormAct(1 → C, kernel=conv\_kernel\_inp (3,3), separable=True, no stride)

&nbsp;                 → \[B, C, T, E]



&nbsp;      erb\_conv1: Conv2dNormAct(C → C, kernel=conv\_kernel (1,3), fstride=2, separable=True)

&nbsp;                 → \[B, C, T, E/2]



&nbsp;      erb\_conv2: Conv2dNormAct(C → C, kernel=conv\_kernel (1,3), fstride=2, separable=True)

&nbsp;                 → \[B, C, T, E/4]



&nbsp;      erb\_conv3: Conv2dNormAct(C → C, kernel=conv\_kernel (1,3), fstride=1, separable=True)

&nbsp;                 → \[B, C, T, E/4]



&nbsp;      Complex Spectrogram Stream (feat\_spec: \[B, 2, T, Fc]):

&nbsp;      df\_conv0: Conv2dNormAct(2 → C, kernel=conv\_kernel\_inp (3,3), separable=True, no stride)

&nbsp;                → \[B, C, T, Fc]



&nbsp;      df\_conv1: Conv2dNormAct(C → C, kernel=conv\_kernel (1,3), fstride=2, separable=True)

&nbsp;                → \[B, C, T, Fc/2]



&nbsp;      Default Parameters:

&nbsp;      - conv\_ch=16 (base channel count)

&nbsp;      - conv\_kernel=(1, 3) (temporal, frequency kernel)

&nbsp;      - conv\_kernel\_inp=(3, 3)

&nbsp;      - Both streams use separable convolutions (depthwise + pointwise)



&nbsp;      Embedding Combination



&nbsp;      cemb = df\_conv1 output permuted + flattened

&nbsp;             \[B, C, T, Fc/2] → \[B, T, C\*Fc/2]



&nbsp;      cemb = df\_fc\_emb(cemb)  # GroupedLinearEinsum with lin\_groups=16

&nbsp;             \[B, T, C\*Fc/2] → \[B, T, C\*E/4]



&nbsp;      emb = e3 permuted + flattened

&nbsp;            \[B, C, T, E/4] → \[B, T, C\*E/4]



&nbsp;      combine:

&nbsp;        If enc\_concat=True:  emb = concat(emb, cemb)  → \[B, T, 2\*C\*E/4]

&nbsp;        If enc\_concat=False: emb = emb + cemb         → \[B, T, C\*E/4]



&nbsp;      Embedding GRU



&nbsp;      emb\_gru: SqueezedGRU\_S(

&nbsp;          input\_size = emb\_in\_dim    (C\*E/4, or 2\*C\*E/4 if concat)

&nbsp;          hidden\_size = emb\_dim      (256)

&nbsp;          output\_size = emb\_out\_dim  (C\*E/4)

&nbsp;          num\_layers = 1

&nbsp;          gru\_skip\_op = None/"identity"/"groupedlinear"

&nbsp;          linear\_groups = 1

&nbsp;      )



&nbsp;      SqueezedGRU\_S Structure:

&nbsp;      1. linear\_in: GroupedLinearEinsum(emb\_in\_dim → hidden\_size) + ReLU

&nbsp;      2. gru: nn.GRU(hidden\_size → hidden\_size, num\_layers=1, batch\_first=True)

&nbsp;      3. linear\_out: GroupedLinearEinsum(hidden\_size → output\_size) + ReLU

&nbsp;      4. gru\_skip: Optional skip connection applied AFTER linear\_out



&nbsp;      Output: emb → \[B, T, C\*E/4]



&nbsp;      Local SNR Estimation



&nbsp;      lsnr\_fc: Sequential(

&nbsp;          Linear(emb\_out\_dim=C\*E/4 → 1),

&nbsp;          Sigmoid()

&nbsp;      )

&nbsp;      lsnr = lsnr\_fc(emb) \* lsnr\_scale + lsnr\_offset

&nbsp;             \[B, T, 1] → scale by (lsnr\_max - lsnr\_min) + add lsnr\_min



&nbsp;      Default: lsnr\_min=-15dB, lsnr\_max=35dB



&nbsp;      Summary of Key Dimensions



&nbsp;      ┌────────────────┬───────────────────────────────┐

&nbsp;      │   Parameter    │         Default Value         │

&nbsp;      ├────────────────┼───────────────────────────────┤

&nbsp;      │ nb\_erb         │ 32 bands                      │

&nbsp;      ├────────────────┼───────────────────────────────┤

&nbsp;      │ conv\_ch (C)    │ 16 channels                   │

&nbsp;      ├────────────────┼───────────────────────────────┤

&nbsp;      │ emb\_hidden\_dim │ 256                           │

&nbsp;      ├────────────────┼───────────────────────────────┤

&nbsp;      │ emb\_out\_dim    │ C × E / 4 = 16 × 32 / 4 = 128 │

&nbsp;      ├────────────────┼───────────────────────────────┤

&nbsp;      │ emb\_in\_dim     │ 128 (or 256 if concat)        │

&nbsp;      └────────────────┴───────────────────────────────┘



&nbsp;      ---

&nbsp;      3. ErbDecoder Class - Architecture



&nbsp;      File: /DeepFilterNet/df/deepfilternet3.py (Lines 188-254)



&nbsp;      Embedding GRU Processing



&nbsp;      emb\_gru: SqueezedGRU\_S(

&nbsp;          input\_size = C\*E/4      (128)

&nbsp;          hidden\_size = 256

&nbsp;          output\_size = C\*E/4     (128)

&nbsp;          num\_layers = emb\_num\_layers - 1  (default: 1)

&nbsp;          gru\_skip\_op = None/"identity"/"groupedlinear"

&nbsp;          linear\_groups = 1

&nbsp;      )



&nbsp;      Input: emb \[B, T, 128]

&nbsp;      Output: emb \[B, T, 128]



&nbsp;      Forward Sequence in ErbDecoder:

&nbsp;      1. Reshape embeddings: emb \[B, T, CE/4] → \[B, T, E/8, C2] → \[B, C\*2, T, E/8]

&nbsp;      2. At each decoder level, process and upsample



&nbsp;      Convolutional Pathway (Skip Connections from Encoder)



&nbsp;      All pathway convolutions are Conv2dNormAct with kernel\_size=1 (1x1 conv):



&nbsp;      Level 3 (E/8):

&nbsp;        conv3p: Conv2dNormAct(C → C, kernel=1x1)

&nbsp;        convt3: Conv2dNormAct(C → C, kernel=conv\_kernel (1,3), no stride)



&nbsp;      Level 2 (E/4):

&nbsp;        conv2p: Conv2dNormAct(C → C, kernel=1x1)

&nbsp;        convt2: ConvTranspose2dNormAct(C → C, kernel=conv\_kernel (1,3), fstride=2)



&nbsp;      Level 1 (E/2):

&nbsp;        conv1p: Conv2dNormAct(C → C, kernel=1x1)

&nbsp;        convt1: ConvTranspose2dNormAct(C → C, kernel=conv\_kernel (1,3), fstride=2)



&nbsp;      Level 0 (E):

&nbsp;        conv0p: Conv2dNormAct(C → C, kernel=1x1)

&nbsp;        conv0\_out: Conv2dNormAct(C → 1, kernel=conv\_kernel (1,3), activation=Sigmoid)



&nbsp;      Forward Pass (Iterative Upsampling)



&nbsp;      # Input: emb \[B, T, 128], e3 \[B, C, T, E/4], e2 \[B, C, T, E/2], e1 \[B, C, T, E], e0 \[B, C, T, E]



&nbsp;      # Step 1: GRU processing

&nbsp;      emb, \_ = emb\_gru(emb)  # \[B, T, 128]

&nbsp;      emb = emb.view(b, t, E/8, C\*2).permute(0, 3, 1, 2)  # \[B, C\*2, T, E/8]



&nbsp;      # Step 2: Level 3 (E/8 → E/4)

&nbsp;      skip3 = conv3p(e3)                    # \[B, C, T, E/4]

&nbsp;      e3\_out = convt3(skip3 + emb)          # \[B, C, T, E/4]



&nbsp;      # Step 3: Level 2 (E/4 → E/2)

&nbsp;      skip2 = conv2p(e2)                    # \[B, C, T, E/2]

&nbsp;      e2\_out = convt2(skip2 + e3\_out)       # \[B, C, T, E/2]



&nbsp;      # Step 4: Level 1 (E/2 → E)

&nbsp;      skip1 = conv1p(e1)                    # \[B, C, T, E]

&nbsp;      e1\_out = convt1(skip1 + e2\_out)       # \[B, C, T, E]



&nbsp;      # Step 5: Level 0 (E → mask)

&nbsp;      skip0 = conv0p(e0)                    # \[B, C, T, E]

&nbsp;      m = conv0\_out(skip0 + e1\_out)         # \[B, 1, T, E] with sigmoid activation



&nbsp;      Output: ERB mask m ∈ \[0, 1] of shape \[B, 1, T, E]



&nbsp;      ---

&nbsp;      4. DfDecoder Class - Architecture



&nbsp;      File: /DeepFilterNet/df/deepfilternet3.py (Lines 278-331)



&nbsp;      Deep Filter GRU



&nbsp;      df\_gru: SqueezedGRU\_S(

&nbsp;          input\_size = emb\_in\_dim      (128 = C\*E/4)

&nbsp;          hidden\_size = df\_hidden\_dim  (256 default)

&nbsp;          output\_size = None           (outputs hidden\_size)

&nbsp;          num\_layers = df\_num\_layers   (3 default)

&nbsp;          gru\_skip\_op = None           (no skip in GRU itself)

&nbsp;          linear\_groups = 1

&nbsp;      )



&nbsp;      Forward: c, \_ = df\_gru(emb)  # \[B, T, 256]



&nbsp;      Skip Connection (Optional)



&nbsp;      df\_skip: None / Identity / GroupedLinearEinsum

&nbsp;        If groupedlinear: GroupedLinearEinsum(C\*E/4 → 256, groups=lin\_groups)



&nbsp;      Applied: c = c + df\_skip(emb)  # \[B, T, 256]



&nbsp;      Pathway Convolution (from c0)



&nbsp;      df\_convp: Conv2dNormAct(

&nbsp;          C → O\*2,  (O=df\_order, default=5, so O\*2=10)

&nbsp;          kernel=(df\_pathway\_kernel\_size\_t, 1)

&nbsp;          fstride=1

&nbsp;      )



&nbsp;      c0\_in: \[B, C, T, Fc] → df\_convp → \[B, O\*2, T, Fc]

&nbsp;      Permute to channels\_last: \[B, T, Fc, O\*2]



&nbsp;      Output Coefficients



&nbsp;      df\_out: Sequential(

&nbsp;          GroupedLinearEinsum(df\_hidden\_dim=256 → nb\_df\*O\*2, groups=lin\_groups),

&nbsp;          Tanh()

&nbsp;      )



&nbsp;      c\_out = df\_out(c)  # \[B, T, nb\_df\*O\*2]

&nbsp;      c\_out = c\_out.view(b, t, nb\_df, O\*2) + c0  # \[B, T, nb\_df, O\*2]



&nbsp;      Where:

&nbsp;      - nb\_df = number of DF frequency bins (96 default)

&nbsp;      - O = df\_order = 5

&nbsp;      - O\*2 = 10 (real and imaginary coefficients)



&nbsp;      Amplitude Scaling (Optional)



&nbsp;      df\_fc\_a: Sequential(

&nbsp;          Linear(df\_hidden\_dim=256 → 1),

&nbsp;          Sigmoid()

&nbsp;      )

&nbsp;      alpha = df\_fc\_a(c)  # \[B, T, 1] ∈ \[0, 1]



&nbsp;      This can be used to scale DF influence per time-frequency bin.



&nbsp;      ---

&nbsp;      5. pad\_feat and pad\_spec - Detailed Explanation



&nbsp;      File: /DeepFilterNet/df/deepfilternet3.py (Lines 357-365)



&nbsp;      pad\_feat



&nbsp;      if p.conv\_lookahead > 0:

&nbsp;          self.pad\_feat = nn.ConstantPad2d((0, 0, -p.conv\_lookahead, p.conv\_lookahead), 0.0)

&nbsp;      else:

&nbsp;          self.pad\_feat = nn.Identity()



&nbsp;      Purpose: Add temporal lookahead to encoder features (ERB and spec features)



&nbsp;      Mechanism:

&nbsp;      - ConstantPad2d((left\_freq, right\_freq, top\_time, bottom\_time))

&nbsp;      - Pads -conv\_lookahead samples at the beginning (top) and +conv\_lookahead at end (bottom)

&nbsp;      - Frequency dimension (axis -1): no padding (0, 0)

&nbsp;      - Padding value: 0.0



&nbsp;      Effect: Allows the encoder to "look ahead" future frames before output is computed

&nbsp;      - Input shape: \[B, C, T, F] → \[B, C, T + 2\*conv\_lookahead, F]



&nbsp;      pad\_spec



&nbsp;      if p.df\_lookahead > 0:

&nbsp;          self.pad\_spec = nn.ConstantPad3d((0, 0, 0, 0, -p.df\_lookahead, p.df\_lookahead), 0.0)

&nbsp;      else:

&nbsp;          self.pad\_spec = nn.Identity()



&nbsp;      Purpose: Add temporal lookahead to DF coefficients for multi-frame filtering



&nbsp;      Mechanism:

&nbsp;      - ConstantPad3d((left, right, top, bottom, front, back))

&nbsp;      - Pads temporal dimension with -df\_lookahead at front and +df\_lookahead at back

&nbsp;      - Other dimensions: no padding (0, 0, 0, 0)



&nbsp;      Effect: Enables DF to use future frames in coefficient computation

&nbsp;      - Input shape: \[B, 1, T, F, 2] → \[B, 1, T + 2\*df\_lookahead, F, 2]



&nbsp;      ---

&nbsp;      6. ModelParams Configuration



&nbsp;      File: /DeepFilterNet/df/deepfilternet3.py (Lines 25-77)



&nbsp;      Model Configuration Parameters



&nbsp;      ┌──────────────────────────┬───────────┬─────────┬─────────────────────────────────────────────────────┐

&nbsp;      │        Parameter         │   Type    │ Default │                     Description                     │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ conv\_lookahead           │ int       │ 0       │ Encoder temporal lookahead in frames                │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ conv\_ch                  │ int       │ 16      │ Base channel count for convolutional layers         │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ conv\_depthwise           │ bool      │ True    │ Use depthwise convolutions in encoder               │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ convt\_depthwise          │ bool      │ True    │ Use depthwise convolutions in decoder               │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ conv\_kernel              │ List\[int] │ (1, 3)  │ Temporal \& frequency kernel size (encoder/decoder)  │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ convt\_kernel             │ List\[int] │ (1, 3)  │ Transposed conv kernel size                         │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ conv\_kernel\_inp          │ List\[int] │ (3, 3)  │ Input convolution kernel size                       │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ emb\_hidden\_dim           │ int       │ 256     │ Embedding GRU hidden dimension                      │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ emb\_num\_layers           │ int       │ 2       │ Number of embedding GRU layers                      │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ emb\_gru\_skip\_enc         │ str       │ "none"  │ Encoder GRU skip: "none"/"identity"/"groupedlinear" │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ emb\_gru\_skip             │ str       │ "none"  │ Decoder GRU skip: "none"/"identity"/"groupedlinear" │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ df\_hidden\_dim            │ int       │ 256     │ Deep filter GRU hidden dimension                    │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ df\_gru\_skip              │ str       │ "none"  │ DF GRU skip: "none"/"identity"/"groupedlinear"      │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ df\_pathway\_kernel\_size\_t │ int       │ 1       │ DF pathway temporal kernel size                     │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ enc\_concat               │ bool      │ False   │ Concatenate ERB \& spec in encoder vs add            │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ df\_num\_layers            │ int       │ 3       │ Number of DF GRU layers                             │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ df\_n\_iter                │ int       │ 1       │ Number of DF iterations (must be 1)                 │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ lin\_groups               │ int       │ 1       │ Linear layer grouping                               │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ enc\_lin\_groups           │ int       │ 16      │ Encoder linear layer grouping                       │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ mask\_pf                  │ bool      │ False   │ Apply post-filter to mask                           │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ pf\_beta                  │ float     │ 0.02    │ Post-filter beta parameter                          │

&nbsp;      ├──────────────────────────┼───────────┼─────────┼─────────────────────────────────────────────────────┤

&nbsp;      │ lsnr\_dropout             │ bool      │ False   │ Zero outputs for low-SNR frames                     │

&nbsp;      └──────────────────────────┴───────────┴─────────┴─────────────────────────────────────────────────────┘



&nbsp;      DfParams Base Configuration



&nbsp;      Inherited from DfParams (config.py, lines 14-39):



&nbsp;      ┌──────────────┬───────┬─────────┬──────────────────────────────────────┐

&nbsp;      │  Parameter   │ Type  │ Default │             Description              │

&nbsp;      ├──────────────┼───────┼─────────┼──────────────────────────────────────┤

&nbsp;      │ sr           │ int   │ 48000   │ Sampling rate (Hz)                   │

&nbsp;      ├──────────────┼───────┼─────────┼──────────────────────────────────────┤

&nbsp;      │ fft\_size     │ int   │ 960     │ FFT size (samples)                   │

&nbsp;      ├──────────────┼───────┼─────────┼──────────────────────────────────────┤

&nbsp;      │ hop\_size     │ int   │ 480     │ STFT hop size (samples)              │

&nbsp;      ├──────────────┼───────┼─────────┼──────────────────────────────────────┤

&nbsp;      │ nb\_erb       │ int   │ 32      │ Number of ERB bands                  │

&nbsp;      ├──────────────┼───────┼─────────┼──────────────────────────────────────┤

&nbsp;      │ nb\_df        │ int   │ 96      │ Number of deep filter frequency bins │

&nbsp;      ├──────────────┼───────┼─────────┼──────────────────────────────────────┤

&nbsp;      │ norm\_tau     │ float │ 1.0     │ Normalization decay factor           │

&nbsp;      ├──────────────┼───────┼─────────┼──────────────────────────────────────┤

&nbsp;      │ lsnr\_max     │ int   │ 35      │ Maximum local SNR (dB)               │

&nbsp;      ├──────────────┼───────┼─────────┼──────────────────────────────────────┤

&nbsp;      │ lsnr\_min     │ int   │ -15     │ Minimum local SNR (dB)               │

&nbsp;      ├──────────────┼───────┼─────────┼──────────────────────────────────────┤

&nbsp;      │ min\_nb\_freqs │ int   │ 2       │ Min frequency bins per ERB band      │

&nbsp;      ├──────────────┼───────┼─────────┼──────────────────────────────────────┤

&nbsp;      │ df\_order     │ int   │ 5       │ Deep filter order                    │

&nbsp;      ├──────────────┼───────┼─────────┼──────────────────────────────────────┤

&nbsp;      │ df\_lookahead │ int   │ 0       │ DF temporal lookahead                │

&nbsp;      └──────────────┴───────┴─────────┴──────────────────────────────────────┘



&nbsp;      ---

&nbsp;      7. SqueezedGRU\_S Class



&nbsp;      File: /DeepFilterNet/df/modules.py (Lines 702-738)



&nbsp;      Purpose



&nbsp;      Wraps nn.GRU with linear input/output layers and optional skip connections, reducing computation.



&nbsp;      Architecture



&nbsp;      SqueezedGRU\_S(input\_size=I, hidden\_size=H, output\_size=O, ...)

&nbsp;      │

&nbsp;      ├─ linear\_in: Sequential(

&nbsp;      │   ├─ GroupedLinearEinsum(I → H, groups=lin\_groups)

&nbsp;      │   └─ ReLU(inplace=True)  \[or other activation]

&nbsp;      │

&nbsp;      ├─ gru: nn.GRU(H → H, num\_layers, batch\_first=True)

&nbsp;      │

&nbsp;      ├─ linear\_out: Sequential(  \[if output\_size is specified]

&nbsp;      │   ├─ GroupedLinearEinsum(H → O, groups=lin\_groups)

&nbsp;      │   └─ ReLU(inplace=True)

&nbsp;      │

&nbsp;      └─ gru\_skip: Optional\[nn.Module]  \[identity or groupedlinear]

&nbsp;         Applied AFTER linear\_out: output = output + gru\_skip(input)



&nbsp;      Forward Pass



&nbsp;      def forward(self, input, h=None):

&nbsp;          x = self.linear\_in(input)           # \[B, T, I] → \[B, T, H]

&nbsp;          x, h = self.gru(x, h)               # \[B, T, H], h: \[num\_layers, B, H]

&nbsp;          x = self.linear\_out(x)              # \[B, T, H] → \[B, T, O]

&nbsp;          if self.gru\_skip is not None:

&nbsp;              x = x + self.gru\_skip(input)    # Residual skip from original input

&nbsp;          return x, h



&nbsp;      Key Difference from SqueezedGRU:

&nbsp;      - Skip connection applied AFTER linear\_out (not before)

&nbsp;      - More flexible for dimension changes



&nbsp;      ---

&nbsp;      8. Rust DFState Implementation



&nbsp;      File: /DeepFilterNet/libDF/src/lib.rs (Lines 50-226)



&nbsp;      DFState Structure



&nbsp;      pub struct DFState {

&nbsp;          pub sr: usize,                              // Sampling rate

&nbsp;          pub frame\_size: usize,                      // hop\_size

&nbsp;          pub window\_size: usize,                     // fft\_size

&nbsp;          pub freq\_size: usize,                       // fft\_size / 2 + 1

&nbsp;          pub fft\_forward: Arc<dyn RealToComplex<f32>>,

&nbsp;          pub fft\_inverse: Arc<dyn ComplexToReal<f32>>,

&nbsp;          pub window: Vec<f32>,                       // Vorbis window

&nbsp;          pub wnorm: f32,                             // Normalization factor

&nbsp;          pub erb: Vec<usize>,                        // ERB band widths (freq bins per band)

&nbsp;          analysis\_mem: Vec<f32>,                     // fft\_size - frame\_size samples

&nbsp;          analysis\_scratch: Vec<Complex32>,           // FFT scratch buffer

&nbsp;          synthesis\_mem: Vec<f32>,                    // fft\_size - frame\_size samples

&nbsp;          synthesis\_scratch: Vec<Complex32>,          // FFT scratch buffer

&nbsp;          mean\_norm\_state: Vec<f32>,                  // One per ERB band

&nbsp;          unit\_norm\_state: Vec<f32>,                  // One per frequency bin

&nbsp;      }



&nbsp;      Frame Analysis (STFT)



&nbsp;      Method: frame\_analysis(input: \&\[f32], output: \&mut \[Complex32], state: \&mut DFState)



&nbsp;      Input: Raw audio frame \[frame\_size]

&nbsp;      Output: Complex spectrum \[freq\_size]



&nbsp;      Process:



&nbsp;      1. OLA Buffer Setup:

&nbsp;        - Create FFT input buffer of size window\_size

&nbsp;        - Split into two halves: \[window\_size - frame\_size] + \[frame\_size]

&nbsp;      2. First Half (Previous Frame Overlap):

&nbsp;      for i in 0..(window\_size - frame\_size):

&nbsp;          buf\[i] = analysis\_mem\[i] \* window\[i]

&nbsp;      3. Second Half (Current Frame):

&nbsp;      for i in 0..frame\_size:

&nbsp;          buf\[window\_size - frame\_size + i] = input\[i] \* window\[window\_size - frame\_size + i]

&nbsp;      4. Update Memory:

&nbsp;      rotate\_left(analysis\_mem, frame\_size)

&nbsp;      copy(input, analysis\_mem\[fft\_size - frame\_size:])

&nbsp;      5. FFT + Normalization:

&nbsp;      fft\_forward.process(buf, output)

&nbsp;      output \*= wnorm  // wnorm = 1 / (window\_size^2 / (2 \* frame\_size))



&nbsp;      Frame Synthesis (ISTFT)



&nbsp;      Method: frame\_synthesis(input: \&mut \[Complex32], output: \&mut \[f32], state: \&mut DFState)



&nbsp;      Input: Complex spectrum \[freq\_size]

&nbsp;      Output: Enhanced audio frame \[frame\_size]



&nbsp;      Process:



&nbsp;      1. IFFT:

&nbsp;      fft\_inverse.process(input, x)  // x: \[window\_size]

&nbsp;      2. Window Application:

&nbsp;      x\[i] \*= window\[i]  for all i in window\_size

&nbsp;      3. Overlap-Add (First Part):

&nbsp;      for i in 0..frame\_size:

&nbsp;          output\[i] = x\[i] + synthesis\_mem\[i]

&nbsp;      4. Update Memory:

&nbsp;      rotate\_left(synthesis\_mem, frame\_size)

&nbsp;      // Copy overlapping tail to memory

&nbsp;      for i in 0..frame\_size:

&nbsp;          synthesis\_mem\[split + i] += x\[frame\_size + split + i]



&nbsp;      Vorbis Window Initialization



&nbsp;      Method: Initialize during DFState::new()



&nbsp;      Window Formula:

&nbsp;      w\[i] = sin(π/2 \* sin²(π \* (i + 0.5) / window\_size\_h))



&nbsp;      Where:

&nbsp;        window\_size\_h = fft\_size / 2

&nbsp;        i ∈ \[0, fft\_size)



&nbsp;      Properties:

&nbsp;      - Perfect reconstruction with 50% overlap

&nbsp;      - Smooth onset/offset for spectral processing



&nbsp;      ERB Band Computation



&nbsp;      Function: erb\_fb(sr, fft\_size, nb\_bands, min\_nb\_freqs) → Vec<usize>



&nbsp;      Algorithm:



&nbsp;      1. ERB Scale Endpoints:

&nbsp;      erb\_low = freq2erb(0 Hz)

&nbsp;      erb\_high = freq2erb(sr/2 Hz)  // Nyquist



&nbsp;      freq2erb(f) = 9.265 \* ln(1 + f / (24.7 \* 9.265))

&nbsp;      2. Uniform ERB Band Distribution:

&nbsp;      step = (erb\_high - erb\_low) / nb\_bands



&nbsp;      for i = 1 to nb\_bands:

&nbsp;          erb\_freq = erb2freq(erb\_low + i \* step)

&nbsp;          fb = round(erb\_freq / freq\_width)  // freq\_width = sr / fft\_size

&nbsp;          nb\_freqs\[i-1] = fb - prev\_fb

&nbsp;      3. Enforce Minimum Bins:

&nbsp;        - If band has < min\_nb\_freqs, extend from previous band

&nbsp;      4. Final Adjustment:

&nbsp;        - Last band includes extra bin for Nyquist: +1

&nbsp;        - Trim if total exceeds fft\_size/2 + 1



&nbsp;      Output: erb: Vec<usize> where erb\[i] = frequency bins in ERB band i



&nbsp;      Feature Extraction: feat\_erb



&nbsp;      Method: feat\_erb(input: \&\[Complex32], alpha: f32, output: \&mut \[f32])



&nbsp;      Process:



&nbsp;      1. Band Correlation (Energy per ERB band):

&nbsp;      for band i:

&nbsp;          output\[i] = mean(|input\[freq\_start..freq\_end]|²)

&nbsp;      1. Using compute\_band\_corr helper.

&nbsp;      2. Log Compression to dB:

&nbsp;      output\[i] = 10 \* log10(output\[i] + 1e-10)

&nbsp;      3. Exponential Mean Normalization:

&nbsp;      for band i:

&nbsp;          state\[i] = output\[i] \* (1 - alpha) + state\[i] \* alpha

&nbsp;          output\[i] = (output\[i] - state\[i]) / 40.0



&nbsp;      Output: Normalized ERB features in dB scale



&nbsp;      Feature Extraction: feat\_cplx



&nbsp;      Method: feat\_cplx(input: \&\[Complex32], alpha: f32, output: \&mut \[Complex32])



&nbsp;      Process:



&nbsp;      1. Copy input to output

&nbsp;      2. Apply band unit normalization:

&nbsp;      for freq i:

&nbsp;          state\[i] = |input\[i]| \* (1 - alpha) + state\[i] \* alpha

&nbsp;          output\[i] /= sqrt(state\[i])



&nbsp;      Output: Unit-normalized complex spectrum



&nbsp;      Apply ERB Gains (Mask Application)



&nbsp;      Method: apply\_interp\_band\_gain(output: \&mut \[Complex32], gains: \&\[f32], erb: \&\[usize])



&nbsp;      Process:

&nbsp;      for band i:

&nbsp;          for freq j in erb\_band\[i]:

&nbsp;              output\[freq\_index] \*= gains\[i]



&nbsp;      Each frequency bin in an ERB band is multiplied by the same gain value.



&nbsp;      Deep Filter Application



&nbsp;      Handled by PyTorch DfOp module (modules.py, lines 307-467)



&nbsp;      The Rust code provides STFT/ISTFT; Deep filtering is applied in PyTorch as:



&nbsp;      Complex Multiplication per Order:

&nbsp;      spec\_f\[t, f] = Σ(o=0 to order-1)

&nbsp;          spec\_pad\[t+o, f] \* coef\[t, o, f]



&nbsp;      Where coef\[t, o, f] is complex and spec\_pad contains df\_order time steps



&nbsp;      ---

&nbsp;      9. Complete Data Flow: Audio to Enhanced Audio



&nbsp;      ┌─────────────────────────────────────────────────────┐

&nbsp;      │ Raw Audio: \[sample\_rate=48kHz, mono]                │

&nbsp;      └──────────────────┬──────────────────────────────────┘

&nbsp;                         │

&nbsp;                         ▼

&nbsp;      ┌──────────────────────────────────────────────────────┐

&nbsp;      │ STFT (Rust DFState.frame\_analysis)                   │

&nbsp;      │ Input: audio frame \[480 samples @ 48kHz = 10ms]    │

&nbsp;      │ Process: OLA windowing + FFT (fft\_size=960)         │

&nbsp;      │ Output: \[freq\_size=481] = \[fft\_size/2+1] complex   │

&nbsp;      └──────────────────┬──────────────────────────────────┘

&nbsp;                         │

&nbsp;                         ▼

&nbsp;      ┌──────────────────────────────────────────────────────┐

&nbsp;      │ Feature Extraction (Rust DFState methods)            │

&nbsp;      │                                                      │

&nbsp;      │ 1. feat\_erb():   complex spectrum → \[32] dB ERB    │

&nbsp;      │    - Band correlation (power per ERB)               │

&nbsp;      │    - Log compression + exponential mean norm        │

&nbsp;      │                                                      │

&nbsp;      │ 2. feat\_cplx():  complex spectrum → unit-normed     │

&nbsp;      │    - Per-frequency normalization                    │

&nbsp;      └──────────────────┬──────────────────────────────────┘

&nbsp;                         │

&nbsp;                         ▼

&nbsp;      ┌──────────────────────────────────────────────────────┐

&nbsp;      │ Deep Learning Model (PyTorch DfNet)                  │

&nbsp;      │ Input: \[feat\_erb, feat\_cplx, spectrum]              │

&nbsp;      │                                                      │

&nbsp;      │ Encoder:                                             │

&nbsp;      │  - Conv layers process ERB \& complex spec           │

&nbsp;      │  - Output: embeddings \[B, T, 128]                   │

&nbsp;      │  - Output: Local SNR estimate \[B, T, 1]            │

&nbsp;      │                                                      │

&nbsp;      │ ERB Decoder:                                         │

&nbsp;      │  - Upsampling decoder from \[B, T, 128]             │

&nbsp;      │  - Output: ERB mask \[B, 1, T, 32] ∈ \[0,1]         │

&nbsp;      │                                                      │

&nbsp;      │ Deep Filter Decoder:                                │

&nbsp;      │  - GRU processes embeddings \[B, T, 256]            │

&nbsp;      │  - Output: filter coefs \[B, T, 96, 10]            │

&nbsp;      │    (96 freq bins × 5 orders × 2 for real/imag)    │

&nbsp;      └──────────────────┬──────────────────────────────────┘

&nbsp;                         │

&nbsp;                         ▼

&nbsp;      ┌──────────────────────────────────────────────────────┐

&nbsp;      │ Spectral Processing (PyTorch)                        │

&nbsp;      │                                                      │

&nbsp;      │ 1. ERB Masking:                                      │

&nbsp;      │    spec\_m = spec \* mask (broadcasted over freq)    │

&nbsp;      │    \[481] freq bins masked by \[32] ERB bands        │

&nbsp;      │                                                      │

&nbsp;      │ 2. Deep Filtering (first 96 freq bins):            │

&nbsp;      │    spec\_df = Σ spec\_windowed\[o] \* coef\[o]         │

&nbsp;      │    Complex convolution over df\_order=5 time steps  │

&nbsp;      │                                                      │

&nbsp;      │ 3. Combine:                                          │

&nbsp;      │    - First 96 bins from DF output                   │

&nbsp;      │    - Remaining 385 bins from ERB masked spectrum   │

&nbsp;      └──────────────────┬──────────────────────────────────┘

&nbsp;                         │

&nbsp;                         ▼

&nbsp;      ┌──────────────────────────────────────────────────────┐

&nbsp;      │ ISTFT (Rust DFState.frame\_synthesis)                 │

&nbsp;      │ Input: \[481] complex enhanced spectrum              │

&nbsp;      │ Process: IFFT + Vorbis window + OLA                │

&nbsp;      │ Output: enhanced audio frame \[480 samples]         │

&nbsp;      └──────────────────┬──────────────────────────────────┘

&nbsp;                         │

&nbsp;                         ▼

&nbsp;      ┌──────────────────────────────────────────────────────┐

&nbsp;      │ Enhanced Audio: \[mono, 48kHz]                        │

&nbsp;      │ Concatenate frames with proper overlap handling      │

&nbsp;      └──────────────────────────────────────────────────────┘



&nbsp;      ---

&nbsp;      10. Key Shape Transformations Summary



&nbsp;      ┌──────────────────────┬────────────────┬────────────────────┬───────────────────────┐

&nbsp;      │        Stage         │  Input Shape   │     Operation      │     Output Shape      │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ STFT                 │ \[480]          │ FFT                │ \[481] complex         │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ ERB Features         │ \[481] complex  │ Correlation + norm │ \[32] float (dB)       │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ Complex Features     │ \[481] complex  │ Unit norm          │ \[481] complex         │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ Encoder ERB Conv     │ \[1, T, 32]     │ 4× conv            │ \[16, T, 8]            │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ Encoder DF Conv      │ \[2, T, 481]    │ 2× conv            │ \[16, T, 240]          │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ Encoder Embedding    │ \[16, T, 240]   │ Flatten+Combine    │ \[B, T, 128]           │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ Encoder GRU          │ \[B, T, 128]    │ GRU                │ \[B, T, 128]           │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ Local SNR            │ \[B, T, 128]    │ Linear+Sigmoid     │ \[B, T, 1]             │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ ERB Decoder GRU      │ \[B, T, 128]    │ GRU                │ \[B, T, 128]           │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ ERB Decoder Reshape  │ \[B, T, 128]    │ Reshape            │ \[B, 16, T, 8]         │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ ERB Decoder Upsample │ \[B, 16, T, 8]  │ 3× TransposeConv   │ \[B, 1, T, 32]         │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ ERB Mask Output      │ \[B, 1, T, 32]  │ Sigmoid            │ \[B, 1, T, 32] ∈ \[0,1] │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ DF Decoder GRU       │ \[B, T, 128]    │ GRU                │ \[B, T, 256]           │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ DF Output Linear     │ \[B, T, 256]    │ Linear+Tanh        │ \[B, T, 960]           │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ DF Reshape           │ \[B, T, 960]    │ Reshape            │ \[B, T, 96, 10]        │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ DF + Pathway         │ \[B, T, 96, 10] │ Add c0             │ \[B, T, 96, 10]        │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ DF Output Transform  │ \[B, T, 96, 10] │ Reshape            │ \[B, 5, T, 96, 2]      │

&nbsp;      ├──────────────────────┼────────────────┼────────────────────┼───────────────────────┤

&nbsp;      │ ISTFT                │ \[481] complex  │ IFFT + window      │ \[480] audio           │

&nbsp;      └──────────────────────┴────────────────┴────────────────────┴───────────────────────┘



&nbsp;      ---

&nbsp;      11. Critical Implementation Notes



&nbsp;      Frequency Bin Mapping



&nbsp;      - Total FFT bins: fft\_size / 2 + 1 = 481 (for fft\_size=960)

&nbsp;      - ERB bins: 32 (parameterized)

&nbsp;      - DF bins: 96 (parameterized)

&nbsp;      - Complex spectrogram encoder sees: 2 × (DF bins or more)



&nbsp;      Multi-Frame DF



&nbsp;      - Uses df\_order=5 historical frames

&nbsp;      - Applies complex multiplication per frame

&nbsp;      - Optional lookahead for future frames



&nbsp;      Normalization



&nbsp;      - Mean normalization (ERB): Exponential moving average subtraction

&nbsp;      - Unit normalization (complex): Per-frequency magnitude normalization

&nbsp;      - Window norm: Compensation for STFT energy loss



&nbsp;      Causality



&nbsp;      - Default conv\_lookahead=0 and df\_lookahead=0 → causal system

&nbsp;      - Temporal padding implemented with negative lookahead offset

&nbsp;      - No computational delay except STFT buffer (< 1 frame)



&nbsp;      ---

&nbsp;      This comprehensive documentation covers all architectural details, data flows, parameter configurations, and

&nbsp;       mathematical operations in DeepFilterNet3.

&nbsp; ⎿  Done (10 tool uses · 61.3k tokens · 1m 30s)



&nbsp;                                                                                       04:28 PM claude-opus-4-6

● Now let me write the comprehensive document:



