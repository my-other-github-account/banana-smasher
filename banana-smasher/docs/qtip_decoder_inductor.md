# Packed QTIP decoder compiler contract

`banana_smasher.qtip_kernel_decompress.decode_compressed` remains the default
`@torch.compile` entry point (Inductor). No eager fallback or deployment-local
monkeypatch is part of this contract.

## Layout defect

On macOS CPU with PyTorch 2.11.0, the K=1 deployment fixture reproduces the
failure reported on Linux CPU with PyTorch 2.11.0+cu130 at canonical baseline
`56cf5aa819048f090240f26771c2241a7488c500`:

    self.stride(-1) must be 1 to view Byte as UInt16
    (different element sizes), but got 4

Inductor lowers the `contiguous()` clone to a buffer with size `[2,2,16,2]`
and stride `[1,2,8,4]` before its `aten.view.dtype` fallback. The physical layout
therefore fails the size-changing bitcast, although eager and aot_eager work.
This is a compiler/layout failure, not evidence of corrupt packed words.

The decoder now assembles each little-endian uint16 value as `lo | (hi << 8)`
in int32 and combines the rolled/current words as `next | (current << 16)`.
This preserves signed int32 shift-and-mask semantics while avoiding both
size-increasing dtype views. The intermediate graph break is unnecessary and
removed. The input byte view, wire swizzle, LUT indexing and output layout are
unchanged.

## Verification

Run the production default backend, not only aot_eager:

    PYTHONPATH=banana-smasher/src TORCHINDUCTOR_COMPILE_THREADS=1 OMP_NUM_THREADS=1 python -m pytest banana-smasher/tests/test_qtip_decoder_inductor.py banana-smasher/tests/test_qtip_kernel_decompress.py -vv

The actual-backend regression uses deterministic packed words for K=1..4,
checks eager equality and checks frozen SHA256s of pre-fix eager output bytes.
Each rate starts with a fresh Dynamo cache: a mixed-rate same-cache run was
stopped after more than six minutes compiling the K=3 auto-dynamic specialization.
No mixed-rate compilation-latency claim is made; this gate covers independent
fixed-rate invocations, including the DS4 Q1 deployment fixture.
The frozen oracle catches a shared semantic regression in the new eager and
compiled paths. The older builder-loader test additionally exercises canonical
module resolution and aot_eager; it is not the Inductor gate.

The separate bounded deployment fixture must also verify `decode_packed`
FP16 bit equality on the target runtime. Local macOS CPU success does not certify
Linux/CUDA execution, GPU performance, producer adoption, model output quality,
or any DS4 scientific acceptance criterion. Deployment owns target-host retry;
protected producers and their code pins must remain unchanged until authorized.
