"""The generic persistent Viterbi must stay bit-identical to a plain fp32 reference.

Covers the K=4 and K=1 geometries that use ``persistent-prefix-generic-aot-v1``
(K=2/K=3 have their own sealed kernels and receipts).  The reference reproduces
the kernel's exact update order: full-branch scan with strict-< replacement,
so ties resolve to the lowest branch index.
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from banana_smasher import qtip_viterbi  # noqa: E402


class _CB:
    def __init__(self, L, K, V):
        self.L, self.K, self.V = L, K, V
        self.lut = torch.randn(V, 1 << L, device="cuda", dtype=torch.float16)


def _reference(cb, x):
    L, K, V = cb.L, cb.K, cb.V
    bb = K * V
    P, Q = 1 << (L - bb), 1 << bb
    q_factor = 1 << (L - 2 * bb)
    steps, B = x.shape[0] // V, x.shape[1]
    lut = cb.lut.float()
    j = torch.arange(P, device="cuda")
    residue = j >> bb
    xs = x.float().view(steps, V, B)
    out = torch.empty(steps, B, dtype=torch.int32, device="cuda")
    for b in range(B):
        best = None
        bp = []
        for t in range(steps):
            nb = torch.full((P,), float("inf"), device="cuda")
            nc = torch.zeros((P,), dtype=torch.int32, device="cuda")
            for q in range(Q):
                state = q * P + j
                c = torch.zeros((P,), device="cuda") if best is None else best[q * q_factor + residue]
                for lane in range(V):
                    c = c + (lut[lane, state] - xs[t, lane, b]) ** 2
                take = c < nb
                nb = torch.where(take, c, nb)
                nc = torch.where(take, state, nc)
            best = nb
            bp.append(nc)
        prefix = int(torch.argmin(best))
        for t in range(steps - 1, -1, -1):
            s = int(bp[t][prefix])
            out[t, b] = s
            prefix = s >> bb
    return out


@pytest.mark.parametrize("K", [4, 1])
def test_generic_persistent_viterbi_matches_reference(K):
    torch.manual_seed(0)
    cb = _CB(16, K, 2)
    x = torch.randn(128 * 2, 4, device="cuda", dtype=torch.float16)
    got = qtip_viterbi.exact_prefix_viterbi(cb, x)
    assert torch.equal(got.cpu(), _reference(cb, x).cpu())
