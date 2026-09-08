from types import SimpleNamespace
import pytest
import torch
from banana_smasher import solver_qtip_profile as solver

@pytest.mark.parametrize('k',[1,2,3,4])
def test_installed_roll_and_overlap_close_memory_contract(k):
    cb=SimpleNamespace(L=16,K=k,V=2,idx_dtype=torch.int32)
    exact=SimpleNamespace(geometry=lambda cb: {'implementation':'fixture'})
    solver._install_profiled_exact_viterbi(cb,exact,solver._ExactTimers(),profile_mode=False)
    source=torch.empty((256,256),device='meta')
    contract=solver._bind_builder_memory_contract(cb,source)
    # Native bitshift.quantize calls installed quantize_seq twice, first rolled,
    # then overlap-conditioned. Both visits count; final states remain one set.
    per_pass=source.numel()//cb.V
    assert contract['state_elements']==2*per_pass
    assert contract['state_storage_bytes']>=per_pass*4
    cb._banana_smasher_observed_state_elements=2*per_pass
    assert solver._verify_builder_memory_contract(cb) is contract
    cb._banana_smasher_observed_state_elements+=1
    with pytest.raises(RuntimeError,match='state output closure mismatch'):
        solver._verify_builder_memory_contract(cb)
