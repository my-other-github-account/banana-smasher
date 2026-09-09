"""Opt-in canonical alphabet publication; original scientific keys preserved."""
from canonical_reference_rebind_run8550 import rebind as _base
PIN='54244dc7aeb7c98895fd7655612f7a69b89464e3'
def rebind(cfg,manifest,**kwargs):
 c,m=_base(cfg,manifest,**kwargs)
 c['viterbi_distance_alphabet']=True
 m['canonical_commit']=PIN
 return c,m
