"""Opt-in device conformance; scientific identities remain unchanged."""
from canonical_alphabet_rebind_run8554 import rebind as _base
PIN='c769d086701769b08fdc85d5191afc2aa321e3e7'
def rebind(cfg,manifest,**kwargs):
 c,m=_base(cfg,manifest,**kwargs)
 c['packed_conformance_on_device']=True
 m['canonical_commit']=PIN
 return c,m
