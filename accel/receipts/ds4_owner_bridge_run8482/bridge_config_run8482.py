from copy import deepcopy

def bind_config(source, pin, runner, manifest, manifest_sha, grouped):
    config=deepcopy(source)
    config['qtip_runner']=runner
    config['exact_solver']='banana_smasher.qtip_viterbi@'+pin
    config['materialization']['run_manifest']=manifest
    config['materialization']['run_manifest_sha256']=manifest_sha
    config['block_ldl_unitwise']=grouped
    return config
