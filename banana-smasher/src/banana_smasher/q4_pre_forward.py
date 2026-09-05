"""Frozen DS4 PRE layer-streamed forward.

Source-derived from authenticated producer 55724cfd18c10dc41acc504a81043cf5421ab12895b0e27935ccaa33540cec50.
Only original Q4 and native control are admitted; no teacher generation or repair.
The 2048-token/mb2 cache geometry and FP16 gathered bank boundary are preserved.
"""
import ctypes
import hashlib
import json
import math
import os
import threading
import time
import torch
DEV = "cuda"
SUP = 8192
E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_VAL = torch.tensor(E2M1_MAG + [-m for m in E2M1_MAG])

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)



def cuda_time():
    """Return a synchronized monotonic timestamp for honest CUDA stage timing."""
    torch.cuda.synchronize()
    return time.perf_counter()



def mem_available_bytes():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing from /proc/meminfo")



def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)



def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()



def jrow(path, **kw):
    kw["ts"] = round(time.time(), 3)
    with open(path, "a") as f:
        f.write(json.dumps(kw, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())



class LocalSource:
    """Bounded one-layer-ahead ext4 reader with observed page residency.

    The parent counted logical payloads and used advisory DONTNEED.  This shard
    gate additionally measures Linux page residency with mincore(2), begins from
    an observed <=1 GiB residual layer-source cache, and refuses to start the
    next read unless the prior source shards are again <=1 GiB resident.  Thus
    the receipt distinguishes the one active source-file payload from the one
    materialized GPU-layer payload instead of treating fadvise as proof.
    """

    # Authenticated measured-residency cap; does not change model numerics.
    RESIDUAL_SOURCE_ALLOWANCE = 1 << 30

    def __init__(self, ckpt_dir, floor_bytes, read_chunk_bytes=16 << 20):
        self.dir = os.path.realpath(ckpt_dir)
        self.floor_bytes = floor_bytes
        self.read_chunk_bytes = read_chunk_bytes
        self.lock = threading.Lock()
        self.job = None
        self.resident_source = None
        self.max_observed_payloads = 0
        self.residency_by_shard = {}
        self.residency_rows = []
        self.preflight_resident_source_bytes = None
        self.max_actual_resident_source_bytes = 0
        self.max_layer_shard_bytes = 0

    def prefetch(self, shard):
        # Static embed/head calls retain the parent no-op behavior.  Layer reads
        # use begin_layer()/wait_layer() explicitly so bounds and timing seal.
        return None

    def get(self, shard):
        return os.path.join(self.dir, shard)

    @staticmethod
    def _resident_bytes(path):
        size = os.path.getsize(path)
        if size == 0:
            return 0, 0, 0
        page = os.sysconf("SC_PAGE_SIZE")
        pages = (size + page - 1) // page
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_ubyte)]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        fd = os.open(path, os.O_RDONLY)
        addr = None
        try:
            addr = libc.mmap(None, size, 0, 1, fd, 0)  # PROT_NONE, MAP_SHARED
            if addr == ctypes.c_void_p(-1).value:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err), path)
            vec = (ctypes.c_ubyte * pages)()
            if libc.mincore(ctypes.c_void_p(addr), size, vec) != 0:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err), path)
            resident_pages = sum(1 for value in vec if value & 1)
            return min(size, resident_pages * page), resident_pages, pages
        finally:
            if addr is not None and addr != ctypes.c_void_p(-1).value:
                libc.munmap(ctypes.c_void_p(addr), size)
            os.close(fd)

    def observe_residency(self, shard, phase, layer):
        path = self.get(shard)
        resident, resident_pages, pages = self._resident_bytes(path)
        self.residency_by_shard[shard] = resident
        total = sum(self.residency_by_shard.values())
        self.max_actual_resident_source_bytes = max(
            self.max_actual_resident_source_bytes, total)
        row = {
            "epoch_ns": time.time_ns(), "phase": phase, "layer": layer,
            "shard": shard, "path": path, "file_bytes": os.path.getsize(path),
            "resident_bytes": resident, "resident_pages": resident_pages,
            "total_pages": pages, "all_layer_sources_resident_upper_bound_bytes": total,
            "memavailable_bytes": mem_available_bytes(),
        }
        self.residency_rows.append(row)
        return row

    def preflight_residency(self, shards):
        unique = list(dict.fromkeys(shards))
        self.max_layer_shard_bytes = max(os.path.getsize(self.get(s)) for s in unique)
        for shard in unique:
            self.drop_untracked(shard)
        for shard in unique:
            self.observe_residency(shard, "preflight_after_fadvise", -1)
        total = sum(self.residency_by_shard.values())
        self.preflight_resident_source_bytes = total
        if total > self.RESIDUAL_SOURCE_ALLOWANCE:
            raise RuntimeError(
                f"preflight source residency {total} exceeds "
                f"{self.RESIDUAL_SOURCE_ALLOWANCE}")
        return total

    def begin_layer(self, layer, shard, active_gpu_layer_payloads):
        path = self.get(shard)
        if not os.path.isfile(path):
            raise RuntimeError(f"missing local shard: {path}")
        size = os.path.getsize(path)
        available = mem_available_bytes()
        if available - size < self.floor_bytes:
            raise MemoryError(
                f"prefetch layer {layer} would breach floor: available={available} "
                f"payload={size} floor={self.floor_bytes}")
        residual = sum(self.residency_by_shard.values())
        if residual > self.RESIDUAL_SOURCE_ALLOWANCE:
            raise RuntimeError(
                f"source residency before layer {layer} is {residual}, above "
                f"{self.RESIDUAL_SOURCE_ALLOWANCE}")
        with self.lock:
            if self.job is not None:
                raise RuntimeError("attempted a second concurrent layer prefetch")
            if self.resident_source is not None:
                raise RuntimeError(
                    f"source payload still resident before prefetch: {self.resident_source}")
            logical_payloads = int(active_gpu_layer_payloads) + 1
            if logical_payloads > 2:
                raise RuntimeError(f"payload bound exceeded: {logical_payloads}")
            self.max_observed_payloads = max(self.max_observed_payloads, logical_payloads)
            row = {
                "layer": layer,
                "shard": shard,
                "path": path,
                "bytes": size,
                "start_abs": time.perf_counter(),
                "start_epoch_ns": time.time_ns(),
                "memavailable_before_bytes": available,
                "resident_source_bytes_before_read": residual,
                "logical_resident_payloads": logical_payloads,
                "error": None,
            }

            def run():
                fd = None
                try:
                    fd = os.open(path, os.O_RDONLY)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
                    offset = 0
                    next_memory_check = 0
                    while offset < size:
                        if offset >= next_memory_check:
                            now_available = mem_available_bytes()
                            if now_available < self.floor_bytes:
                                raise MemoryError(
                                    f"prefetch memory floor: layer={layer} "
                                    f"available={now_available} floor={self.floor_bytes}")
                            next_memory_check = offset + (256 << 20)
                        data = os.pread(fd, min(self.read_chunk_bytes, size - offset), offset)
                        if not data:
                            raise OSError(f"short pread layer={layer} offset={offset} size={size}")
                        offset += len(data)
                    row["bytes_read"] = offset
                except BaseException as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                finally:
                    if fd is not None:
                        os.close(fd)
                    row["end_abs"] = time.perf_counter()
                    row["end_epoch_ns"] = time.time_ns()
                    row["io_seconds"] = row["end_abs"] - row["start_abs"]
                    row["memavailable_after_bytes"] = mem_available_bytes()

            thread = threading.Thread(
                target=run, name=f"layer-prefetch-{layer}", daemon=True)
            row["thread"] = thread
            self.job = row
            self.resident_source = shard
            thread.start()

    def wait_layer(self, layer):
        with self.lock:
            row = self.job
        if row is None or row["layer"] != layer:
            raise RuntimeError(
                f"prefetch job mismatch expected={layer} got={None if row is None else row['layer']}")
        wait_started = time.perf_counter()
        wait_started_epoch_ns = time.time_ns()
        row["thread"].join()
        wait_ended = time.perf_counter()
        row["wait_interval_epoch_ns"] = [wait_started_epoch_ns, time.time_ns()]
        row["exposed_wait_seconds"] = wait_ended - wait_started
        row.pop("thread", None)
        with self.lock:
            self.job = None
        if row["error"] is not None:
            raise RuntimeError(f"layer prefetch failed: {row['error']}")
        if row.get("bytes_read") != row["bytes"]:
            raise RuntimeError(f"layer prefetch byte mismatch: {row}")
        observed = self.observe_residency(row["shard"], "after_prefetch", layer)
        row["resident_bytes_after_read"] = observed["resident_bytes"]
        row["all_layer_sources_resident_upper_bound_bytes"] = observed[
            "all_layer_sources_resident_upper_bound_bytes"]
        allowed = self.max_layer_shard_bytes + self.RESIDUAL_SOURCE_ALLOWANCE
        if observed["all_layer_sources_resident_upper_bound_bytes"] > allowed:
            raise RuntimeError(
                f"active source residency exceeds one-file bound: {observed} allowed={allowed}")
        return row

    def drop_untracked(self, shard):
        path = self.get(shard)
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    def evict(self, shard, layer=None):
        observed = None
        for attempt in range(3):
            self.drop_untracked(shard)
            observed = self.observe_residency(
                shard, f"after_fadvise_attempt_{attempt + 1}",
                -1 if layer is None else layer)
            if sum(self.residency_by_shard.values()) <= self.RESIDUAL_SOURCE_ALLOWANCE:
                break
            time.sleep(0.02)
        if sum(self.residency_by_shard.values()) > self.RESIDUAL_SOURCE_ALLOWANCE:
            raise RuntimeError(
                f"source pages remain resident before next read: {observed}; "
                f"allowance={self.RESIDUAL_SOURCE_ALLOWANCE}")
        with self.lock:
            if self.resident_source != shard:
                raise RuntimeError(
                    f"source eviction mismatch expected={self.resident_source} got={shard}")
            self.resident_source = None



def e8m0(t):
    return torch.exp2(t.view(torch.uint8).to(torch.float32) - 127.0)



def deq_fp8_block(w, s, block=128):
    """fp8 e4m3 [N,K] + e8m0 scale [ceil(N/128),ceil(K/128)] -> bf16 [N,K]."""
    w = w.to(DEV)
    sc = e8m0(s.to(DEV))
    N, K = w.shape
    sc = sc.repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return (w.to(torch.float32) * sc).to(torch.bfloat16)



_BYTE_LUT = {}


def byte_lut(kind):
    # Native source only: no alternate snapping/codebook policy.
    if kind != "e2m1":
        raise ValueError("Balanced64 native source requires e2m1")
    if kind not in _BYTE_LUT:
        vals = E2M1_VAL
        b = torch.arange(256)
        t = torch.stack([vals[(b & 0xF)], vals[(b >> 4)]], -1)
        _BYTE_LUT[kind] = t.to(DEV)
    return _BYTE_LUT[kind]



def deq_fp4_block32(wb, sb, kind):
    """packed nibbles [.., N, K/2] u8 + e8m0 [.., N, K/32] -> bf16 [.., N, K].

    Nibble order: low nibble = even k (matches mxfp4 packing / vllm-moet).
    """
    lut = byte_lut(kind)
    vals = lut[wb.long()].flatten(-2)          # [.., N, K]
    sc = e8m0(sb).repeat_interleave(32, -1)    # [.., N, K]
    return (vals * sc).to(torch.bfloat16)



def build_layer_sd(L, wm, get_tensor, mode, planes=None, qtip=None):
    """Native ckpt keys for layer L -> HF-named tensor dict (on DEV)."""
    pre = f"layers.{L}."
    keys = [k for k in wm if k.startswith(pre)]
    sd = {}
    consumed = set()

    def T(name):
        consumed.add(pre + name)
        return get_tensor(pre + name)

    def has(name):
        return (pre + name) in wm

    def fp8(name):
        return deq_fp8_block(T(name + ".weight"), T(name + ".scale"))

    f32 = lambda name: T(name).to(DEV).to(torch.float32)
    bf = lambda name: T(name).to(DEV).to(torch.bfloat16)

    # attention core (fp8 -> bf16)
    sd["self_attn.q_a_proj.weight"] = fp8("attn.wq_a")
    sd["self_attn.q_b_proj.weight"] = fp8("attn.wq_b")
    sd["self_attn.kv_proj.weight"] = fp8("attn.wkv")
    sd["self_attn.o_a_proj.weight"] = fp8("attn.wo_a")
    sd["self_attn.o_b_proj.weight"] = fp8("attn.wo_b")
    sd["self_attn.sinks"] = f32("attn.attn_sink")
    sd["self_attn.q_a_norm.weight"] = bf("attn.q_norm.weight")
    sd["self_attn.kv_norm.weight"] = bf("attn.kv_norm.weight")
    sd["input_layernorm.weight"] = bf("attn_norm.weight")
    sd["post_attention_layernorm.weight"] = bf("ffn_norm.weight")

    # compressor (CSA/HCA layers)
    if has("attn.compressor.wkv.weight"):
        sd["self_attn.compressor.position_bias"] = f32("attn.compressor.ape")
        sd["self_attn.compressor.kv_norm.weight"] = bf("attn.compressor.norm.weight")
        sd["self_attn.compressor.kv_proj.weight"] = bf("attn.compressor.wkv.weight")
        sd["self_attn.compressor.gate_proj.weight"] = bf("attn.compressor.wgate.weight")
    # indexer (CSA layers)
    if has("attn.indexer.wq_b.weight"):
        idx = "self_attn.compressor.indexer."
        sd[idx + "position_bias"] = f32("attn.indexer.compressor.ape")
        sd[idx + "kv_norm.weight"] = bf("attn.indexer.compressor.norm.weight")
        sd[idx + "kv_proj.weight"] = bf("attn.indexer.compressor.wkv.weight")
        sd[idx + "gate_proj.weight"] = bf("attn.indexer.compressor.wgate.weight")
        sd[idx + "q_b_proj.weight"] = fp8("attn.indexer.wq_b")
        sd[idx + "scorer.weights_proj.weight"] = bf("attn.indexer.weights_proj.weight")

    # router
    sd["mlp.gate.weight"] = bf("ffn.gate.weight")
    if has("ffn.gate.tid2eid"):
        sd["mlp.gate.tid2eid"] = T("ffn.gate.tid2eid").to(DEV)
    if has("ffn.gate.bias"):
        sd["mlp.gate.e_score_correction_bias"] = f32("ffn.gate.bias")

    # hyper-connections (module floats internally; keep fp32)
    sd["attn_hc.fn"] = f32("hc_attn_fn")
    sd["attn_hc.base"] = f32("hc_attn_base")
    sd["attn_hc.scale"] = f32("hc_attn_scale")
    sd["ffn_hc.fn"] = f32("hc_ffn_fn")
    sd["ffn_hc.base"] = f32("hc_ffn_base")
    sd["ffn_hc.scale"] = f32("hc_ffn_scale")

    # shared experts (fp8, teacher path in BOTH modes)
    sd["mlp.shared_experts.gate_proj.weight"] = fp8("ffn.shared_experts.w1")
    sd["mlp.shared_experts.up_proj.weight"] = fp8("ffn.shared_experts.w3")
    sd["mlp.shared_experts.down_proj.weight"] = fp8("ffn.shared_experts.w2")

    # routed experts (fp4 block-32; W2 snap in cand mode; shipped bytes
    # in planes mode)
    E = 256
    gu = torch.empty(E, 4096, 4096, dtype=torch.bfloat16, device=DEV)
    dn = torch.empty(E, 4096, 2048, dtype=torch.bfloat16, device=DEV)
    if mode == "q4":
        qtip.fill_layer(L, gu, dn)
        for k in keys:
            if ".ffn.experts." in k:
                consumed.add(k)
    else:
        kind = "e2m1"
        CH = 8
        for e0 in range(0, E, CH):
            es = range(e0, min(e0 + CH, E))
            for wname, dst, rows in (("w1", gu, slice(0, 2048)),
                                     ("w3", gu, slice(2048, 4096)),
                                     ("w2", dn, slice(0, 4096))):
                wb = torch.stack([T(f"ffn.experts.{e}.{wname}.weight").view(torch.uint8)
                                  for e in es]).to(DEV)
                sb = torch.stack([T(f"ffn.experts.{e}.{wname}.scale").view(torch.uint8)
                                  for e in es]).to(DEV)
                dst[e0:e0 + len(es), rows] = deq_fp4_block32(wb, sb, kind)
                del wb, sb
    sd["mlp.experts.gate_up_proj"] = gu
    sd["mlp.experts.down_proj"] = dn

    missed = set(keys) - consumed
    if missed:
        raise RuntimeError(f"layer {L}: unconsumed ckpt keys: {sorted(missed)[:8]}")
    return sd



def materialize_layer(model, L, sd, config):
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4RotaryEmbedding)
    lay = model.model.layers[L]
    missing, unexpected = lay.load_state_dict(sd, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"layer {L} unexpected: {unexpected[:8]}")
    # rebuild rotary submodules (buffers were meta; deterministic from config)
    for name, mod in list(lay.named_modules()):
        if isinstance(mod, DeepseekV4RotaryEmbedding):
            parent = lay.get_submodule(name.rsplit(".", 1)[0]) if "." in name else lay
            setattr(parent, name.rsplit(".", 1)[-1],
                    DeepseekV4RotaryEmbedding(config).to(DEV))
    bad = [n for n, p in lay.named_parameters() if p.is_meta]
    bad += [n for n, b in lay.named_buffers() if b.is_meta]
    if bad:
        raise RuntimeError(f"layer {L} still meta: {bad[:8]}")
    return lay



def dematerialize_layer(model, L):
    lay = model.model.layers[L]
    for mod in lay.modules():
        for n, p in list(mod._parameters.items()):
            if p is not None:
                mod._parameters[n] = torch.nn.Parameter(
                    torch.empty(p.shape, device="meta", dtype=p.dtype),
                    requires_grad=False)
        for n, b in list(mod._buffers.items()):
            if b is not None:
                mod._buffers[n] = torch.empty(
                    b.shape, device="meta", dtype=b.dtype)
    torch.cuda.empty_cache()



def forward(a):
    process_started_epoch_ns = time.time_ns()
    process_started = time.perf_counter()
    qtip = a.source
    planes = None
    os.makedirs(a.out, exist_ok=True)
    done_path = os.path.join(a.out, "DONE.jsonl")
    corpus = json.load(open(a.corpus))
    todo = list(a.windows)
    if not todo:
        log("nothing to do")
        return 0
    log(f"mode={a.mode} todo={len(todo)} windows out={a.out}")

    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.masking_utils import create_sliding_window_causal_mask
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4RotaryEmbedding)
    from transformers.cache_utils import DynamicCache
    from safetensors import safe_open

    config = AutoConfig.from_pretrained(a.meta_dir)
    wm = json.load(open(os.path.join(a.meta_dir,
                   "model.safetensors.index.json")))["weight_map"]

    log(
        f"transformers {transformers.__version__} torch {torch.__version__} "
        f"attn={a.attn_implementation} mb={a.mb}"
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, attn_implementation=a.attn_implementation)
    model.eval()

    floor_bytes = int(a.mem_floor_gib * (1 << 30))
    if not a.local_dir:
        raise RuntimeError("overlap candidate requires the existing local checkpoint")
    cache = LocalSource(os.path.expanduser(a.local_dir), floor_bytes=floor_bytes)
    handles = {}

    def close_handles():
        while handles:
            _, handle = handles.popitem()
            handle.__exit__(None, None, None)

    def get_tensor(name):
        shard = wm[name]
        path = cache.get(shard)
        if path not in handles:
            close_handles()
            handles[path] = safe_open(path, framework="pt")
        return handles[path].get_tensor(name)

    # static parts
    log("materializing embed/head/norm/hc_head")
    for s in (wm["embed.weight"], wm["head.weight"]):
        cache.prefetch(s)
    model.model.embed_tokens.weight = torch.nn.Parameter(
        get_tensor("embed.weight").to(DEV).to(torch.bfloat16), requires_grad=False)
    model.lm_head.weight = torch.nn.Parameter(
        get_tensor("head.weight").to(DEV).to(torch.bfloat16), requires_grad=False)
    model.model.norm.weight = torch.nn.Parameter(
        get_tensor("norm.weight").to(DEV).to(torch.bfloat16), requires_grad=False)
    model.model.hc_head.hc_fn = torch.nn.Parameter(
        get_tensor("hc_head_fn").to(DEV).to(torch.float32), requires_grad=False)
    model.model.hc_head.hc_base = torch.nn.Parameter(
        get_tensor("hc_head_base").to(DEV).to(torch.float32), requires_grad=False)
    model.model.hc_head.hc_scale = torch.nn.Parameter(
        get_tensor("hc_head_scale").to(DEV).to(torch.float32), requires_grad=False)
    model.model.rotary_emb = DeepseekV4RotaryEmbedding(config).to(DEV)
    close_handles()

    NL = a.limit_layers or config.num_hidden_layers
    layer_shards = {L: sorted({wm[k] for k in wm if k.startswith(f"layers.{L}.")})
                    for L in range(NL)}
    bad_layer_shards = {L: shards for L, shards in layer_shards.items()
                        if len(shards) != 1}
    if bad_layer_shards:
        raise RuntimeError(f"one-shard-per-layer invariant failed: {bad_layer_shards}")
    # Drop all clean layer-source pages and independently observe the result with
    # mincore(2).  The gate fails before GPU work if residual source residency
    # exceeds 64 MiB; fadvise alone is not treated as proof.
    preflight_dropped_source_shards = len(layer_shards)
    preflight_resident_source_bytes = cache.preflight_residency(
        [shards[0] for shards in layer_shards.values()])

    hb = os.path.join(a.out, "HEARTBEAT")
    timing_path = a.timing_json or os.path.join(a.out, "TIMING.json")
    stage_totals = {
        "process_setup": time.perf_counter() - process_started,
        "input_prepare": 0.0,
        "weight_io": 0.0,
        "weight_io_forward_overlap": 0.0,
        "weight_io_other_overlap": 0.0,
        "weight_io_exposed_wait": 0.0,
        "layer_weight_load": 0.0,
        "layer_materialize": 0.0,
        "residency_gate": 0.0,
        "forward": 0.0,
        "layer_dematerialize": 0.0,
        "hc_norm": 0.0,
        "lm_head": 0.0,
        "normalization_nll": 0.0,
        "topk_or_sort": 0.0,
        "device_to_cpu": 0.0,
        "torch_save_rename": 0.0,
        "hash_ledger": 0.0,
        "empty_cache": 0.0,
    }
    layer_rows = []
    prefetch_rows = []
    window_rows = []
    chunk_rows = []
    memory_rows = []
    minimum_memavailable = None

    def sample_memory(phase, **detail):
        nonlocal minimum_memavailable
        available = mem_available_bytes()
        minimum_memavailable = (available if minimum_memavailable is None
                                else min(minimum_memavailable, available))
        row = {"phase": phase, "memavailable_bytes": available,
               "elapsed_seconds": time.perf_counter() - process_started, **detail}
        memory_rows.append(row)
        log(f"MEM phase={phase} available={available / (1 << 30):.2f}GiB")
        if available < floor_bytes:
            atomic_json(os.path.join(a.out, "MEMORY_STOP.json"), {
                "status": "STOP_MEMORY_FLOOR", "floor_bytes": floor_bytes,
                "observation": row, "minimum_memavailable_bytes": minimum_memavailable,
            })
            raise MemoryError(
                f"MemAvailable {available} below floor {floor_bytes} before {phase}")
        return available

    def seal_timing(status):
        wall = time.perf_counter() - process_started
        nonadditive = {"weight_io", "weight_io_forward_overlap",
                       "weight_io_other_overlap"}
        known = sum(value for name, value in stage_totals.items()
                    if name not in nonadditive)
        atomic_json(timing_path, {
            "schema": "teacher-bank-two-host-shard-profile-v1",
            "status": status,
            "host": os.uname().nodename,
            "tag": a.tag,
            "process_interval_epoch_ns": [process_started_epoch_ns, time.time_ns()],
            "clock_basis": "CLOCK_REALTIME via Python time.time_ns; CUDA stage boundaries synchronized",
            "mode": a.mode,
            "readout_mode": a.readout_mode,
            "windows": todo,
            "window_count": len(todo),
            "microbatch": a.mb,
            "chunk": a.chunk,
            "layers": NL,
            "overlap_mode": "bounded-one-layer-ahead-ext4-pread-pagecache-mincore-observed",
            "preflight_dropped_source_shards": preflight_dropped_source_shards,
            "preflight_resident_source_bytes": preflight_resident_source_bytes,
            "max_resident_layer_payloads_allowed": 2,
            "max_resident_layer_payloads_observed": cache.max_observed_payloads,
            "resident_source_allowance_bytes": cache.RESIDUAL_SOURCE_ALLOWANCE,
            "max_layer_shard_bytes": cache.max_layer_shard_bytes,
            "max_actual_resident_source_bytes": cache.max_actual_resident_source_bytes,
            "stage_seconds": stage_totals,
            "accounted_seconds": known,
            "wall_seconds": wall,
            "other_unaccounted_seconds": wall - known,
            "minimum_memavailable_bytes": minimum_memavailable,
            "memory_floor_bytes": floor_bytes,
            "memory_rows": memory_rows,
            "residency_rows": cache.residency_rows,
            "prefetch_rows": prefetch_rows,
            "layer_rows": layer_rows,
            "window_rows": window_rows,
            "chunk_rows": chunk_rows,
            "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        })

    sample_memory("profile_start")
    t_run = time.perf_counter()
    for c0 in range(0, len(todo), a.chunk):
        wins = todo[c0:c0 + a.chunk]
        t_chunk = time.perf_counter()
        chunk_started_epoch_ns = time.time_ns()
        log(f"chunk {c0//a.chunk}: windows {wins[0]}..{wins[-1]} (n={len(wins)})")
        tp = time.perf_counter()
        ids = torch.full((len(wins), 2048), 1, dtype=torch.long)
        rlens = []
        for i, k in enumerate(wins):
            t = corpus[k]["token_ids"]
            ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
            rlens.append(corpus[k]["real_len"])
        ids = ids.to(DEV)
        pos = torch.arange(2048, device=DEV).unsqueeze(0)
        torch.cuda.synchronize()
        stage_totals["input_prepare"] += time.perf_counter() - tp

        mbs = [slice(i, min(i + a.mb, len(wins)))
               for i in range(0, len(wins), a.mb)]
        with torch.no_grad():
            embeds = model.model.embed_tokens(ids)
            pe = {
                "main": model.model.rotary_emb(
                    embeds[:1], position_ids=pos, layer_type="main"),
                "compress": model.model.rotary_emb(
                    embeds[:1], position_ids=pos, layer_type="compress"),
            }
            caches = [DynamicCache(config=config) for _ in mbs]
            masks, hidden = [], []
            for mi, s in enumerate(mbs):
                masks.append(create_sliding_window_causal_mask(
                    config=config, inputs_embeds=embeds[s],
                    attention_mask=None, past_key_values=caches[mi],
                    position_ids=pos))
                hidden.append(embeds[s].unsqueeze(2).expand(
                    -1, -1, config.hc_mult, -1).contiguous())
            del embeds

            previous_forward_interval = None
            cache.begin_layer(0, layer_shards[0][0], active_gpu_layer_payloads=0)
            for L in range(NL):
                sample_memory("before_layer", chunk=c0 // a.chunk, layer=L)
                prefetch = cache.wait_layer(L)
                if previous_forward_interval is None:
                    forward_overlap = 0.0
                else:
                    forward_overlap = max(
                        0.0,
                        min(prefetch["end_abs"], previous_forward_interval[1])
                        - max(prefetch["start_abs"], previous_forward_interval[0]),
                    )
                other_overlap = max(
                    0.0,
                    prefetch["io_seconds"] - forward_overlap
                    - prefetch["exposed_wait_seconds"],
                )
                prefetch["forward_overlap_seconds"] = forward_overlap
                prefetch["other_overlap_seconds"] = other_overlap
                prefetch["chunk"] = c0 // a.chunk
                prefetch_rows.append(prefetch)
                stage_totals["weight_io"] += prefetch["io_seconds"]
                stage_totals["weight_io_forward_overlap"] += forward_overlap
                stage_totals["weight_io_other_overlap"] += other_overlap
                stage_totals["weight_io_exposed_wait"] += prefetch["exposed_wait_seconds"]

                t0 = cuda_time()
                e0 = time.time_ns()
                sd = build_layer_sd(L, wm, get_tensor, a.mode, planes, qtip)
                close_handles()
                t1 = cuda_time()
                e1 = time.time_ns()
                lay = materialize_layer(model, L, sd, config)
                del sd
                t2 = cuda_time()
                e2 = time.time_ns()
                residency_gate_started = time.perf_counter()
                er0 = time.time_ns()
                cache.evict(layer_shards[L][0], layer=L)
                er1 = time.time_ns()
                residency_gate_seconds = time.perf_counter() - residency_gate_started
                if L + 1 < NL:
                    cache.begin_layer(
                        L + 1, layer_shards[L + 1][0],
                        active_gpu_layer_payloads=1)
                tf0 = cuda_time()
                ef0 = time.time_ns()
                for mi, s in enumerate(mbs):
                    hidden[mi] = lay(
                        hidden[mi], position_embeddings=pe, position_ids=pos,
                        attention_mask=masks[mi], input_ids=ids[s],
                        past_key_values=caches[mi])
                t3 = cuda_time()
                e3 = time.time_ns()
                previous_forward_interval = (tf0, t3)
                dematerialize_layer(model, L)
                t4 = cuda_time()
                e4 = time.time_ns()
                row = {"chunk": c0 // a.chunk, "layer": L,
                       "weight_io_seconds": prefetch["io_seconds"],
                       "weight_io_forward_overlap_seconds": forward_overlap,
                       "weight_io_exposed_wait_seconds": prefetch["exposed_wait_seconds"],
                       "weight_load_seconds": t1 - t0,
                       "materialize_seconds": t2 - t1,
                       "residency_gate_seconds": residency_gate_seconds,
                       "forward_seconds": t3 - tf0,
                       "dematerialize_seconds": t4 - t3,
                       "intervals_epoch_ns": {
                           "weight_build": [e0, e1],
                           "materialize": [e1, e2],
                           "residency_gate": [er0, er1],
                           "forward": [ef0, e3],
                           "dematerialize": [e3, e4],
                       },
                       "memavailable_bytes": mem_available_bytes()}
                layer_rows.append(row)
                stage_totals["layer_weight_load"] += t1 - t0
                stage_totals["layer_materialize"] += t2 - t1
                stage_totals["residency_gate"] += residency_gate_seconds
                stage_totals["forward"] += t3 - tf0
                stage_totals["layer_dematerialize"] += t4 - t3
                log(f"  L{L:02d} io {prefetch['io_seconds']:5.1f}s "
                    f"ovl {forward_overlap:4.1f}s wait {prefetch['exposed_wait_seconds']:4.1f}s "
                    f"load {t1-t0:5.1f}s mat {t2-t1:4.1f}s "
                    f"fwd {t3-tf0:5.1f}s demat {t4-t3:4.1f}s "
                    f"gpu_res {torch.cuda.memory_reserved()>>30}G")
                with open(hb, "w") as f:
                    json.dump({"chunk": c0 // a.chunk, "layer": L,
                               "ts": time.time()}, f)

            # readout
            for mi, s in enumerate(mbs):
                sample_memory("before_readout_batch", chunk=c0 // a.chunk,
                              microbatch_index=mi)
                tr0 = cuda_time()
                h = model.model.norm(model.model.hc_head(hidden[mi]))
                tr1 = cuda_time()
                stage_totals["hc_norm"] += tr1 - tr0
                for j in range(h.shape[0]):
                    k = wins[s.start + j]
                    rl = rlens[s.start + j]
                    tw0 = cuda_time()
                    readout_started_epoch_ns = time.time_ns()
                    logits = model.lm_head(h[j, :rl].to(torch.bfloat16)).float()
                    tw1 = cuda_time()
                    npos = min(1024, rl - 1)
                    tgt = ids[s.start + j, 1:npos + 1]
                    lp = torch.log_softmax(logits, dim=-1)
                    nll = -lp[:npos].gather(1, tgt.unsqueeze(1)).mean().item()
                    tw2 = cuda_time()
                    ref = torch.load(os.path.join(
                        a.ref_dir, f"t8192_win{k}.pt"), map_location=DEV, weights_only=True)
                    P = min(rl, a.cand_pos_limit) if a.cand_pos_limit else rl
                    ridx = ref["idx"].long()[:P]
                    obj = {"q_lp_at_ref": lp[:P].gather(1, ridx).to(
                               torch.float16).cpu(),
                           "q_argmax": lp[:P].gather(1, ridx).to(torch.float16).argmax(-1).to(torch.int32).cpu()}
                    torch.cuda.synchronize()
                    tw3 = time.perf_counter()
                    tw4 = tw3
                    fname = f"q8192_win{k}.pt"
                    del ref, ridx
                    out_p = os.path.join(a.out, fname)
                    ts0 = time.perf_counter()
                    torch.save(obj, out_p + ".tmp")
                    os.replace(out_p + ".tmp", out_p)
                    ts1 = time.perf_counter()
                    jrow(done_path, win=k, file=fname, md5=md5(out_p),
                         real_len=rl, npos=npos, nll1024=round(nll, 5),
                         mode=a.mode, tag=a.tag)
                    ts2 = time.perf_counter()
                    readout_ended_epoch_ns = time.time_ns()
                    stage_totals["lm_head"] += tw1 - tw0
                    stage_totals["normalization_nll"] += tw2 - tw1
                    stage_totals["topk_or_sort"] += tw3 - tw2
                    stage_totals["device_to_cpu"] += tw4 - tw3
                    stage_totals["torch_save_rename"] += ts1 - ts0
                    stage_totals["hash_ledger"] += ts2 - ts1
                    window_rows.append({
                        "window": k, "real_len": rl, "npos": npos,
                        "readout_interval_epoch_ns": [
                            readout_started_epoch_ns, readout_ended_epoch_ns],
                        "nll1024": nll, "lm_head_seconds": tw1 - tw0,
                        "normalization_nll_seconds": tw2 - tw1,
                        "topk_or_sort_seconds": tw3 - tw2,
                        "device_to_cpu_seconds": tw4 - tw3,
                        "torch_save_rename_seconds": ts1 - ts0,
                        "hash_ledger_seconds": ts2 - ts1,
                        "output_bytes": os.path.getsize(out_p),
                    })
                    del logits, obj
                    del lp
                del h
            del hidden, caches, masks
            te0 = cuda_time()
            torch.cuda.empty_cache()
            te1 = cuda_time()
            stage_totals["empty_cache"] += te1 - te0
        sample_memory("chunk_done", chunk=c0 // a.chunk)
        chunk_ended_epoch_ns = time.time_ns()
        chunk_rows.append({
            "chunk": c0 // a.chunk, "windows": wins,
            "interval_epoch_ns": [chunk_started_epoch_ns, chunk_ended_epoch_ns],
            "wall_seconds": time.perf_counter() - t_chunk,
        })
        seal_timing("RUNNING")
        log(f"chunk done in {(time.perf_counter()-t_chunk)/60:.1f} min "
            f"({len(wins)} windows)")

    seal_timing("PASS")
    log(f"ALL DONE {len(todo)} windows in {(time.perf_counter()-t_run)/60:.1f} min")
    return 0


