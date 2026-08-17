# green-run1978-u5

> REPRODUCTION

```text
green-run1978-u5: REPRODUCTION
parent: UPDATE_005 sha256=d798a6535d1d9009d584e706d295147c112fcb1c816517685a4a2cdb6c7a3fbd; next_update=5
groups: first U1 W20-23; U5 W36-39; next W40-43; 4 windows/step
LR: Adam LUT/gain=0.00986514063938 norm=9.86514063938e-05; warmup=0; cosine min=0.1 over 64
data: prompt=green-training-prompt-binding; corpus=green-training-corpus-binding; teacher=green-training-teacher-binding; batch=4; scaling=mean; no extra 4/64 factor
scorer: evaluations/configs/balanced64-v1.json sha256=d5610f11c23b75f81e196e74407cb7e642a4f4a2e12f55925e13e5a7fe43ffb9; BALANCED64_V1 suite lock; KLD and Top1 semantics are owned by the referenced lock
scientific_identity_sha256: 47bcd256b62555ed1cc39604d6a5d82433385d3b75359c87e949b6672b6588ae
execution: device=cuda; kernel=QTIP-V7 public update/repair
```

## Scientific surfaces

- Mutable: 43 layer LUTs, 235 RMSNorm masters, 43 output gains
- Frozen: codes, assignments, scales, geometry
- Evaluation authority: `evaluations/configs/balanced64-v1.json` (`d5610f11c23b75f81e196e74407cb7e642a4f4a2e12f55925e13e5a7fe43ffb9`)

The evaluation protocol is referenced by suite-lock identity rather than copied here.
