# DeepSeek-V4-Flash-0731 HumanEval results

| Model | HumanEval pass@1 | HumanEval+ pass@1 | Weight size (decimal GB) | Effective BPW |
|---|---:|---:|---:|---:|
| **Official, Thinking ON** | **96.95%** (159/164) | **92.68%** (152/164) | 166.887 | 4.695 |
| **UD-IQ4_XS, Thinking ON** | **96.95%** (159/164) | **92.68%** (152/164) | 136.662 | 3.845 |
| **UD-IQ3_XXS, Thinking ON** | **94.51%** (155/164) | **90.24%** (148/164) | 104.208 | 2.932 |
| **DwarfStar IQ2XXS, target-only** | **93.90%** (154/164) | **87.80%** (144/164) | 90.861 | 2.556 |

Each score is actual benchmark accuracy from one generated solution per task. It is not token Top-1 agreement or best-of-n.

Effective BPW is exact weight bytes × 8 divided by the common 284,334,567,511-parameter base-model denominator. The DwarfStar result is target-only; its complete base+drafter shipment is 93.691 GB and 2.636 effective BPW.
