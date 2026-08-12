# MMLU-500 Evals intelligence density

All seven rows use the immutable `mmlu500-v1` bank: 500 ordered zero-shot literal prompts, no chat template or answer generation, and final-position A/B/C/D logits normalized over the four choices. The original Unsloth IQ4, IQ3, IQ2, and DwarfStar Q2 aggregates and public evidence remain unchanged. The Official native MXFP4 reference and the routed-only EXL3 K2/native-rest and K3/native-rest rows were independently reaggregated from sealed 500-row terminals.

| Evals row | MMLU | MMLU % | Gold CE (bits) | Complete bytes | Decimal GB | Base-eq BPW | MMLU/BPW | MMLU/GB | MMLU/BPW vs Unsloth IQ4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Official native MXFP4 | 423/500 | 84.60% | 20.897371 | 156035165824 | 156.035165824 | 4.3901849061799633842692039291812057173773172588621 | 13.575737986822021169770638279576235636557835123104 | 0.38196517871635341134624870650594092581056761872936 | 0.8938x |
| Unsloth IQ4 | 417/500 | 83.40% | 0.701727 | 136662446656 | 136.662446656 | 3.8451166272834685 | 15.188095878709130567915577771526674434593682062595 | 0.42733026832895515404579703590229275164660783928765 | 1.0000x |
| EXL3 K3 routed-only + native rest | 426/500 | 85.20% | 0.697600 | 123999250168 | 123.999250168 | 3.488881932423359811648345096334173619526617322555388135469206037221313139582164 | 17.254811474283792491553855724714163461485791821051 | 0.48548680672212304889491934657389903386984165073471 | 1.1361x |
| Unsloth IQ3 | 416/500 | 83.20% | 0.749950 | 104207848032 | 104.207848032 | 2.931978308348837 | 19.850078642899549759031453844135583807795782420652 | 0.55849920230698963792224489259665129479410378541344 | 1.3069x |
| EXL3 K2 routed-only + native rest | 418/500 | 83.60% | 0.750380 | 89371076344 | 89.371076344 | 2.5145328512486971484262613667868966546438084310621785627887683259240843040121566 | 23.304527507325944441799885144281350158156488410122 | 0.65569312127831566311520532536017993199609797014576 | 1.5344x |
| Unsloth IQ2 | 409/500 | 81.80% | 0.842842 | 90860736928 | 90.860736928 | 2.556445745541928 | 22.218347523725466492605930658855055502031785729095 | 0.62513250409810719777744834097016272881268524681363 | 1.4629x |
| DwarfStar Q2 0731 | 403/500 | 80.60% | 0.809176 | 93691352992 | 93.691352992 | 2.6360875868777476 | 21.091863668253194105236973215575887670359061422882 | 0.59343790247908473709236195891779784278856964251876 | 1.3887x |

`MMLU/BPW` is `(MMLU percentage - 25) / exact base-equivalent comparison BPW`. `MMLU/GB` is `(MMLU percentage - 25) / complete decimal artifact GB`. Both report MMLU percentage points above chance per denominator. The exact machine-readable denominators, not rounded display values, are used. DwarfStar's complete-GB denominator remains the complete base-plus-drafter Evals payload for the original row. The Official row is base-only with native MTP and any drafter excluded. Its BALANCED64 Top-1/KLD cells remain blank because only its same-bank MMLU terminal is sealed; no comparison score is inferred from its reference role.

Machine-readable aggregates and public-safe provenance are in [`results.json`](results.json), with the seven-row contract in [`results.schema.json`](results.schema.json). The frozen prompts are [`items.jsonl`](items.jsonl), and the original four-row model basis remains [`four-row-mission-basis.json`](four-row-mission-basis.json).

The three new rows bind only public-safe identities: the fixed bank hash `df6704c4d02550b9155e106bc9a9e1bfe1164a663d509e41a76736bb60d01ded`, source model index hash `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`, and per-terminal/qrow hashes recorded in `results.json`. No task IDs, private paths, hostnames, or transfer paths are part of the public record.
