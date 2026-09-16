# Finite max2 forkserver experiment

Exact physically exercised scripts, not a production API. Task-local absolute paths and ownership/CAS gates are intentional. Do not execute on other hosts or substitute production claims. CPU-only Torch forkserver preload; each serial GPU child owns two original jobs.

One enclosing observation: fresh-process control 36.22119022498373s; forkserver (including startup) 35.46539227501489s; ratio 1.0213108583181043. All four matched runtime quality ratios1.0. Historical E096 PRE remains failed; these controls do not replace original K1 references. No promotion from one observation. Runtime cap30GiB AS/4GiB CUDA; parent admission adds768MiB. Actual executor/group closure was verified.
