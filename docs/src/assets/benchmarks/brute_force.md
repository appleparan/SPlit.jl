## Twinning: search structure wall time

First call: one warm-up per `p`, on a 500-row/100-group slice, in this process.

| p | k-d tree first call (s) | brute tree first call (s) | matrix first call (s) |
|---:|---:|---:|---:|
| 50 | 0.72 | 0.151 | 0.286 |
| 200 | 1.6 | 0.179 | 0.00146 |
| 768 | 13.5 | 0.188 | 0.0253 |

| N | p | k-d tree (s) | brute tree (s) | matrix (s) | brute/matrix | kdtree/matrix |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 50 | 0.0081 | 0.00445 | 0.00353 | 1.26 | 2.3 |
| 10000 | 50 | 0.632 | 0.314 | 0.323 | 0.973 | 1.95 |
| 1000 | 200 | 0.0279 | 0.0114 | 0.00555 | 2.06 | 5.02 |
| 10000 | 200 | 1.45 | 1.09 | 0.663 | 1.65 | 2.19 |
| 1000 | 768 | 0.235 | 0.0359 | 0.0158 | 2.27 | 14.9 |
| 10000 | 768 | 8.89 | 4.15 | 2.17 | 1.91 | 4.09 |
| 100000 | 50 | 70.0 | 39.6 | 34.8 | 1.14 | 2.01 |

## select_nearest: search structure wall time

First call: one warm-up per row, on a 500-row/100-point slice, in this process.

| N | p | k-d tree first call (s) | matrix first call (s) | k-d tree (s) | matrix (s) | kdtree/matrix |
|---:|---:|---:|---:|---:|---:|---:|
| 10000 | 2 | 0.00499 | 7.71e-5 | 0.00262 | 0.0305 | 0.0857 |
| 10000 | 10 | 0.335 | 0.000231 | 0.00793 | 0.0921 | 0.0861 |
| 10000 | 50 | 0.167 | 0.000546 | 0.0411 | 0.221 | 0.185 |
| 10000 | 200 | 0.321 | 0.000924 | 0.363 | 0.429 | 0.847 |
| 10000 | 768 | 1.62 | 0.00345 | 4.95 | 1.73 | 2.86 |
| 100000 | 10 | – | – | 0.173 | 9.4 | 0.0184 |
| 100000 | 50 | – | – | 2.32 | 22.9 | 0.102 |

First-call columns show "–" for the 100000-row rows: their widths (10, 50) already ran, and compiled, earlier in this process at the 10000-row rows above, so no genuine first call remains to measure there.

## First call at extreme width (`:matrix` only — the widths BruteTree/KDTree could not compile)

Each width runs `selectrows` in a fresh Julia process, so both columns are genuine first calls.

| p | twinning first call (s) | select_nearest first call (s) |
|---:|---:|---:|
| 3072 | 1.18 | 1.84 |
| 6144 | 1.27 | 1.82 |
| 12288 | 1.35 | 1.85 |
