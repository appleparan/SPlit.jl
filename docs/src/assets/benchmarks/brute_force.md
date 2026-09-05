## Twinning: search structure wall time

First call: one warm-up per `p`, on a 500-row/100-group slice, in this process.

| p | k-d tree first call (s) | brute tree first call (s) | matrix first call (s) |
|---:|---:|---:|---:|
| 50 | 0.77 | 0.159 | 0.403 |
| 200 | 1.73 | 0.181 | 0.00183 |
| 768 | 13.7 | 0.201 | 0.0293 |

| N | p | k-d tree (s) | brute tree (s) | matrix (s) | brute/matrix | kdtree/matrix |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 50 | 0.0081 | 0.00462 | 0.00497 | 0.93 | 1.63 |
| 10000 | 50 | 0.64 | 0.318 | 0.36 | 0.884 | 1.78 |
| 1000 | 200 | 0.0267 | 0.0121 | 0.00623 | 1.94 | 4.29 |
| 10000 | 200 | 1.49 | 1.12 | 0.647 | 1.72 | 2.3 |
| 1000 | 768 | 0.247 | 0.0379 | 0.0171 | 2.22 | 14.4 |
| 10000 | 768 | 8.43 | 4.16 | 2.4 | 1.74 | 3.52 |
| 100000 | 50 | 74.7 | 39.3 | 39.3 | 1.0 | 1.9 |

## select_nearest: search structure wall time

First call: one warm-up per row, on a 500-row/100-point slice, in this process.

| N | p | k-d tree first call (s) | matrix first call (s) | k-d tree (s) | matrix (s) | kdtree/matrix |
|---:|---:|---:|---:|---:|---:|---:|
| 10000 | 2 | 0.00482 | 7.58e-5 | 0.00222 | 0.0287 | 0.0774 |
| 10000 | 10 | 0.355 | 0.000366 | 0.0074 | 0.0681 | 0.109 |
| 10000 | 50 | 0.174 | 0.000563 | 0.0417 | 0.217 | 0.192 |
| 10000 | 200 | 0.333 | 0.00087 | 0.375 | 0.371 | 1.01 |
| 10000 | 768 | 1.69 | 0.00321 | 5.72 | 1.73 | 3.31 |
| 100000 | 10 | 0.000239 | 0.000166 | 0.174 | 6.88 | 0.0254 |
| 100000 | 50 | 0.000747 | 0.000538 | 2.3 | 23.0 | 0.1 |

## First call at extreme width (`:matrix` only — the widths BruteTree/KDTree could not compile)

| p | twinning first call (s) | select_nearest first call (s) |
|---:|---:|---:|
| 3072 | 0.103 | 0.00247 |
| 6144 | 0.00658 | 0.00506 |
| 12288 | 0.0161 | 0.0135 |
