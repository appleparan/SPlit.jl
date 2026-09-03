# Rounding-step experiment

`SupportPointSplitter` optimizes continuous support points and then maps
each one to its nearest unclaimed data row (`select_nearest`, sequential
nearest-neighbor, Joseph & Vakayil 2022). Points are initialized at a
random sample of data rows, jittered by 0.1% of the per-dimension range
(`_initial_points`). This experiment measures, per (dataset, N), how far
the optimizer moves the points relative to the spacing between data rows,
whether that leaves the rounded selection identical to the initial
sample, and whether starting away from data rows (uniform in the
bounding box, or heavy jitter around the initial sample) changes the
outcome. It also reproduces `run.jl`'s own `datasplit` path for
`support points · gaussian` (rows marked "(datasplit path)"): `datasplit`
resolves the `:median` bandwidth from the splitter's `rng` before
`_initial_points` draws from it, so its initial sample is a different
draw from the fresh-`rng` "initial sample" row above it. `*` marks
iteration counts that hit `max_iterations` without converging. `rows
kept` counts selected rows shared with the relevant initial sample (the
fresh-`rng` one, or `init_ds` for the "(datasplit path)" row) out of the
method's own subset size.

| dataset | N | method | iterations | median move | continuous MMD | rows kept | test-vs-train MMD | test-vs-train energy distance |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| mixture-2d | 1000 | random (5 seeds) | – | – | – | – | 0.00258 | 0.012 |
| mixture-2d | 1000 | initial sample | – | – | – | – | 0.0012 | 0.00642 |
| mixture-2d | 1000 | initial sample (datasplit path) | – | – | – | – | 0.0012 | 0.00642 |
| mixture-2d | 1000 | support points · gaussian (datasplit path) | 25 | – | – | 68/200 | 3.21e-5 | 0.00264 |
| mixture-2d | 1000 | support points · gaussian | 25 | 0.0913 | 5.71e-7 | 68/200 | 3.21e-5 | 0.00264 |
| mixture-2d | 1000 | support points · energy | 500* | 0.109 | 1.81e-7 | 37/200 | 1.57e-6 | 0.000323 |
| mixture-2d | 1000 | gaussian, uniform-box init | 95 | 0.226 | 4.17e-6 | 44/200 | 2.14e-5 | 0.00837 |
| mixture-2d | 1000 | gaussian, heavy-jitter init | 117 | 0.412 | 2.26e-6 | 43/200 | 2.4e-5 | 0.00843 |
| normal-10d | 1000 | random (5 seeds) | – | – | – | – | 0.00227 | 0.0255 |
| normal-10d | 1000 | initial sample | – | – | – | – | 0.00224 | 0.0247 |
| normal-10d | 1000 | initial sample (datasplit path) | – | – | – | – | 0.00224 | 0.0247 |
| normal-10d | 1000 | support points · gaussian (datasplit path) | 138 | – | – | 200/200 | 0.00224 | 0.0247 |
| normal-10d | 1000 | support points · gaussian | 138 | 0.575 | 1.19e-5 | 200/200 | 0.00224 | 0.0247 |
| normal-10d | 1000 | support points · energy | 500* | 0.682 | 2.28e-5 | 185/200 | 0.00187 | 0.0215 |
| normal-10d | 1000 | gaussian, uniform-box init | 96 | 2.99 | 2.4e-5 | 39/200 | 0.00333 | 0.0325 |
| normal-10d | 1000 | gaussian, heavy-jitter init | 139 | 3.25 | 1.84e-5 | 46/200 | 0.00381 | 0.0372 |
| uniform-5d | 1000 | random (5 seeds) | – | – | – | – | 0.00217 | 0.0173 |
| uniform-5d | 1000 | initial sample | – | – | – | – | 0.00194 | 0.0161 |
| uniform-5d | 1000 | initial sample (datasplit path) | – | – | – | – | 0.00194 | 0.0161 |
| uniform-5d | 1000 | support points · gaussian (datasplit path) | 38 | – | – | 199/200 | 0.00189 | 0.0158 |
| uniform-5d | 1000 | support points · gaussian | 38 | 0.248 | 6.08e-6 | 199/200 | 0.00189 | 0.0158 |
| uniform-5d | 1000 | support points · energy | 500* | 0.491 | 1.89e-6 | 96/200 | 0.000255 | 0.0047 |
| uniform-5d | 1000 | gaussian, uniform-box init | 64 | 0.354 | 4.82e-6 | 37/200 | 0.000316 | 0.00685 |
| uniform-5d | 1000 | gaussian, heavy-jitter init | 71 | 0.587 | 3.9e-6 | 36/200 | 0.000243 | 0.0067 |
| t3-3d | 1000 | random (5 seeds) | – | – | – | – | 0.00375 | 0.0161 |
| t3-3d | 1000 | initial sample | – | – | – | – | 0.00397 | 0.0149 |
| t3-3d | 1000 | initial sample (datasplit path) | – | – | – | – | 0.00397 | 0.0149 |
| t3-3d | 1000 | support points · gaussian (datasplit path) | 104 | – | – | 114/200 | 0.000841 | 0.00586 |
| t3-3d | 1000 | support points · gaussian | 104 | 0.154 | 6.7e-5 | 114/200 | 0.000841 | 0.00586 |
| t3-3d | 1000 | support points · energy | 500* | 0.244 | 4.4e-5 | 42/200 | 0.000116 | 0.00163 |
| t3-3d | 1000 | gaussian, uniform-box init | 200* | 2.75 | 0.105 | 40/200 | 0.0578 | 0.234 |
| t3-3d | 1000 | gaussian, heavy-jitter init | 200* | 3.18 | 0.0978 | 43/200 | 0.0449 | 0.151 |
| mixture-2d | 10000 | random (5 seeds) | – | – | – | – | 0.000166 | 0.000885 |
| mixture-2d | 10000 | initial sample | – | – | – | – | 0.000267 | 0.00125 |
| mixture-2d | 10000 | initial sample (datasplit path) | – | – | – | – | 0.000168 | 0.000813 |
| mixture-2d | 10000 | support points · gaussian (datasplit path) | 14 | – | – | 675/2000 | 1.74e-6 | 0.0003 |
| mixture-2d | 10000 | support points · gaussian | 38 | 0.0405 | 1.38e-7 | 504/2000 | 6.36e-7 | 0.000493 |
| mixture-2d | 10000 | support points · energy | 500* | 0.0421 | 2.25e-5 | 411/2000 | 3.2e-5 | 0.000173 |
| mixture-2d | 10000 | support points · energy, full data | 100* | 0.0362 | 4.79e-8 | 437/2000 | 1.88e-7 | 1.98e-5 |
| mixture-2d | 10000 | gaussian, uniform-box init | 78 | 0.223 | 5.01e-6 | 385/2000 | 7.85e-6 | 0.00735 |
| mixture-2d | 10000 | gaussian, heavy-jitter init | 100* | 0.522 | 4.42e-6 | 403/2000 | 9.18e-6 | 0.0084 |
| normal-10d | 10000 | random (5 seeds) | – | – | – | – | 0.000164 | 0.00208 |
| normal-10d | 10000 | initial sample | – | – | – | – | 0.000215 | 0.0025 |
| normal-10d | 10000 | initial sample (datasplit path) | – | – | – | – | 0.000289 | 0.00299 |
| normal-10d | 10000 | support points · gaussian (datasplit path) | 35 | – | – | 2000/2000 | 0.000289 | 0.00299 |
| normal-10d | 10000 | support points · gaussian | 52 | 0.144 | 2.29e-6 | 2000/2000 | 0.000215 | 0.0025 |
| normal-10d | 10000 | support points · energy | 500* | 0.29 | 7.72e-5 | 2000/2000 | 0.000215 | 0.0025 |
| normal-10d | 10000 | support points · energy, full data | 100* | 0.199 | 1.88e-6 | 2000/2000 | 0.000215 | 0.0025 |
| normal-10d | 10000 | gaussian, uniform-box init | 100* | 4.06 | 1.84e-5 | 395/2000 | 0.003 | 0.0258 |
| normal-10d | 10000 | gaussian, heavy-jitter init | 100* | 4.25 | 2.33e-5 | 409/2000 | 0.00268 | 0.0243 |
| uniform-5d | 10000 | random (5 seeds) | – | – | – | – | 0.00016 | 0.00146 |
| uniform-5d | 10000 | initial sample | – | – | – | – | 0.000151 | 0.00147 |
| uniform-5d | 10000 | initial sample (datasplit path) | – | – | – | – | 0.000244 | 0.00187 |
| uniform-5d | 10000 | support points · gaussian (datasplit path) | 40 | – | – | 1998/2000 | 0.000243 | 0.00187 |
| uniform-5d | 10000 | support points · gaussian | 40 | 0.0845 | 5.1e-7 | 1998/2000 | 0.000151 | 0.00146 |
| uniform-5d | 10000 | support points · energy | 500* | 0.195 | 4.41e-5 | 1731/2000 | 7.24e-5 | 0.000844 |
| uniform-5d | 10000 | support points · energy, full data | 100* | 0.142 | 9.03e-8 | 1929/2000 | 0.000131 | 0.0013 |
| uniform-5d | 10000 | gaussian, uniform-box init | 44 | 0.106 | 6.08e-7 | 378/2000 | 3.37e-5 | 0.000723 |
| uniform-5d | 10000 | gaussian, heavy-jitter init | 39 | 0.406 | 1.06e-6 | 404/2000 | 1.48e-5 | 0.000695 |
| t3-3d | 10000 | random (5 seeds) | – | – | – | – | 0.000305 | 0.00151 |
| t3-3d | 10000 | initial sample | – | – | – | – | 0.000161 | 0.000919 |
| t3-3d | 10000 | initial sample (datasplit path) | – | – | – | – | 0.000219 | 0.00114 |
| t3-3d | 10000 | support points · gaussian (datasplit path) | 55 | – | – | 1521/2000 | 8.06e-5 | 0.000583 |
| t3-3d | 10000 | support points · gaussian | 43 | 0.0448 | 5.06e-6 | 1605/2000 | 7.1e-5 | 0.000575 |
| t3-3d | 10000 | support points · energy | 500* | 0.0895 | 6.52e-5 | 789/2000 | 4.82e-5 | 0.000262 |
| t3-3d | 10000 | support points · energy, full data | 100* | 0.0701 | 3.4e-6 | 1105/2000 | 2.87e-5 | 0.00023 |
| t3-3d | 10000 | gaussian, uniform-box init | 100* | 0.473 | 0.477 | 383/2000 | 0.227 | 0.517 |
| t3-3d | 10000 | gaussian, heavy-jitter init | 100* | 0.465 | 0.484 | 382/2000 | 0.23 | 0.514 |

| dataset | N | median nearest-neighbor spacing |
|---|---:|---:|
| mixture-2d | 1000 | 0.036 |
| normal-10d | 1000 | 1.8 |
| uniform-5d | 1000 | 0.621 |
| t3-3d | 1000 | 0.161 |
| mixture-2d | 10000 | 0.0112 |
| normal-10d | 10000 | 1.37 |
| uniform-5d | 10000 | 0.376 |
| t3-3d | 10000 | 0.0772 |
