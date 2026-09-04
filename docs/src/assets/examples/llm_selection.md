| setting | method | energy distance to the optimized measure | energy distance to the data | seconds |
|---|---|---:|---:|---:|
| plain | random | 0.0024 | 0.0024 | – |
| plain | k-center greedy | 0.0228 | 0.0228 | 3.6 |
| plain | herding · energy | 0.00107 | 0.00107 | 1.5 |
| plain | twinning | 0.00166 | 0.00166 | 0.28 |
| plain | kernel thinning · energy | 0.00123 | 0.00123 | 3.1 |
| plain | support points · energy | 0.00231 | 0.00231 | 8.3 |
| weights = length | random | 0.00504 | 0.0024 | – |
| weights = length | k-center greedy | 0.0336 | 0.0228 | 3.6 |
| weights = length | herding · energy | 0.00103 | 0.00439 | 1.5 |
| weights = length | kernel thinning · energy | 0.00109 | 0.00449 | 5 |
| weights = length | support points · energy | 0.00632 | 0.00231 | 8.4 |
| reference = cs | random | 0.147 | 0.0024 | – |
| reference = cs | k-center greedy | 0.0909 | 0.0228 | 3.6 |
| reference = cs | herding · energy | 0.00818 | 0.0994 | 0.96 |
| reference = cs | kernel thinning · energy | 0.00812 | 0.101 | 4.7 |
| reference = cs | support points · energy | 0.133 | 0.00243 | 4.6 |
| plain, n = 250 | kernel thinning · compress = :never | 0.00245 | 0.00245 | 4.5 |
| plain, n = 250 | kernel thinning · compress = :always | 0.00261 | 0.00261 | 3.1 |
| plain, n = 250 | random | 0.00485 | 0.00485 | – |
