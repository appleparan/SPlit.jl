| setting | method | energy distance to the optimized measure | energy distance to the data | seconds |
|---|---|---:|---:|---:|
| plain | random | 0.0024 | 0.0024 | – |
| plain | k-center greedy | 0.0228 | 0.0228 | 1.9 |
| plain | herding · energy | 0.00107 | 0.00107 | 1.5 |
| plain | twinning | 0.00166 | 0.00166 | 0.43 |
| plain | kernel thinning · energy | 0.00122 | 0.00122 | 3.1 |
| plain | support points · energy | 0.00239 | 0.00239 | 8.6 |
| weights = length | random | 0.00504 | 0.0024 | – |
| weights = length | k-center greedy | 0.0336 | 0.0228 | 2 |
| weights = length | herding · energy | 0.00103 | 0.00439 | 1.5 |
| weights = length | kernel thinning · energy | 0.00107 | 0.00448 | 4.9 |
| weights = length | support points · energy | 0.00594 | 0.00239 | 8.5 |
| reference = cs | random | 0.147 | 0.0024 | – |
| reference = cs | k-center greedy | 0.0909 | 0.0228 | 2.1 |
| reference = cs | herding · energy | 0.00818 | 0.0994 | 0.94 |
| reference = cs | kernel thinning · energy | 0.00812 | 0.101 | 4.7 |
| reference = cs | support points · energy | 0.14 | 0.00242 | 4.6 |
| plain, n = 250 | kernel thinning · compress = :never | 0.00244 | 0.00244 | 4.5 |
| plain, n = 250 | kernel thinning · compress = :always | 0.00258 | 0.00258 | 3.1 |
| plain, n = 250 | random | 0.00485 | 0.00485 | – |
