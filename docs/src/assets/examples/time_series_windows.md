## Fixture

`X` (N = 5 rows, p = 2 variables), `L = 2`, `stride = 2`:

```text
X = [1.0 10.0; 2.0 20.0; 3.0 30.0; 4.0 40.0; 5.0 50.0]
windows(X, 2; stride = 2) = [1.0 2.0 10.0 20.0; 3.0 4.0 30.0 40.0]
starts = [1, 3]
```

Row `i` is `vec(X[starts[i]:starts[i]+L-1, :])`: variable-major, all `L`
offsets of variable 1 then variable 2. The trailing row 5 does not fill a
full window and is dropped, not padded.

## Point vs window level

M = 1000 windows of length 32 (regime A share ≈ 0.7). Point-level
per-variable mean and variance should match across regimes:

| variable | mean A | mean B | var A | var B |
|---|---:|---:|---:|---:|
| 1 | 0.033 | 0.005 | 1.18 | 1.2 |
| 2 | 0.0195 | -0.00413 | 0.663 | 0.657 |
| 3 | 0.041 | -0.0102 | 1.88 | 1.9 |

Energy distance, A vs B, point level: 0.00175

Energy distance, A vs B, window level (flattened, standardized): 0.529

Null scale (two random halves of regime A): point level 0.000271, window level 0.0724

The temporal dependence that distinguishes the regimes is invisible at the
point level, where the A-vs-B distance stays within an order of magnitude of the null, and only shows up
once each window is kept as one sample, where A-vs-B separates well above the
null.

## Selectors vs random

n = 100 of M = 1000 windows (L = 32, p = 3), evaluated on the
standardized flattened windows. Random: 20 draws; stochastic
selectors: 3 seeds.

| method | energy distance | regime-proportion error | mean lag-1 autocorrelation |
|---|---:|---:|---:|
| random | 0.119 ± 0.018 | 0.036 ± 0.031 | 0.13 ± 0.0634 |
| twinning | 0.0627 | 0.02 | 0.114 |
| herding · energy | 0.0466 | 0 | 0.147 |
| kernel thinning · energy | 0.0483 ± 0.000496 | 0.00667 ± 0.00577 | 0.144 ± 0.0238 |
| support points · energy | 0.121 ± 0.0167 | 0.0533 ± 0.00577 | 0.0589 ± 0.0058 |

## datasplit

`datasplit(TwinningSplitter(), Zs; standardize = false)`: 200 of
1000 windows on the `test` side.

Recovering the original time-series slice for the first 3 selected windows
(each stays a separate L × p sample; selected windows are never concatenated):

```text
window 594 (rows 18977:19008):
[-1.0366530634837612 -0.8949494709509489 -1.6978493146942308; 0.9826741714011777 0.9136687735542621 1.2905526228666901; -1.1020468922644833 -0.803930117904336 -0.718832638994183] …
window 617 (rows 19713:19744):
[-0.7940421281103217 -0.21452221218324696 -0.8632301469761647; 0.5143972623539675 1.1973811538814654 0.9065161386430288; -0.2254081865606382 -0.9850816248911023 -1.163538605162607] …
window 399 (rows 12737:12768):
[-1.3307604984135937 -0.9950025789242789 -1.0185232221567513; 0.4456488607770851 0.27963309280792953 1.8241116872239527; -1.005473010961746 -0.6022163614361392 -1.663942434801514] …
```

## Contrast 1: representation below the dependence length

TwinningSplitter, n = 100 of M = 1000 windows, representation built from
only the first `L_short` rows of each length-32 segment, evaluated in the
full L = 32 space (dependence length ≈ 1/(1-stay_a) ≈ 16), averaged over
5 independently generated datasets (mean ± sd over data seeds).

| L_short | ratio to random | regime-proportion error |
|---:|---:|---:|
| 1 | 0.895 ± 0.0964 | 0.028 ± 0.0164 |
| 2 | 0.896 ± 0.201 | 0.024 ± 0.00894 |
| 4 | 0.854 ± 0.0793 | 0.01 ± 0.00707 |
| 8 | 0.872 ± 0.159 | 0.01 ± 0.01 |
| 16 | 0.656 ± 0.0738 | 0.008 ± 0.011 |
| 32 | 0.565 ± 0.034 | 0.006 ± 0.00894 |

## Contrast 2: the L·p dimension ladder

M = 2000, n = 200, L ∈ [8, 64, 512, 1024, 4096]. Support points: 3
seeds (min time, mean energy distance); random: 10 draws.
The cached energy distance at L = 8 was checked to agree with
`energydistance` directly. "compile seconds" is the first call of that splitter
at that width in this process, on a throwaway 60-row matrix; with the matrix
brute-force search, compilation is flat from 200 columns on; below that, the
k-d tree of `select_nearest` still pays a width-specific compile (the
192-column rung). Twinning is warmed up first, and the
support-point warm-up follows. This ladder runs before Contrast 1 so no ladder
width has been compiled earlier in the process.

| L | L·p | method | compile seconds | seconds | energy distance | ratio to random |
|---:|---:|---|---:|---:|---:|---:|
| 8 | 24 | random | – | – | 0.0297 | 1 |
| 8 | 24 | twinning | 0.48 | 0.043 | 0.0112 | 0.378 |
| 8 | 24 | support points · energy | 0.17 | 0.13 | 0.0273 | 0.918 |
| 64 | 192 | random | – | – | 0.0927 | 1 |
| 64 | 192 | twinning | 8.8e-05 | 0.012 | 0.0551 | 0.595 |
| 64 | 192 | support points · energy | 1.2 | 0.95 | 0.08 | 0.863 |
| 512 | 1536 | random | – | – | 0.245 | 1 |
| 512 | 1536 | twinning | 0.0013 | 0.094 | 0.2 | 0.817 |
| 512 | 1536 | support points · energy | 0.0062 | 12 | 0.257 | 1.05 |
| 1024 | 3072 | random | – | – | 0.355 | 1 |
| 1024 | 3072 | twinning | 0.00061 | 0.33 | 0.317 | 0.893 |
| 1024 | 3072 | support points · energy | 0.0064 | 43 | 0.343 | 0.966 |
| 4096 | 12288 | random | – | – | 0.706 | 1 |
| 4096 | 12288 | twinning | 0.0025 | 3.9 | 0.658 | 0.932 |
| 4096 | 12288 | support points · energy | 0.021 | 2.2e+02 | 0.701 | 0.993 |
