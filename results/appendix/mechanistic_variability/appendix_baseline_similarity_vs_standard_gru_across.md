# Two-unit similarity: Across subjects

Selection: Minimum block-reset train + validation NLL. One model per subject/configuration/outer fold.

| Configuration | Parameters | Hidden states | Update gate | Reset gate | Readout |
|---|---|---|---|---|---|
| GRU · tanh · bias on | 0.453 (reference) | 0.680 (reference) | 0.583 (reference) | 0.247 (reference) | 0.919 (reference) |
| GRU · tanh · bias off | 0.447; Δ -0.006; dz -0.22 | 0.702; Δ +0.023; dz +0.65 | 0.583; Δ -0.000; dz -0.00 | 0.289; Δ +0.042; dz +0.63 | 0.919; Δ +0.000; dz +0.10 |
| GRU · relu · bias on | 0.383; Δ -0.070; dz -0.52 | 0.563; Δ -0.116; dz -0.51 | 0.147; Δ -0.436; dz -3.79** | 0.096; Δ -0.152; dz -0.98 | 0.916; Δ -0.002; dz -0.18 |
| GRU · relu · bias off | 0.420; Δ -0.033; dz -0.38 | 0.613; Δ -0.067; dz -0.47 | 0.208; Δ -0.375; dz -3.16* | 0.089; Δ -0.159; dz -1.03 | 0.919; Δ +0.000; dz +0.05 |
| lightGRU · tanh · bias on | 0.571; Δ +0.118; dz +1.19 | 0.623; Δ -0.057; dz -0.47 | 0.548; Δ -0.035; dz -0.20 | N/A | 0.916; Δ -0.003; dz -0.38 |
| lightGRU · tanh · bias off | 0.511; Δ +0.058; dz +0.63 | 0.633; Δ -0.047; dz -0.46 | 0.453; Δ -0.129; dz -0.87 | N/A | 0.911; Δ -0.007; dz -1.01 |
| lightGRU · relu · bias on | 0.419; Δ -0.033; dz -0.25 | 0.512; Δ -0.168; dz -0.85 | 0.119; Δ -0.463; dz -3.69** | N/A | 0.909; Δ -0.009; dz -0.54 |
| lightGRU · relu · bias off | 0.520; Δ +0.067; dz +0.68 | 0.630; Δ -0.050; dz -0.40 | 0.187; Δ -0.396; dz -2.62 | N/A | 0.907; Δ -0.012; dz -0.42 |
| monoGRU · tanh · bias on | 0.492; Δ +0.040; dz +0.22 | 0.710; Δ +0.030; dz +0.25 | 0.395; Δ -0.187; dz -0.64 | N/A | 0.916; Δ -0.002; dz -0.14 |
| monoGRU · tanh · bias off | 0.449; Δ -0.004; dz -0.02 | 0.775; Δ +0.095; dz +0.99 | 0.332; Δ -0.250; dz -0.81 | N/A | 0.915; Δ -0.004; dz -0.19 |
| monoGRU · relu · bias on | 0.599; Δ +0.146; dz +0.83 | 0.562; Δ -0.117; dz -0.55 | 0.521; Δ -0.062; dz -0.40 | N/A | 0.927; Δ +0.009; dz +0.97 |
| monoGRU · relu · bias off | 0.783; Δ +0.330; dz +3.82** | 0.891; Δ +0.211; dz +1.96 | 0.568; Δ -0.015; dz -0.12 | N/A | 0.945; Δ +0.026; dz +1.22 |

Means are raw correlations/cosines. Δ = model − reference. dz = mean paired subject-level difference / its SD.
Stars use BH correction over all 94 non-reference comparisons. Across-subject inference is approximate node-jackknife t(10); descriptive dz does not supply its p-value.
Only 11 subjects; treat significance as exploratory. Reset gate is absent in lightGRU/monoGRU.
