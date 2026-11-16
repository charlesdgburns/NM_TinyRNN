# Two-unit similarity: Within subject

Selection: Minimum block-reset train + validation NLL. One model per subject/configuration/outer fold.

| Configuration | Parameters | Hidden states | Update gate | Reset gate | Readout |
|---|---|---|---|---|---|
| GRU · tanh · bias on | 0.747 (reference) | 0.892 (reference) | 0.816 (reference) | 0.605 (reference) | 0.985 (reference) |
| GRU · tanh · bias off | 0.748; Δ +0.001; dz +0.02 | 0.902; Δ +0.011; dz +0.25 | 0.823; Δ +0.006; dz +0.05 | 0.636; Δ +0.032; dz +0.23 | 0.983; Δ -0.001; dz -0.22 |
| GRU · relu · bias on | 0.762; Δ +0.015; dz +0.05 | 0.842; Δ -0.050; dz -0.23 | 0.719; Δ -0.097; dz -0.25 | 0.625; Δ +0.020; dz +0.05 | 0.985; Δ +0.000; dz +0.02 |
| GRU · relu · bias off | 0.664; Δ -0.083; dz -0.26 | 0.767; Δ -0.124; dz -0.59 | 0.580; Δ -0.236; dz -0.59 | 0.507; Δ -0.098; dz -0.22 | 0.978; Δ -0.007; dz -0.58 |
| lightGRU · tanh · bias on | 0.847; Δ +0.100; dz +0.50 | 0.863; Δ -0.028; dz -0.14 | 0.805; Δ -0.011; dz -0.04 | N/A | 0.986; Δ +0.002; dz +0.25 |
| lightGRU · tanh · bias off | 0.817; Δ +0.070; dz +0.34 | 0.892; Δ +0.001; dz +0.01 | 0.778; Δ -0.038; dz -0.17 | N/A | 0.983; Δ -0.001; dz -0.14 |
| lightGRU · relu · bias on | 0.729; Δ -0.018; dz -0.07 | 0.767; Δ -0.125; dz -0.64 | 0.593; Δ -0.224; dz -0.66 | N/A | 0.979; Δ -0.006; dz -0.39 |
| lightGRU · relu · bias off | 0.783; Δ +0.036; dz +0.16 | 0.807; Δ -0.085; dz -0.43 | 0.666; Δ -0.150; dz -0.48 | N/A | 0.973; Δ -0.012; dz -0.35 |
| monoGRU · tanh · bias on | 0.860; Δ +0.113; dz +0.56 | 0.899; Δ +0.007; dz +0.05 | 0.817; Δ +0.001; dz +0.01 | N/A | 0.985; Δ +0.001; dz +0.06 |
| monoGRU · tanh · bias off | 0.906; Δ +0.159; dz +1.17 | 0.925; Δ +0.033; dz +0.38 | 0.876; Δ +0.059; dz +0.51 | N/A | 0.988; Δ +0.003; dz +0.26 |
| monoGRU · relu · bias on | 0.830; Δ +0.083; dz +0.33 | 0.822; Δ -0.069; dz -0.38 | 0.813; Δ -0.003; dz -0.01 | N/A | 0.985; Δ +0.001; dz +0.07 |
| monoGRU · relu · bias off | 0.906; Δ +0.159; dz +0.79 | 0.962; Δ +0.071; dz +0.74 | 0.809; Δ -0.007; dz -0.03 | N/A | 0.992; Δ +0.007; dz +1.60 |

Means are raw correlations/cosines. Δ = model − reference. dz = mean paired subject-level difference / its SD.
Stars use BH correction over all 94 non-reference comparisons. Across-subject inference is approximate node-jackknife t(10); descriptive dz does not supply its p-value.
Only 11 subjects; treat significance as exploratory. Reset gate is absent in lightGRU/monoGRU.
