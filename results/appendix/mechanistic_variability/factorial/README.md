# Exploratory factorial mechanistic variability

## attribution

Mean marginal changes over the 12 intervention orders in which reset removal precedes scalar gating. Uses only observed configurations; conditional exploratory attribution, not unconstrained Shapley or causal effects. Contributions sum to target minus baseline. Error bars use subject-deletion SE and t(10).

## source_run

nested_DA_S32_B8

## selection

block_reset

## design

Architecture(3) × BC(2) × decoder-bias-removal(2)

## gate_estimability

Reset removal: lightGRU−GRU; scalar gating conditional on no reset: monoGRU−lightGRU. reset_removed×scalar_gate aliases scalar_gate and is not estimable independently.

## BC_confound

BC is ReLU versus tanh; training energy-regularization grid also changes. Interpret as bundled configuration contrast, not isolated biological-constraint causality.

## within

11 repeated subjects, full factorial ANOVA via orthonormal contrasts; Greenhouse-Geisser corrected dfs/p when effect df>1; partial eta squared descriptive

## across

No ordinary ANOVA treating 55 dyads as independent. Scalar factorial contrasts use deletion of all incident dyads, approximate t(10); exploratory with 11 nodes.

## reset_metric

Reset-gate similarity exists only in GRU: BC × DB_removed only

## correction

BH over31 within omnibus tests; separate BH over94 planned mean contrasts spanning within/across; optional sign sensitivity corrected over47 within contrasts

## assumptions

RM ANOVA requires distributional assumptions; GG adjusts sphericity, not arbitrary nonnormality. Across jackknife approximate. Within sign sensitivity tests direction, not the mean ANOVA estimand.
