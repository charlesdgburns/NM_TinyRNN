# Final monoGRU +BC-DB alignment

Protocol: balanced_medoid_reference_fit_v3_choice0_first. All 110 selected two-unit monoGRU ReLU models have no decoder bias. The final convention is h1 toward choice 0 and h2 toward choice 1, giving an identity-like decoder.

Independently balanced models determine the subject-weighted medoid (WS14, outer fold 3, inner fold 4). Each model is subsequently fitted to this fixed balanced reference by positive-scale grid search and continuous refinement, preserving the final semantic labels. Reference distance decreases in 109/110 fits; the medoid is unchanged apart from its global unit relabeling. Mean distance before and after fitting: 1.091121 and 0.970440.

All 110 full-subject histories were regenerated with jobs=-1. Original trial identities and split labels were retained. All source-prediction checks passed; maximum probability difference is 3.32e-07. Mean and SD statistics match the global unit permutation of the prior reference-fitted export within numerical tolerance.

The means and sample SDs were independently recomputed from all saved final reference-fitted model parameters. First average each subject's ten folds, then compute the mean and sample SD (ddof=1) across the 11 subjects. All table entries agree within 1e-12. SD is not computed on unaligned models or by treating all 110 folds as independent.

The open appendix document was updated in place: complete mean equations, then matching parameter-SD layouts. The SD layout is a coefficient comparison, not a model for SDs of activations or predictions. The built-in document compiler was attempted but its environment cannot load pdflatex.fmt, so PDF layout could not be verified.

Nine alignment tests and three export integration tests pass, including rejection of references using the old unit order and an explicit subject-SD weighting check.
