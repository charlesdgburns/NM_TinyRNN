# monoGRU +BC-DB reference-fitted alignment

Protocol: balanced_medoid_reference_fit_v2. All 110 two-unit monoGRU ReLU models have no decoder bias (11 subjects × ten block-reset train+validation-selected outer winners).

First independently minimize the full parameter norm and label h1 toward choice 1, h2 toward choice 0. Select the equal-subject-weighted medoid in that balanced space: WS14, outer fold 3, inner fold 4. Then optimize each fold's positive hidden-unit scales against that **fixed** reference using the existing core-parameter Euclidean objective, grid search and continuous TRF refinement. Unit roles are constrained. All 110 numerical refinements converged.

The medoid's parameter values were checked to be exactly unchanged from the preceding export. Distances decrease for 109/110 models; the medoid remains unchanged. Mean distance before refinement: 1.091121; after: 0.970440. No model's distance increased. Initial balancing scales, reference-fitting scales and composed original-to-final transforms are recorded in aligned_models.htsv.

All original/aligned synthetic checks pass for logits, update gates and transformed hidden states. All 110 whole-subject trial histories were also regenerated in parallel and checked against source histories. Maximum full-history absolute logit difference: 2.09e-06; probability difference: 4.21e-07. The updated mean is over the final reference-fitted parameters, with equal subject weights. NLL is preserved; hidden-state scales and mean parameters change.

The pre-reference-fitting outputs were preserved in a local backup before this update. Current data/rnns/subject_examples_aligned files supersede the earlier fixed-scale outputs. The frozen reference model and its selection-space leave-subject-out analysis are unchanged. Eight alignment tests and three export integration tests pass.
