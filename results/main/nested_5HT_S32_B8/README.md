# Cohort validation: nested_5HT_S32_B8

Included subjects: WS03, WS04, WS06, WS07, WS11, WS12, WS15, WS17, WS18, WS21, WS23, WS26, WS27, WS28, WS29, WS30, WS31, WS32, WS33, WS34, WS35, WS36.

Selection: minimum free-choice block-reset train + validation NLL; no test targets or penalties. Each configuration has ten outer winners per subject. Evaluation uses complete continuous saved inputs of sorted reference subject A, including forced trials; model B is aligned to A. Tanh uses permutation/sign alignment; ReLU uses the canonical grid plus continuous positive-scale refinement. Metrics and inferential tests match the maintained main analysis. Within: independent subject means, exact two-sided sign tests. Across: dyad means, subject-deletion jackknife SE and approximate t(21); dyads are dependent. BH correction spans the four metrics × two scopes. Cohen dz is descriptive; deletion ranges are influence diagnostics.

Intermediate data: /home/charlesdgburns/Coding/NM_TinyRNN/data/analyses/cohort_validation/nested_5HT_S32_B8

Reproduce:
```sh
.venv/bin/python code/analyses/cohort_validation.py --folder nested_5HT_S32_B8
```
