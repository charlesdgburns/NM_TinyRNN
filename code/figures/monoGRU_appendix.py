"""Comparable mean and between-subject parameter-SD equation displays."""
import json
import numpy as np


def parameter_arrays(table, column):
    arrays = {}
    for name, group in table.groupby('parameter'):
        entries = [(tuple(json.loads(r.index)), getattr(r, column)) for r in group.itertuples()]
        ndim = len(entries[0][0])
        shape = tuple(max(index[j] for index, value in entries)+1 for j in range(ndim))
        array = np.zeros(shape)
        for index, value in entries:
            array[index] = value
        arrays[name] = array
    return arrays


def equation_block(table, column='mean'):
    """Repeat equation geometry; SD display uses template symbols, not state SDs."""
    arrays = parameter_arrays(table, column)
    means = column == 'mean'
    def number(x):
        text = f'{x:.3f}'
        return r'\boldsymbol{'+text+'}' if means else text
    def matrix(a):
        return r'\begin{bmatrix}'+r' \\ '.join(' & '.join(number(x) for x in row)
            for row in np.atleast_2d(a))+r'\end{bmatrix}'
    x = r'\begin{bmatrix}c_t \\ a_t \\ o_t\end{bmatrix}'
    hp = r'\begin{bmatrix}h_{1,t-1} \\ h_{2,t-1}\end{bmatrix}'
    ht = r'\begin{bmatrix}h_{1,t} \\ h_{2,t}\end{bmatrix}'
    lhs = ([r'\mathbf n_t', 'z_t', r'\mathbf h_t', r'\hat{\mathbf a}_{t+1}', r'\mathbf h_0']
           if means else [r'\mathcal T_n', r'\mathcal T_z', r'\mathcal T_h', r'\mathcal T_a', r's(\mathbf h_0)'])
    return '\n'.join([
        r'\begingroup\small\setlength{\arraycolsep}{3pt}',
        r'\begin{align*}',
        lhs[0]+r' &= \operatorname{ReLU}\!\left('+matrix(arrays['rnn.W_ih'].T)+x+' + '+matrix(arrays['rnn.W_hh'].T)+hp+' + '+matrix(arrays['rnn.bias_h'][:, None])+r'\right), \\[1em]',
        lhs[1]+r' &= \sigma\!\left('+matrix(arrays['rnn.W_iz'].T)+x+' + '+matrix(arrays['rnn.W_hz'].T)+hp+' + '+number(float(arrays['rnn.bias_z']))+r'\right), \\[1em]',
        lhs[2]+r' &= (1-z_t)\mathbf n_t + z_t\mathbf h_{t-1}, \\[1em]',
        lhs[3]+r' &= \operatorname{Softmax}\!\left('+matrix(arrays['decoder.weight'])+ht+r'\right), \\[1em]',
        lhs[4]+' &= '+matrix(arrays['rnn.hidden_0'][:, None])+'.',
        r'\end{align*}\endgroup', ''])


def write_appendix(table, path):
    """Edit the existing appendix document in place from final aligned statistics."""
    text = r'''\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage[margin=22mm]{geometry}
\pagestyle{empty}
\begin{document}
\section*{Average parameters of monoGRU}
We selected one two-unit monoGRU model with ReLU activations and no decoder bias per subject and outer fold by minimizing block-reset training-plus-validation negative log-likelihood (11 subjects, ten outer folds). Positive hidden-unit scales were initially balanced by minimizing the total squared parameter norm. The subject-weighted medoid in this balanced parameter space selected WS14, outer fold~3. Every fold was then fitted to this fixed reference by minimizing core-parameter distance over positive scales, with unit order constrained so that $h_1$ favors choice~0 and $h_2$ favors choice~1. Parameters were averaged across the ten aligned folds within each subject, and then equally across subjects.

\subsection*{Mean parameters}
Bold entries are the means across 11 subject-level parameter averages, rounded to three decimal places. The equations describe the mean-parameter network, rather than mean activations across networks.
'''
    text += equation_block(table, 'mean')
    text += r'''
\newpage
\section*{Between-subject standard deviations}
Entries are sample standard deviations ($n-1$ denominator) across the 11 subject-level parameter averages, calculated \emph{after each fold was aligned to the same reference}. They are not standard errors or SDs treating 110 overlapping folds as independent observations.

The display repeats the equation layout for direct coefficient comparison. Symbols $\mathcal T$ indicate a coefficient-display template: substituting parameter SDs into nonlinear equations does \emph{not} give the SD of hidden states, gates, or predicted choices. The hidden-state update is repeated for structural reference and has no fitted coefficients.
'''
    text += equation_block(table, 'subject_sd')
    text += '\n\\end{document}\n'
    path.write_text(text)
