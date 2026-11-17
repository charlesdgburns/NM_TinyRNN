"""Small reusable ReLU norm-balancing helpers.

Extracted from scaling_sensitivity, activation_comparability and
canonical_weight_comparison so monoGRU exports do not depend on exploratory
analysis runners. Transformations match mechanistic_variability.align_rnns.
"""
import numpy as np
from scipy.special import logsumexp

def parameters(model):
    return {k: v.detach().cpu().numpy().astype(float).copy()
            for k, v in model.named_parameters() if v.requires_grad}


def transform(params, perm, scales):
    """Match align_rnns and transform_gate_weights, including row-vector recurrence."""
    out = {k: v.copy() for k, v in params.items()}
    h = len(perm)
    p = np.eye(h)[list(perm)]
    pt = np.diag(scales) @ p
    inv = p.T @ np.diag(1 / np.asarray(scales))
    w = params['rnn.W_ih']
    out['rnn.W_ih'] = pt @ w if w.shape[0] == h else (pt @ w.T).T
    for key in ('rnn.bias_h', 'rnn.hidden_0'):
        if key in params:
            out[key] = pt @ params[key]
    out['rnn.W_hh'] = inv.T @ params['rnn.W_hh'] @ pt.T
    out['decoder.weight'] = params['decoder.weight'] @ inv
    for gate in ('z', 'r', 'i'):
        key = f'rnn.W_h{gate}'
        if key in params:
            w = params[key]
            if w.shape == (h, h):
                out[key] = inv.T @ w @ p.T
            elif w.shape == (h, 1):
                out[key] = inv.T @ w
        key = f'rnn.W_i{gate}'
        if key in params and params[key].shape[1] == h:
            out[key] = params[key] @ p.T
        key = f'rnn.bias_{gate}'
        if key in params and params[key].shape == (h,):
            out[key] = p @ params[key]
    return out


def core(params, h):
    w = params['rnn.W_ih']
    if w.shape[0] != h:
        w = w.T
    return np.concatenate((w.ravel(), params.get('rnn.bias_h', np.zeros(h)).ravel(),
                           params['rnn.W_hh'].ravel(), params['decoder.weight'].ravel()))


def flat(params):
    return np.concatenate([v.ravel() for v in params.values()])


def power_matrix(params):
    """Exact monomial exponents of ALL trainable parameters under positive scaling."""
    base = flat(params)
    columns = []
    h = params['rnn.W_hh'].shape[0]
    for j in range(h):
        scale = np.ones(h); scale[j] = np.e
        scaled = flat(transform(params, tuple(range(h)), scale))
        ratio = np.divide(scaled, base, out=np.ones_like(base), where=base != 0)
        column = np.log(ratio)
        assert np.allclose(column, np.round(column), atol=1e-10)
        columns.append(np.round(column))
    return np.array(columns).T


def canonicalize(params,activation):
    if activation=='tanh':return {k:v.copy() for k,v in params.items()},dict(log_scale_1=0.,log_scale_2=0.,gradient=0.,iterations=0,anchored=True,hessian_min=1.,norm_before=float(np.linalg.norm(flat(params))),norm_after=float(np.linalg.norm(flat(params))))
    p=power_matrix({k:np.ones_like(v) for k,v in params.items()})
    values=flat(params);active=(values!=0)&np.any(p!=0,axis=1)
    powers,indices=np.unique(p[active],axis=0,return_inverse=True)
    logs=np.array([logsumexp(2*np.log(np.abs(values[active][indices==i]))) for i in range(len(powers))])
    anchors=[]
    for i in range(2):
        ei=np.eye(2)[i]
        plus=np.flatnonzero(np.all(powers==ei,axis=1));minus=np.flatnonzero(np.all(powers==-ei,axis=1))
        if len(plus)!=1 or len(minus)!=1:raise ValueError('Unanchored unit: unique finite balancing needs a separate degeneracy treatment')
        anchors.append(.25*(logs[minus[0]]-logs[plus[0]]))
    origin=np.array(anchors);q=2*powers;center=logs+q@origin
    def objective(z):
        v=center+q@z;l=logsumexp(v);w=np.exp(v-l)
        gradient=w@q;hessian=(q.T*w)@q-np.outer(gradient,gradient)
        return l,gradient,hessian
    z=np.zeros(2)
    for iteration in range(100):
        loss,grad,hess=objective(z)
        if np.max(np.abs(grad))<1e-12:break
        step=np.linalg.solve(hess,grad)
        # Armijo descent on the log of the variable part of the squared norm.
        alpha=1.
        while alpha>1e-12:
            candidate=z-alpha*step
            if objective(candidate)[0]<=loss-1e-4*alpha*(grad@step)+1e-14:break
            alpha*=.5
        z=candidate
    loss,grad,hess=objective(z)
    if np.max(np.abs(grad))>1e-9:raise RuntimeError(f'Balancing did not converge: {grad}')
    x=origin+z;balanced=transform(params,(0,1),np.exp(x))
    info=dict(log_scale_1=x[0],log_scale_2=x[1],gradient=float(np.max(np.abs(grad))),iterations=iteration+1,
        anchored=True,hessian_min=float(np.linalg.eigvalsh(hess).min()),norm_before=float(np.linalg.norm(values)),norm_after=float(np.linalg.norm(flat(balanced))))
    if info['norm_after']>info['norm_before']+1e-8:raise RuntimeError('Balancing increased the parameter norm')
    return balanced,info
