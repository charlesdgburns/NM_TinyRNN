"""Evidence likelihood constants and boundary models agree with dense Gaussians."""
import itertools
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from NM_TinyRNN.code.analyses import mechanistic_variability_appendix as study


class EvidenceTests(unittest.TestCase):
    def test_all_model_variants_against_dense_normal(self):
        rng=np.random.default_rng(8)
        for within in [True,False]:
            rows=[]
            for a,b in ([(s,s) for s in ['a','b','c']] if within else itertools.combinations(['a','b','c'],2)):
                for i,j in (itertools.combinations(range(4),2) if within else itertools.product(range(4),repeat=2)):
                    rows.append(dict(subject_A=a,subject_B=b,model_A_outer_n=i,model_B_outer_n=j,score=np.tanh(rng.normal())))
            d=pd.DataFrame(rows);s=study.prepare_likelihood(d,'score',within);x=study.design(d);N=len(d)
            def incidence(a,b,k,sign=1):
                z=np.zeros((N,k));z[np.arange(N),a]+=1;z[np.arange(N),b]+=sign;return z
            A=np.eye(3)[x['sa']] if within else incidence(x['sa'],x['sb'],3)
            B=incidence(x['fa'],x['fb'],12);D=incidence(x['fa'],x['fb'],12,-1)
            C=np.eye(len(x['dyads']))[x['dyad']]
            for variant,active in study.VARIANTS.items():
                sd=np.array([.7,.3,.4,.2,.5]);sd[[i for i in range(5) if i not in active]]=0
                if within:sd[1]=0
                cov=9*np.ones((N,N))+sd[0]**2*(A@A.T)+sd[1]**2*(C@C.T)+sd[2]**2*(B@B.T)+sd[3]**2*(D@D.T)+sd[4]**2*np.eye(N)
                expected=multivariate_normal.logpdf(np.arctanh(d.score),cov=cov)
                with self.subTest(within=within,variant=variant):
                    self.assertAlmostEqual(study.log_likelihood(s,sd),expected,places=8)


if __name__=='__main__':unittest.main()
