"""Checks for variance accounting and shared endpoint identity."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from NM_TinyRNN.code.analyses import mechanistic_variability_appendix as study


class VarianceTests(unittest.TestCase):
    def test_group_budget_preserves_weighting_and_total(self):
        groups=pd.DataFrame(dict(n=[2,3],mean=[1.,6.],endpoint_variance=[1.,0.],interaction_variance=[0.,8/3]))
        result=study.group_variance_budget(groups)
        self.assertAlmostEqual(result['between_subjects'],6.)
        self.assertAlmostEqual(result['outer_fold'],.4)
        self.assertAlmostEqual(result['model_pair'],1.6)
        self.assertAlmostEqual(result['total'],8.)

    def test_reference_matched_contrast_uses_A_baseline_and_saved_direction(self):
        import tempfile
        rows=[]
        for a,b,within,value in [('WS01','WS01',True,.9),('WS20','WS20',True,.1),('WS01','WS20',False,.4)]:
            rows.append(dict(model_id=study.MODELS['GRU'],subject_A=a,subject_B=b,
                             trial_subject=a,within_subject=within,**{m:value for m in study.METRICS}))
        with tempfile.TemporaryDirectory() as directory:
            contrasts,summary=study.reference_matched_contrasts(pd.DataFrame(rows),Path(directory))
        self.assertEqual(len(contrasts),len(study.METRICS))
        self.assertTrue(contrasts.reference_subject.eq('WS01').all())
        self.assertTrue(contrasts.comparison_subject.eq('WS20').all())
        np.testing.assert_allclose(contrasts.difference,.5)
        np.testing.assert_allclose(summary.mean_difference,.5)

    def test_unbalanced_groups_exact_identity(self):
        result = study.decomposition(np.array([0., 2., 4., 6., 8.]), ['a', 'a', 'b', 'b', 'b'])
        self.assertAlmostEqual(result['total_variance'], 8.)
        self.assertAlmostEqual(result['between_group_variance'], 6.)
        self.assertAlmostEqual(result['within_group_variance'], 2.)

    def test_endpoint_identity_does_not_depend_on_role(self):
        d = pd.DataFrame(dict(subject_A=['s', 's'], subject_B=['s', 's'],
                              model_A_outer_n=[1, 2], model_B_outer_n=[2, 3]))
        x = study.design(d)
        self.assertEqual(x['fb'][0], x['fa'][1])
        self.assertEqual(x['sa'][0], x['sb'][0])

    def test_variance_multipliers_match_population_simulation(self):
        rng = np.random.default_rng(18)
        sd = np.array([.7, .3, .5, .4, .8])
        n = 200000
        for within in [True, False]:
            effects = rng.normal(size=(n, 8))
            score = sd[0]*effects[:, 0] + sd[2]*(effects[:, 2]+effects[:, 3]) + sd[3]*(effects[:, 4]-effects[:, 5]) + sd[4]*effects[:, 6]
            if not within:
                score += sd[0]*effects[:, 1] + sd[1]*effects[:, 7]
            expected = (sd**2 * study.contribution_multipliers(within)).sum()
            self.assertAlmostEqual(np.var(score) / expected, 1., delta=.015)

    def test_bayes_factor_uses_prior_odds(self):
        post = np.arange(10000) < 8000
        prior = np.arange(10000) < 5000
        result = study.evidence(post, prior, np.ones(10000)/10000, 10000)
        self.assertAlmostEqual(result['bayes_factor'], 4.)
        unresolved = study.evidence(np.ones(10000, dtype=bool), prior, np.ones(10000)/10000, 10000)
        self.assertTrue(np.isnan(unresolved['bayes_factor']))

    def test_endpoint_projection_recovers_pure_endpoint_effects(self):
        import itertools
        edges = list(itertools.combinations(range(10), 2))
        d = pd.DataFrame(edges, columns=['model_A_outer_n', 'model_B_outer_n'])
        u = np.arange(10) / 10
        d['score'] = [u[i]+u[j] for i, j in edges]
        part = study.endpoint_decomposition(d, 'score', True)
        self.assertAlmostEqual(part['interaction_variance'], 0.)
        self.assertAlmostEqual(part['endpoint_variance'], d.score.var(ddof=0))
        edges = list(itertools.product(range(10), repeat=2))
        d = pd.DataFrame(edges, columns=['model_A_outer_n', 'model_B_outer_n'])
        d['score'] = [u[i]-2*u[j] for i, j in edges]
        part = study.endpoint_decomposition(d, 'score', False)
        self.assertAlmostEqual(part['interaction_variance'], 0.)
        self.assertAlmostEqual(part['candidate_endpoint_variance'], 4 * part['reference_endpoint_variance'])

    def test_collapsed_likelihood_matches_dense_gaussian(self):
        import itertools
        from scipy.stats import multivariate_normal
        rng = np.random.default_rng(13)
        rows = []
        for a,b in itertools.combinations(['a','b','c'],2):
            for i,j in itertools.product(range(3),repeat=2):
                rows.append(dict(subject_A=a,subject_B=b,model_A_outer_n=i,model_B_outer_n=j,
                                 score=np.tanh(rng.normal())))
        d = pd.DataFrame(rows)
        s = study.across_sufficient_statistics(d,'score')
        x = s['design']
        N = len(d)
        def incidence(a,b,k,sign=1):
            z = np.zeros((N,k));z[np.arange(N),a]+=1;z[np.arange(N),b]+=sign
            return z
        subject = incidence(x['sa'],x['sb'],3)
        common = incidence(x['fa'],x['fb'],9)
        role = incidence(x['fa'],x['fb'],9,-1)
        dyad = np.eye(3)[x['dyad']]
        sd = np.array([.7,.4,.5,.2,.3])
        cov = 9*np.ones((N,N))+sd[0]**2*(subject@subject.T)+sd[1]**2*(dyad@dyad.T)+sd[2]**2*(common@common.T)+sd[3]**2*(role@role.T)+sd[4]**2*np.eye(N)
        expected = multivariate_normal.logpdf(np.arctanh(d.score),mean=np.zeros(N),cov=cov)
        self.assertAlmostEqual(study.collapsed_logpdf(s,sd),expected,places=8)
        # Validate the conditional predictive reconstruction, not just its density.
        predictions,_ = study.conditional_across_predictions(s,d,np.repeat(sd[None,:],3000,axis=0),np.random.default_rng(29))
        z = np.arctanh(predictions)
        latent_cov = cov-sd[4]**2*np.eye(N)
        target_mean = latent_cov@np.linalg.solve(cov,np.arctanh(d.score))
        target_cov = latent_cov-latent_cov@np.linalg.solve(cov,latent_cov)+sd[4]**2*np.eye(N)
        np.testing.assert_allclose(z.mean(axis=0),target_mean,atol=.035)
        np.testing.assert_allclose(z.var(axis=0),np.diag(target_cov),atol=.015)


if __name__ == '__main__':
    unittest.main()
