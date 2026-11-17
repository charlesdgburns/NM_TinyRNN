"""Checks of function preservation, scale invariance and semantic constraints."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from NM_TinyRNN.code.models.rnns import TinyRNN
from NM_TinyRNN.code.analyses import monoGRU_alignment as ma
from NM_TinyRNN.code.measures import mechanistic_variability as mv
from NM_TinyRNN.code.measures.monogru_balancing import parameters, transform


class MonoGRUAlignment(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = TinyRNN(rnn_type='monoGRU', input_size=3, hidden_size=2,
                             nonlinearity='relu', input_forced_choice=True, decoder_bias=False)
        gen=torch.Generator().manual_seed(341)
        with torch.no_grad():
            for p in self.model.parameters():p.copy_(torch.randn(p.shape,generator=gen)*.3)
            self.model.rnn.hidden_0.copy_(torch.tensor([.2,.4]))
            self.model.decoder.weight.copy_(torch.tensor([[-.7,.8],[.9,-.6]]))

    def test_equivalent_rescaling_and_permutation_have_same_canonical_model(self):
        reference,_=ma.align_model(self.model,self.model)
        changed=ma.write_parameters(self.model,transform(parameters(self.model),(1,0),np.array([.03,40.])))
        canonical,_=ma.align_model(changed,reference)
        for name,p in reference.named_parameters():
            np.testing.assert_allclose(p.detach(),dict(canonical.named_parameters())[name].detach(),atol=2e-5,rtol=2e-5)
        ma.check_equivalence(self.model,changed,ma.probes(self.model),(1,0),[.03,40.])

    def test_positive_scale_and_unit_labels_preserve_gates_initial_state_and_logits(self):
        original={k:v.clone() for k,v in self.model.state_dict().items()}
        b,info=ma.balance_model(self.model)
        aligned,ai=ma.align_balanced_model(b,b)
        perm=ai['perm'];scales=np.exp([info['log_scale_1'],info['log_scale_2']])[perm]
        ma.check_equivalence(self.model,aligned,ma.probes(self.model),perm,scales)
        contrast=(aligned.decoder.weight[1]-aligned.decoder.weight[0]).detach().numpy()
        self.assertGreater(contrast[0],0);self.assertLess(contrast[1],0)
        for k,v in original.items():self.assertTrue(torch.equal(v,self.model.state_dict()[k]))

    def test_restricted_permutation_cannot_fall_back_to_identity(self):
        aligned,info=mv.align_rnns(self.model,self.model,scale_values=np.array([1.]),
                                  refine_scaling=False,allowed_permutations=[(1,0)])
        self.assertEqual(list(info['perm']),[1,0])
        self.assertGreater(info['aligned_distance'],info['unaligned_distance'])
        ma.check_equivalence(self.model,aligned,ma.probes(self.model),(1,0),[1.,1.])
        with self.assertRaises(ValueError):mv.align_rnns(self.model,self.model,allowed_permutations=[(0,0)])

    def test_decoder_bias_on_is_rejected(self):
        wrong=TinyRNN(rnn_type='monoGRU',input_size=3,hidden_size=2,nonlinearity='relu',decoder_bias=True)
        with self.assertRaisesRegex(ValueError,'decoder bias off'):ma.balance_model(wrong)

    def test_unanchored_unit_is_flagged(self):
        with torch.no_grad():
            self.model.rnn.W_ih[:,0].zero_();self.model.rnn.W_hh[0,:].zero_();self.model.rnn.W_hh[:,0].zero_()
            self.model.rnn.bias_h[0]=0;self.model.rnn.hidden_0[0]=0
            self.model.rnn.W_hz[0,:].zero_();self.model.decoder.weight[:,0].zero_()
        balanced,info=ma.balance_model(self.model)
        self.assertEqual(info['balance_status'],'unanchored_bounded')
        self.assertTrue(all(torch.isfinite(p).all() for p in balanced.parameters()))

    def test_subject_and_bias_weights_are_equal(self):
        rows=pd.DataFrame(dict(subject_id=['A','A','A','B','B'],decoder_bias=[True,True,False,True,False]))
        weights=ma.cohort_weights(rows)
        np.testing.assert_allclose(weights,[.125,.125,.25,.25,.25])
        vectors=np.array([[0.],[0.],[1.],[2.],[3.]])
        winner,_,_=ma.choose_medoid(vectors,rows,np.ones(5,dtype=bool))
        self.assertIn(winner,[2,3])


if __name__=='__main__':unittest.main()
