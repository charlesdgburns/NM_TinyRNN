"""Behavioral invariants and non-invariants under coordinate changes."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from NM_TinyRNN.code.models.rnns import TinyRNN
from NM_TinyRNN.code.analyses import monoGRU_alignment as ma
from NM_TinyRNN.code.analyses.monogru_activation_comparison import paired_metrics


class ActivationComparison(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model=TinyRNN(rnn_type='monoGRU',hidden_size=2,input_size=3,nonlinearity='relu',decoder_bias=False,input_forced_choice=True)
        with torch.no_grad():
            for p in self.model.parameters():p.zero_()
            self.model.rnn.W_ih.copy_(torch.tensor([[.1,.1],[1.,-1.],[.2,1.1]]))
            self.model.rnn.bias_h.copy_(torch.tensor([.1,.8]))
            self.model.rnn.W_hh.copy_(torch.eye(2)*.2)
            self.model.rnn.hidden_0.copy_(torch.tensor([.1,.2]))
            self.model.decoder.weight.copy_(torch.tensor([[1.,-1.],[-1.,1.]]))
        gen=torch.Generator().manual_seed(822)
        self.inputs=torch.randint(0,2,(1,256,3),generator=gen).float()

    def forward(self,model):
        with torch.no_grad():logits,h=model(self.inputs)
        return logits,h[0].numpy()

    def test_permutation_only_preserves_total_and_predictions(self):
        flipped=ma.write_parameters(self.model,ma.transform(ma.parameters(self.model),(1,0),np.ones(2)))
        relabeled,info=ma.align_balanced_model(flipped,self.model)
        np.testing.assert_array_equal(info['transform_vec'],np.ones(2))
        p,x=self.forward(flipped);q,y=self.forward(relabeled)
        torch.testing.assert_close(p,q)
        np.testing.assert_allclose(x.sum(1),y.sum(1),atol=1e-6)

    def test_common_gain_changes_amplitude_but_preserves_sum_time_course(self):
        scaled=ma.write_parameters(self.model,ma.transform(ma.parameters(self.model),(0,1),np.array([3.,3.])))
        p,x=self.forward(self.model);q,y=self.forward(scaled)
        torch.testing.assert_close(p,q)
        for record in paired_metrics(x,y):
            self.assertAlmostEqual(record['trajectory_r'],1.,places=6)
            self.assertAlmostEqual(record['mean_ratio'],3.,places=5)

    def test_unequal_gains_change_sum_shape_without_changing_output(self):
        scaled=ma.write_parameters(self.model,ma.transform(ma.parameters(self.model),(0,1),np.array([.25,4.])))
        p,x=self.forward(self.model);q,y=self.forward(scaled)
        torch.testing.assert_close(p,q)
        records={r['component']:r for r in paired_metrics(x,y)}
        for component in ['h1','h2']:self.assertAlmostEqual(records[component]['trajectory_r'],1.,places=6)
        self.assertLess(records['sum']['trajectory_r'],.99)


if __name__=='__main__':unittest.main()
