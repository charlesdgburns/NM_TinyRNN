from pathlib import Path
import sys
from copy import deepcopy
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch
from NM_TinyRNN.code.models.rnns import TinyRNN
from NM_TinyRNN.code.measures import mechanistic_variability as mv
from NM_TinyRNN.code.measures.scaling_sensitivity import parameters,transform


class GridRefinement(unittest.TestCase):
    def test_recovers_known_scale_outside_grid(self):
        b=TinyRNN(rnn_type='GRU',input_size=3,hidden_size=1,nonlinearity='relu',weight_seed=17)
        a=deepcopy(b);values=transform(parameters(b),(0,),np.array([3.1]))
        with torch.no_grad():
            for name,p in a.named_parameters():
                if name in values:p.copy_(torch.as_tensor(values[name],dtype=p.dtype))
        _,grid=mv.align_rnns(a,b)
        aligned,refined=mv.align_rnns(a,b,refine_scaling=True)
        self.assertTrue(refined['refinement_success'])
        self.assertGreater(grid['aligned_distance'],1e-5)
        self.assertLess(refined['aligned_distance'],1e-6)
        self.assertAlmostEqual(refined['transform_vec'][0],3.1,places=5)
        np.testing.assert_array_equal(grid['perm'],refined['perm'])
        np.testing.assert_allclose(grid['aligned_distance'],refined['grid_distance'],rtol=1e-5)

    def test_refinement_preserves_function_and_inputs(self):
        torch.set_num_threads(1)
        for kind in ['GRU','monoGRU','lightGRU']:
            for h in [1,2]:
                a=TinyRNN(rnn_type=kind,input_size=3,hidden_size=h,nonlinearity='relu',weight_seed=17)
                b=TinyRNN(rnn_type=kind,input_size=3,hidden_size=h,nonlinearity='relu',weight_seed=31)
                before={k:v.clone() for k,v in b.state_dict().items()}
                grid,gi=mv.align_rnns(a,b);refined,ri=mv.align_rnns(a,b,refine_scaling=True)
                self.assertLessEqual(ri['aligned_distance'],gi['aligned_distance']+1e-6)
                x=torch.tensor([[[0.,0.,1.],[1.,1.,0.],[0.,1.,1.],[1.,0.,0.]]])
                with torch.no_grad():np.testing.assert_allclose(b.decoder(b.rnn(x)[0]),refined.decoder(refined.rnn(x)[0]),atol=1e-6,rtol=2e-5)
                for k,v in b.state_dict().items():self.assertTrue(torch.equal(v,before[k]))

    def test_custom_grids(self):
        b=TinyRNN(rnn_type='GRU',input_size=3,hidden_size=1,nonlinearity='relu',weight_seed=17)
        a=deepcopy(b);values=transform(parameters(b),(0,),np.array([3.1]))
        with torch.no_grad():
            for name,p in a.named_parameters():
                if name in values:p.copy_(torch.as_tensor(values[name],dtype=p.dtype))
        aligned,info=mv.align_rnns(a,b,refine_scaling=True,
            scale_values=np.geomspace(.25,4.,9))
        self.assertTrue(info['refinement_success'])
        self.assertLess(info['aligned_distance'],1e-6)
        self.assertEqual(info['refinement_method'],'trf')
        for values in [[],[0.,1.],[-1.,1.],[float('nan')]]:
            with self.assertRaises(ValueError):mv.align_rnns(a,b,scale_values=values)

    def test_tanh_is_unchanged(self):
        a=TinyRNN(rnn_type='GRU',input_size=3,hidden_size=2,nonlinearity='tanh',weight_seed=17)
        b=TinyRNN(rnn_type='GRU',input_size=3,hidden_size=2,nonlinearity='tanh',weight_seed=31)
        old,oi=mv.align_rnns(a,b);new,ni=mv.align_rnns(a,b,refine_scaling=True)
        self.assertIsNone(ni['refinement_success'])
        for k,v in old.state_dict().items():self.assertTrue(torch.equal(v,new.state_dict()[k]))


if __name__=='__main__':unittest.main()
