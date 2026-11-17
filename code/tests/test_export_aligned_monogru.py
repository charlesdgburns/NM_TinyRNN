"""Integration checks for regenerated aligned trajectories and subject means."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from NM_TinyRNN.code.analyses import export_aligned_monoGRU as export
from NM_TinyRNN.code.analyses import monoGRU_alignment as alignment
from NM_TinyRNN.code.models.rnns import TinyRNN
from NM_TinyRNN.code.models.save_data import get_model_trial_by_trial_df
from types import SimpleNamespace


class ExportAlignedMonoGRU(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model=TinyRNN(rnn_type='monoGRU',hidden_size=2,input_size=3,
            input_forced_choice=True,nonlinearity='relu',decoder_bias=False)
        gen=torch.Generator().manual_seed(812)
        with torch.no_grad():
            for p in self.model.parameters():p.copy_(torch.randn(p.shape,generator=gen)*.25)
            self.model.decoder.weight.copy_(torch.tensor([[-.8,.9],[.8,-.9]]))
        self.data=pd.DataFrame(dict(forced_choice=[0,0,1,0]*3,choice=[0,1,0,1]*3,
            outcome=[1,0,1,1]*3,good_poke=[0]*12,session_folder_name=['S/session']*12,
            session_trial_idx=range(12),sequence_block_idx=np.repeat([0.,1.,2.],4),
            block_trial_idx=range(12)))

    def test_regeneration_matches_native_export_and_preserves_aligned_function(self):
        saved=get_model_trial_by_trial_df(self.model,SimpleNamespace(subject_df=self.data),dict(train=[0],val=[1],eval=[2]))
        same=export.regenerate_trials(self.model,self.data,saved)
        pd.testing.assert_frame_equal(same,saved,check_dtype=False,atol=1e-6,rtol=1e-6)
        aligned,info=alignment.align_model(self.model,self.model)
        regenerated=export.regenerate_trials(aligned,self.data,saved)
        perm=info['perm'];scales=info['scales']
        np.testing.assert_allclose(regenerated[['hidden_1','hidden_2']],saved[['hidden_1','hidden_2']].to_numpy()[:,perm]*scales,atol=1e-6)
        np.testing.assert_allclose(regenerated[['prob_A','prob_B']],saved[['prob_A','prob_B']],atol=1e-6)
        self.assertEqual(regenerated.split.tolist(),saved.split.tolist())

    def test_mixed_split_labels_in_block_are_rejected(self):
        saved=get_model_trial_by_trial_df(self.model,SimpleNamespace(subject_df=self.data),dict(train=[0],val=[1],eval=[2]))
        saved.loc[0,'split']='val'
        with self.assertRaises(AssertionError):export.regenerate_trials(self.model,self.data,saved)

    def test_parameter_mean_weights_subjects_equally_even_with_unequal_fit_counts(self):
        models=[]
        for value in [1.,3.,10.]:
            m=alignment.write_parameters(self.model,{k:np.full_like(v,value) for k,v in alignment.parameters(self.model).items()})
            models.append(m)
        average,table=export.average_parameters(models,pd.DataFrame({'subject_id':['A','A','B']}))
        for p in average.parameters():np.testing.assert_allclose(p.detach(),6.)
        np.testing.assert_allclose(table['mean'],6.)
        self.assertTrue(table.n_subjects.eq(2).all())
        self.assertTrue(table.n_fits.eq(3).all())


if __name__=='__main__':unittest.main()
