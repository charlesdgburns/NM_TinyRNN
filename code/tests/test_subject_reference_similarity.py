"""Restored subject-reference scores must reproduce the original notebook routine."""
from pathlib import Path
import pickle,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import pandas as pd
import numpy as np
import torch
from NM_TinyRNN.code.models.rnns import TinyRNN
from NM_TinyRNN.code.measures import mechanistic_variability as mv

class SubjectReferenceSimilarity(unittest.TestCase):
    def test_matches_original_subject_b_reference_scores(self):
        torch.set_num_threads(1)
        raw={'S0':torch.tensor([[[0.,0.,1.],[1.,1.,0.],[0.,1.,1.],[1.,0.,0.]]]),
             'S1':torch.tensor([[[1.,0.,1.],[0.,1.,0.],[1.,1.,1.],[0.,0.,0.],[1.,0.,0.]]])}
        for kind in ['GRU','monoGRU','lightGRU']:
            for act in ['relu','tanh']:
                with tempfile.TemporaryDirectory() as tmp:
                    root=Path(tmp);rows=[]
                    for i in range(4):
                        subject=f'S{i//2}'
                        model=TinyRNN(rnn_type=kind,input_size=3,hidden_size=2,nonlinearity=act,
                                      input_forced_choice=True,weight_seed=10+i)
                        with torch.no_grad():
                            for name,p in model.named_parameters():
                                if 'W' in name or name=='decoder.weight':p.mul_(200)
                            model.rnn.hidden_0.copy_(torch.tensor([.2,.1]))
                        model_path=root/f'{i}.pickle';trial_path=root/f'{i}.htsv'
                        with open(model_path,'wb') as f:pickle.dump(model,f)
                        pd.DataFrame(raw[subject][0].numpy(),columns=['forced_choice','choice','outcome']).to_csv(trial_path,sep='\t',index=False)
                        rows.append(dict(model_id=model.get_model_id(),model_type=kind,model_type2=kind,
                            model_pickle_path=str(model_path),trials_data_path=str(trial_path),
                            subject_id=subject,outer_loop_n=i%2+1,inner_loop_idx=0,nonlinearity=act,decoder_bias=True))
                    rows=pd.DataFrame(rows)
                    legacy=pd.DataFrame(mv._process_group(rows,rows.model_id.iloc[0]))
                    mv.compute_reference_group(rows,raw,root)
                    current=pd.read_csv(next((root/'groups').glob('*.htsv')),sep='\t')
                    self.assertTrue(current.trial_subject.eq(current.subject_B).all())
                    self.assertEqual(current.n_reference_steps.tolist(),[4,5,5,5,5,5])
                    for metric in mv.SIMILARITY_METRICS:
                        np.testing.assert_allclose(current[metric],legacy[metric],atol=2e-5,equal_nan=True,
                                                   err_msg=f'{kind}/{act}/{metric}')

if __name__=='__main__':unittest.main()
