"""Selection, matched aggregation, cache provenance and standalone stage routing."""
from pathlib import Path
import sys,tempfile,pickle,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from NM_TinyRNN.code.studies import weight_similarity_sensitivity as study
from NM_TinyRNN.code.models.rnns import TinyRNN


def selected_frame(root=None):
    rows=[]
    for activation in ['relu','tanh']:
        for subject in ['S1','S2']:
            for outer in [1,2]:
                for inner,score in [(1,.2),(0,.2),(2,.3)]:
                    rows.append(dict(model_id='GRU_'+activation,model_type='GRU',hidden_size=2,
                        nonlinearity=activation,decoder_bias=True,subject_id=subject,outer_loop_n=outer,
                        inner_loop_idx=inner,train_val_CE=score,model_pickle_path='data/rnns/run/f.pickle'))
    return pd.DataFrame(rows)


def pairs_frame():
    rows=[]
    import itertools
    for activation in ['relu','tanh']:
        subjects=['S1','S1','S2','S2']
        for ia,ib in itertools.combinations(range(4),2):
            within=subjects[ia]==subjects[ib]
            base=(.8 if within else .4) if activation=='relu' else (.7 if within else .3)
            delta=.01 if activation=='relu' else .02
            rows.append(dict(model_id='GRU_'+activation,architecture='GRU',activation=activation,
                hidden_size=2,decoder_bias=True,index_A=ia,index_B=ib,subject_A=subjects[ia],subject_B=subjects[ib],
                outer_A=ia%2+1,outer_B=ib%2+1,inner_A=0,inner_B=0,
                comparison='within_subject' if within else 'across_subject',baseline_similarity=base,
                scaled_similarity=base+delta,delta_similarity=delta,baseline_distance=1.,scaled_distance=.9))
    return pd.DataFrame(rows)


class WeightSensitivityStudy(unittest.TestCase):
    def test_trainval_selection_and_stable_ties(self):
        frame=selected_frame();selected=study.select_models(frame,2,folds=2,subjects=['S1','S2'])
        self.assertEqual(len(selected),8)
        self.assertTrue(selected.inner_loop_idx.eq(0).all())
        self.assertTrue(selected.train_val_CE.eq(.2).all())
        with self.assertRaisesRegex(ValueError,'Duplicate'):study.select_models(pd.concat([frame,frame.iloc[[0]]]),2,2)
        with self.assertRaisesRegex(ValueError,'outer-fold'):study.select_models(frame,2,3)

    def test_matched_subject_means_percentages_and_modes(self):
        pairs=pairs_frame();selected=study.select_models(selected_frame(),2,2)
        study.validate_pairs(pairs,selected)
        blocks,summary,ranges,contrasts=study.aggregate(pairs)
        self.assertTrue(blocks[blocks.comparison.eq('within_subject')].fit_pairs.eq(1).all())
        self.assertTrue(blocks[blocks.comparison.eq('across_subject')].fit_pairs.eq(4).all())
        r=summary[summary.activation.eq('relu')&summary.comparison.eq('across_subject')].iloc[0]
        self.assertAlmostEqual(r.percent_change,2.5)
        np.testing.assert_allclose(contrasts['discrete'].difference,.1)
        np.testing.assert_allclose(contrasts['continuous'].difference,.11)
        with self.assertRaisesRegex(ValueError,'Unmatched'):
            study.aggregate(pairs[~(pairs.activation.eq('tanh')&pairs.comparison.eq('across_subject'))])
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            study.validate_pairs(pairs.iloc[1:],selected)

    def test_group_cache_requires_identity_and_unchanged_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'groups').mkdir();rows=[]
            for outer,seed in [(1,17),(2,31)]:
                model=TinyRNN(rnn_type='GRU',input_size=3,hidden_size=2,nonlinearity='relu',weight_seed=seed)
                path=root/f'{outer}.pickle';path.write_bytes(pickle.dumps(model))
                rows.append(dict(model_id='GRU_relu',model_type='GRU',nonlinearity='relu',hidden_size=2,
                    decoder_bias=True,subject_id='S1',outer_loop_n=outer,inner_loop_idx=0,model_pickle_path=str(path)))
            frame=pd.DataFrame(rows)
            result=study.compute_group(frame,root,'identity-one')
            with patch.object(study.mv,'align_rnns',wraps=study.mv.align_rnns) as align:
                study.compute_group(frame,root,'identity-one');align.assert_not_called()
                study.compute_group(frame,root,'identity-two');self.assertEqual(align.call_count,2)
            result.write_text(result.read_text()+'\n')
            with patch.object(study.mv,'align_rnns',wraps=study.mv.align_rnns) as align:
                study.compute_group(frame,root,'identity-two');self.assertEqual(align.call_count,2)

    def test_plots_only_uses_frozen_computation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(study,'REPO',Path(tmp)),patch.object(study,'compute') as compute,\
                    patch.object(study,'load_computation',return_value=(pd.DataFrame(),pd.DataFrame(),{})) as load,\
                    patch.object(study,'plots') as plots:
                study.main(['--folder','run','--stage','plots'])
            compute.assert_not_called();load.assert_called_once();plots.assert_called_once()
        with self.assertRaises(ValueError):study.paths_for('../other',2)


if __name__=='__main__':unittest.main()
