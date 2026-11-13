"""Cached similarities must correspond to the model rows supplied by the caller."""
import itertools
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import pandas as pd
from NM_TinyRNN.code.measures import mechanistic_variability as mv


class SimilarityCacheSelection(unittest.TestCase):
    def frames(self):
        selected=pd.DataFrame([dict(model_id='model',subject_id=f'S{i//2}',outer_loop_n=i%2+1,
            inner_loop_idx=0,save_path='NM_TinyRNN/data/rnns/run/S0/GRU/sparsity') for i in range(4)])
        pairs=[]
        for ia,ib in itertools.combinations(range(4),2):
            a,b=selected.iloc[ia],selected.iloc[ib]
            pairs.append(dict(model_id='model',subject_A=a.subject_id,subject_B=b.subject_id,
                model_A_outer_n=a.outer_loop_n,model_B_outer_n=b.outer_loop_n,
                model_A_inner_n=0,model_B_inner_n=0))
        return selected,pd.DataFrame(pairs)

    def test_correct_cache_loads_without_recomputing(self):
        selected,pairs=self.frames()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'similarities.htsv';pairs.to_csv(path,sep='\t',index=False)
            with patch.object(mv,'_process_group',side_effect=AssertionError('Should use cache')):
                actual=mv.compute_similarities(selected,cache_path=path,use_cache=True)
            pd.testing.assert_frame_equal(actual,pairs)

    def test_changed_inner_winner_is_rejected(self):
        selected,pairs=self.frames();selected.loc[0,'inner_loop_idx']=1
        with self.assertRaisesRegex(ValueError,'does not match.*selection'):
            mv.validate_similarity_cache_models(pairs,selected)

    def test_extra_models_or_missing_pairs_are_rejected(self):
        selected,pairs=self.frames()
        with self.assertRaisesRegex(ValueError,'does not match'):
            mv.validate_similarity_cache_models(pairs,selected.iloc[:-1])
        with self.assertRaisesRegex(ValueError,'does not match'):
            mv.validate_similarity_cache_models(pairs.iloc[:-1],selected)

if __name__=='__main__':unittest.main()
