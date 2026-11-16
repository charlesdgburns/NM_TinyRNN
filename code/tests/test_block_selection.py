"""Protect the public outer-winner selection policy from silent score changes."""
import sys
from pathlib import Path
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from NM_TinyRNN.code.measures.performance import select_best_outer


class BlockSelection(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(model_id=['m','m'],hidden_size=[2,2],subject_id=['s','s'],
            outer_loop_n=[1,1],inner_loop_idx=[0,1],train_val_CE=[.2,.3],block_train_val_CE=[.4,.3]))

    def test_default_and_historical_selection_can_choose_different_models(self):
        self.assertEqual(select_best_outer(self.frame()).inner_loop_idx.iloc[0],1)
        self.assertEqual(select_best_outer(self.frame(),selection='continuous').inner_loop_idx.iloc[0],0)

    def test_missing_reset_scores_never_silently_fall_back(self):
        with self.assertRaisesRegex(ValueError,'Missing block_train_val_CE'):
            select_best_outer(self.frame().drop(columns='block_train_val_CE'))

    def test_nonfinite_scores_are_rejected(self):
        frame=self.frame();frame.loc[0,'block_train_val_CE']=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            select_best_outer(frame)


if __name__=='__main__':unittest.main()
