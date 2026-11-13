"""Missing run folders should fail before performance accesses absent columns."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import pandas as pd
from NM_TinyRNN.code.measures import analysis,performance

class EmptyAnalysis(unittest.TestCase):
    def test_missing_run_is_reported_by_path_discovery(self):
        frame=pd.DataFrame([dict(save_path='NM_TinyRNN/data/rnns/missing_run/WS01/GRU/energy',completed=False)])
        with self.assertRaisesRegex(FileNotFoundError,'No completed models.*missing_run'):
            analysis.get_analysis_df(frame,n_jobs=1,use_cache=False)

    def test_empty_analysis_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError,'Analysis dataframe is empty'):
            performance.get_performance_df(pd.DataFrame())
        with self.assertRaisesRegex(ValueError,'Experiment dataframe is empty'):
            analysis.get_analysis_df(pd.DataFrame())

if __name__=='__main__':unittest.main()
