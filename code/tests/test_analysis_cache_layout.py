"""Dataset-free cache paths must not allow cross-run or fold-selection reuse."""
import sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pandas as pd
from NM_TinyRNN.code.measures import analysis, performance, mechanistic_variability as mv
from NM_TinyRNN.code.analyses import mechanistic_variability as study
from NM_TinyRNN.code.analyses import weight_similarity_sensitivity as weight


class AnalysisCacheLayout(unittest.TestCase):
    def frame(self, run='run_a'):
        return pd.DataFrame([dict(save_path=f'data/rnns/{run}/S1/GRU', model_id='GRU',
            outer_loop_n=1, inner_loop_idx=0, info_path=f'data/rnns/{run}/S1/GRU/info.json')])

    def test_identical_fold_labels_from_another_run_are_rejected(self):
        self.assertTrue(analysis.cache_matches_models(self.frame(), self.frame()))
        self.assertFalse(analysis.cache_matches_models(self.frame(), self.frame('run_b')))
        wrong = self.frame(); wrong.loc[0, 'inner_loop_idx'] = 1
        self.assertFalse(analysis.cache_matches_models(self.frame(), wrong))

    def test_outer_manifest_matches_expanded_inner_models(self):
        expanded = pd.concat([self.frame(), self.frame()], ignore_index=True)
        expanded.loc[1, 'inner_loop_idx'] = 1
        requested = self.frame().drop(columns=['inner_loop_idx', 'info_path'])
        self.assertTrue(analysis.cache_matches_models(expanded, requested, inner=False))
        self.assertFalse(analysis.cache_matches_models(expanded, requested))

    def test_analysis_output_paths_do_not_embed_dataset_or_hidden_size(self):
        tables, _, cache = study.paths_for('run_a')
        self.assertEqual(tables, study.paths_for('run_b')[0])
        self.assertEqual(cache, tables/'pairwise_similarities.htsv')
        self.assertEqual(weight.paths_for('run_a', 2)[0], weight.paths_for('run_b', 1)[0])

    def test_study_rejects_other_dataset_on_plot_only(self):
        import os,json
        cwd=Path.cwd();self.addCleanup(os.chdir,cwd)
        with tempfile.TemporaryDirectory() as temp:
            repo=Path(temp)/'repo';tables=repo/'data/analysis/mechanistic_variability'
            tables.mkdir(parents=True)
            (tables/'manifest.json').write_text(json.dumps({'folder':'run_a'}))
            with patch.object(study,'REPO',repo),patch.object(study,'load_computation') as load:
                with self.assertRaisesRegex(ValueError,'different training run'):
                    study.main(['--folder','run_b','--stage','plots'])
                load.assert_not_called()

if __name__ == '__main__':
    unittest.main()
