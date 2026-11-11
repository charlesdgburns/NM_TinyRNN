"""Shared experiment manifests and submission/analysis boundaries."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))
from NM_TinyRNN.code.models import experiments
from NM_TinyRNN.code.servers.SLURM import submit_jobs
from NM_TinyRNN.code.measures import analysis


class ExperimentManifests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_root = self.root / 'data'
        (self.data_root / 'WS01').mkdir(parents=True)
        self.experiment_df = experiments.get_DA_info_df(
            processed_data_path=self.data_root, save_path=self.root / 'results',
            sequence_length=17, batch_size=3)

    def test_legacy_builders_are_shared_not_duplicated(self):
        from NM_TinyRNN.code.models import submit_jobs as legacy
        for name in ['get_DA_info_df', 'get_job_info_df', 'get_test_info_df',
                     'get_ws_all_info_df', 'training_folder_name']:
            self.assertIs(getattr(experiments, name), getattr(submit_jobs, name))
            self.assertIs(getattr(experiments, name), getattr(legacy, name))
        self.assertTrue(self.experiment_df.sequence_length.eq(17).all())
        self.assertTrue(self.experiment_df.batch_size.eq(3).all())
        self.assertFalse({'RAM', 'time_limit', 'script_path'} & set(self.experiment_df.columns))

    def test_supplied_manifest_is_used_without_rebuilding_or_test_limit(self):
        experiment_df = self.experiment_df.query('outer_loop_n == 1').head(4).copy()
        experiment_df.loc[experiment_df.index[0], 'completed'] = True
        before = experiment_df.copy(deep=True)
        with patch.object(experiments, 'get_test_info_df') as preset, \
                patch.object(submit_jobs, 'get_NM_TinyRNN_SLURM_script',
                             return_value=self.root / 'job script.sh') as script, \
                patch.object(submit_jobs.os, 'system', return_value=0) as system:
            # Default test=True applies only when selecting a preset, not to this table.
            submit_jobs.run_training(experiment_df)
            preset.assert_not_called()
            self.assertEqual(script.call_count, 3)
            self.assertEqual(system.call_count, 3)
            self.assertEqual({c.args[0].model_id for c in script.call_args_list},
                             set(experiment_df.loc[experiment_df.completed == False, 'model_id']))
            for call in script.call_args_list:
                self.assertEqual(call.args[0].sequence_length, 17)
                self.assertEqual(call.args[0].batch_size, 3)
            script.reset_mock()
            submit_jobs.run_training(experiment_df, overwrite=True)
            self.assertEqual(script.call_count, 4)
        pd.testing.assert_frame_equal(experiment_df, before)

    def test_legacy_preset_submission_still_forwards_settings(self):
        with patch.object(experiments, 'get_DA_info_df', return_value=self.experiment_df) as preset, \
                patch.object(submit_jobs, 'get_NM_TinyRNN_SLURM_script', return_value=self.root / 'job.sh'), \
                patch.object(submit_jobs.os, 'system', return_value=0) as system:
            submit_jobs.run_training(test=False, sequence_length=17, batch_size=3)
            preset.assert_called_once_with(sequence_length=17, batch_size=3, folder_name=None)
            self.assertEqual(system.call_count, len(self.experiment_df.query('outer_loop_n == 1')))

    def test_analysis_expands_same_manifest_and_retains_legacy_keyword(self):
        experiment_df = self.experiment_df.iloc[[2]].copy() # filtered manifests need not start at index 0
        experiment_df['completed'] = True
        row = experiment_df.iloc[0]
        for fold in [0, 1]:
            folder = row.save_path / 'outer_fold_1' / f'inner_fold_{fold}'
            folder.mkdir(parents=True)
            (folder / f'{row.model_id}_info.json').write_text('{}')
        with patch.object(analysis, 'DATA_PATH', self.root / 'analysis_data'):
            analysis_df = analysis.get_analysis_df(experiment_df=experiment_df, n_jobs=1, use_cache=False)
            legacy_df = analysis.get_analysis_df(info_df=experiment_df, n_jobs=1, use_cache=False)
        pd.testing.assert_frame_equal(analysis_df, legacy_df)
        self.assertEqual(len(analysis_df), 2)
        self.assertTrue(analysis_df.sequence_length.eq(17).all())
        self.assertTrue(analysis_df.batch_size.eq(3).all())
        self.assertEqual(set(analysis_df.inner_loop_idx), {0, 1})
        self.assertTrue(all(Path(p).is_file() for p in analysis_df.info_path))

    def test_models_experiments_does_not_import_server_tools(self):
        code = ('import sys; from NM_TinyRNN.code.models import experiments; '
                'assert not any(n.startswith("NM_TinyRNN.code.servers") for n in sys.modules)')
        subprocess.run([sys.executable, '-c', code], cwd=REPO.parent,
                       capture_output=True, text=True, check=True)


if __name__ == '__main__':
    unittest.main()
