"""Sequence-length plumbing and output naming; synthetic data, no SLURM jobs."""
import ast
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))
from NM_TinyRNN.code.models import datasets as ds, nested_cv as nc, submit_jobs as sj
from NM_TinyRNN.code.models import rnns, training


class SequenceSettings(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.behavior = self.root / 'data' / 'AB_behaviour'
        for subject in ['WS01', 'simulation']:
            for day, n in enumerate([67, 70, 73], start=1):
                folder = self.behavior / subject / f'2026-01-{day:02d}-120000'
                folder.mkdir(parents=True)
                pd.DataFrame({'forced_choice': np.arange(n) % 3 == 0,
                              'choice': np.arange(n) % 2,
                              'outcome': np.arange(n) % 2,
                              'good_poke': np.arange(n) % 2}).to_csv(
                                  folder / 'trials.htsv', sep='\t', index=False)
        self.data_patch = patch.object(ds, 'DATA_PATH', self.root / 'data')
        self.data_patch.start()
        self.addCleanup(self.data_patch.stop)

    def dataset(self, **kwargs):
        return ds.AB_Dataset(self.behavior / 'WS01', device=torch.device('cpu'), **kwargs)

    def test_block_lengths_alignment_and_default(self):
        for length in [4, 7, ds.SEQUENCE_LENGTH]:
            dataset = self.dataset(sequence_length=length)
            self.assertEqual(dataset.inputs.shape[1], length)
            self.assertEqual(len(dataset), sum(n // (length + 1) for n in [67, 70, 73]))
            valid = dataset.subject_df.dropna(subset=['sequence_block_idx'])
            for index, (_, rows) in enumerate(valid.groupby('sequence_block_idx', sort=True)):
                self.assertEqual(rows.session_folder_name.nunique(), 1)
                self.assertEqual(len(rows), length + 1)
                torch.testing.assert_close(dataset.targets[index], torch.tensor(rows.choice.values[1:]))
                torch.testing.assert_close(dataset.forced_choice_mask[index],
                                           torch.tensor(rows.forced_choice.values[1:], dtype=torch.float32))
        self.assertEqual(self.dataset().sequence_length, ds.SEQUENCE_LENGTH)
        for invalid in [0, -1, 1.5, True]:
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                self.dataset(sequence_length=invalid)

    def test_job_factories_name_and_record_settings(self):
        for factory, prefix in [(sj.get_DA_info_df, 'nested_DA'),
                                (sj.get_job_info_df, 'nested_DA'),
                                (sj.get_test_info_df, 'nested_DA'),
                                (sj.get_ws_all_info_df, 'test_input_encodings')]:
            frame = factory(processed_data_path=self.behavior, save_path=self.root,
                            sequence_length=7, batch_size=3)
            self.assertFalse(frame.empty)
            self.assertTrue(frame.sequence_length.eq(7).all())
            self.assertTrue(frame.batch_size.eq(3).all())
            self.assertEqual({p.relative_to(self.root).parts[0] for p in frame.save_path},
                             {f'{prefix}_S7_B3'})
        frame = sj.get_DA_info_df(processed_data_path=self.behavior, save_path=self.root)
        self.assertEqual(frame.iloc[0].save_path.relative_to(self.root).parts[0],
                         f'nested_DA_S{ds.SEQUENCE_LENGTH}_B{training.BATCH_SIZE}')
        legacy = sj.get_DA_info_df(processed_data_path=self.behavior, save_path=self.root,
                                   folder_name='nested_DA_128')
        self.assertIn('vanilla', legacy.model_type.unique())
        self.assertEqual(legacy.iloc[0].save_path.relative_to(self.root).parts[0], 'nested_DA_128')
        for kwargs in [dict(sequence_length=0), dict(batch_size=0)]:
            with self.assertRaises(ValueError):
                sj.get_DA_info_df(processed_data_path=self.behavior, **kwargs)

    def test_generated_command_reaches_nested_training(self):
        frame = sj.get_DA_info_df(processed_data_path=self.behavior, save_path=self.root,
                                 sequence_length=7, batch_size=3)
        row = frame.iloc[0]
        with patch.object(sj, 'JOBS_PATH', self.root / 'Jobs'):
            script = sj.get_NM_TinyRNN_SLURM_script(row)
            text = script.read_text()
            subprocess.run(['bash', '-n', str(script)], check=True)
            self.assertIn('_S7_B3.sh', script.name)
            command = shlex.split(next(line for line in text.splitlines() if line.startswith('python -c ')))
            ast.parse(command[2])
            original_options = rnns.OPTIONS_DICT.copy()
            with patch.object(nc, 'run_outer_fold') as run_fold:
                exec(command[2], {})
            self.assertEqual(run_fold.call_count, nc.N_OUTER_LOOPS)
            for call in run_fold.call_args_list:
                self.assertEqual(call.args[1].sequence_length, 7)
                self.assertEqual(call.args[1].inputs.shape[1], 7)
                self.assertEqual(call.kwargs['trainer_kwargs']['batch_size'], 3)
                self.assertEqual(call.kwargs['save_path'], str(row.save_path))
            self.assertEqual(rnns.OPTIONS_DICT, original_options)

    def test_train_outers_defaults_and_direct_fit(self):
        with patch.object(nc, 'run_outer_fold') as run_fold:
            nc.train_outers(self.behavior / 'WS01', self.root / 'results', 'GRU')
        self.assertEqual(run_fold.call_args.args[1].sequence_length, ds.SEQUENCE_LENGTH)
        self.assertEqual(run_fold.call_args.kwargs['trainer_kwargs']['batch_size'], training.BATCH_SIZE)
        dataset = self.dataset(sequence_length=4)
        trainer = training.Trainer(weight_seeds=[0], sparsity_lambdas=[0],
                                   energy_lambdas=[0], batch_size=3, max_epochs=1)
        state, _, _ = trainer.fit(rnns.TinyRNN(input_size=2, hidden_size=2), dataset)
        self.assertTrue(state)
        self.assertTrue(torch.isfinite(trainer._last_best_val_loss))

    def test_small_nested_run_saves_sequence_and_batch_sizes(self):
        dataset = self.dataset(sequence_length=7)
        result = nc.run_outer_fold(
            rnns.TinyRNN(input_size=2, hidden_size=2), dataset,
            n_outer_loops=3, n_jobs=1, save_path=self.root / 'nested',
            trainer_kwargs=dict(weight_seeds=[0], sparsity_lambdas=[0],
                                energy_lambdas=[0], batch_size=3, max_epochs=1))
        self.assertEqual(len(result['inner_results']), 2)
        paths = list((self.root / 'nested').rglob('*_info.json'))
        self.assertEqual(len(paths), 2)
        for path in paths:
            info = json.loads(path.read_text())
            self.assertEqual(info['sequence_length'], 7)
            self.assertEqual(info['hparam_ranges']['batch_size'], 3)

    def test_local_runner_cli_defaults_and_overrides(self):
        # Import these CPU-only runners in subprocesses to isolate their env setup.
        for module in ['session_learning_curve', 'pooled_ws_cv']:
            code = (f'from NM_TinyRNN.code.models import {module} as runner; '
                    'args=runner.parse_args(); print(args.sequence_length, args.batch_size)')
            for options, expected in [([], f'{ds.SEQUENCE_LENGTH} {training.BATCH_SIZE}'),
                                      (['--sequence-length', '7', '--batch-size', '3'], '7 3')]:
                proc = subprocess.run([sys.executable, '-c', code, *options],
                                      cwd=REPO.parent, capture_output=True, text=True, check=True)
                self.assertEqual(proc.stdout.strip(), expected)
            proc = subprocess.run([sys.executable, '-c', code, '--sequence-length', '0'],
                                  cwd=REPO.parent, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)

    def test_pooled_dry_run_uses_custom_block_length(self):
        output = self.root / 'pooled'
        subprocess.run([sys.executable, str(REPO / 'code/models/pooled_ws_cv.py'),
                        '--data-root', str(self.behavior), '--subjects', 'WS01',
                        '--sequence-length', '7', '--batch-size', '3',
                        '--models', 'GRU_tanh_biasoff', '--output', str(output), '--dry-run'],
                       capture_output=True, text=True, check=True)
        manifest = json.loads((output / 'run.json').read_text())
        self.assertEqual(manifest['sequence_length'], 7)
        self.assertEqual(manifest['trainer']['batch_size'], 3)
        self.assertEqual(manifest['subject_plan']['WS01']['n_blocks'],
                         sum(n // 8 for n in [67, 70, 73]))


if __name__ == '__main__':
    unittest.main()
