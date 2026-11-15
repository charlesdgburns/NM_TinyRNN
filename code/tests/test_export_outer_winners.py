"""Standard-library checks for outer-winner artifact export."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'export_outer_winners', Path(__file__).parents[1] / 'models/export_outer_winners.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def itertuples(self):
        return iter(self.rows)


class ExportTests(unittest.TestCase):
    def test_outer_folds_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            rows = []
            for fold in (1, 2):
                source = root / 'source' / str(fold)
                source.mkdir(parents=True)
                for suffix in exporter.SUFFIXES:
                    (source / ('model' + suffix)).write_text(str(fold))
                rows.append(SimpleNamespace(subject_id='WS01', model_type='GRU',
                    model_id='model', outer_loop_n=fold, info_path=source / 'model_info.json'))
            artifacts = exporter.copy_winners(Rows(rows), root / 'export')
            self.assertEqual(len(artifacts), 6)
            for fold in (1, 2):
                self.assertEqual((root / 'export/WS01/GRU' / f'outer_fold_{fold}'
                                  / 'model_info.json').read_text(), str(fold))

    def test_missing_source_prevents_partial_export(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            (root / 'model_info.json').write_text('{}')
            row = SimpleNamespace(subject_id='WS01', model_type='GRU', model_id='model',
                                  outer_loop_n=1, info_path=root / 'model_info.json')
            with self.assertRaises(FileNotFoundError):
                exporter.copy_winners(Rows([row]), root / 'export')
            self.assertFalse((root / 'export').exists())


if __name__ == '__main__':
    unittest.main()
