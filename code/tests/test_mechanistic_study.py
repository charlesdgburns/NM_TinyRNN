"""The study owns ordered stages; plot-only runs must not rerun computation."""
import os,json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import pandas as pd
from NM_TinyRNN.code.analyses import mechanistic_variability as study


class MechanisticStudy(unittest.TestCase):
    def setUp(self):
        self.cwd=Path.cwd();self.addCleanup(os.chdir,self.cwd)
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.repo=Path(temporary.name)/'repo';self.repo.mkdir()

    def test_full_workflow_orders_compute_statistics_plots(self):
        calls=[];frame=pd.DataFrame()
        def compute(folder,tables,cache,manifest,*args):
            calls.append('compute');manifest['computation']={'cache_sha256':'test'}
            return frame,frame
        def statistics(selected,pairs,tables,manifest):
            self.assertIn('computation',manifest);calls.append('statistics');manifest['statistics']={'test':True}
        def plots(selected,pairs,tables,figures,manifest):
            self.assertIn('statistics',manifest);calls.append('plots');manifest['plots']={'test':True}
        with patch.object(study,'REPO',self.repo),patch.object(study,'compute',side_effect=compute),\
                patch.object(study,'statistics',side_effect=statistics),patch.object(study,'plots',side_effect=plots):
            study.main(['--folder','run'])
        self.assertEqual(calls,['compute','statistics','plots'])
        manifest=json.loads((self.repo/'data/analysis/mechanistic_variability/manifest.json').read_text())
        self.assertEqual(manifest['study'],'mechanistic_variability')

    def test_plot_only_never_calls_computation_or_statistics(self):
        tables=self.repo/'data/analysis/mechanistic_variability';tables.mkdir(parents=True)
        (tables/'manifest.json').write_text('{}')
        with patch.object(study,'REPO',self.repo),patch.object(study,'compute') as compute,\
                patch.object(study,'statistics') as statistics,\
                patch.object(study,'load_computation',return_value=(pd.DataFrame(),pd.DataFrame())) as load,\
                patch.object(study,'plots') as plots:
            study.main(['--folder','run','--stage','plots'])
        compute.assert_not_called();statistics.assert_not_called();load.assert_called_once();plots.assert_called_once()

    def test_run_name_cannot_escape_output_directory(self):
        with self.assertRaises(ValueError):study.paths_for('../other')

if __name__=='__main__':unittest.main()
