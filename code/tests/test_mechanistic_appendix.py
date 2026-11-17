"""Observed-design attribution and metadata-only posterior reuse guards."""
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from NM_TinyRNN.code.analyses.mechanistic_variability_appendix import ordered_contributions
from NM_TinyRNN.code.analyses import mechanistic_variability_appendix as appendix
from NM_TinyRNN.code.analyses import mechanistic_variability_appendix as variance


class AppendixTests(unittest.TestCase):
    def test_additive_observed_design_recovers_four_effects(self):
        values=np.empty((3,2,2))
        for a,b,d in itertools.product(range(3),range(2),range(2)):
            values[a,b,d]=.1+2*b+3*(a>=1)+5*(a==2)+7*d
        np.testing.assert_allclose(ordered_contributions(values),[2,3,5,7])

    def test_interactions_share_only_over_permitted_orders(self):
        # This interaction becomes possible only after reset removal. Its whole
        # increment belongs to scalar gating under the declared precedence rule.
        values=np.zeros((3,2,2));values[2,:,:]=4
        np.testing.assert_allclose(ordered_contributions(values),[0,0,4,0])
        rng=np.random.default_rng(8);values=rng.normal(size=(20,3,2,2))
        np.testing.assert_allclose(ordered_contributions(values).sum(axis=-1),values[:,2,1,1]-values[:,0,0,0])

    def test_metadata_adoption_preserves_posterior_identity_and_rejects_changed_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            original=dict(source='old',input_sha256={'pairwise_similarities.htsv':'p','selected_models.htsv':'s','manifest.json':'old'},
                          source_manifest={'folder':'run','computation':{'fingerprint':'protocol'}},fingerprint='posterior')
            path=output/'manifest.json';path.write_text(json.dumps(original))
            current=dict(source='new',input_sha256={**original['input_sha256'],'manifest.json':'new'},source_manifest=original['source_manifest'])
            with patch.object(variance,'load_inputs',return_value=(None,None,current)):
                appendix.adopt_metadata_only(output,output)
            saved=json.loads(path.read_text());self.assertEqual(saved['fingerprint'],'posterior')
            self.assertEqual(saved['input_sha256']['manifest.json'],'new')
            current['input_sha256']['pairwise_similarities.htsv']='changed'
            with patch.object(variance,'load_inputs',return_value=(None,None,current)),self.assertRaises(ValueError):
                appendix.adopt_metadata_only(output,output)
            self.assertEqual(json.loads(path.read_text()),saved)


if __name__=='__main__':unittest.main()
