from pathlib import Path
import sys,unittest,itertools
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from NM_TinyRNN.code.measures.mechanistic_variability import contrast,holm

class SimilarityStatistics(unittest.TestCase):
    def test_within_jackknife_matches_paired_t(self):
        differences=np.array([.1,.3,-.1,.4,.2])
        frame=pd.DataFrame(dict(subject_A=list('abcde'),subject_B=list('abcde'),reference=np.zeros(5),model=differences))
        result=contrast(frame)
        self.assertAlmostEqual(result['p_raw'],ttest_1samp(differences,0).pvalue)
        self.assertAlmostEqual(result['cohens_dz'],differences.mean()/differences.std(ddof=1))
        self.assertAlmostEqual(result['jackknife_se'],differences.std(ddof=1)/np.sqrt(5))

    def test_across_jackknife_deletes_every_incident_pair(self):
        records=[dict(subject_A=str(i),subject_B=str(j),reference=0.,model=float(i+j)) for i,j in itertools.combinations(range(5),2)]
        f=pd.DataFrame(records);result=contrast(f)
        means=np.array([f.loc[~(f.subject_A.eq(str(i))|f.subject_B.eq(str(i))),'model'].mean() for i in range(5)])
        self.assertAlmostEqual(result['jackknife_se'],np.sqrt(4/5*np.sum((means-means.mean())**2)))
        reversed_=f.rename(columns={'model':'reference','reference':'model'})
        self.assertAlmostEqual(contrast(reversed_)['cohens_dz'],-result['cohens_dz'])
        self.assertAlmostEqual(contrast(reversed_)['p_raw'],result['p_raw'])

    def test_holm_adjustment(self):
        np.testing.assert_allclose(holm([.04,.01,.03]),[.06,.03,.06])

if __name__=='__main__':unittest.main()
