"""Physical and cross-file consistency checks for the appendix calculations."""
import hashlib
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import HERE, observables, project, read_json


class TestAppendix(unittest.TestCase):
    def test_symmetric_two_particle_observables(self):
        angle = .1
        # Equal massless momenta at opposite azimuths around the x axis.
        event = np.array([[22,np.cos(angle),np.sin(angle),0,1],
                          [22,np.cos(angle),-np.sin(angle),0,1]])
        result = observables(event)
        expected = [2,1/np.sqrt(2),-np.log(.5+1e-12),.5,.25,.0625,
                    .25*np.sqrt(.5),.125,np.log(4*np.sin(angle)**2+1e-12)]
        np.testing.assert_allclose(result,expected,rtol=1e-12,atol=1e-12)
        # An azimuthal rotation across the branch cut preserves all observables.
        phi = 3.13
        rotated = event.copy()
        rotated[:,1] = event[:,1]*np.cos(phi)-event[:,2]*np.sin(phi)
        rotated[:,2] = event[:,1]*np.sin(phi)+event[:,2]*np.cos(phi)
        np.testing.assert_allclose(observables(rotated),result,atol=1e-12)

    def test_full_constituents_and_one_particle_mass(self):
        # Observables must not inherit the encoder's 80-particle cap.
        event = np.tile([22,1,0,0,1],(81,1))
        self.assertEqual(observables(event)[0],81)
        single = observables(event[:1])
        self.assertEqual(single[0],1)
        self.assertAlmostEqual(single[-1],np.log(1e-12))
        self.assertTrue(np.isfinite(single).all())

    def test_projection_is_not_recentered(self):
        basis = dict(a=[2.,3.],b=[.5,1.5],active_indices=[0,1])
        lv = np.array(basis['a']) + 4*np.array(basis['b'])
        np.testing.assert_allclose(project(lv[None,:],basis),[4.])

    def test_published_plot_inputs(self):
        manifest=read_json(HERE/'data/provenance.json')
        for name,expected in manifest['plot_data_sha256'].items():
            self.assertEqual(hashlib.sha256((HERE/name).read_text(encoding="utf-8").encode("utf-8")).hexdigest(),expected)
        targets=read_json(HERE/'RAMBO/data/multiplicity_targets.json')['groups']
        response=read_json(HERE/'RAMBO/data/fig10_response.json')
        sizes=[]
        for group,mean_n in zip(targets,response['mean_N']):
            counts=np.array(group['counts_after_N1_cut'])
            self.assertEqual(counts[:2].sum(),0)
            self.assertEqual(sum(group['original_counts']),20000)
            self.assertAlmostEqual(counts@np.arange(81)/counts.sum(),mean_n)
            sizes.append(int(counts.sum()))
        self.assertEqual(sum(sizes),99975)
        self.assertEqual(sizes,response['n_events'])
        corr=np.array(read_json(HERE/'data/fig11_correlations.json')['correlations'])
        self.assertEqual(corr.shape,(33,9))
        np.testing.assert_allclose(corr[-1,[0,1,2,6,8]],[-.74,.66,-.75,-.79,-.70],atol=.005)


if __name__=='__main__':
    unittest.main()
