"""Checks for the shared model comparison and published numerical inputs."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shared import BASE,models,read,image_observables,select_widths


class TestComparison(unittest.TestCase):
    def test_registry_and_distinct_runs(self):
        registry=models();self.assertEqual(len(registry),12)
        self.assertEqual(sum(m['representation']=='image' for m in registry),5)
        particle=[m for m in registry if m['representation']=='particle']
        self.assertTrue(all(m['beta']!=1e-3 for m in particle))
        repeated=[m for m in particle if m['beta']==1e-5]
        self.assertEqual({m['seed'] for m in repeated},{1,42})
        self.assertEqual(len({m['checkpoint_sha256'] for m in repeated}),2)

    def test_pixel_observables_scale_invariance(self):
        image=np.zeros((1,32,32));image[0,14,16]=.5;image[0,18,16]=.5
        x=image_observables(image)
        np.testing.assert_allclose(image_observables(17*image),x,atol=1e-14)
        self.assertAlmostEqual(x[0],1/np.sqrt(2));self.assertAlmostEqual(x[1],np.log(2))
        self.assertTrue(np.all(x[2:7]>=0));self.assertTrue(np.isfinite(x).all())

    def test_no_bulk_width_is_not_invented(self):
        rng=np.random.default_rng(3);lv=rng.normal(size=(10000,4))
        choice=select_widths(lv,np.arange(4))
        self.assertEqual(choice['pairs'],[]);self.assertIsNone(choice['selected'])

    def test_all_plot_data(self):
        for m in models():
            with self.subTest(model=m['id']):
                d=read(BASE/'data'/f"{m['id']}.json");self.assertEqual(d['model'],m)
                r=d['reference'];active=r['active_indices'];n=len(active)
                for name in ['corr_mu','corr_logvar']:
                    matrix=np.asarray(r[name]);self.assertEqual(matrix.shape,(n,n))
                    np.testing.assert_allclose(matrix,matrix.T,atol=1e-13)
                    np.testing.assert_allclose(np.diag(matrix),1,atol=1e-13)
                for pair in r['pairs']:self.assertEqual(np.array(pair['counts']).sum(),10000)
                resp=d['response'];delta=np.array(resp['delta']);self.assertTrue(np.isfinite(delta).all())
                np.testing.assert_array_equal(delta[:,2],0)
                self.assertEqual(len(resp['source_sha256']),66)
                if 'rambo' in d:
                    ra=d['rambo'];self.assertEqual(ra['distributions']['rambo']['n_events'],79000)
                    self.assertEqual(sum(ra['response']['n_events']),99975)
                    np.testing.assert_array_equal(ra['response']['delta'][2],0)
                    for sample in ra['distributions'].values():self.assertEqual(np.asarray(sample['counts']).sum(),sample['n_events'])


if __name__=='__main__':unittest.main()
