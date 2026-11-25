"""Physics and interface checks; run: python -m unittest discover -s RAMBO/tests -v."""
from pathlib import Path
import json
import math
import subprocess
import sys
import tempfile
import unittest
import numpy as np

RAMBO_DIR=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RAMBO_DIR))
from rambo import (generate_massless,validate_events,validate_total_p4,
                   total_from_mass_pt,log_phase_space_volume)


class TestRambo(unittest.TestCase):
    def test_kinematics_rest_and_boosted(self):
        targets=[[200,0,0,0],total_from_mass_pt(20,200),total_from_mass_pt(35,200,.8,1.1)]
        for n in (2,3,20,80):
            for target in targets:
                with self.subTest(n=n,target=target):
                    p=generate_massless(1500,n,target,seed=1700+n)
                    check=validate_events(p,target)
                    self.assertEqual(check['status'],'PASS')
                    self.assertLess(check['max_four_momentum_error_GeV'],1e-7)

    def test_two_body_exact_energies(self):
        p=generate_massless(10000,2,[200,0,0,0],seed=51)
        np.testing.assert_allclose(p[...,0],100,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(p[:,0,1:],-p[:,1,1:],rtol=1e-9,atol=1e-8)

    def test_three_and_four_body_phase_space_distributions(self):
        # Analytic labeled-particle spectrum x=2E/M is Beta(2,N-2).
        # N=3: F=x^2. N=4: F=3x^2-2x^3. This tests the measure,
        # independently of energy-momentum conservation.
        for n in (3,4):
            p=generate_massless(60000,n,[200,0,0,0],seed=810+n)
            x=np.sort(p[:,0,0]/100)
            self.assertGreater(x.min(),0);self.assertLessEqual(x.max(),1+1e-12)
            cdf=x*x if n==3 else 3*x*x-2*x*x*x
            m=len(x)
            d=max(np.max(np.arange(1,m+1)/m-cdf),np.max(cdf-np.arange(m)/m))
            self.assertLess(d,.012,(n,d))
            self.assertAlmostEqual(float(x.mean()),2/n,delta=.006)
            self.assertAlmostEqual(float(np.mean(x*x)),6/(n*(n+1)),delta=.006)
            if n==3:
                # Pair invariant s12/M^2 = 1-x3 has density 2*(1-z).
                pair=p[:,0]+p[:,1]
                z=(pair[:,0]**2-np.sum(pair[:,1:]**2,axis=1))/200**2
                self.assertAlmostEqual(float(z.mean()),1/3,delta=.006)

    def test_cm_isotropy(self):
        p=generate_massless(60000,4,[200,0,0,0],seed=111)
        directions=p[:,0,1:]/p[:,0,0,None]
        np.testing.assert_allclose(directions.mean(axis=0),0,atol=.012)
        np.testing.assert_allclose((directions**2).mean(axis=0),1/3,atol=.012)

    def test_lorentz_boost_and_pair_invariants(self):
        target=total_from_mass_pt(20,200,.4,.7)
        _,mass=validate_total_p4(target)
        rest=generate_massless(3000,8,[mass,0,0,0],seed=68)
        lab=generate_massless(3000,8,target,seed=68)
        # Independent textbook boost expression using beta and gamma.
        beta=target[1:]/target[0];beta2=beta@beta;gamma=target[0]/mass
        dot=np.sum(rest[...,1:]*beta,axis=-1)
        expected_e=gamma*(rest[...,0]+dot)
        expected_v=rest[...,1:]+beta*(gamma*rest[...,0]+(gamma-1)*dot/beta2)[...,None]
        np.testing.assert_allclose(lab[...,0],expected_e,rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(lab[...,1:],expected_v,rtol=1e-11,atol=1e-11)
        def s12(p):
            q=p[:,0]+p[:,1]
            return q[:,0]**2-np.sum(q[:,1:]**2,axis=1)
        np.testing.assert_allclose(s12(lab),s12(rest),rtol=1e-9,atol=1e-9)

    def test_reproducibility_and_batches(self):
        p=generate_massless(130,20,total_from_mass_pt(20,200),seed=75,batch_size=17)
        q=generate_massless(130,20,total_from_mass_pt(20,200),seed=75,batch_size=130)
        np.testing.assert_array_equal(p,q)
        self.assertFalse(np.array_equal(p,generate_massless(130,20,total_from_mass_pt(20,200),seed=76)))

    def test_phase_space_volume_normalization(self):
        self.assertAlmostEqual(math.exp(log_phase_space_volume(2,200)),1/(8*math.pi),places=14)
        self.assertAlmostEqual(math.exp(log_phase_space_volume(3,200)),200**2/(256*math.pi**3),places=12)

    def test_bad_inputs(self):
        for target in ([200,200,0,0],[100,200,0,0],[-200,0,0,0],[0,0,0,0],[math.nan,0,0,0],[1,2,3]):
            with self.assertRaises(ValueError):generate_massless(10,3,target)
        for n in (0,1,-2,2.5,True):
            with self.assertRaises(ValueError):generate_massless(10,n,[200,0,0,0])
        for count in (0,-1,1.5,True):
            with self.assertRaises(ValueError):generate_massless(count,3,[200,0,0,0])
        with self.assertRaises(ValueError):generate_massless(2,3,[200,0,0,0],seed=1,rng=np.random.default_rng(1))
        with self.assertRaises(ValueError):total_from_mass_pt(0,200)
        with self.assertRaises(ValueError):total_from_mass_pt(20,-200)

    def test_cli_dat_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='test_',dir=RAMBO_DIR) as temp:
            out=Path(temp).resolve()
            assert out.is_relative_to(RAMBO_DIR.resolve())
            command=[sys.executable,'-B','-X','utf8',str(RAMBO_DIR/'rambo.py'),
                     '--n-events','16','--n-particles','2','7','--mass','20','--pt','200',
                     '--seed','37','--batch-size','5','--write-dat','--output',str(out)]
            result=subprocess.run(command,text=True,encoding='utf-8',capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            for n in (2,7):
                stem=out/f'rambo_N{n:03d}'
                with np.load(stem.with_suffix('.npz')) as z:
                    p=z['p4'];weights=z['weights']
                    self.assertEqual(p.shape,(16,n,4));np.testing.assert_array_equal(weights,np.ones(16))
                events=[];current=[]
                for line in stem.with_suffix('.dat').read_text().splitlines():
                    if line.startswith('<event_begin>'):current=[]
                    elif line.startswith('<event_end>'):events.append(current)
                    else:
                        pid,px,py,pz,e=line.split();self.assertEqual(pid,'0')
                        current.append([float(e),float(px),float(py),float(pz)])
                np.testing.assert_array_equal(np.array(events),p)
                meta=json.loads(stem.with_suffix('.json').read_text())
                self.assertEqual(meta['validation']['status'],'PASS')
                expected=generate_massless(16,n,total_from_mass_pt(20,200),rng=np.random.default_rng(np.random.SeedSequence([37,n])),batch_size=16)
                np.testing.assert_array_equal(p,expected)
            before=(out/'rambo_N002.npz').read_bytes()
            second=subprocess.run(command,text=True,encoding='utf-8',capture_output=True)
            self.assertEqual(second.returncode,2)
            self.assertEqual((out/'rambo_N002.npz').read_bytes(),before)


if __name__=='__main__':unittest.main()
