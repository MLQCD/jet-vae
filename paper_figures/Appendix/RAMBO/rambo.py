"""Massless RAMBO: uniform Lorentz-invariant N-body phase space.

Four-vector convention: (E, px, py, pz), all in GeV, metric (+---).
Each event has N labeled massless particles and a specified future-timelike
sum. Fixed-N events have equal relative weights. No shower or jet selection.

Algorithm: Kleiss, Stirling, Ellis, Comput. Phys. Commun. 40 (1986) 359,
https://doi.org/10.1016/0010-4655(86)90119-0 .
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import sys

import numpy as np


def _integer(value: int, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return int(value)


def validate_total_p4(total_p4) -> tuple[np.ndarray, float]:
    """Return a float64 four-vector and its strictly positive invariant mass."""
    p = np.asarray(total_p4, dtype=np.float64)
    if p.shape != (4,) or not np.isfinite(p).all():
        raise ValueError('total_p4 must be four finite numbers: (E, px, py, pz).')
    spatial_norm = math.hypot(*p[1:])
    if p[0] <= spatial_norm:
        raise ValueError('Require E > |P|: a future-timelike total four-vector. '
                         'E=|P| only admits exactly collinear massless particles, '
                         'not a nondegenerate RAMBO phase space.')
    # Factored form reduces cancellation when the boost is large.
    mass = math.sqrt((p[0] - spatial_norm) * (p[0] + spatial_norm))
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('Total invariant mass is not numerically well resolved.')
    return p, mass


def total_from_mass_pt(mass: float, pt: float, rapidity: float = 0.0,
                       phi: float = 0.0) -> np.ndarray:
    """Construct P from system mass, vector-sum pT, true rapidity and azimuth."""
    if not all(math.isfinite(x) for x in (mass, pt, rapidity, phi)) or mass <= 0 or pt < 0:
        raise ValueError('Require finite mass > 0, pT >= 0, rapidity, and phi.')
    mt = math.hypot(mass, pt)
    try:
        p = np.array([mt * math.cosh(rapidity), pt * math.cos(phi),
                      pt * math.sin(phi), mt * math.sinh(rapidity)], dtype=np.float64)
    except OverflowError as exc:
        raise ValueError('Rapidity is too large for float64 kinematics.') from exc
    return validate_total_p4(p)[0]


def log_phase_space_volume(n_particles: int, mass: float) -> float:
    """Log Phi_N for dPhi=(2pi)^4 delta^4 prod[d^3p/((2pi)^3 2E)].

    Labeled particles; no 1/N! symmetry factor. Units: GeV^(2*N-4).
    Relative event weights within each fixed-N sample can simply be unity.
    This volume does not supply a physical probability distribution over N.
    """
    n = _integer(n_particles, 'n_particles', 2)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError('mass must be finite and strictly positive.')
    return ((2*n-4)*math.log(mass) - math.log(2)
            - (2*n-3)*math.log(4*math.pi) - math.lgamma(n) - math.lgamma(n-1))


def _rest_batch(n_events: int, n_particles: int, mass: float,
                rng: np.random.Generator) -> np.ndarray:
    r = rng.random((n_events, n_particles, 4))
    cos_theta = 2*r[..., 0]-1
    phi = 2*math.pi*r[..., 1]
    sin_theta = np.sqrt(np.maximum(0, 1-cos_theta*cos_theta))
    # Two independent exponential draws give density q0*exp(-q0).
    tiny = np.finfo(np.float64).tiny
    energy = -np.log(np.maximum(r[..., 2], tiny))-np.log(np.maximum(r[..., 3], tiny))
    qvec = energy[..., None]*np.stack([sin_theta*np.cos(phi), sin_theta*np.sin(phi), cos_theta], axis=-1)
    qsum0 = energy.sum(axis=1)
    qsumv = qvec.sum(axis=1)
    qnorm = np.linalg.norm(qsumv, axis=1)
    qmass2 = (qsum0-qnorm)*(qsum0+qnorm)
    if np.any(qmass2 <= 0) or not np.isfinite(qmass2).all():
        raise FloatingPointError('Numerically degenerate auxiliary momenta; use another seed.')
    qmass = np.sqrt(qmass2)
    # Boost auxiliary momenta to their common rest frame.
    rest0 = (qsum0[:, None]*energy-np.sum(qvec*qsumv[:, None, :], axis=-1))/qmass[:, None]
    restv = qvec-qsumv[:, None, :]*((energy+rest0)/(qsum0+qmass)[:, None])[..., None]
    out = np.concatenate([rest0[..., None], restv], axis=-1)
    out *= (mass/qmass)[:, None, None]
    return out


def generate_massless(n_events: int, n_particles: int, total_p4, *,
                      seed: int | None = None, rng: np.random.Generator | None = None,
                      batch_size: int = 1024) -> np.ndarray:
    """Return float64 p4 of shape (events, particles, 4).

    Exactly one of seed/rng may be supplied. Same seed and particle count
    give the same random stream regardless of batch size. The returned array
    occupies 32*n_events*n_particles bytes; batch_size limits scratch space.
    """
    n_events = _integer(n_events, 'n_events', 1)
    n_particles = _integer(n_particles, 'n_particles', 2)
    batch_size = _integer(batch_size, 'batch_size', 1)
    target, mass = validate_total_p4(total_p4)
    if seed is not None and rng is not None:
        raise ValueError('Supply seed or rng, not both.')
    if seed is not None:
        seed = _integer(seed, 'seed', 0)
    if rng is None:
        rng = np.random.default_rng(seed)
    elif not isinstance(rng, np.random.Generator):
        raise ValueError('rng must be a numpy.random.Generator.')
    result = np.empty((n_events, n_particles, 4), dtype=np.float64)
    for start in range(0, n_events, batch_size):
        stop = min(start+batch_size, n_events)
        rest = _rest_batch(stop-start, n_particles, mass, rng)
        # Boost from the total-system rest frame to the requested lab P.
        energy = (target[0]*rest[..., 0]+np.sum(rest[..., 1:]*target[1:], axis=-1))/mass
        vec = rest[..., 1:]+target[1:]*(
            (rest[..., 0]+energy)/(target[0]+mass))[..., None]
        result[start:stop, :, 0] = energy
        result[start:stop, :, 1:] = vec
    return result


def validate_events(p4: np.ndarray, total_p4, tolerance: float = 2e-10) -> dict:
    """Check momentum conservation, positive energies and the massless shell."""
    p4 = np.asarray(p4, dtype=np.float64)
    target, mass = validate_total_p4(total_p4)
    if p4.ndim != 3 or p4.shape[-1] != 4 or p4.shape[0] < 1 or p4.shape[1] < 2:
        raise ValueError('p4 must have shape (events>=1, particles>=2, 4).')
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('tolerance must be positive and finite.')
    if not np.isfinite(p4).all() or np.any(p4[..., 0] <= 0):
        raise ValueError('Nonfinite momenta or nonpositive energies.')
    sums = p4.sum(axis=1)
    norm = np.linalg.norm(p4[..., 1:], axis=-1)
    conservation = float(np.max(np.abs(sums-target)))
    shell = float(np.max(np.abs(p4[..., 0]-norm)/p4[..., 0]))
    scale = max(float(target[0]), 1.0)
    if conservation > tolerance*scale or shell > tolerance:
        raise ValueError(f'Kinematics failed: max sum error={conservation:g} GeV; '
                         f'max relative |E-|p||/E={shell:g}.')
    return {'n_events':int(len(p4)), 'n_particles':int(p4.shape[1]),
            'target_p4_GeV':target.tolist(), 'system_mass_GeV':mass,
            'max_four_momentum_error_GeV':conservation,
            'max_relative_massless_shell_error':shell,
            'max_abs_particle_mass_squared_GeV2':float(np.max(np.abs(p4[..., 0]**2-norm**2))),
            'min_particle_energy_GeV':float(p4[..., 0].min()), 'status':'PASS'}


def write_dat(path: Path, p4: np.ndarray) -> None:
    """Write the repository's event format: placeholder PID=0, px,py,pz,E.

    Every RAMBO event is one particle list. No clustering or angular cut is
    implied. PID 0 is a neutral placeholder for the no-PID model, not a species.
    """
    with Path(path).open('w', encoding='utf-8', newline='\n') as f:
        for i, event in enumerate(p4):
            f.write(f'<event_begin>:{i}\n')
            for energy, px, py, pz in event:
                f.write(f'0\t{px:.17e}\t{py:.17e}\t{pz:.17e}\t{energy:.17e}\n')
            f.write(f'<event_end>:{i}\n')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-events', type=int, default=1000, help='Events for EACH specified multiplicity (default: 1000).')
    parser.add_argument('--n-particles', type=int, nargs='+', required=True, help='One or several fixed multiplicities, each >=2.')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cm-energy', type=float, help='System rest energy: total P=(ECM,0,0,0) GeV.')
    mode.add_argument('--mass', type=float, help='System invariant mass in GeV; combine with --pt etc.')
    mode.add_argument('--total-p4', type=float, nargs=4, metavar=('E','PX','PY','PZ'), help='Explicit total four-momentum in GeV.')
    parser.add_argument('--pt', type=float, help='Vector-sum transverse momentum in GeV; only with --mass (default 0).')
    parser.add_argument('--rapidity', type=float, help='System true rapidity; only with --mass (default 0).')
    parser.add_argument('--phi', type=float, help='System azimuth in radians; only with --mass (default 0).')
    parser.add_argument('--seed', type=int, default=20261115)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--output', type=Path, required=True, help='Directory for NPZ and JSON files, one pair per multiplicity.')
    parser.add_argument('--write-dat', action='store_true', help='Also export particle lists in the existing repository DAT format.')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly allow replacing files for requested multiplicities.')
    args = parser.parse_args(argv)
    try:
        _integer(args.n_events, 'n_events', 1); _integer(args.seed, 'seed', 0)
        _integer(args.batch_size, 'batch_size', 1)
        for n in args.n_particles:_integer(n, 'n_particles', 2)
        if len(set(args.n_particles)) != len(args.n_particles):
            raise ValueError('Duplicate multiplicities are not allowed.')
        if args.mass is None and any(v is not None for v in (args.pt,args.rapidity,args.phi)):
            raise ValueError('--pt/--rapidity/--phi must be used with --mass.')
        if args.cm_energy is not None:
            target, mass = validate_total_p4([args.cm_energy,0,0,0])
        elif args.mass is not None:
            target = total_from_mass_pt(args.mass, args.pt if args.pt is not None else 0,
                                       args.rapidity if args.rapidity is not None else 0,
                                       args.phi if args.phi is not None else 0)
            target, mass = validate_total_p4(target)
        else:
            target, mass = validate_total_p4(args.total_p4)
        exts = ['.npz','.json'] + (['.dat'] if args.write_dat else [])
        planned = [args.output/f'rambo_N{n:03d}{ext}' for n in args.n_particles for ext in exts]
        existing = [str(p) for p in planned if p.exists()]
        if existing and not args.overwrite:
            raise ValueError('Output already exists; use a new directory or --overwrite: '+', '.join(existing))
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for n in args.n_particles:
        # N-specific streams make a scan independent of the multiplicity order.
        rng = np.random.default_rng(np.random.SeedSequence([args.seed,n]))
        p4 = generate_massless(args.n_events, n, target, rng=rng, batch_size=args.batch_size)
        check = validate_events(p4,target)
        prefix = args.output/f'rambo_N{n:03d}'
        np.savez_compressed(prefix.with_suffix('.npz'), p4=p4, weights=np.ones(args.n_events),
                            total_p4=target, n_particles=np.int64(n))
        if args.write_dat:write_dat(prefix.with_suffix('.dat'),p4)
        report = {'created_utc':datetime.now(timezone.utc).isoformat(), 'algorithm':'massless RAMBO',
                  'p4_order':['E','px','py','pz'], 'units':'GeV', 'dtype':'float64',
                  'numpy_version':np.__version__, 'code_sha256':code_hash,
                  'seed':args.seed, 'seed_sequence_components':[args.seed,n], 'batch_size':args.batch_size,
                  'system_mass_input_GeV':args.mass,
                  'relative_weight':1.0, 'log_phase_space_volume':log_phase_space_volume(n,mass),
                  'phase_space_volume_energy_power':2*n-4,
                  'phase_space_convention':'(2pi)^4 delta^4(P-sum p) product d^3p/((2pi)^3 2E); labeled particles, no 1/N!',
                  'sample_scope':'Uniform phase space at fixed P and N. No QCD, PID, shower, hadronization, jet clustering, or cone cut.',
                  'validation':check}
        prefix.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps({'output':str(prefix),'validation':check},ensure_ascii=False),flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
