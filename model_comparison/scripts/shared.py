"""Common model-comparison definitions; reuse the paper preprocessing and physics."""
import json
import os
import sys
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
sys.path.insert(0, str(ROOT / "paper_figures"))
sys.path.insert(0, str(ROOT / "paper_figures/Appendix"))
sys.path.insert(0, str(ROOT / "paper_figures/Appendix/RAMBO"))
from analysis import digest, observables, summarize_distribution


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,default=convert,allow_nan=False,separators=(",",":"))+"\n",encoding="utf-8",newline="\n")


def models():
    return read(BASE/"models.json")["models"]


def setup_torch():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False


def image_observables(image):
    """Eight normalized-image observables, R_ref=.4; mass is dimensionless.

    Pixel-center massless vectors define the image axis. This is distinct from
    particle multiplicity and the dimensional raw-constituent jet mass.
    """
    centers=(np.arange(32)+.5)/32-.5
    eta,phi=np.meshgrid(centers,centers,indexing="xy")
    flat=np.asarray(image,dtype=float).reshape(-1);keep=flat>0
    z=flat[keep];z=z/z.sum();eta=eta.ravel()[keep];phi=phi.ravel()[keep]
    px,py,pz=float(z@np.cos(phi)),float(z@np.sin(phi)),float(z@np.sinh(eta))
    axis_eta=np.arcsinh(pz/np.hypot(px,py));axis_phi=np.arctan2(py,px)
    wrap=lambda x:(x+np.pi)%(2*np.pi)-np.pi
    theta=np.hypot(eta-axis_eta,wrap(phi-axis_phi))/.4
    i,j=np.triu_indices(len(z),1);de=eta[i]-eta[j];dp=wrap(phi[i]-phi[j]);w=z[i]*z[j]
    pair=np.hypot(de,dp)/.4
    mass2=4*np.sum(w*(np.sinh(de/2)**2+np.sin(dp/2)**2))
    return np.array([np.sqrt(z@z),-z@np.log(z),z@np.sqrt(theta),z@theta,z@(theta**2),
                     w@np.sqrt(pair),w@pair,np.log(mass2+1e-12)])


def select_widths(logvar, active):
    """Same bulk-correlation rule for every model; no physical axis matching."""
    active=np.asarray(active,dtype=int)
    if len(active)<2:return dict(pairs=[],eligible=[],selected=int(active[0]) if len(active) else None,fraction=[])
    corr=np.corrcoef(np.asarray(logvar,dtype=float)[:,active],rowvar=False)
    fraction=((corr>.8).sum(1)-1)/(len(active)-1)
    eligible=active[fraction>.5]
    # Up to five disjoint pairs spread through original latent-index order.
    count=min(5,len(eligible)//2)
    pairs=eligible[np.round(np.linspace(0,len(eligible)-1,2*count)).astype(int)].reshape(count,2) if count else []
    return dict(pairs=np.asarray(pairs,dtype=int).reshape(-1,2).tolist(),eligible=eligible.tolist(),
                selected=int(eligible[0]) if len(eligible) else None,fraction=fraction.tolist())


def width_distribution(n,u):
    """Same profiles as the appendix; extend u bins if another checkpoint needs it."""
    n,u=np.asarray(n),np.asarray(u)
    low=min(-1.5,np.floor(u.min()*10)/10-.1);high=max(6.,np.ceil(u.max()*10)/10+.1)
    edges=np.linspace(low,high,int(round((high-low)/.1))+1)
    counts=np.histogram2d(n,u,bins=(np.arange(.5,81.5),edges))[0].astype(int)
    if counts.sum()!=len(n):raise ValueError("Events outside multiplicity range 1..80")
    profiles=[]
    for value in np.unique(n):
        v=u[n==value]
        if len(v)>=20:profiles.append(dict(N=int(value),count=len(v),mean=float(v.mean()),sem=float(v.std(ddof=1)/np.sqrt(len(v))),q16=float(np.quantile(v,.16)),q84=float(np.quantile(v,.84))))
    return dict(n_events=len(n),counts=counts,yedges=edges,profiles=profiles)
