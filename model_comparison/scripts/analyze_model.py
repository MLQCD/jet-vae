"""Encode one registered model and rebuild its comparison data (no training)."""
import argparse
from pathlib import Path
import numpy as np
from shared import BASE,ROOT,read,write,models,setup_torch,digest,select_widths,width_distribution
from common import (load_cnn,load_transformer,encode_cnn,encode_transformer,decode_cnn,decode_transformer,
                    particles_to_image,cnn_input_image,load_jets,active_dimensions,sigma_scalar_coefficients)
from fig4_5_parameter_response import PARAMS,CHANGES,setting_paths,pooled_moments,responses
from analysis import encode_p4
from rambo import generate_massless,total_from_mass_pt,validate_events


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True);p.add_argument("--checkpoint",required=True)
    p.add_argument("--reference",type=Path,required=True);p.add_argument("--scan-dir",type=Path,required=True)
    p.add_argument("--work-dir",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    a=p.parse_args();registry={m["id"]:m for m in models()}
    if a.model not in registry:p.error("Unknown model ID")
    info=registry[a.model];kind=info["representation"]
    if digest(a.checkpoint)!=info["checkpoint_sha256"]:raise ValueError("Checkpoint does not match models.json")
    setup_torch();a.work_dir.mkdir(parents=True,exist_ok=True)
    identity=dict(checkpoint_sha256=info["checkpoint_sha256"],reference_sha256=digest(a.reference),format_version=1)
    identity_file=a.work_dir/"identity.json"
    if identity_file.exists() and read(identity_file)!=identity:raise ValueError("Work directory belongs to different inputs/model")
    write(identity_file,identity)
    model,checkpoint=load_cnn(a.checkpoint) if kind=="image" else load_transformer(a.checkpoint,a.device)
    model=model.to(a.device).eval().requires_grad_(False)
    with np.load(a.reference,allow_pickle=False) as z:ref={k:z[k].copy() for k in z.files}
    def encode(data):
        if kind=="image":return encode_cnn(model,data["images"],batch=256,device=a.device)
        return encode_transformer(model,data["particles"],data["mask"],batch=256,device=a.device)
    ref_cache=a.work_dir/"reference_posteriors.npz"
    if ref_cache.exists():
        with np.load(ref_cache) as z:mu,lv=z["mu"],z["logvar"]
    else:
        mu,lv=encode(ref);np.savez_compressed(ref_cache,mu=mu,logvar=lv)
    active=np.flatnonzero(active_dimensions(mu,lv));m64,l64=mu.astype(float),lv.astype(float)
    if len(active)<2:raise ValueError("Fewer than two active coordinates; inspect before publishing")
    cm=np.corrcoef(m64[:,active],rowvar=False);cl=np.corrcoef(l64[:,active],rowvar=False)
    selection=select_widths(lv,active)
    pairs=[]
    for j,k in selection["pairs"]:
        counts,xe,ye=np.histogram2d(l64[:,j],l64[:,k],bins=42)
        pairs.append(dict(indices=[j,k],counts=counts.astype(int),xedges=xe,yedges=ye,r=float(np.corrcoef(l64[:,j],l64[:,k])[0,1])))
    selected=selection["selected"]
    x=m64[:,active] if selected is None else np.column_stack([m64[:,active],l64[:,selected]])
    y=ref[kind+"_observables"]
    corr_obs=np.corrcoef(x.T,y.T)[:x.shape[1],x.shape[1]:]
    # Illustrations use CPU decode(mu), with fixed original file indices.
    model=model.to("cpu")
    if kind=="image":
        emu,_=encode_cnn(model,ref["example_images"],device="cpu")
        reco=decode_cnn(model,emu)[:,0].transpose(0,2,1)
    else:
        emu,_=encode_transformer(model,ref["example_particles"],ref["example_mask"],device="cpu")
        reco=particles_to_image(decode_transformer(model,emu))
    model=model.to(a.device)
    # Evaluate a common image-space SSE on the matched reference, independent of the training objective.
    sums=[]
    for start in range(0,len(mu),256):
        if kind=="image":pred=decode_cnn(model,mu[start:start+256])[:,0].transpose(0,2,1)
        else:pred=particles_to_image(decode_transformer(model,mu[start:start+256]))
        truth=ref["images"][start:start+256,0].transpose(0,2,1)
        sums.extend(np.square(pred.astype(float)-truth).sum((1,2)).tolist())
    result=dict(model=info,provenance=identity,reference_metadata=read_reference_metadata(ref),
        reference=dict(active_indices=active,activity=m64.var(0)/np.exp(l64).mean(0),corr_mu=cm,corr_logvar=cl,
                       selection=selection,pairs=pairs,observable_correlations=corr_obs,mean_pixel_sse=float(np.mean(sums))),
        reconstruction=dict(input=ref["example_images"][:,0].transpose(0,2,1),reconstruction=reco,indices=[45050,77395]))
    # Keep the fixed 50:50 class composition and the shared nominal MC covariance.
    scan={};source_hashes={}
    for param,_,_ in PARAMS:
        for change in CHANGES:
            for flavor,path in zip(("u","g"),setting_paths(a.scan_dir,param,change)):
                key=f"{flavor}/"+("nominal" if change==0 else f"{param}/{change:+.1f}")
                if key in scan:continue
                h=digest(path);source_hashes[key]=h
                cache=a.work_dir/(key.replace("/","_")+".npz")
                if cache.exists():
                    with np.load(cache) as z:
                        if str(z["source_sha256"])!=h:raise ValueError("Cached scan input changed")
                        mean,var=z["mean"],z["variance"]
                        if change==0:nom_mu,nom_lv=z["mu"],z["logvar"]
                else:
                    if kind=="particle":
                        with np.load(path,allow_pickle=False) as z:jets=dict(particles=z["particles80"],mask=z["mask80"].astype(bool))
                    else:
                        jets=load_jets(path)
                        jets["images"]=np.stack([cnn_input_image(e) for e in jets["events"]])
                    if len(jets["particles"])!=10000:raise ValueError("Expected 10000 scan jets per class")
                    sm,sl=encode(jets);mean=sl.mean(0,dtype=float);var=sl.astype(float).var(0,ddof=1)/len(sl)
                    saved=dict(mean=mean,variance=var,source_sha256=np.array(h))
                    if change==0:saved.update(mu=sm,logvar=sl);nom_mu,nom_lv=sm,sl
                    np.savez_compressed(cache,**saved)
                scan[key]=(mean,var)
                if change==0:scan[flavor+"_nominal_posterior"]=(nom_mu,nom_lv)
        print(a.model,param,"complete",flush=True)
    scan_active=active_dimensions(np.concatenate([scan[k+"_nominal_posterior"][0] for k in ("u","g")]),
                                   np.concatenate([scan[k+"_nominal_posterior"][1] for k in ("u","g")]))
    means=np.zeros((8,5,32));variances=means.copy()
    for ip,(param,_,_) in enumerate(PARAMS):
        for ic,change in enumerate(CHANGES):
            key="nominal" if change==0 else f"{param}/{change:+.1f}"
            left,right=scan['u/'+key],scan['g/'+key]
            means[ip,ic]=(left[0]+right[0])/2;variances[ip,ic]=(left[1]+right[1])/4
    delta,slopes=responses(means,scan_active,variances)
    included=np.ones(slopes.shape,dtype=bool)
    if kind=="image":included[0]=np.abs(delta[0]).max(0)>=.1
    summaries=[]
    for row,keep in zip(slopes,included):
        summaries.append(dict(mean=float(row[keep].mean()) if keep.any() else None,
                              sd=float(row[keep].std()) if keep.any() else None,count=int(keep.sum())))
    result["response"]=dict(active_indices=np.flatnonzero(scan_active),mean_logvar=means,variance_mean=variances,
        delta=delta,slopes=slopes,summary=summaries,summary_included=included,source_sha256=source_hashes)
    if kind=="particle":result["rambo"]=rambo_analysis(model,a,ref,l64,active)
    write(a.out,result)
    print("SAVED",a.model,"active",len(active),"pixel SSE",np.mean(sums),flush=True)


def read_reference_metadata(ref):
    import json
    return json.loads(str(ref["metadata"]))


def rambo_analysis(model,args,ref,lv,active):
    # Each checkpoint has its own QCD-fitted basis, held fixed on RAMBO.
    b,_=sigma_scalar_coefficients(lv[:,active]);center=lv[:,active].mean(0)
    def project(values):return (values[:,active].astype(float)-center)@b/(b@b)
    target=total_from_mass_pt(20,200);mus=[];widths=[];ns=[]
    for n in range(2,81):
        path=args.work_dir/f"rambo_N{n}.npz"
        if path.exists():
            with np.load(path) as z:values=z["logvar"]
        else:
            p4=generate_massless(1000,n,target,rng=np.random.default_rng(np.random.SeedSequence([20261115,n])))
            if validate_events(p4,target)["status"]!="PASS":raise ValueError("RAMBO kinematics failed")
            _,values=encode_p4(model,p4,args.device);np.savez_compressed(path,logvar=values)
        widths.append(values);ns.append(np.full(1000,n))
    widths=np.concatenate(widths);ns=np.concatenate(ns)
    u=project(lv);rambo_u=project(widths)
    distributions={name:width_distribution(ref["particle_observables"][ref["labels"]==k,0],u[ref["labels"]==k]) for k,name in enumerate(("u","g"))}
    distributions['rambo']=width_distribution(ns,rambo_u)
    targets=read(ROOT/'paper_figures/Appendix/RAMBO/data/multiplicity_targets.json')['groups']
    means=[];mean_n=[];counts_total=[]
    for group,row in enumerate(targets):
        all_lv=[];all_n=[];counts=np.asarray(row['counts_after_N1_cut'])
        path=args.work_dir/f"rambo_matched{group}.npz"
        if path.exists():
            with np.load(path) as z:all_lv=z['logvar'];all_n=z['n_particles']
        else:
            for n in np.flatnonzero(counts):
                p4=generate_massless(int(counts[n]),int(n),target,rng=np.random.default_rng(np.random.SeedSequence([20261116,3906,group,int(n)])))
                if validate_events(p4,target)['status']!='PASS':raise ValueError('RAMBO kinematics failed')
                _,v=encode_p4(model,p4,args.device);all_lv.append(v);all_n.append(np.full(int(counts[n]),n))
            all_lv=np.concatenate(all_lv);all_n=np.concatenate(all_n)
            np.savez_compressed(path,logvar=all_lv,n_particles=all_n)
        np.testing.assert_array_equal(np.bincount(all_n,minlength=81),counts)
        means.append(all_lv.mean(0,dtype=float));mean_n.append(float(all_n.mean()));counts_total.append(len(all_n))
    means=np.asarray(means)
    print(args.model,'RAMBO complete',flush=True)
    return dict(basis=dict(a=center,b=b,active_indices=active),distributions=distributions,
                response=dict(mean_N=mean_n,mean_logvar=means,delta=(means-means[2])[:,active],n_events=counts_total))


if __name__=="__main__":main()
