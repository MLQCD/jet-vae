"""Draw all model comparisons from small JSON tables; no weights or jets required."""
import argparse
from pathlib import Path
import numpy as np
from shared import BASE,models,read,write
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.interpolate import UnivariateSpline

PARAM_NAMES=["alphaS","aLund","bLund","sigma","pTmin","mesonUDvector","probStoUD","probQQtoQ"]
PARAM_LABELS=[r"$\alpha_s$",r"$a_{\rm Lund}$",r"$b_{\rm Lund}$",r"$\sigma_{\rm string}$",r"$p_{T,\min}$",r"$r_{V/P}$",r"$r_{s/(u,d)}$",r"$r_{qq/q}$"]
COLORS=["#2184a1","#eb7e6e",(.1875,.54687,.347656),"#b47e8b","#ffb838","#7b3294","#ff0000",(.36,.39,.42)]
OBS=[r"$p_TD$",r"$H_{p_T}$",r"$\lambda_{1/2}$",r"$\lambda_1$",r"$\lambda_2$",r"$e_2^{(1/2)}$",r"$e_2^{(1)}$"]
CORR=LinearSegmentedColormap.from_list("correlation",["#51A7C0","white","#FF8000"],N=4097)


def style():
    plt.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","font.size":11,
        "axes.labelsize":12,"axes.titlesize":12,"axes.linewidth":.7,"xtick.direction":"in",
        "ytick.direction":"in","pdf.fonttype":42,"savefig.facecolor":"white"})


def label(m):
    level="Image" if m['representation']=='image' else "Particle"
    seed="seed unrecorded" if m['seed'] is None else f"seed {m['seed']}"
    return f"{level} | beta = {m['beta']:.0e} | {seed} | epoch {m['epoch']}"+(" | paper reference" if m['paper_reference'] else "")


def title(fig,topic,m):
    fig.suptitle(topic+"\n"+label(m),fontsize=13,y=.98)


def reconstruction(d,limits):
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.3));fig.subplots_adjust(left=.10,right=.86,bottom=.09,top=.83,wspace=.23,hspace=.25)
    for row,key in enumerate(['input','reconstruction']):
        for col,flavor in enumerate(['u jet','g jet']):
            ax=axes[row,col];im=ax.imshow(np.array(d['reconstruction'][key])[col],origin='lower',extent=(-.5,.5,-.5,.5),cmap='Blues' if row==0 else 'Oranges',vmin=0,vmax=limits['reconstruction'])
            ax.set(xlabel=r'$\Delta\phi$' if row else '',ylabel=r'$\Delta\eta$',title=f'{flavor}: {key}')
        cax=fig.add_axes([.89,.535 if row==0 else .105,.018,.30]);cb=fig.colorbar(im,cax=cax);cb.set_label('Normalized pT per pixel',fontsize=10)
    title(fig,'Input and deterministic reconstruction',d['model']);return fig


def posterior(d,limits):
    r=d['reference'];active=np.array(r['active_indices']);fig,axes=plt.subplots(1,2,figsize=(8.5,4.5))
    fig.subplots_adjust(left=.08,right=.88,bottom=.16,top=.78,wspace=.26)
    for ax,key,t in zip(axes,['corr_mu','corr_logvar'],['Posterior means', 'Log variances']):
        im=ax.imshow(r[key],vmin=-1,vmax=1,cmap=CORR,origin='upper',interpolation='nearest')
        ticks=np.arange(0,len(active),max(1,len(active)//8));ax.set_xticks(ticks,active[ticks]);ax.set_yticks(ticks,active[ticks])
        ax.set(xlabel='Original latent index',ylabel='Original latent index',title=t)
    cb=fig.colorbar(im,cax=fig.add_axes([.91,.18,.018,.58]));cb.set_label('Pearson correlation')
    title(fig,f"Posterior correlations: {len(active)} active coordinates",d['model']);return fig


def joint(d,limits):
    pairs=d['reference']['pairs'];fig=plt.figure(figsize=(11,3.1));title(fig,'Joint log-variance distributions',d['model'])
    if not pairs:fig.text(.5,.45,'No coordinate pair satisfies the common bulk-correlation rule.',ha='center');return fig
    axes=fig.subplots(1,len(pairs),squeeze=False)[0];fig.subplots_adjust(left=.065,right=.90,bottom=.23,top=.68,wspace=.50)
    for ax,p in zip(axes,pairs):
        h=np.asarray(p['counts']);j,k=p['indices']
        im=ax.pcolormesh(p['xedges'],p['yedges'],np.ma.masked_equal(h.T,0),cmap='Greens',norm=LogNorm(1,limits['joint']),rasterized=True)
        ax.set(xlabel=rf'$\ln\sigma_{{{j}}}^2$',ylabel=rf'$\ln\sigma_{{{k}}}^2$',title=f"r = {p['r']:.3f}")
        ax.locator_params(axis='both',nbins=3);ax.tick_params(labelsize=9)
    cb=fig.colorbar(im,cax=fig.add_axes([.935,.23,.012,.45]));cb.set_label('Jets / bin',fontsize=10)
    return fig


def response(d,limits):
    r=d['response'];delta=np.array(r['delta']);fig,axes=plt.subplots(1,2,figsize=(8.6,4.4),gridspec_kw={'width_ratios':[1.25,1]})
    fig.subplots_adjust(left=.10,right=.98,bottom=.16,top=.70,wspace=.38)
    for ip,color in enumerate(COLORS):
        for j in range(delta.shape[-1]):axes[0].plot([-20,-10,0,10,20],delta[ip,:,j],color=color,lw=.6,alpha=.4)
        s=r['summary'][ip]
        if s['mean'] is not None:axes[1].errorbar(s['mean'],ip,xerr=s['sd'],fmt='o',ms=3,color=color,capsize=2)
    axes[0].axhline(0,color='.6',lw=.5);axes[0].plot(0,0,'ko',ms=3)
    axes[0].set(xlabel='Parameter relative change (%)',ylabel=r'$\Delta\langle\ln\sigma_j^2\rangle$',ylim=(-limits['response'],limits['response']),title=f"All {delta.shape[-1]} active coordinates")
    axes[1].set_yticks(range(8),PARAM_LABELS);axes[1].set(ylim=(7.7,-.7),xlim=limits['slopes'],xlabel='Mean slope (coordinate SD)',title='Average response')
    axes[1].axvline(0,color='.6',lw=.5)
    handles=[Line2D([],[],color=c,label=t) for c,t in zip(COLORS,PARAM_LABELS)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.82),ncol=8,frameon=False,fontsize=10,handlelength=1.4,columnspacing=1.0)
    title(fig,'Dijet parameter response',d['model']);return fig


def training(d,limits):
    m=d['model'];t=d["training_data"]
    fig,axes=plt.subplots(1,2,figsize=(8.5,4.1));fig.subplots_adjust(left=.10,right=.97,bottom=.16,top=.72,wspace=.30)
    approximate=t['kind']=='approximate_png_digitization'
    for ax,metric in zip(axes,['reco','kld']):
        if approximate:
            rows=t['curves'][metric];x=[r['epoch_approx'] for r in rows]
            for key,name,color in [('train_approx','Training','#2184a1'),('validation_approx','Validation','#e07822')]:ax.plot(x,[np.nan if r[key] is None else r[key] for r in rows],label=name,color=color,lw=1)
        else:
            series=t['series'];key='emd' if metric=='reco' else 'kld'
            for phase,name,color in [('train','Training','#2184a1'),('val','Validation','#e07822')]:ax.plot(series['epoch'],series[f'{phase}_{key}'],label=name,color=color,lw=1)
        ax.axvline(m['epoch'],color='.4',ls=':',lw=.8)
        ax.set(xlabel='Epoch',ylabel=('Pixel reconstruction loss' if m['representation']=='image' else 'Sinkhorn EMD') if metric=='reco' else 'KLD')
        ax.legend(frameon=False,fontsize=9)
    title(fig,'Training history'+(' (approximate digitization from PNG)' if approximate else ' (recorded epoch values)'),m)
    fig.text(.5,.77,'Dotted line: selected checkpoint. Training objectives differ between representations.',ha='center',fontsize=9)
    return fig


def observable(d,limits):
    r=d['reference'];matrix=np.asarray(r['observable_correlations']);active=r['active_indices'];selected=r['selection']['selected'];m=d['model']
    rows=[rf'$\mu_{{{j}}}$' for j in active]+([] if selected is None else [rf'$\ln\sigma_{{{selected}}}^2$'])
    cols=([r'Multiplicity $N$']+OBS+[r'$\ln(m_J^2/\mathrm{GeV}^2)$']) if m['representation']=='particle' else OBS+[r'$\ln\widehat m^2_{\rm img}$']
    fig,ax=plt.subplots(figsize=(9.5,max(5.4,.24*len(rows)+2.5)));fig.subplots_adjust(left=.10,right=.86,bottom=.16,top=.87)
    im=ax.imshow(matrix,cmap=CORR,vmin=-1,vmax=1,aspect='auto',interpolation='nearest')
    ax.set_xticks(range(len(cols)),cols,rotation=42,ha='right',rotation_mode='anchor');ax.set_yticks(range(len(rows)),rows);ax.tick_params(length=0,labelsize=10)
    ax.set_ylabel('Latent feature')
    if selected is not None:ax.axhline(len(rows)-1.5,color='.25',lw=1)
    for i in range(len(rows)):
        for j in range(len(cols)):ax.text(j,i,f'{matrix[i,j]:.2f}' if abs(matrix[i,j])>=.005 else '0.00',ha='center',va='center',fontsize=8.5)
    cb=fig.colorbar(im,cax=fig.add_axes([.90,.25,.018,.53]));cb.set_label('Signed Pearson correlation')
    title(fig,('Particle' if m['representation']=='particle' else 'Image')+' observable correlations',m)
    return fig


def rambo_distribution(d,limits):
    samples=d['rambo']['distributions'];fig,axes=plt.subplots(2,2,figsize=(8.7,6.5));fig.subplots_adjust(left=.08,right=.94,bottom=.09,top=.84,wspace=.38,hspace=.36)
    colors={'u':np.array([81,167,192])/255*.9,'g':np.array([1,.5,0])*.9,'rambo':np.array([.1875,.54687,.347656])}
    for ax in axes.flat:ax.set(xlim=(.5,80.5),ylim=limits['u_range'],xlabel='Particle multiplicity N',ylabel='u(x)')
    for ax,key in zip(axes.flat,['u','g','rambo']):
        sample=samples[key];color=colors[key];h=np.asarray(sample['counts'])
        cmap=LinearSegmentedColormap.from_list(key,[1-.18*(1-color),color],N=4097,gamma=.7);cmap.set_bad('white')
        im=ax.pcolormesh(np.arange(.5,81.5),sample['yedges'],np.ma.masked_equal(h.T/sample['n_events'],0),cmap=cmap,norm=LogNorm(1e-5,1e-2),rasterized=True)
        fig.colorbar(im,ax=ax,pad=.015,fraction=.035,ticks=[1e-5,1e-4,1e-3,1e-2]);ax.set_title({'u':'u jets','g':'g jets','rambo':'RAMBO'}[key])
        rows=sample['profiles'];n=np.array([r['N'] for r in rows]);means=np.array([r['mean'] for r in rows]);sem=np.array([r['sem'] for r in rows])
        grid=np.linspace(n[0],n[-1],max(2,int(n[-1]-n[0])*30+1))
        line=UnivariateSpline(n,means,w=1/sem,s=.6*len(n),k=3,ext=2)(grid) if len(n)>=4 and np.all(sem>0) else np.interp(grid,n,means)
        ax.plot(grid,line,color='black',lw=.9)
        axes[1,1].fill_between(n,[r['q16'] for r in rows],[r['q84'] for r in rows],color=color,alpha=.16,lw=0)
        axes[1,1].plot(grid,line,color=color,lw=1.2,label=key)
    axes[1,1].set_title('Conditional means and 68% event bands');axes[1,1].legend(frameon=False)
    title(fig,'RAMBO and QCD width versus multiplicity',d['model']);return fig


def rambo_response(d,limits):
    r=d['rambo']['response'];fig,ax=plt.subplots(figsize=(6.4,4));fig.subplots_adjust(left=.16,right=.96,bottom=.18,top=.76)
    for values in np.asarray(r['delta']).T:ax.plot(r['mean_N'],values,color='#2184a1',alpha=.40,lw=.7,marker='o',ms=2)
    ax.axhline(0,color='.6',lw=.6);ax.plot(r['mean_N'][2],0,'ko',ms=3)
    ax.set_xticks(r['mean_N'],[f'{v:.2f}' for v in r['mean_N']]);ax.set(xlabel=r'Mean RAMBO multiplicity $\langle N\rangle$',ylabel=r'$\Delta\langle\ln\sigma_j^2\rangle$',ylim=(-limits['rambo_response'],limits['rambo_response']))
    title(fig,'Multiplicity-matched RAMBO response',d['model']);return fig


TOPICS=[('reconstruction',reconstruction),('posterior_correlations',posterior),('joint_log_variances',joint),
        ('parameter_response',response),('training',training),('observable_correlations',observable),
        ('rambo_multiplicity',rambo_distribution),('rambo_response',rambo_response)]


def overview(all_data):
    fig=plt.figure(figsize=(11,8.5));fig.text(.5,.95,'Jet VAE: model comparison across beta settings',ha='center',fontsize=19)
    fig.text(.5,.90,'Matched 5,000 u + 5,000 g jets | 32 latent coordinates | frozen checkpoints',ha='center',fontsize=12)
    rows=[]
    for d in all_data:
        m=d['model'];r=d['reference'];alpha=d['response']['summary'][0]
        rows.append([m['representation'],f"{m['beta']:.0e}",str(m['seed']) if m['seed'] is not None else '?',m['epoch'],len(r['active_indices']),f"{r['mean_pixel_sse']:.6f}",f"{alpha['mean']:.3f} +/- {alpha['sd']:.3f}" if alpha['mean'] is not None else 'No selected coordinate','paper' if m['paper_reference'] else 'comparison'])
    ax=fig.add_axes([.035,.30,.93,.53]);ax.axis('off')
    table=ax.table(cellText=rows,colLabels=['Representation','Beta','Seed','Epoch','Active','Pixel SSE','alpha_s slope +/- SD','Role'],loc='center',cellLoc='center',colWidths=[.14,.09,.06,.07,.07,.12,.24,.12])
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.75)
    fig.text(.06,.24,'Pixel SSE: unweighted summed squared pixel errors, averaged over the 10,000 reference jets.',fontsize=11)
    fig.text(.06,.205,'Beta values multiply different training objectives in the two representations; they are not directly equivalent.',fontsize=11)
    fig.text(.06,.17,'These are individual checkpoints, not averages over random seeds. The two beta=1e-5 particle runs remain distinct.',fontsize=11)
    fig.text(.06,.135,'Image training histories are approximate PNG digitizations. Particle histories use recorded epoch values.',fontsize=11)
    fig.text(.06,.10,'Topics and checkpoint labels replace manuscript figure numbers. Full settings and numerical inputs accompany this report.',fontsize=11)
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,default=BASE/'data');p.add_argument('--out-dir',type=Path,default=BASE/'figures');p.add_argument('--report',type=Path,default=BASE/'beta_comparison.pdf');a=p.parse_args()
    style();all_data=[read(a.data_dir/f"{m['id']}.json") for m in models()]
    for d in all_data:d["training_data"]=read(a.data_dir/f"{d['model']['id']}_training.json")
    a.out_dir.mkdir(parents=True,exist_ok=True);a.report.parent.mkdir(parents=True,exist_ok=True)
    limits=dict(reconstruction=max(.3,max(np.max(d['reconstruction']['reconstruction']) for d in all_data)),joint=max(np.max(p['counts']) for d in all_data for p in d['reference']['pairs']),response=max(np.max(np.abs(d['response']['delta'])) for d in all_data)*1.12)
    endpoints=[(s['mean']-s['sd'],s['mean']+s['sd']) for d in all_data for s in d['response']['summary'] if s['mean'] is not None]
    limits['slopes']=[min(v[0] for v in endpoints)-.1,max(v[1] for v in endpoints)+.1]
    particles=[d for d in all_data if 'rambo' in d]
    limits['u_range']=[min(s['yedges'][0] for d in particles for s in d['rambo']['distributions'].values()),max(s['yedges'][-1] for d in particles for s in d['rambo']['distributions'].values())]
    limits['rambo_response']=max(.2983057705650325,max(np.max(np.abs(d['rambo']['response']['delta'])) for d in particles)*1.12)
    page_index=[]
    with PdfPages(a.report,metadata={'Title':'Jet VAE model comparison across beta settings','Author':'MLQCD','Subject':'Dijet and RAMBO checkpoint benchmarks'}) as pdf:
        fig=overview(all_data);pdf.savefig(fig);fig.savefig(a.out_dir/'overview.png',dpi=160);plt.close(fig)
        page_index.append(dict(page=1,topic='overview',model=None))
        for topic,plot in TOPICS:
            folder=a.out_dir/topic;folder.mkdir(exist_ok=True)
            for d in all_data:
                if topic.startswith('rambo') and 'rambo' not in d:continue
                fig=plot(d,limits);name=d['model']['id'];pdf.savefig(fig)
                fig.savefig(folder/f'{name}.pdf');fig.savefig(folder/f'{name}.png',dpi=140);plt.close(fig)
                page_index.append(dict(page=len(page_index)+1,topic=topic,model=name,figure=f'figures/{topic}/{name}.pdf'))
            print('Plotted',topic,flush=True)
    # Topic and checkpoint bookmarks make the multipage report navigable.
    from pypdf import PdfReader,PdfWriter
    writer=PdfWriter();writer.clone_document_from_reader(PdfReader(a.report))
    parent=None;previous=None
    for row in page_index:
        if row['topic']!=previous:
            parent=writer.add_outline_item(row['topic'].replace('_',' ').title(),row['page']-1)
            previous=row['topic']
        if row['model']:writer.add_outline_item(row['model'],row['page']-1,parent=parent)
    staged=a.report.with_suffix('.bookmarked.pdf')
    with staged.open('wb') as stream:writer.write(stream)
    staged.replace(a.report)
    write(a.out_dir/'page_index.json',page_index);write(a.out_dir/'shared_plot_limits.json',limits)
    rows=['# Comparison figures','', '[Full comparison report](../beta_comparison.pdf)','']
    for topic,_ in TOPICS:
        rows.extend([f'## {topic.replace("_"," ").title()}',''])
        rows += [f"- [{r['model']}]({topic}/{r['model']}.pdf) (report page {r['page']})" for r in page_index if r['topic']==topic]
        rows.append('')
    (a.out_dir/'README.md').write_text('\n'.join(rows),encoding='utf-8')
    print('Saved',len(page_index),'report pages to',a.report)


if __name__=='__main__':main()
