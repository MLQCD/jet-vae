"""
app.py -Streamlit interactive latent-space explorer for the CNN Beta-VAE.

Run from the vae_cnn directory:
    streamlit run visualize/app.py -- \\
        --checkpoint 20260508_021610/best_model_betavae_cone05_20260508_021610.pth

Or from anywhere by giving the full path:
    streamlit run /path/to/vae_cnn/visualize/app.py -- \\
        --checkpoint /path/to/best_model_betavae_cone05_20260508_021610.pth

If running on a remote machine, forward the port:
    ssh -L 8501:localhost:8501 <user>@<host>
Then open: http://localhost:8501
"""

import argparse
import io
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

# -- resolve import path ------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core import (
    load_checkpoint,
    decode_z,
    sweep_dimension,
    compute_image_observables,
    compute_impact_heatmap,
    parse_training_log,
    load_dataset_split,
    encode_image,
    reconstruct_image,
    batch_encode,
    compute_active_units,
    make_channel_grid,
    make_total_image,
    make_sweep_figure,
    make_impact_figure,
    make_compare_figure,
    make_channel_compare_figure,
    parse_dat_file,
    constituents_to_kinematics,
    constituents_to_image,
    jet_axis,
    make_scatter_figure,
    make_centered_scatter_figure,
    make_scatter_channel_grid,
    normalise_to_unit_sum,
    CHANNEL_NAMES,
    CHANNEL_COLORS,
    CHANNEL_PIDS,
    LABEL_NAMES,
    _PID_TO_CH,
    uses_no_pid_channels,
)

# -- CLI arg: default checkpoint path -----------------------------------------
def _cli_defaults():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--checkpoint', default='')
    p.add_argument('--cache-dir',  default='')
    p.add_argument('--g-jet', default='')
    p.add_argument('--u-jet', default='')
    args, _ = p.parse_known_args()
    return args.checkpoint, args.cache_dir, args.g_jet, args.u_jet

DEFAULT_CKPT, DEFAULT_CACHE_DIR, DEFAULT_G_JET, DEFAULT_U_JET = _cli_defaults()
RAW_JETS_FROM_CLI = bool(DEFAULT_G_JET or DEFAULT_U_JET)


def _default_raw_path(filename):
    path = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'data', filename))
    return path if os.path.exists(path) else ''


if not RAW_JETS_FROM_CLI and not DEFAULT_G_JET:
    DEFAULT_G_JET = _default_raw_path('g_jet_real_sample.dat')
if not RAW_JETS_FROM_CLI and not DEFAULT_U_JET:
    DEFAULT_U_JET = _default_raw_path('u_jet_real_sample.dat')

# -- page config ---------------------------------------------------------------
st.set_page_config(
    page_title='CNN-VAE Latent Explorer',
    layout='wide',
    initial_sidebar_state='expanded',
)

# --------------------------------------
# SIDEBAR
# --------------------------------------
with st.sidebar:
    st.title('CNN-VAE Explorer')
    st.caption('Jet image latent space')
    st.markdown('---')

    # -- Load model ----------------------------------------------------------
    st.subheader('1. Load checkpoint')
    ckpt_path = st.text_input(
        'Path to .pth file', value=DEFAULT_CKPT,
        placeholder='vae_cnn/20260508_021610/best_model_betavae_cone05_20260508_021610.pth',
    )
    load_btn = st.button('Load', type='primary', use_container_width=True)

    if load_btn and ckpt_path:
        if not os.path.exists(ckpt_path):
            st.error(f'File not found:\n{ckpt_path}')
        else:
            with st.spinner('Loading checkpoint...'):
                model, meta = load_checkpoint(ckpt_path)
            st.session_state['model'] = model
            st.session_state['meta']  = meta
            st.success(
                f'Loaded  latent_dim={meta["latent_dim"]}  '
                f'img={meta["img_size"]}x{meta["img_size"]}  '
                f'channels={meta["n_channels"]}'
            )

    if 'model' not in st.session_state:
        st.info('Enter a checkpoint path and click **Load**.')
        st.stop()

    model      = st.session_state['model']
    meta       = st.session_state['meta']
    latent_dim = meta['latent_dim']
    channel_names = meta.get('channel_names', CHANNEL_NAMES[:meta['n_channels']])
    channel_colors = meta.get('channel_colors', CHANNEL_COLORS[:meta['n_channels']])

    # -- Checkpoint summary ---------------------------------------------------
    with st.expander('Checkpoint info', expanded=False):
        st.markdown(
            f'**epoch** {meta["epoch"]}  \n'
            f'**val_loss** {meta["val_loss"]:.4f}  \n'
            f'**latent_dim** {meta["latent_dim"]}  \n'
            f'**img_size** {meta["img_size"]}  \n'
            f'**n_channels** {meta["n_channels"]}  \n'
            f'**channel_mode** {meta.get("channel_mode", "pid_channels")}  \n'
        )
        pid_map = meta.get('pid_channel_map', {})
        if pid_map and not uses_no_pid_channels(meta):
            rows = '\n'.join(f'| {pid} | {ch} | {CHANNEL_NAMES[ch]} |'
                             for pid, ch in sorted(pid_map.items(), key=lambda x: x[1]))
            st.markdown(f'**PID -> channel map**\n\n| PID | ch | name |\n|---|---|---|\n{rows}')

    st.markdown('---')

    # -- Baseline z ----------------------------------------------------------
    st.subheader('2. Baseline z')
    st.caption(
        'Reference point **z = 0** in latent space. '
        'All sweep and slider controls perturb individual dimensions away from this baseline.'
    )
    baseline_mode = st.radio(
        'Start from',
        ['zeros  (prior mean)', 'random sample from prior'],
    )

    if 'zeros' in baseline_mode:
        z_base = np.zeros(latent_dim, dtype=np.float32)
        st.caption('**z = 0** is the center of the N(0, I) prior and represents an average learned jet.')
    else:
        seed   = st.number_input('Random seed', value=42, step=1)
        z_base = np.random.default_rng(int(seed)).standard_normal(latent_dim).astype(np.float32)
        st.caption('**z ~ N(0, I)** is a random sample from the learned prior.')

    # -- Channel legend ------------------------------------------------------
    st.markdown('---')
    st.caption('**Channel legend**')
    for ch, (name, color) in enumerate(zip(channel_names, channel_colors)):
        st.markdown(
            f'<span style="background:{color};padding:2px 8px;border-radius:3px;'
            f'font-size:12px">&nbsp;</span> '
            f'<span style="font-family:monospace;font-size:12px">ch {ch}</span> {name}',
            unsafe_allow_html=True,
        )

    st.markdown('---')
    with st.expander('Physics glossary', expanded=False):
        st.markdown(
            '**eta**: pseudorapidity; proxy for polar angle from the beam axis.\n\n'
            '**phi**: azimuthal angle around the beam axis in radians.\n\n'
            '**pT**: transverse momentum; each pixel is pT density in one eta-phi cell.\n\n'
            '**Width**: pT-weighted standard deviation in eta or phi; larger values mean a more diffuse jet.\n\n'
            '**Multiplicity**: number of pixels with pT > 0.0001, used as a constituent-count proxy.\n\n'
            '**beta-VAE**: VAE with a scaled KL term for latent regularization.\n\n'
            '**Latent z**: compressed latent representation; the prior is z ~ N(0, I).'
        )

# --------------------------------------
# TABS
# --------------------------------------
tab_base, tab_sweep, tab_manual, tab_impact, tab_data, tab_raw, tab_au, tab_curves = st.tabs([
    'Baseline Jet',
    'Dimension Sweep',
    'Manual Control',
    'Dimension Impact',
    'Real Data',
    'Raw Events',
    'Active Units',
    'Training Curves',
])

# --------------------------------------
# TAB 0 -Baseline jet image
# --------------------------------------
with tab_base:
    st.subheader('Decoded jet at baseline z')

    img_base = decode_z(model, z_base, meta)
    obs_base = compute_image_observables(img_base)

    col_total, col_metrics = st.columns([2, 1])
    with col_total:
        vmax_global = float(img_base.max()) if img_base.max() > 0 else 1.0
        cmap_choice = st.selectbox('Color map', ['Greys', 'inferno', 'hot', 'viridis', 'plasma'], index=0,
                                   key='base_cmap')
        fig_total = make_total_image(img_base, title='pT-summed jet (all channels)', cmap=cmap_choice)
        st.pyplot(fig_total, use_container_width=True)
        plt.close(fig_total)

    with col_metrics:
        st.markdown('**Jet observables**')
        st.metric('std eta', f'{obs_base["std_eta"]:.3f}',
                  help='pT-weighted standard deviation in eta.')
        st.metric('std phi', f'{obs_base["std_phi"]:.3f}',
                  help='pT-weighted standard deviation in phi.')
        st.metric('std R', f'{obs_base["std_R"]:.3f}',
                  help='pT-weighted radial width around the jet centroid.')
        st.metric('pT entropy', f'{obs_base["pT_entropy"]:.3f}',
                  help='-sum(pT_i * ln(pT_i)), using pT_i normalized over pixels.')
        st.markdown('---')
        st.markdown('**pT per channel**')
        ch_vals = obs_base['total_pt_per_channel']
        for c in range(meta['n_channels']):
            pct = 100 * ch_vals[c] / (obs_base['total_pt'] + 1e-8)
            st.markdown(
                f'<span style="color:{channel_colors[c]}">{channel_names[c]}</span>'
                f': {ch_vals[c]:.3f}  ({pct:.1f}%)',
                unsafe_allow_html=True,
            )

    st.markdown('---')
    st.subheader('All channels')
    vmax_ch = st.slider('Color scale max (all channels)', 0.01, float(max(img_base.max(), 0.1)),
                        float(img_base.max() or 0.1), step=0.001, key='base_vmax',
                        format='%.3f')
    fig_grid = make_channel_grid(img_base, vmax=vmax_ch, cmap=cmap_choice,
                                 channel_names=channel_names, channel_colors=channel_colors)
    st.pyplot(fig_grid, use_container_width=True)
    plt.close(fig_grid)

    buf = io.BytesIO()
    fig_grid2 = make_channel_grid(img_base, vmax=vmax_ch, cmap=cmap_choice,
                                  channel_names=channel_names, channel_colors=channel_colors)
    fig_grid2.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig_grid2)
    st.download_button('Download channel grid', buf.getvalue(),
                       file_name='baseline_channels.png', mime='image/png')

# --------------------------------------
# TAB 1 -Dimension Sweep
# --------------------------------------
with tab_sweep:
    st.subheader('Fix all dimensions, sweep one')
    st.caption(
        'Each column shows the decoded jet when dimension **d** takes a different value '
        'while all other dimensions stay at the baseline z.'
    )

    with st.expander('Parameter definitions', expanded=False):
        st.markdown(
            '**Dimension to sweep**: index *d* of z[d] to vary. '
            'All other z dimensions stay at the baseline.\n\n'
            '**Value range**: interval [z_min, z_max]. '
            'The prior is N(0,1), so +/-3 covers ~99.7% of likely values.\n\n'
            '**Steps**: number of evenly spaced samples across the range.'
        )

    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        dim     = st.slider('Dimension to sweep', 0, latent_dim - 1, 0, key='sw_dim')
    with c2:
        v_range = st.slider('Value range', -5.0, 5.0, (-3.0, 3.0), step=0.5, key='sw_range')
    with c3:
        n_steps = st.slider('Steps', 3, 9, 7, key='sw_steps')

    values  = np.linspace(v_range[0], v_range[1], n_steps)
    results = sweep_dimension(model, z_base, dim, values, meta)

    fig_sw = make_sweep_figure(results, values, dim)
    st.pyplot(fig_sw, use_container_width=True)
    plt.close(fig_sw)

    buf = io.BytesIO()
    fig_sw2 = make_sweep_figure(results, values, dim)
    fig_sw2.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig_sw2)
    st.download_button('Download sweep figure', buf.getvalue(),
                       file_name=f'sweep_dim{dim}.png', mime='image/png')

    # -- per-step channel grids ----------------------------------------------
    with st.expander('Show all-channel grids for each sweep step', expanded=False):
        vmax_sw = max(r[0].max() for r in results) or 1.0
        for i, (v, (img, obs)) in enumerate(zip(values, results)):
            st.markdown(f'**z[{dim}] = {v:+.2f}** -std_R={obs["std_R"]:.3f}  '
                        f'pT_entropy={obs["pT_entropy"]:.3f}')
            fig_ch = make_channel_grid(img, vmax=vmax_sw,
                                       channel_names=channel_names, channel_colors=channel_colors)
            st.pyplot(fig_ch, use_container_width=True)
            plt.close(fig_ch)

# --------------------------------------
# TAB 2 -Manual Control
# --------------------------------------
with tab_manual:
    st.subheader('Drag any latent dimension and watch the jet update')

    # Sync sliders to z_base when baseline changes
    if not np.array_equal(st.session_state.get('_z_base_last'), z_base):
        st.session_state['_z_base_last'] = z_base.copy()
        for d in range(latent_dim):
            st.session_state[f'mz_{d}'] = float(z_base[d])

    z_manual = z_base.copy()

    with st.expander('Latent sliders', expanded=True):
        n_cols  = 4
        dim_ids = list(range(latent_dim))
        for row_dims in [dim_ids[i:i+n_cols] for i in range(0, latent_dim, n_cols)]:
            cols = st.columns(n_cols)
            for col, d in zip(cols, row_dims):
                with col:
                    z_manual[d] = st.slider(
                        f'z[{d}]', -4.0, 4.0,
                        value=float(z_base[d]),
                        step=0.05, key=f'mz_{d}',
                    )

    img_m = decode_z(model, z_manual, meta)
    obs_m = compute_image_observables(img_m)

    col_img, col_obs = st.columns([3, 1])
    with col_img:
        cmap_m  = st.selectbox('Color map', ['Greys', 'inferno', 'hot', 'viridis', 'plasma'],
                               index=0, key='man_cmap')
        view_m  = st.radio('View', ['Total pT image', 'All channels'], horizontal=True,
                           key='man_view')
        if view_m == 'Total pT image':
            fig_m = make_total_image(img_m, title='Generated jet (total pT)', cmap=cmap_m)
            st.pyplot(fig_m, use_container_width=True)
            plt.close(fig_m)
        else:
            fig_m = make_channel_grid(img_m, cmap=cmap_m,
                                      channel_names=channel_names, channel_colors=channel_colors)
            st.pyplot(fig_m, use_container_width=True)
            plt.close(fig_m)

    with col_obs:
        st.markdown('**Jet observables**')
        st.metric('std eta',        f'{obs_m["std_eta"]:.3f}')
        st.metric('std phi',        f'{obs_m["std_phi"]:.3f}')
        st.metric('std R',        f'{obs_m["std_R"]:.3f}')
        st.metric('pT entropy',   f'{obs_m["pT_entropy"]:.3f}')
        st.markdown('---')
        st.caption('**Current z vector**')
        st.dataframe(
            {'dim': list(range(latent_dim)),
             'value': [round(float(v), 3) for v in z_manual]},
            height=300, use_container_width=True,
        )

    # reset button
    if st.button('Reset to baseline z', key='man_reset'):
        for d in range(latent_dim):
            st.session_state[f'mz_{d}'] = float(z_base[d])
        st.rerun()

# --------------------------------------
# TAB 3 -Dimension Impact Heatmap
# --------------------------------------
with tab_impact:
    st.subheader('Which latent dimensions affect which observables most?')
    st.caption(
        'For each dimension d, decode z with z[d] = +sigma and z[d] = -sigma. '
        'The heatmap shows |Delta observable|; brighter cells indicate stronger effects.'
    )

    with st.expander('Observable definitions', expanded=False):
        st.markdown(
            '| Observable | Definition |\n|---|---|\n'
            '| **std_eta** | pT-weighted standard deviation in eta |\n'
            '| **std_phi** | pT-weighted standard deviation in phi |\n'
            '| **std_R** | pT-weighted radial width around the jet centroid |\n'
            '| **pT_entropy** | -sum(pT_i * ln(pT_i)), using pT_i normalized over pixels |\n\n'
            '**Probe sigma**: each dimension is pushed +/-sigma from baseline; larger sigma reveals non-linear effects.'
        )

    probe_sigma = st.slider('Probe distance (sigma)', 0.5, 4.0, 2.0, step=0.5, key='imp_sigma')

    if st.button('Compute heatmap', type='primary'):
        with st.spinner(f'Sweeping all {latent_dim} dimensions...'):
            impact, obs_keys = compute_impact_heatmap(model, z_base, meta, probe_sigma)
            st.session_state['impact']   = impact
            st.session_state['obs_keys'] = obs_keys

    if 'impact' in st.session_state:
        impact   = st.session_state['impact']
        obs_keys = st.session_state['obs_keys']

        col_heat, col_top = st.columns([2, 1])
        with col_heat:
            fig_imp = make_impact_figure(impact, obs_keys)
            st.pyplot(fig_imp, use_container_width=True)
            plt.close(fig_imp)

        with col_top:
            st.markdown('**Top 5 dims per observable**')
            for j, key in enumerate(obs_keys):
                top5 = np.argsort(impact[:, j])[::-1][:5]
                st.markdown(f'**{key}**: ' + ', '.join(f'z[{d}]' for d in top5))

        import pandas as pd
        df = pd.DataFrame(impact, columns=obs_keys)
        df.index.name = 'latent_dim'
        with st.expander('Raw impact values'):
            st.dataframe(df.style.background_gradient(cmap='YlOrRd', axis=0), height=400)

        buf = io.BytesIO()
        fig_imp2 = make_impact_figure(impact, obs_keys)
        fig_imp2.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig_imp2)
        st.download_button('Download heatmap', buf.getvalue(),
                           file_name='impact_heatmap.png', mime='image/png')

# --------------------------------------
# --------------------------------------
# TAB 4 -Real Data
# --------------------------------------
with tab_data:
    st.subheader('Real jet images from the dataset')

    # -- Cache dir input ------------------------------------------------------
    cache_dir = st.text_input(
        'Cache directory', value=DEFAULT_CACHE_DIR,
        placeholder='/path/to/cache_jet_images_32_real',
        key='cache_dir_input',
    )
    data_ready = bool(cache_dir and os.path.isdir(cache_dir))
    if data_ready and uses_no_pid_channels(meta):
        st.info('Real Data cache view is skipped for no-PID checkpoints. Use Raw Events for raw .dat input.')
        data_ready = False
    if not data_ready:
        st.info('Enter the cache directory path above to load data.')
    else:

        # -- Split selector -------------------------------------------------------
        split_choice = st.radio('Split', ['val', 'test', 'train'], horizontal=True,
                                help='train may be partially written (~52% complete).')

        load_data_btn = st.button('Reload split', type='primary', key='load_data_btn')
        cache_key = f'dataset_{split_choice}'
        cache_dir_key = f'{cache_key}_cache_dir'
        cache_dir_abs = os.path.abspath(cache_dir)

        should_load_split = (
            load_data_btn
            or cache_key not in st.session_state
            or st.session_state.get(cache_dir_key) != cache_dir_abs
        )

        if should_load_split:
            with st.spinner(f'Loading {split_choice} split...'):
                images, labels, pt_sums, n_total = load_dataset_split(cache_dir, split_choice)
            st.session_state[cache_key] = (images, labels, pt_sums, n_total)
            st.session_state[cache_dir_key] = cache_dir_abs

        if cache_key not in st.session_state:
            st.info(f'Could not load the {split_choice} images.')
            st.stop()

        images, labels, pt_sums, n_total = st.session_state[cache_key]
        N = len(images)
        n_quark = int((labels == 0).sum())
        n_gluon = int((labels == 1).sum())

        if N < n_total:
            st.warning(f'Truncated file: loaded {N:,} of {n_total:,} declared images.')
        else:
            st.success(f'Loaded {N:,} images  ({n_quark:,} quark - {n_gluon:,} gluon)')

        # ------------------------------------
        # Sub-tabs within Real Data
        # ------------------------------------
        sub_browse, sub_stats, sub_recon = st.tabs([
            ' Jet Browser',
            ' Dataset Statistics',
            'Input Input vs Reconstruction',
        ])

        # -- Jet Browser ----------------------------------------------------------
        with sub_browse:
            st.markdown('Browse individual jets from the dataset.')

            col_ctrl, col_filter = st.columns([2, 1])
            with col_filter:
                filter_label = st.selectbox('Filter by type', ['all', 'quark', 'gluon'],
                                            key='browse_filter')
            with col_ctrl:
                if filter_label == 'all':
                    valid_idx = np.arange(N)
                elif filter_label == 'quark':
                    valid_idx = np.where(labels == 0)[0]
                else:
                    valid_idx = np.where(labels == 1)[0]

                rand_btn = st.button('Random jet', key='browse_rand')
                jet_idx  = st.number_input('Jet index (within filtered set)',
                                           min_value=0, max_value=len(valid_idx)-1,
                                           value=0, step=1, key='browse_idx')

            if rand_btn:
                st.session_state['browse_idx'] = int(np.random.randint(0, len(valid_idx)))
                st.rerun()

            real_idx = int(valid_idx[st.session_state.get('browse_idx', 0)])
            img_sel  = images[real_idx]
            obs_sel  = compute_image_observables(img_sel)
            lbl_sel  = LABEL_NAMES.get(int(labels[real_idx]), str(labels[real_idx]))
            pt_sel   = float(pt_sums[real_idx])

            st.markdown(f'**Jet #{real_idx}** -type: `{lbl_sel}` - scalar pT: `{pt_sel:.1f} GeV`')

            cmap_br = st.selectbox('Color map', ['Greys', 'inferno', 'hot', 'viridis'], key='browse_cmap')

            col_tot, col_met = st.columns([2, 1])
            with col_tot:
                fig_br = make_total_image(img_sel, title=f'{lbl_sel} jet #{real_idx}  (total pT)',
                                          cmap=cmap_br)
                st.pyplot(fig_br, use_container_width=True)
                plt.close(fig_br)
            with col_met:
                st.markdown('**Observables**')
                st.metric('std eta',        f'{obs_sel["std_eta"]:.3f}')
                st.metric('std phi',        f'{obs_sel["std_phi"]:.3f}')
                st.metric('std R',        f'{obs_sel["std_R"]:.3f}')
                st.metric('pT entropy',   f'{obs_sel["pT_entropy"]:.3f}')
                st.markdown('---')
                ch_vals = obs_sel['total_pt_per_channel']
                for c in range(meta['n_channels']):
                    pct = 100 * ch_vals[c] / (obs_sel['total_pt'] + 1e-8)
                    st.markdown(
                        f'<span style="color:{channel_colors[c]}">{channel_names[c]}</span>'
                        f': {ch_vals[c]:.3f} ({pct:.1f}%)',
                        unsafe_allow_html=True,
                    )

            vmax_br = st.slider('Channel color scale max', 0.001,
                                float(max(img_sel.max(), 0.01)),
                                float(img_sel.max() or 0.01),
                                step=0.001, format='%.3f', key='browse_vmax')
            fig_bch = make_channel_grid(img_sel, vmax=vmax_br, cmap=cmap_br,
                                        title=f'All channels -{lbl_sel} jet #{real_idx}')
            st.pyplot(fig_bch, use_container_width=True)
            plt.close(fig_bch)

        # -- Dataset Statistics ---------------------------------------------------
        with sub_stats:
            st.markdown('Distributions computed over the loaded split.')

            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            n_sample = min(N, 5000)
            rng      = np.random.default_rng(0)
            sample_idx = rng.choice(N, n_sample, replace=False)
            imgs_s   = images[sample_idx]
            labs_s   = labels[sample_idx]
            pts_s    = pt_sums[sample_idx]

            # -- total pT histogram -----------------------------------------------
            st.markdown(f'**Sample size:** {n_sample:,} (random subset)')

            fig_pt = go.Figure()
            for lbl_id, lbl_name, color in [(0, 'quark', '#4e9af1'), (1, 'gluon', '#f5a623')]:
                sel = labs_s == lbl_id
                if sel.any():
                    fig_pt.add_trace(go.Histogram(
                        x=pts_s[sel], name=lbl_name,
                        opacity=0.65, nbinsx=60,
                        marker_color=color,
                    ))
            fig_pt.update_layout(
                barmode='overlay', template='plotly_dark',
                title='Scalar jet pT distribution', height=320,
                xaxis_title='pT [GeV]', yaxis_title='count',
                legend=dict(orientation='h'),
            )
            st.plotly_chart(fig_pt, use_container_width=True)

            # -- per-channel total pT fractions ----------------------------------
            ch_totals = imgs_s.sum(axis=(0, 2, 3))   # (C,)
            ch_frac   = ch_totals / (ch_totals.sum() + 1e-8)

            fig_ch = go.Figure(go.Bar(
                x=channel_names, y=ch_frac,
                marker_color=channel_colors,
                text=[f'{v*100:.1f}%' for v in ch_frac],
                textposition='outside',
            ))
            fig_ch.update_layout(
                template='plotly_dark', title='Per-channel pT fraction (sample mean)',
                height=320, yaxis_title='fraction of total pT',
            )
            st.plotly_chart(fig_ch, use_container_width=True)

            # -- mean and std images ----------------------------------------------
            st.markdown('**Mean and std pT images** (across sample, all channels summed)')
            mean_img_sample = imgs_s.mean(axis=0)   # (C,H,W)
            std_img_sample  = imgs_s.std(axis=0)

            col_mn, col_sd = st.columns(2)
            with col_mn:
                fig_mn = make_total_image(mean_img_sample, title='Mean pT image (sample)')
                st.pyplot(fig_mn, use_container_width=True)
                plt.close(fig_mn)
            with col_sd:
                fig_sd = make_total_image(std_img_sample, title='Std pT image (sample)', cmap='Greys')
                st.pyplot(fig_sd, use_container_width=True)
                plt.close(fig_sd)

            # -- multiplicity distribution ----------------------------------------
            mults = [compute_image_observables(imgs_s[i])['multiplicity']
                     for i in range(min(500, n_sample))]
            fig_mult = go.Figure()
            for lbl_id, lbl_name, color in [(0, 'quark', '#4e9af1'), (1, 'gluon', '#f5a623')]:
                sel_m = [m for m, l in zip(mults, labs_s[:len(mults)]) if l == lbl_id]
                if sel_m:
                    fig_mult.add_trace(go.Histogram(
                        x=sel_m, name=lbl_name, opacity=0.65,
                        marker_color=color, nbinsx=40,
                    ))
            fig_mult.update_layout(
                barmode='overlay', template='plotly_dark',
                title='Pixel multiplicity distribution (first 500 samples)',
                height=300, xaxis_title='multiplicity', yaxis_title='count',
                legend=dict(orientation='h'),
            )
            st.plotly_chart(fig_mult, use_container_width=True)

        # -- Input vs Reconstruction ----------------------------------------------
        with sub_recon:
            st.markdown('Pick a real jet, encode it, decode it, compare.')

            col_r1, col_r2, col_r3 = st.columns([2, 1, 1])
            with col_r1:
                recon_idx = st.number_input('Jet index', min_value=0, max_value=N-1,
                                            value=0, step=1, key='recon_idx')
            with col_r2:
                rand_recon = st.button('Random', key='rand_recon')
            with col_r3:
                use_sample = st.checkbox('Sample z (vs mu)', value=False, key='recon_sample',
                                         help='Use sampled z ~ N(mu,sigma^2) instead of the mean mu.')

            if rand_recon:
                st.session_state['recon_idx'] = int(np.random.randint(0, N))
                st.rerun()

            r_idx   = int(st.session_state.get('recon_idx', 0))
            r_img   = images[r_idx]
            r_label = LABEL_NAMES.get(int(labels[r_idx]), str(labels[r_idx]))
            r_pt    = float(pt_sums[r_idx])

            st.markdown(f'**Jet #{r_idx}** -`{r_label}` - scalar pT `{r_pt:.1f} GeV`')

            with st.spinner('Encoding & decoding...'):
                recon_img, mu = reconstruct_image(model, r_img, meta,
                                                  use_sample=use_sample,
                                                  already_normalized=True)

            cmap_rc   = st.selectbox('Color map', ['Greys', 'inferno', 'hot', 'viridis'], key='recon_cmap')
            view_mode = st.radio(
                'Display space',
                ['Normalized  (model input/output space)',
                 'Physical pT  (denormalized, clipped >=0)'],
                horizontal=True, key='rd_view_mode',
                help='Normalized = what the model actually sees. '
                     'Physical pT = denormalized back to GeV units; values below zero are clipped.',
            )

            std_c = meta['std_img']; mean_c = meta['mean_img']
            if 'Normalized' in view_mode:
                disp_o = r_img                           # already (raw - mean) / std
                disp_r = recon_img                       # decoder output (normalized)
                lbl_sc = 'normalized space'
                # clip min for display so color scale isn't dominated by large negatives
                vmin_clip = float(np.percentile(disp_o, 5))
                disp_o_show = np.clip(disp_o, vmin_clip, None)
                disp_r_show = np.clip(disp_r, vmin_clip, None)
            else:
                disp_o_show = np.clip(r_img    * std_c + mean_c, 0, None)
                disp_r_show = np.clip(recon_img * std_c + mean_c, 0, None)
                lbl_sc = 'pT [GeV]'

            # -- side-by-side total images ----------------------------------------
            col_o_img, col_r_img = st.columns(2)
            with col_o_img:
                fig_o = make_total_image(disp_o_show,
                                         title=f'Original  ({r_label} #{r_idx})  [{lbl_sc}]',
                                         cmap=cmap_rc)
                st.pyplot(fig_o, use_container_width=True); plt.close(fig_o)
            with col_r_img:
                fig_r = make_total_image(disp_r_show,
                                         title=f'Reconstruction  [{lbl_sc}]',
                                         cmap=cmap_rc)
                st.pyplot(fig_r, use_container_width=True); plt.close(fig_r)

            # -- per-channel comparison -------------------------------------------
            st.markdown('**Per-channel  (top: original - bottom: reconstruction)**')
            fig_cch = make_channel_compare_figure(
                disp_o_show, disp_r_show, cmap=cmap_rc,
                channel_names=channel_names,
                channel_colors=channel_colors,
            )
            st.pyplot(fig_cch, use_container_width=True)
            plt.close(fig_cch)

            # -- observables comparison -------------------------------------------
            obs_orig  = compute_image_observables(np.clip(r_img    * std_c + mean_c, 0, None))
            obs_recon = compute_image_observables(np.clip(recon_img * std_c + mean_c, 0, None))

            obs_keys_show = ['std_eta', 'std_phi', 'std_R', 'pT_entropy']
            col_o, col_r = st.columns(2)
            col_o.markdown('**Original**')
            col_r.markdown('**Reconstruction**')
            for k in obs_keys_show:
                vo = obs_orig[k];  vr = obs_recon[k]
                delta = float(vr) - float(vo)
                col_o.metric(k, f'{float(vo):.3f}')
                col_r.metric(k, f'{float(vr):.3f}', delta=f'{delta:+.3f}')

            # -- latent z bar chart -----------------------------------------------
            with st.expander('Latent vector mu', expanded=False):
                import plotly.graph_objects as go
                fig_z = go.Figure(go.Bar(
                    x=list(range(len(mu))), y=mu,
                    marker_color=['#4e9af1' if v >= 0 else '#f5a623' for v in mu],
                ))
                fig_z.update_layout(
                    template='plotly_dark', height=280,
                    xaxis_title='dimension', yaxis_title='mu value',
                    title=f'Encoded mu for jet #{r_idx}',
                )
                st.plotly_chart(fig_z, use_container_width=True)

            # -- download ---------------------------------------------------------
            buf = io.BytesIO()
            fig_dl = make_compare_figure(r_img, recon_img, cmap=cmap_rc,
                                         title_orig=f'Original ({r_label} #{r_idx})',
                                         title_recon='Reconstruction')
            fig_dl.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig_dl)
            st.download_button('Download comparison', buf.getvalue(),
                               file_name=f'recon_jet{r_idx}.png', mime='image/png')


    # --------------------------------------
    # --------------------------------------
# TAB 5 -Raw Events
# --------------------------------------
with tab_raw:
    st.subheader('Raw jet events - before any normalization')
    st.caption('Constituents are read directly from the .dat files (pid, px, py, pz, E).')

    import plotly.graph_objects as go

    # -- file paths ------------------------------------------------------------
    col_gp, col_up = st.columns(2)
    with col_gp:
        g_path = st.text_input('Gluon jet .dat', key='raw_g',
                               value=DEFAULT_G_JET)
    with col_up:
        u_path = st.text_input('Quark jet .dat', key='raw_u',
                               value=DEFAULT_U_JET)

    max_load = st.number_input('Max events to load per file (for statistics)',
                               min_value=100, max_value=20000, value=5000, step=500,
                               key='raw_max_load')

    load_raw_btn = st.button('Load files', type='primary', key='load_raw_btn')

    if load_raw_btn:
        loaded = {}
        for label, path in [('gluon', g_path), ('quark', u_path)]:
            path = path.strip()
            if not path:
                continue
            if os.path.exists(path):
                with st.spinner(f'Parsing {label} jets...'):
                    loaded[label] = parse_dat_file(path, max_events=int(max_load))
            else:
                st.warning(f'{label} file not found: {path}')
        if loaded:
            st.session_state['raw_events'] = loaded
            st.success(
                '  -  '.join(f'{k}: {len(v):,} events' for k, v in loaded.items())
            )

    if 'raw_events' not in st.session_state:
        st.info('Enter file paths above and click **Load files**.')
    else:

        raw_events = st.session_state['raw_events']

        sub_raw_jet, sub_raw_stats, sub_raw_recon = st.tabs([
            'Jet Browser', 'Statistics', 'Input vs Reconstruction',
        ])

        # -- Jet Browser -----------------------------------------------------------
        with sub_raw_jet:
            col_jt, col_ji = st.columns([1, 2])
            with col_jt:
                jet_type_raw = st.radio('Jet type', list(raw_events.keys()),
                                        horizontal=True, key='raw_type')
            with col_ji:
                n_avail = len(raw_events[jet_type_raw])
                raw_idx = st.number_input('Event index', min_value=0,
                                          max_value=n_avail - 1, value=0,
                                          step=1, key='raw_idx')
                if st.button('Random', key='raw_rand'):
                    st.session_state['raw_idx'] = int(np.random.randint(0, n_avail))
                    st.rerun()

            evt_idx   = int(st.session_state.get('raw_idx', 0))
            const_arr = raw_events[jet_type_raw][evt_idx]
            kin       = constituents_to_kinematics(const_arr)
            ax_j      = jet_axis(kin)

            # -- summary metrics ---------------------------------------------------
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric('Constituents', len(kin['pid']))
            mc2.metric('Jet pT [GeV]', f'{ax_j["pt"]:.1f}')
            mc3.metric('Jet mass [GeV]', f'{ax_j["mass"]:.1f}')
            mc4.metric('Jet eta', f'{ax_j["eta"]:.2f}')
            mc5.metric('Jet phi', f'{ax_j["phi"]:.2f}')

            cmap_raw = st.selectbox('Image color map', ['Greys', 'inferno', 'hot', 'viridis'],
                                    key='raw_cmap')

            # -- two columns: lab scatter | jet image -----------------------------
            col_lab, col_img = st.columns(2)

            with col_lab:
                st.markdown('**Lab frame eta-phi**')
                fig_lab = make_scatter_figure(kin, ax_j,
                                              title=f'{jet_type_raw} #{evt_idx}  (lab)')
                st.pyplot(fig_lab, use_container_width=True)
                plt.close(fig_lab)

            with col_img:
                st.markdown('**Jet image - log(1+pT)**')
                img_raw, _, _ = constituents_to_image(
                    const_arr,
                    img_size=meta['img_size'],
                    n_channels=meta['n_channels'],
                    no_pid=uses_no_pid_channels(meta),
                )
                fig_img = make_total_image(np.log1p(img_raw),
                                           title='log(1+pT) image', cmap=cmap_raw)
                st.pyplot(fig_img, use_container_width=True)
                plt.close(fig_img)

            # -- 2x4 channel scatter grid ------------------------------------------
            st.markdown('**Per-channel Delta eta-Delta phi scatter  (dot size proportional to log(1+pT))**')
            fig_ch_sc = make_scatter_channel_grid(
                kin, ax_j, title=f'{jet_type_raw} #{evt_idx}')
            st.pyplot(fig_ch_sc, use_container_width=True)
            plt.close(fig_ch_sc)

            # -- 2x4 channel image grid --------------------------------------------
            with st.expander('Per-channel jet image  log(1+pT)', expanded=False):
                fig_ch_raw = make_channel_grid(np.log1p(img_raw), cmap=cmap_raw,
                                               title=f'{jet_type_raw} #{evt_idx} - all channels  log(1+pT)')
                st.pyplot(fig_ch_raw, use_container_width=True)
                plt.close(fig_ch_raw)

            # -- constituent table -------------------------------------------------
            with st.expander('Constituent list', expanded=False):
                import pandas as pd
                ch_col = [CHANNEL_NAMES[_PID_TO_CH.get(int(p), 0)]
                          if int(p) in _PID_TO_CH else '?' for p in kin['pid']]
                df_const = pd.DataFrame({
                    'pid':     kin['pid'],
                    'particle': ch_col,
                    'pT [GeV]': kin['pt'].round(3),
                    'eta':        kin['eta'].round(4),
                    'phi':        kin['phi'].round(4),
                    'Delta eta':      (kin['eta'] - ax_j['eta']).round(4),
                    'Delta phi':      (((kin['phi'] - ax_j['phi'] + np.pi) % (2*np.pi)) - np.pi).round(4),
                    'E [GeV]': const_arr[:, 4].round(3),
                }).sort_values('pT [GeV]', ascending=False)
                st.dataframe(df_const, height=300, use_container_width=True)

        # -- Statistics ------------------------------------------------------------
        with sub_raw_stats:
            st.markdown('Statistics over all loaded events.')

            # collect per-event observables (fast scan, no image building)
            @st.cache_data(show_spinner='Computing statistics...')
            def _compute_raw_stats(events_g, events_u):
                result = {}
                for label, evts in [('gluon', events_g), ('quark', events_u)]:
                    if evts is None:
                        continue
                    pts, mults, masses, girths = [], [], [], []
                    ch_fracs = np.zeros(len(CHANNEL_PIDS), dtype=np.float64)
                    for ev in evts:
                        k  = constituents_to_kinematics(ev)
                        aj = jet_axis(k)
                        pts.append(aj['pt'])
                        mults.append(len(k['pid']))
                        masses.append(aj['mass'])
                        # girth
                        deta = k['eta'] - aj['eta']
                        dphi = ((k['phi'] - aj['phi'] + np.pi) % (2*np.pi)) - np.pi
                        dr   = np.sqrt(deta**2 + dphi**2)
                        girth = float(np.sum(k['pt'] * dr) / (aj['pt'] + 1e-8))
                        girths.append(girth)
                        # channel fractions
                        for ci, pid_val in enumerate(CHANNEL_PIDS):
                            mask = k['pid'] == pid_val
                            ch_fracs[ci] += k['pt'][mask].sum()
                    ch_fracs /= (ch_fracs.sum() + 1e-8)
                    result[label] = dict(pts=np.array(pts), mults=np.array(mults),
                                         masses=np.array(masses), girths=np.array(girths),
                                         ch_fracs=ch_fracs)
                return result

            # pass as tuples so streamlit can hash them
            g_evts = tuple(map(tuple, [e.flatten().tolist() for e in raw_events.get('gluon', [])])) if 'gluon' in raw_events else None
            u_evts = tuple(map(tuple, [e.flatten().tolist() for e in raw_events.get('quark', [])])) if 'quark' in raw_events else None

            # cache_data needs hashable -compute directly without cache for simplicity
            stats_raw = {}
            for label, evts in raw_events.items():
                pts, mults, masses, girths = [], [], [], []
                ch_fracs = np.zeros(len(CHANNEL_PIDS), dtype=np.float64)
                for ev in evts:
                    k  = constituents_to_kinematics(ev)
                    aj = jet_axis(k)
                    pts.append(aj['pt'])
                    mults.append(len(k['pid']))
                    masses.append(aj['mass'])
                    deta = k['eta'] - aj['eta']
                    dphi = ((k['phi'] - aj['phi'] + np.pi) % (2*np.pi)) - np.pi
                    dr   = np.sqrt(deta**2 + dphi**2)
                    girths.append(float(np.sum(k['pt'] * dr) / (aj['pt'] + 1e-8)))
                    for ci, pid_val in enumerate(CHANNEL_PIDS):
                        ch_fracs[ci] += k['pt'][k['pid'] == pid_val].sum()
                ch_fracs /= (ch_fracs.sum() + 1e-8)
                stats_raw[label] = dict(pts=np.array(pts), mults=np.array(mults),
                                        masses=np.array(masses), girths=np.array(girths),
                                        ch_fracs=ch_fracs)

            COLORS_LBL = {'gluon': '#f5a623', 'quark': '#4e9af1'}

            def _hist_fig(title, xlabel, data_dict, nbins=60, log_x=False):
                fig = go.Figure()
                for lbl, vals in data_dict.items():
                    fig.add_trace(go.Histogram(
                        x=np.log10(vals + 1e-3) if log_x else vals,
                        name=lbl, opacity=0.65, nbinsx=nbins,
                        marker_color=COLORS_LBL.get(lbl, '#888'),
                    ))
                fig.update_layout(
                    barmode='overlay', template='plotly_dark', height=300,
                    title=title, xaxis_title=('log10(' + xlabel + ')') if log_x else xlabel,
                    yaxis_title='count', legend=dict(orientation='h'),
                )
                return fig

            st.plotly_chart(
                _hist_fig('Jet pT', 'pT [GeV]',
                          {l: s['pts'] for l, s in stats_raw.items()}, log_x=True),
                use_container_width=True,
            )

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.plotly_chart(
                    _hist_fig('Constituent multiplicity', 'N constituents',
                              {l: s['mults'] for l, s in stats_raw.items()}, nbins=50),
                    use_container_width=True,
                )
            with col_s2:
                st.plotly_chart(
                    _hist_fig('Jet mass', 'mass [GeV]',
                              {l: s['masses'] for l, s in stats_raw.items()}, log_x=True),
                    use_container_width=True,
                )

            col_s3, col_s4 = st.columns(2)
            with col_s3:
                st.plotly_chart(
                    _hist_fig('Jet girth  (pT-weighted mean DeltaR)', 'girth',
                              {l: s['girths'] for l, s in stats_raw.items()}, nbins=50),
                    use_container_width=True,
                )
            with col_s4:
                # per-channel pT fraction bar chart
                fig_ch_stat = go.Figure()
                for lbl, s in stats_raw.items():
                    fig_ch_stat.add_trace(go.Bar(
                        name=lbl, x=CHANNEL_NAMES, y=s['ch_fracs'],
                        marker_color=COLORS_LBL.get(lbl, '#888'),
                        opacity=0.8,
                        text=[f'{v*100:.1f}%' for v in s['ch_fracs']],
                        textposition='outside',
                    ))
                fig_ch_stat.update_layout(
                    barmode='group', template='plotly_dark', height=300,
                    title='Per-channel pT fraction', yaxis_title='fraction',
                    legend=dict(orientation='h'),
                )
                st.plotly_chart(fig_ch_stat, use_container_width=True)

            # -- mean jet images ---------------------------------------------------
            st.markdown('**Mean pT image** (averaged over first 500 events per type)')
            img_cols = st.columns(len(raw_events))
            for col_img_s, (label, evts) in zip(img_cols, raw_events.items()):
                with col_img_s:
                    n_avg = min(500, len(evts))
                    mean_img_raw = np.mean(
                        [
                            constituents_to_image(
                                evts[i],
                                img_size=meta['img_size'],
                                n_channels=meta['n_channels'],
                                no_pid=uses_no_pid_channels(meta),
                            )[0]
                            for i in range(n_avg)
                        ],
                        axis=0,
                    )
                    fig_mean = make_total_image(np.log1p(mean_img_raw),
                                                title=f'Mean {label} jet - log(1+pT)  (n={n_avg})',
                                                 cmap='Greys')
                    st.pyplot(fig_mean, use_container_width=True)
                    plt.close(fig_mean)

        # -- vs Reconstruction -----------------------------------------------------
        with sub_raw_recon:
            st.markdown(
                'Pick a raw event, build its jet image, encode it through the VAE, '
                'then compare the original image with the reconstruction. '
                'All images are shown as **log(1+pT)**.'
            )

            col_rr1, col_rr2, col_rr3 = st.columns([1, 1, 1])
            with col_rr1:
                rr_type = st.radio('Jet type', list(raw_events.keys()),
                                   horizontal=True, key='rr_type')
            with col_rr2:
                n_rr = len(raw_events[rr_type])
                rr_idx = st.number_input('Event index', min_value=0,
                                         max_value=n_rr - 1, value=0,
                                         step=1, key='rr_idx')
            with col_rr3:
                if st.button('Random', key='rr_rand'):
                    st.session_state['rr_idx'] = int(np.random.randint(0, n_rr))
                    st.rerun()
                rr_sample = st.checkbox('Sample z', value=False, key='rr_sample',
                                        help='Use z ~ N(mu,sigma^2) instead of mu.')

            rr_idx   = int(st.session_state.get('rr_idx', 0))
            rr_const = raw_events[rr_type][rr_idx]
            rr_kin   = constituents_to_kinematics(rr_const)
            rr_ax    = jet_axis(rr_kin)

            # build image, encode, reconstruct
            rr_img_raw, _, _ = constituents_to_image(
                rr_const,
                img_size=meta['img_size'],
                n_channels=meta['n_channels'],
                no_pid=uses_no_pid_channels(meta),
            )
            with st.spinner('Encoding & decoding...'):
                rr_recon, rr_mu = reconstruct_image(model, rr_img_raw, meta,
                                                    use_sample=rr_sample)

            cmap_rr = st.selectbox('Color map', ['Greys', 'inferno', 'hot', 'viridis'],
                                   key='rr_cmap')

            # -- headline metrics --------------------------------------------------
            obs_rr_orig  = compute_image_observables(rr_img_raw)
            obs_rr_recon = compute_image_observables(rr_recon)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric('Constituents (raw)', len(rr_kin['pid']))
            mc2.metric('Jet pT [GeV]', f'{rr_ax["pt"]:.1f}')
            mc3.metric('Jet eta', f'{rr_ax["eta"]:.3f}')
            mc4.metric('Jet phi', f'{rr_ax["phi"]:.3f}')

            norm_toggle = st.checkbox(
                'Normalise both images by total pT  (compare shape, not scale)',
                value=True, key='rr_norm_toggle',
                help='Divides each image by its total pT sum so color scales are comparable.',
            )

            disp_orig  = normalise_to_unit_sum(rr_img_raw) if norm_toggle else rr_img_raw
            disp_recon = normalise_to_unit_sum(rr_recon)   if norm_toggle else rr_recon
            scale_lbl  = 'normalized by total pT' if norm_toggle else 'absolute pT [GeV]'

            # -- two-column: original image | reconstruction -----------------------
            col_orig, col_rec = st.columns(2)
            with col_orig:
                st.markdown(f'**Original image  log(1 + {scale_lbl})**')
                fig_orig_rr = make_total_image(
                    np.log1p(disp_orig),
                    title=f'Original  ({scale_lbl})', cmap=cmap_rr,
                )
                st.pyplot(fig_orig_rr, use_container_width=True)
                plt.close(fig_orig_rr)

            with col_rec:
                st.markdown(f'**VAE reconstruction  log(1 + {scale_lbl})**')
                fig_rec_rr = make_total_image(
                    np.log1p(disp_recon),
                    title=f'Reconstruction  ({scale_lbl})', cmap=cmap_rr,
                )
                st.pyplot(fig_rec_rr, use_container_width=True)
                plt.close(fig_rec_rr)

            # -- per-channel 2x8 comparison ----------------------------------------
            st.markdown('**Per-channel comparison  (top: original - bottom: reconstruction)**')
            fig_cch_rr = make_channel_compare_figure(
                np.log1p(disp_orig), np.log1p(disp_recon), cmap=cmap_rr,
                channel_names=channel_names,
                channel_colors=channel_colors,
            )
            st.pyplot(fig_cch_rr, use_container_width=True)
            plt.close(fig_cch_rr)

            # -- channel scatter grid for the raw event ----------------------------
            with st.expander('Per-channel constituent scatter  (raw event)', expanded=False):
                fig_ch_sc_rr = make_scatter_channel_grid(
                    rr_kin, rr_ax, title=f'{rr_type} #{rr_idx}  - constituents by PID')
                st.pyplot(fig_ch_sc_rr, use_container_width=True)
                plt.close(fig_ch_sc_rr)

            # -- observable comparison ---------------------------------------------
            st.markdown('**Observable comparison**')
            obs_keys_rr = ['std_eta', 'std_phi', 'std_R', 'pT_entropy']
            col_oo, col_rr_obs = st.columns(2)
            col_oo.markdown('**Original**')
            col_rr_obs.markdown('**Reconstruction**')
            for k in obs_keys_rr:
                vo = float(obs_rr_orig[k])
                vr = float(obs_rr_recon[k])
                col_oo.metric(k, f'{vo:.3f}')
                col_rr_obs.metric(k, f'{vr:.3f}', delta=f'{vr - vo:+.3f}')

            # -- residual image ----------------------------------------------------
            with st.expander('Residual  (reconstruction -original)  normalized', expanded=False):
                residual = disp_recon - disp_orig
                vabs = float(np.abs(residual).max()) or 1e-8
                fig_res, ax_res = plt.subplots(figsize=(4, 3.5))
                im_res = ax_res.imshow(
                    residual.sum(0), origin='lower', cmap='RdBu_r',
                    vmin=-vabs, vmax=vabs, aspect='auto',
                    extent=[-0.5, 0.5, -0.5, 0.5],
                )
                ax_res.set_xlabel('phi', fontsize=9)
                ax_res.set_ylabel('eta', fontsize=9)
                ax_res.set_title('Residual  (all channels summed)', fontsize=9)
                plt.colorbar(im_res, ax=ax_res, label=scale_lbl, fraction=0.046, pad=0.04)
                fig_res.tight_layout()
                st.pyplot(fig_res, use_container_width=True)
                plt.close(fig_res)

            # -- latent mu vector ---------------------------------------------------
            with st.expander('Encoded latent mu', expanded=False):
                fig_z_rr = go.Figure(go.Bar(
                    x=list(range(len(rr_mu))), y=rr_mu,
                    marker_color=['#4e9af1' if v >= 0 else '#f5a623' for v in rr_mu],
                ))
                fig_z_rr.update_layout(
                    template='plotly_dark', height=260,
                    xaxis_title='dimension', yaxis_title='mu',
                    title=f'Latent mu  - {rr_type} #{rr_idx}',
                )
                st.plotly_chart(fig_z_rr, use_container_width=True)


# TAB 6 -Active Units
# --------------------------------------
with tab_au:
    st.subheader('Active latent units')
    st.markdown(
        'A dimension **d** is *active* if its posterior mean varies across data: '
        '**A_d = Var_x[mu_d(x)] > threshold** (Burda et al. 2015). '
        'Inactive dimensions have collapsed to the prior N(0,1) - the encoder ignores them.'
    )

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # -- data source ----------------------------------------------------------
    st.caption('Active Units are computed directly from raw .dat events; cached images are not required.')
    col_src_g, col_src_u = st.columns(2)
    with col_src_g:
        au_g_path = st.text_input('Gluon jet .dat', value=DEFAULT_G_JET, key='au_raw_g')
    with col_src_u:
        au_u_path = st.text_input('Quark jet .dat', value=DEFAULT_U_JET, key='au_raw_u')

    col_au1, col_au2 = st.columns([1, 1])
    with col_au1:
        au_n = st.number_input('Max raw events per file', min_value=100,
                               max_value=50000, value=5000, step=500, key='au_n')
    with col_au2:
        au_threshold = st.number_input('Active threshold', min_value=1e-6, value=0.01,
                                       step=0.001, format='%.4f', key='au_thresh',
                                       help='Var_x[mu_d] must exceed this to count as active.')

    run_au = st.button('Compute active units', type='primary')

    if run_au:
        raw_specs = [
            ('gluon', 1, au_g_path.strip()),
            ('quark', 0, au_u_path.strip()),
        ]
        images_list = []
        labels_list = []
        loaded_counts = []

        for label_name, label_id, path in raw_specs:
            if not path:
                continue
            if not os.path.exists(path):
                st.warning(f'{label_name} file not found: {path}')
                continue
            with st.spinner(f'Parsing {label_name} raw events...'):
                events = parse_dat_file(path, max_events=int(au_n))
            if not events:
                st.warning(f'No events found in {path}')
                continue
            with st.spinner(f'Building {label_name} jet images...'):
                for ev in events:
                    img_ev, _, _ = constituents_to_image(
                        ev,
                        img_size=meta['img_size'],
                        n_channels=meta['n_channels'],
                        no_pid=uses_no_pid_channels(meta),
                    )
                    images_list.append(img_ev)
                    labels_list.append(label_id)
            loaded_counts.append(f'{label_name}: {len(events):,}')

        if not images_list:
            st.error('Enter at least one valid raw .dat file.')
        else:
            images_au = np.asarray(images_list, dtype=np.float32)
            labels_au = np.asarray(labels_list, dtype=np.int64)
            n_use = len(images_au)
            st.success('Loaded raw events for Active Units: ' + '  -  '.join(loaded_counts))

            with st.spinner(f'Encoding {n_use:,} raw-event images...'):
                mus_au, lvs_au = batch_encode(
                    model,
                    images_au,
                    meta,
                    already_normalized=False,
                )

            stats = compute_active_units(mus_au, lvs_au, threshold=au_threshold)
            st.session_state['au_stats']    = stats
            st.session_state['au_mus']      = mus_au
            st.session_state['au_lvs']      = lvs_au
            st.session_state['au_labels']   = labels_au
            st.session_state['au_n_use']    = n_use
            st.session_state['au_threshold']= au_threshold

    # -- display results -------------------------------------------------------
    if 'au_stats' in st.session_state:
        stats     = st.session_state['au_stats']
        mus_au    = st.session_state['au_mus']
        lvs_au    = st.session_state['au_lvs']
        labels_au = st.session_state['au_labels']
        n_use     = st.session_state['au_n_use']
        thresh    = au_threshold

        # recompute if threshold changed
        stats = compute_active_units(mus_au, lvs_au, threshold=thresh)

        D        = latent_dim
        au_mask  = stats['au_mask']
        var_mu   = stats['var_mu']
        kld_dim  = stats['mean_kld_per_dim']
        n_active = stats['n_active']

        # -- headline metrics -------------------------------------------------
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Active units',   f'{n_active} / {D}')
        c2.metric('Inactive units', f'{D - n_active} / {D}')
        c3.metric('Total KLD',      f'{kld_dim.sum():.2f}')
        c4.metric('Samples used',   f'{n_use:,}')

        # -- Var_x[mu_d] bar chart ---------------------------------------------
        colors_au = ['#4e9af1' if a else '#555555' for a in au_mask]
        fig_au = go.Figure()
        fig_au.add_trace(go.Bar(
            x=list(range(D)), y=var_mu,
            marker_color=colors_au,
            name='Var_x[mu_d]',
            hovertemplate='dim %{x}<br>Var=%{y:.4f}<extra></extra>',
        ))
        fig_au.add_hline(y=thresh, line_dash='dash', line_color='red',
                         annotation_text=f'threshold = {thresh}',
                         annotation_position='top right')
        fig_au.update_layout(
            template='plotly_dark', height=320,
            title=f'Variance of posterior mean mu_d across {n_use:,} jets  '
                  f'-<span style="color:#4e9af1">active</span>  '
                  f'<span style="color:#555">inactive</span>',
            xaxis_title='Latent dimension d',
            yaxis_title='Var_x[mu_d]',
        )
        st.plotly_chart(fig_au, use_container_width=True)

        # -- KLD per dimension -------------------------------------------------
        fig_kld = go.Figure()
        fig_kld.add_trace(go.Bar(
            x=list(range(D)), y=kld_dim,
            marker_color=colors_au,
            name='KLD_d',
            hovertemplate='dim %{x}<br>KLD=%{y:.4f}<extra></extra>',
        ))
        fig_kld.update_layout(
            template='plotly_dark', height=280,
            title='Mean KL divergence per dimension  KLD_d = 0.5 * (mu_d^2 + sigma_d^2 - log sigma_d^2 -1)',
            xaxis_title='Latent dimension d',
            yaxis_title='mean KLD_d',
        )
        st.plotly_chart(fig_kld, use_container_width=True)

        # -- Var_mu vs KLD scatter ---------------------------------------------
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=var_mu, y=kld_dim,
            mode='markers+text',
            text=[str(d) for d in range(D)],
            textposition='top center',
            textfont=dict(size=8),
            marker=dict(
                color=['#4e9af1' if a else '#888888' for a in au_mask],
                size=8, line=dict(width=0.5, color='white'),
            ),
            hovertemplate='dim %{text}<br>Var=%{x:.4f}  KLD=%{y:.4f}<extra></extra>',
        ))
        fig_sc.add_vline(x=thresh, line_dash='dash', line_color='red')
        fig_sc.update_layout(
            template='plotly_dark', height=360,
            title='Var_x[mu_d] vs mean KLD_d  (blue = active, grey = inactive)',
            xaxis_title='Var_x[mu_d]', yaxis_title='mean KLD_d',
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # -- mu distributions for active dims ----------------------------------
        active_dims = np.where(au_mask)[0]
        with st.expander(f'mu distributions for {n_active} active dimensions', expanded=False):
            n_cols_ad = 4
            rows_ad   = [active_dims[i:i+n_cols_ad].tolist()
                         for i in range(0, len(active_dims), n_cols_ad)]
            for row_dims in rows_ad:
                cols_ad = st.columns(n_cols_ad)
                for col_ad, d in zip(cols_ad, row_dims):
                    with col_ad:
                        fig_h, ax_h = plt.subplots(figsize=(2.5, 1.8))
                        for lbl_id, lbl_name, color in [(0,'quark','#4e9af1'),(1,'gluon','#f5a623')]:
                            sel = labels_au == lbl_id
                            if sel.any():
                                ax_h.hist(mus_au[sel, d], bins=30, alpha=0.6,
                                          color=color, label=lbl_name, density=True)
                        ax_h.set_title(f'z[{d}]  Var={var_mu[d]:.3f}', fontsize=7)
                        ax_h.tick_params(labelsize=6)
                        ax_h.set_xlabel('mu_d', fontsize=6)
                        ax_h.legend(fontsize=5, loc='upper right')
                        fig_h.tight_layout()
                        st.pyplot(fig_h, use_container_width=True)
                        plt.close(fig_h)
                # fill empty columns if last row is short
                for _ in range(n_cols_ad - len(row_dims)):
                    cols_ad[len(row_dims) + _].empty()

        # -- raw table --------------------------------------------------------
        with st.expander('Raw values per dimension', expanded=False):
            import pandas as pd
            df_au = pd.DataFrame({
                'dim':       range(D),
                'Var_x[mu]':  var_mu.round(5),
                'std_x[mu]':  stats['std_mu'].round(5),
                'mean_x[mu]': stats['mean_mu'].round(5),
                'mean_KLD':  kld_dim.round(5),
                'active':    au_mask,
            })
            st.dataframe(
                df_au.style
                    .background_gradient(subset=['Var_x[mu]', 'mean_KLD'], cmap='Blues')
                    .map(lambda v: 'color: #4e9af1' if v else 'color: #888', subset=['active']),
                height=400, use_container_width=True,
            )


# TAB 6 -Training Curves
# --------------------------------------
with tab_curves:
    st.subheader('Training history')

    log_path = meta.get('log_path')
    if log_path:
        st.caption(f'Parsed from: `{log_path}`')
    else:
        log_path = st.text_input('Log file path (.log)', value='', key='log_path_input',
                                 placeholder='vae_cnn/bvae_beta_b001_*.log')

    if log_path and os.path.exists(log_path):
        log = parse_training_log(log_path)
        if not log['epoch']:
            st.warning('No epoch lines found in log.')
        else:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            epochs = log['epoch']
            best_e = meta['epoch']

            fig_curves = make_subplots(
                rows=1, cols=3,
                subplot_titles=['Total Loss', 'Reconstruction Loss', 'KLD'],
                shared_xaxes=False,
            )

            TRAIN_LINE = dict(color='#4e9af1', width=1.5)
            VAL_LINE   = dict(color='#f5a623', width=1.5)

            # clip extreme val spikes for readability
            val_loss  = np.array(log['val_loss'])
            val_recon = np.array(log['val_recon'])
            val_kld   = np.array(log['val_kld'])
            p95_l  = np.percentile(val_loss,  95)
            p95_r  = np.percentile(val_recon, 95)
            p95_k  = np.percentile(val_kld,   95)

            fig_curves.add_trace(go.Scatter(x=epochs, y=log['train_loss'],  name='Train', mode='lines', line=TRAIN_LINE), row=1, col=1)
            fig_curves.add_trace(go.Scatter(x=epochs, y=log['val_loss'],    name='Val',   mode='lines', line=VAL_LINE),   row=1, col=1)
            fig_curves.add_trace(go.Scatter(x=epochs, y=log['train_recon'], name='Train', mode='lines', line=TRAIN_LINE, showlegend=False), row=1, col=2)
            fig_curves.add_trace(go.Scatter(x=epochs, y=log['val_recon'],   name='Val',   mode='lines', line=VAL_LINE,   showlegend=False), row=1, col=2)
            fig_curves.add_trace(go.Scatter(x=epochs, y=log['train_kld'],   name='Train', mode='lines', line=TRAIN_LINE, showlegend=False), row=1, col=3)
            fig_curves.add_trace(go.Scatter(x=epochs, y=log['val_kld'],     name='Val',   mode='lines', line=VAL_LINE,   showlegend=False), row=1, col=3)

            # mark best checkpoint
            for col_i, y_key, p95 in [
                (1, 'val_loss',  p95_l),
                (2, 'val_recon', p95_r),
                (3, 'val_kld',   p95_k),
            ]:
                if best_e in epochs:
                    best_idx = epochs.index(best_e)
                    y_val = log[y_key][best_idx]
                    fig_curves.add_trace(
                        go.Scatter(
                            x=[best_e], y=[y_val],
                            mode='markers',
                            marker=dict(color='lime', size=10, symbol='star'),
                            name='Best ckpt' if col_i == 1 else None,
                            showlegend=(col_i == 1),
                        ),
                        row=1, col=col_i,
                    )
                # clip y-axis to p95 of val so spikes don't squash the view
                fig_curves.update_yaxes(range=[0, p95 * 1.05], row=1, col=col_i)

            fig_curves.update_layout(
                height=400, template='plotly_dark',
                legend=dict(orientation='h', x=0.5, xanchor='center', y=1.12),
                margin=dict(t=60, b=40),
            )
            fig_curves.update_xaxes(title_text='Epoch')

            st.plotly_chart(fig_curves, use_container_width=True)

            # -- summary stats ----------------------------------------------
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Epochs trained',  max(epochs))
            c2.metric('Best val_loss',   f'{min(log["val_loss"]):.4f}',
                      help=f'Epoch {epochs[np.argmin(log["val_loss"])]}')
            c3.metric('Final train_loss', f'{log["train_loss"][-1]:.4f}')
            c4.metric('Final KLD',        f'{log["train_kld"][-1]:.2f}')
    elif log_path:
        st.warning(f'File not found: {log_path}')
    else:
        st.info('Log file path not found automatically. Enter it above to load training curves.')
