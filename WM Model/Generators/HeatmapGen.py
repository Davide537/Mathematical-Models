import Generators.BaseModel as bm
import contextlib
import os
from itertools import product

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# Base Case: Heat maps
def _format_condition_suffix(inhi, distract, lesion):
    """
    Build a filename-safe suffix describing non-default inhi/lesion settings.
    Returns '' if both are at their runsim() defaults, so the base case is untouched.
    """
    parts = []

    on, off, strength, areas = inhi
    is_default_inhi = (areas is None or strength == 0)
    if not is_default_inhi:
        area_str = areas if isinstance(areas, str) else '-'.join(areas)
        area_str = area_str.replace('/', '-')
        parts.append(f"inhi{on}-{off}_{strength}_{area_str}")

    perm, lesioned_areas = lesion
    if lesioned_areas:  # None or [] -> no lesion
        area_str = lesioned_areas if isinstance(lesioned_areas, str) else '-'.join(lesioned_areas)
        area_str = area_str.replace('/', '-')
        parts.append(f"lesion{'perm' if perm else 'temp'}_{area_str}")

    on, off, strength = distract
    if strength:  # 0/None -> no distractor
        parts.append(f"distract{on}-{off}_{strength}")

    return ('_' + '_'.join(parts)) if parts else ''

def _run_one(age, J_max, G, mech, trials,
             stim=[4, 4.5, 3, 'V1'],
             inhi=[9, 9.5, 0.3, None],
             distract=[9, 9.5, 0],
             lesion=[False, None],
             Jcrit='cutFB'):
    B = bm.BuildModel(J_max=J_max, G=G, aging_mechanism=mech, age=age, Jcrit=Jcrit)
    B.runsim(stim=stim, inhi=inhi, distract=distract, lesion=lesion,
             trials=trials, trial_length=20, save_fac=10)

    suffix = _format_condition_suffix(inhi, distract, lesion)
    key = f"{age}_J{J_max:.2f}_G{G:.2f}{suffix}"
    tmp_path = f"data/tmp/{key}_{mech}.npy"
    np.save(tmp_path, B.R)
    return {'Age': age, 'G': G, 'J_max': J_max, 'tmp_path': tmp_path,
            'inhi': inhi, 'distract': distract, 'lesion': lesion, 'suffix': suffix}

# After parallel section — merge into one HDF5
def merge_to_hdf5(rows, h5_path):
    with h5py.File(h5_path, 'w') as f:
        for row in rows:
            R = np.load(row['tmp_path'])
            key = f"{row['Age']}/J{row['J_max']:.2f}_G{row['G']:.2f}{row['suffix']}"
            f.create_dataset(key, data=R, compression='gzip', compression_opts=4)
            # Store parameters as metadata
            f[key].attrs['age']   = row['Age']
            f[key].attrs['J_max'] = row['J_max']
            f[key].attrs['G']     = row['G']

            on, off, strength, areas = row['inhi']
            f[key].attrs['inhi_on']       = on
            f[key].attrs['inhi_off']      = off
            f[key].attrs['inhi_strength'] = strength
            f[key].attrs['inhi_areas']    = str(areas) if areas else ''

            perm, lesioned_areas = row['lesion']
            f[key].attrs['lesion_perm']  = bool(perm)
            f[key].attrs['lesion_areas'] = str(lesioned_areas) if lesioned_areas else ''
            
            d_on, d_off, d_strength = row['distract']
            f[key].attrs['distract_on']       = d_on
            f[key].attrs['distract_off']      = d_off
            f[key].attrs['distract_strength'] = d_strength
            os.remove(row['tmp_path'])  # clean up

def Heatmap_gen(trials=100, mech={'FI', 'spine'}, n_jobs=-1,
                 inhi=[9, 9.5, 0.3, None], lesion=[False, None], distract=[9, 9.5, 0], Jcrit='cutFB'):
    """
    n_jobs: -1 = use all cores, -2 = all cores minus one, 8 = exactly 8, etc.
    inhi: [start(s), end(s), strength(nA), area(s)] — leave areas=None for no inhibition (default/base case)
    lesion: [perm(bool), area(s)] — leave area(s)=None for no lesion (default/base case)
    """
    os.makedirs('data/tmp', exist_ok=True)

    ages    = ['Y', 'MA', 'A']
    j_maxes = [round(i * 0.01 + 0.21, 2) for i in range(31)]
    Gs      = [round(j * 0.02 + 0.2,  2) for j in range(16)]

    tasks = list(product(ages, j_maxes, Gs))   # 1488 combinations
    print(f"Total tasks: {len(tasks)}")
    rows = Parallel(n_jobs=n_jobs, verbose=10, backend='loky')(
        delayed(_run_one)(age, j_max, G, mech, trials, inhi=inhi, lesion=lesion, distract=distract, Jcrit=Jcrit)
        for age, j_max, G in tasks)

    # Merge all tmp .npy files into one HDF5
    print('Merging the files')
    mech_str = '_'.join(mech) if isinstance(mech, set) else mech
    cond_suffix = _format_condition_suffix(inhi, distract, lesion)
    h5_path = f'data/heatmap_R_{mech_str}{cond_suffix}_{Jcrit}.h5'
    merge_to_hdf5(rows, h5_path)
    print(f"R saved to {h5_path}")

def compute_metric(R, metric):    
    # Remove zero trials
    mean_fr_E1_post = np.mean(R[-50:, :30], axis=0)       # Compute the mean fr at the end of the trial
    mean_fr_E2_post = np.mean(R[-50:, 30:60], axis=0)     # (n_areas,trials)
    fr_diff_post = mean_fr_E1_post - mean_fr_E2_post              # (n_areas,trials)    

    act_areas = np.count_nonzero(np.abs(fr_diff_post) > 5, axis=0) # Compute whether there is activity per area; (trials,)
    act_mask = act_areas > 0                          # Only consider nonzero trials; (trials,) bool

    R_a = R[:,:,act_mask]                             # We only want values where the boolean is true (areas active at the end of the trial)

    if R_a.shape[-1] == 0:
        match metric:
            case 'perc_act'  :     return 0, False
            case 'perc_nhall':     return 0, False
            case 'mean':           return 0, False
            case 'std':            return 0, False
            case _: raise ValueError(f"Unknown metric: {metric}")

    # Remove hallucinations, only consider active trials
    mean_fr_E1_pre_a = np.mean(R_a[30:40, :30], axis=0)      # Compute the mean fr before the stimulus
    mean_fr_E2_pre_a = np.mean(R_a[30:40, 30:60], axis=0)    # (n_areas,trials)
    fr_diff_pre_a  = mean_fr_E1_pre_a - mean_fr_E2_pre_a     # (n_areas,trials)

    mean_fr_E1_post_a = np.mean(R_a[-50:, :30], axis=0)        # Compute the mean fr after the stimulus
    mean_fr_E2_post_a = np.mean(R_a[-50:, 30:60], axis=0)      # (n_areas,trials)
    fr_diff_post_a    = mean_fr_E1_post_a - mean_fr_E2_post_a  # (n_areas,trials)
    hall_areas = np.count_nonzero( (np.abs(fr_diff_pre_a) > 5) | (fr_diff_post_a < -5), axis=0 ) # Compute whether there is activity per area; (trials,)
    # In some cases, there is activity in E1 or E2 before any stimulus is put in the system, that is a hallucination --> (np.abs(fr_diff_pre_a) > 5) 
    # There are cases where we see at the end of the trial that E2 is active, clearly this is due to hallucinations, we remove those as well --> (fr_diff_post_a < -5)

    no_hall_mask = hall_areas == 0   # Only consider non hallucinating trials (no activity before stimulus); (trials,) bool
    R_h_a = R_a[:, :, no_hall_mask]   # active AND non-hallucinating; we only want values where the boolean is true

    if R_h_a.shape[-1] == 0:   # can't compute anything on empty array
        match metric:
            case 'perc_act'   :   return sum(act_mask)*100/R.shape[-1],  False
            case 'perc_nhall' :   return sum(no_hall_mask)*100/R_a.shape[-1], False
            case 'mean':          return np.nan, True
            case 'std' :          return np.nan, True
            case _: raise ValueError(f"Unknown metric: {metric}")

    # Compute means
    mean_fr_E1_post = np.mean(R_h_a[-50:, :30], axis=0)    # Compute the mean fr at the end of the trial
    mean_fr_E2_post = np.mean(R_h_a[-50:, 30:60], axis=0)  # (n_areas,trials)
    fr_diff_post    = mean_fr_E1_post - mean_fr_E2_post               # (n_areas,trials)

    act_areas  = np.count_nonzero(np.abs(fr_diff_post) > 5, axis=0)*100/30 # (trials,)

    if R_h_a.shape[-1] < 0.4 * R_a.shape[-1]:
        match metric:
            case 'perc_act'  :     return sum(act_mask)*100/R.shape[-1], False
            case 'perc_nhall':     return sum(no_hall_mask)*100/R_a.shape[-1], False
            case 'mean':           return act_areas.mean(), True
            case 'std':            return act_areas.std(), True
            case _: raise ValueError(f"Unknown metric: {metric}")

    match metric:
        case 'perc_act'   :     return sum(act_mask)*100/R.shape[-1], False
        case 'perc_nhall' :     return sum(no_hall_mask)*100/R_a.shape[-1], False
        case 'mean':            return act_areas.mean() , False
        case 'std' :            return act_areas.std(), False
        case _: raise ValueError(f"Unknown metric: {metric}")

def compute_metric_dist(R, metric):    
    # Remove zero trials
    mean_fr_E1_post = np.mean(R[-50:, :30], axis=0)       # Compute the mean fr at the end of the trial
    mean_fr_E2_post = np.mean(R[-50:, 30:60], axis=0)     # (n_areas,trials)
    fr_diff_post = mean_fr_E1_post - mean_fr_E2_post              # (n_areas,trials)    

    act_areas = np.count_nonzero(np.abs(fr_diff_post) > 5, axis=0) # Compute whether there is activity per area; (trials,)
    act_mask = act_areas > 0                          # Only consider nonzero trials; (trials,) bool

    # Require sustained activity between cue and distractor (7-8 s window), 
    mean_fr_E1_inter = np.mean(R[70:80, :30], axis=0)       # Compute the mean fr between stim and dist
    mean_fr_E2_inter = np.mean(R[70:80:, 30:60], axis=0)    # (n_areas,trials)
    fr_diff_inter = mean_fr_E1_inter - mean_fr_E2_inter        # (n_areas,trials)    
    act_areas = np.count_nonzero(np.abs(fr_diff_inter) > 5, axis=0) # Compute whether there is activity per area; (trials,)
    act_mask &= act_areas > 0

    R_a = R[:,:,act_mask]                             # We only want values where the boolean is true (areas active at the end of the trial)

    if R_a.shape[-1] == 0:
        match metric:
            case 'perc_act'  :     return 0, False
            case 'perc_nhall':     return 0, False
            case 'mean':           return 0, False
            case 'resist':         return np.nan, False
            case 'std':            return 0, False
            case _: raise ValueError(f"Unknown metric: {metric}")

    # Remove hallucinations, only consider active trials
    mean_fr_E1_pre_a = np.mean(R_a[30:40, :30], axis=0)      # Compute the mean fr before the stimulus
    mean_fr_E2_pre_a = np.mean(R_a[30:40, 30:60], axis=0)    # (n_areas,trials)
    fr_diff_pre_a  = mean_fr_E1_pre_a - mean_fr_E2_pre_a     # (n_areas,trials)

    # E2 can legitimately win at the end of a distractor trial, so only the pre-cue check applies
    # But we also add a filter after stimulus and before distractor to remove E2 activity
    mean_fr_E1_post_a = np.mean(R_a[70:80:, :30], axis=0)        # Compute the mean fr after the stimulus
    mean_fr_E2_post_a = np.mean(R_a[70:80:, 30:60], axis=0)      # (n_areas,trials)
    fr_diff_post_a    = mean_fr_E1_post_a - mean_fr_E2_post_a  # (n_areas,trials)
    hall_areas = np.count_nonzero( (np.abs(fr_diff_pre_a) > 5) | (fr_diff_post_a < -5), axis=0 ) # Compute whether there is activity per area; (trials,)

    no_hall_mask = hall_areas == 0   # Only consider non hallucinating trials (no activity before stimulus); (trials,) bool
    R_h_a = R_a[:, :, no_hall_mask]   # active AND non-hallucinating; we only want values where the boolean is true

    if R_h_a.shape[-1] == 0:   # can't compute anything on empty array
        match metric:
            case 'perc_act'   :   return sum(act_mask)*100/R.shape[-1],  False
            case 'perc_nhall' :   return sum(no_hall_mask)*100/R_a.shape[-1], False
            case 'mean':          return np.nan, True
            case 'resist':        return np.nan, True
            case 'std' :          return np.nan, True
            case _: raise ValueError(f"Unknown metric: {metric}")

    # Compute means
    mean_fr_E1_post = np.mean(R_h_a[-50:, :30], axis=0)    # Compute the mean fr at the end of the trial
    mean_fr_E2_post = np.mean(R_h_a[-50:, 30:60], axis=0)  # (n_areas,trials)
    fr_diff_post    = mean_fr_E1_post - mean_fr_E2_post               # (n_areas,trials)

    act_areas  = np.count_nonzero(np.abs(fr_diff_post) > 5, axis=0)*100/30 # (trials,)

    mean_fr_E1_dist = np.mean(R_h_a[70:80, :30], axis=0)    # Compute the mean fr at the end of the trial
    mean_fr_E2_dist = np.mean(R_h_a[70:80, 30:60], axis=0)  # (n_areas,trials)
    fr_diff_pre_dist = mean_fr_E1_dist - mean_fr_E2_dist  # 7-8 s, (n_areas,trials)
    switched   = np.sign(fr_diff_pre_dist.mean(axis=0)) != np.sign(fr_diff_post.mean(axis=0))
    resist_val = 100 * (1 - switched.mean())

    if R_h_a.shape[-1] < 0.4 * R_a.shape[-1]:
        match metric:
            case 'perc_act'  :     return sum(act_mask)*100/R.shape[-1], False
            case 'perc_nhall':     return sum(no_hall_mask)*100/R_a.shape[-1], False
            case 'mean':           return act_areas.mean(), True
            case 'resist':         return resist_val , True
            case 'std':            return act_areas.std(), True
            case _: raise ValueError(f"Unknown metric: {metric}")

    match metric:
        case 'perc_act'   :     return sum(act_mask)*100/R.shape[-1], False
        case 'perc_nhall' :     return sum(no_hall_mask)*100/R_a.shape[-1], False
        case 'mean':            return act_areas.mean() , False
        case 'resist':          return resist_val , False
        case 'std' :            return act_areas.std(), False
        case _: raise ValueError(f"Unknown metric: {metric}")

METRIC_CFG = {
    'perc_act':    dict(label='% active trials',                       cmap='Purples',  vmin=0, vmax=100),
    'perc_nhall':  dict(label='% non-hallucinating trials',            cmap='Greens',  vmin=0, vmax=100),
    'mean':     dict(label='% Active Areas, mean',                  cmap='Blues',   vmin=0, vmax=100),
    'std':      dict(label='% Active Areas, std',                   cmap='Oranges', vmin=0, vmax=100),
    'resist':   dict(label='Population Dominance after distractor', cmap=LinearSegmentedColormap.from_list(
                                'orange_blue', ['tab:orange', 'tab:blue']), vmin=0, vmax=100)
}

def Plot_Heatmaps(mech, metric='mean', comb_plot=True,
                   inhi=[9, 9.5, 0.3, None],
                   distract=[9, 9.5, 0],
                   lesion=[False, None],
                   condition_label=None,
                   diff=False, Jcrit='cutFB'):

    cfg      = METRIC_CFG[metric]
    mech_str = '_'.join(mech) if isinstance(mech, set) else mech
    cond_suffix = _format_condition_suffix(inhi, distract, lesion)
    h5_path  = f'data/heatmap_R_{mech_str}{cond_suffix}_{Jcrit}.h5'

    if diff:
        base_suffix = _format_condition_suffix([9, 9.5, 0.3, None], [9, 9.5, 0], [False, None])
        h5_path_base = f'data/heatmap_R_{mech_str}{base_suffix}.h5'

    marked_pairs = [(0.468, 0.21), (0.26, 0.48), (0.23, 0.5)]
    full_ages    = {'Y': 'Young', 'MA': 'Middle Aged', 'A': 'Aged'}
    full_mech    = {
            'spine':                'Spine Count Aging Mechanism',
            'FI':                   'Neuronal Hyperexcitability Aging Mechanism',
            frozenset({'spine','FI'}) :             'Both Aging Mechanisms',
            frozenset({'spine','FI','longrange'}) :  'Three Aging Mechanisms'
            }
    if type(mech) == set:
        mech = frozenset(mech)

    ages    = ['Y', 'MA', 'A']
    j_maxes = [round(i * 0.01 + 0.21, 2) for i in range(31)]
    Gs      = [round(j * 0.02 + 0.2,  2) for j in range(16)]

    title_suffix = f" ({condition_label} - Base)" if condition_label else ""
    metric_label = f'Δ {cfg["label"]}' if diff else cfg['label']

    # --- diff mode: pre-compute diff grids for all ages up front, for shared color scale ---
    all_diffs = None
    if diff:
        all_diffs = {}
        with h5py.File(h5_path, 'r') as f_cond, h5py.File(h5_path_base, 'r') as f_base:
            for age in ages:
                records = []
                for J_max in j_maxes:
                    for G in Gs:
                        key_cond = f"{age}/J{J_max:.2f}_G{G:.2f}{cond_suffix}"
                        key_base = f"{age}/J{J_max:.2f}_G{G:.2f}{base_suffix}"
                        R_cond = f_cond[key_cond][:]
                        R_base = f_base[key_base][:]
                        val_cond, flag_cond = compute_metric(R_cond, metric)
                        val_base, flag_base = compute_metric(R_base, metric)
                        records.append({
                            'J_max': J_max, 'G': G,
                            'val': val_cond - val_base,
                            'flag': flag_cond or flag_base
                        })
                all_diffs[age] = pd.DataFrame(records)

        max_abs = max(
            grid['val'].replace([np.inf, -np.inf], np.nan).abs().max()
            for grid in all_diffs.values()
        )
        diff_vmax = 1 if not np.isfinite(max_abs) or max_abs == 0 else np.ceil(max_abs)

    if comb_plot:
        fig, axes = plt.subplots(1, 3, figsize=(22, 5), sharey=True, layout='constrained')
        if diff:
            fig.suptitle(f'{metric_label}{title_suffix}', fontsize=14, x=0.5, ha='center')
        else:
            fig.suptitle(f'{full_mech[mech]}, {metric_label}{title_suffix}', fontsize=14, x=0.5, ha='center')

    # In diff mode we already have all data in `all_diffs`; in absolute mode we stream from h5.
    # Use a null-context so the same loop body works whether or not a live h5.File is open.
    f_ctx = h5py.File(h5_path, 'r') if not diff else contextlib.nullcontext()

    with f_ctx as f:
        for i, age in enumerate(ages):
            if diff:
                grid = all_diffs[age]
            else:
                records = []
                for J_max in j_maxes:
                    for G in Gs:
                        key = f"{age}/J{J_max:.2f}_G{G:.2f}{cond_suffix}"
                        R   = f[key][:]
                        val, flag = compute_metric(R, metric) if distract[-1] == 0 else compute_metric_dist(R,metric)
                        records.append({'J_max': J_max, 'G': G, 'val': val, 'flag': flag})
                grid = pd.DataFrame(records)

            Z      = grid.pivot(index='G', columns='J_max', values='val')
            Z_flag = grid.pivot(index='G', columns='J_max', values='flag')
            x      = Z.columns.values
            y      = Z.index.values
            Zvals  = np.ma.masked_invalid(Z.values.astype(float))
            Zflag  = Z_flag.values.astype(bool)

            if diff:
                cmap = plt.get_cmap('RdBu_r').copy()
                vmin, vmax = -diff_vmax, diff_vmax
            else:
                cmap = plt.get_cmap(cfg['cmap']).copy()
                vmin, vmax = cfg['vmin'], cfg['vmax']
            cmap.set_bad(color='white' if metric =='resist' else 'black')

            
            if not comb_plot:
                fig, ax = plt.subplots(1, 1, figsize=(7, 4))
                subtitle = f"\n{metric_label} of {full_mech[mech_str]}{title_suffix}" if diff \
                           else f"\nwith {full_mech[mech_str]}{title_suffix}"
                ax.set_title(f"{full_ages[age]} WM Network{subtitle}", fontsize=13)
            else:
                ax = axes[i]
                ax.set_title(f"{full_ages[age]} WM Network", fontsize=13)

            im = ax.pcolormesh(x, y, Zvals, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)

            flag_rows, flag_cols = np.where(Zflag)
            ax.scatter(x[flag_cols], y[flag_rows], marker='x', color='black', s=60, linewidths=1.2, zorder=5)

            ax.set_xlabel(r"$J_{max}$")
            if i == 0 or not comb_plot:
                ax.set_ylabel("G")

            if not comb_plot:
                cbar = fig.colorbar(im, ax=ax, shrink=0.9)
                if not diff:
                    cbar.set_ticks(np.linspace(cfg['vmin'], cfg['vmax'], 5))
                cbar.set_label(metric_label)


            # Marked pairs + regime annotations only make sense for absolute dynamics
            if not diff or metric!='resist':
                for (jm, g) in marked_pairs:
                    ax.plot(jm, g, marker='x', markersize=5, color='black')
                    ax.annotate(f"({jm}, {g})", xy=(jm, g), xytext=(5, -5),
                                textcoords='offset points', fontsize=7, color='black')

            ax.axvline(x=0.4653, color='black', linestyle='--', alpha=0.75)

            if age == 'Y':

                if not diff:
                    ax.plot((0.22, 0.4653), (0.48, 0.3955), color='black', linestyle='--', alpha=0.75)
                    ax.annotate('Distributed WM Regime', xy=(0.35, 0.49), xytext=(0.3, 0.35),
                                arrowprops=dict(arrowstyle='->'))
                    ax.annotate('Localized WM Regime',   xy=(0.49, 0.25), xytext=(0.3, 0.25),
                                arrowprops=dict(arrowstyle='->'))

            if not comb_plot:
                plt.tight_layout()
                prefix = 'heatmap_diff' if diff else 'heatmap'
                plt.savefig(f'Fig/Heatmaps/{prefix}_{age}_{mech_str}_{metric}{cond_suffix}.svg',
                            dpi=150, bbox_inches='tight')
                plt.show()

    if comb_plot:
        cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)   # one bar spanning all axes
        if not diff:
            cbar.set_ticks(np.linspace(cfg['vmin'], cfg['vmax'], 5))
        cbar.set_label(metric_label)
        prefix = 'heatmaps_diff' if diff else 'heatmaps'
        plt.savefig(f'Fig/Heatmaps/{prefix}_{mech_str}_{metric}{cond_suffix}_{Jcrit}.svg',
                     dpi=150, bbox_inches='tight')
        plt.show()

## Cortical Areas
def plot_cortical_map_overlay(mech,J_max,G,
                            inhi=[9, 9.5, 0.3, None],
                            distract=[9, 9.5, 0],
                            lesion=[False, None],
                            condition_label=None):
    """
    inhi/lesion: same spec as Heatmap_gen — leave at defaults for the base case
                 (no inhibition, no lesion), matching the .h5 file produced without them.
    condition_label: optional short string appended to plot titles (e.g. "prefrontal inhibition"),
                      purely cosmetic — doesn't affect what data is loaded.
    """
    
    age_jitter = {'Y': 0, 'MA': 0.01, 'A': 0.02}
    mech_str = '_'.join(mech) if isinstance(mech, set) else mech
    cond_suffix = _format_condition_suffix(inhi, distract, lesion)
    h5_path  = f'data/heatmap_R_{mech_str}{cond_suffix}.h5'
    title_suffix = f" ({condition_label})" if condition_label else ""

    ages      = ['Y', 'MA', 'A']
    full_ages = {'Y': 'Young', 'MA': 'Middle Aged', 'A': 'Aged'}
    age_colors = {'Y': 'tab:blue', 'MA': 'tab:orange', 'A': 'tab:green'}

    region_names = ['Occipital', 'Parietal', 'Temporal', 'Frontal', 'Prefrontal']
    _B = bm.BuildModel(J_max=0.3)
    region_idx = [_B.occipital, _B.parietal, _B.temporal, _B.frontal, _B.prefrontal]

    N           = len(region_names)
    angles      = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_plot = angles + [angles[0]]
    # linestyles  = ['-', '--', '-.', ':']

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(np.linspace(0, 2*np.pi, 200), [r]*200,
                color='gray', linewidth=0.5, linestyle=':', alpha=0.4)

    with h5py.File(h5_path, 'r') as f:
            for age in ages:
                key = f"{age}/J{J_max:.2f}_G{G:.2f}{cond_suffix}"
                if key not in f:
                    print(f"Warning: key {key} not found, skipping.")
                    continue
                R = f[key][:]

                # Active mask
                diff_post = np.mean(R[-50:, :30], axis=0) - np.mean(R[-50:, 30:60], axis=0)
                act_mask  = np.count_nonzero(np.abs(diff_post) > 5, axis=0) > 0

                # Hallucination mask
                diff_pre   = np.mean(R[30:40, :30], axis=0) - np.mean(R[30:40, 30:60], axis=0)
                no_hall_mask  = np.count_nonzero( (np.abs(diff_pre) > 5) | (diff_post < -5), axis=0 ) == 0

                R_valid = R[:, :, act_mask & no_hall_mask]
                if R_valid.shape[-1] == 0:
                    # print(f"Warning: no valid trials for {key}, skipping.")
                    plt.close(fig)
                    return 0

                diff  = np.mean(R_valid[-50:, :30], axis=0) - np.mean(R_valid[-50:, 30:60], axis=0)
                fracs = [float(np.mean(np.abs(diff[idx]) > 5) + age_jitter[age]) for idx in region_idx]

                color = age_colors[age]
                ax.plot(angles_plot, fracs + [fracs[0]], color=color, linewidth=2)
                ax.fill(angles_plot, fracs + [fracs[0]], color=color, alpha=0.07)
                ax.scatter(angles, fracs, color=color, s=35, zorder=5)

    ax.set_thetagrids(np.degrees(angles), region_names, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8, color='gray')
    ax.set_rlabel_position(22.5)

    age_handles = [Line2D([0],[0], color=age_colors[a], linewidth=2,
                          label=full_ages[a]) for a in ages]

    leg1 = ax.legend(handles=age_handles, loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=9, title='Age', framealpha=0.7)
    ax.add_artist(leg1)
    ax.set_title(f'Cortical activity, $J_{{max}}$={J_max},G={G}\n {title_suffix}', va='bottom', fontsize=12, pad=20)
    # plt.savefig(f'Fig/.svg',
                    #  dpi=150, bbox_inches='tight')
    plt.tight_layout()

    plt.show()