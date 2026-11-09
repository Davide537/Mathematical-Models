# --- Standard library ---
import os

# Must be set BEFORE importing numpy
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# --- Third-party ---
import matlab.engine
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import brentq, curve_fit

def aging_a_e(perc):
    # ── Data ─────────────────────────────────────────────────────────────
    full_df = pd.read_excel("data/EPHYS_Aver_byMonkey.xlsx")
    df = full_df.iloc[:, 14:22].copy()  # Only look at the FI and rheobase data
    rb = full_df[['Age Group', 'rheobase']].copy()

    age_order = ["Y", "MA", "Aged"]

    # Split into ages, averaging
    df["Age Group"] = pd.Categorical(full_df["Age Group"], categories=age_order, ordered=True)
    rb["Age Group"] = pd.Categorical(rb["Age Group"], categories=age_order, ordered=True)
    rb["rheobase"] = rb["rheobase"] / 1000

    group_mean = df.groupby("Age Group",observed=False).mean()
    # group_sem = df.groupby("Age Group",observed=False).sem()
    rheo_mean = rb.groupby("Age Group",observed=False).mean()
    # rheo_sem = rb.groupby("Age Group",observed=False).sem()

    # Take the Y monkey as baseline
    young_fi = group_mean.loc["Y"].values
    young_rheo = rheo_mean.loc["Y"].values.squeeze()

    # ── Model ────────────────────────────────────────────────────────────
    a_Y, b_Y, d_E = 135, 54, 0.308

    def phi_E(I, a_E, b_E):
        x = a_E * I - b_E
        return x / (1 - np.exp(-d_E * x))

    # 1 Hz rheobase definition
    def rheobase_1hz(a_E, b_E, I_min=0.0, I_max=2.0):
        def f(I):
            return phi_E(I, a_E, b_E) - 1.0
        try:
            return brentq(f, I_min, I_max)
        except ValueError:
            return np.nan

    # Define the Y monkey rheobase
    I_rheo_Y = rheobase_1hz(a_Y, b_Y)

    # Define the Y monkey curve
    xc = np.linspace(0, 1, 500)
    y_young = phi_E(xc, a_Y, b_Y)

    # ── Fit ──────────────────────────────────────────────────────────────
    results = {"Y": {"a": a_Y, "b": b_Y, "label": "Young"}}

    for g in ["MA", "Aged"]:
        # Compute percent change for MA and A with Y as baseline
        pct_fi = np.mean((group_mean.loc[g].values - young_fi) / young_fi * 100)
        pct_rheo = (rheo_mean.loc[g].values.squeeze() - young_rheo) / young_rheo * 100

        # Define the target values of the curves and rheobase with MA and A aging
        y_target = y_young * (1 + pct_fi * perc/ 100)
        # print(y_target)
        target_rheo = I_rheo_Y * (1 + pct_rheo * perc / 100)

        # (starting guess)
        a0 = a_Y * (1 + pct_fi / 100)

        # for each a, solve b such that rheobase matches target
        def solve_b(a_E):
            def err(b):
                r = rheobase_1hz(a_E, b)
                return r - target_rheo
            try:
                return brentq(err, 0.1, 500)
            except ValueError:
                return b_Y
            
        def constrained_model(I, a_E):
            b_E = solve_b(a_E)
            return phi_E(I, a_E, b_E)

        try:
            popt, _ = curve_fit(
                constrained_model,
                xc,
                y_target,
                p0=[a0],
                bounds=([0], [500]),
                maxfev=10000
            )
            a_E = popt[0]
        except RuntimeError:
            a_E = a0

        b_E = solve_b(a_E)

        results[g] = {"a": a_E, "b": b_E}

        # print(f"{proper_age_labels[g]}: a_E = {a_E:.3f}, b_E = {b_E:.3f}")
    return results

class BuildModel:
    def __init__(self, J_max = 0.42, G = 0.48, age = "Y", aging_mechanism = None, Jcrit='cutFB', ALPHA=1):
    
        if aging_mechanism == None:
            aging_mechanism = set()
        elif 'FI' not in aging_mechanism and 'spine' not in aging_mechanism and 'longrange' not in aging_mechanism:
            raise Exception("Aging factor naming error")
        
        ######## ANATOMICAL DATA #########
        subgraph_data = sio.loadmat('data/subgraphData30.mat')
        fln = subgraph_data['fln']  # FLN (fraction of labelled neurons)
        sln = subgraph_data['sln']  # SLN (supragranular layered neurons)

        area_list = np.array(['V1','V2','V4','DP','MT','8m','5','8l','2','TEO','F1','STPc','7A','46d','10','9/46v','9/46d','F5','TEpd','PBr','7m','LIP','F2','7B','ProM','STPi','F7','8b','STPr','24c'])
        self.num_areas = fln.shape[0]

        # DEFINE SPINE HIERARCHY:
        # SPINE COUNT DATA (Elston, several papers):
        spinec = np.array([643, 1201, 2429, -10, 2077, 3200, 4689, 3200, -10, 4812, -10, 8337, 2572, 6600,  # 46d
            6488, 7800, 7800, -10, 6200, -10, 2294, 2316, -10, 6841, -10, 8337, -10, -10, 8337, 6825])  #24c
        # AGE FACTOR:
        # Level the age factors for a young monkey Y
        AF = np.ones((self.num_areas,)) 
        AF[[13,14,15,16,18,29]] = AF[[13,14,15,16,18,29]] * 1.15  # 5yo, 15% increase
        AF[[5,7,20]] = AF[[5,7,20]] * 1.30  # 10yo, 30% increase
        spinec = spinec * AF  # age correction
        hierVals2 = spinec / np.max(spinec)
        # Now the ones based on hierarchical positions:
        hierVals2[spinec < 0] = np.squeeze(subgraph_data['hierVals'])[spinec < 0]

        # SWITCH TO CORRECT ORDER
        self.spine_order = np.argsort(hierVals2, kind='stable')   # cort order -> spine order
        self.cort_order  = np.argsort(self.spine_order)           # spine order -> cort order

        self.hierarchy = hierVals2[self.spine_order]
        self.area_list = area_list[self.spine_order]
        fln = fln[np.ix_(self.spine_order, self.spine_order)]
        sln = sln[np.ix_(self.spine_order, self.spine_order)]

        lobes = { 'occipital':  [0,1,2,4],                 # V1, V2, V4, MT        
                    'parietal':   [3,6,8,12,20,21,23],       # DP, 5, 2, 7A, 7m, LIP, 7B
                    'temporal':   [9,11,18,19,25,28],        # TEO, STPc, TEpd, PBr, STPi, STPr
                    'frontal':    [10,13,14,17,22,24,26,29], # F1, 46d, 10, F5, F2, ProM, 24c, F7
                    'prefrontal': [5,7,15,16,27]             # 8m, 8l, 9/46v, 9/46d, 8b
                    }
        
        for name, ids in lobes.items():
            setattr(self, name, np.sort(self.cort_order[ids]))        

        frontal = np.array([5,7,10,13,14,15,16,17,22,24,26,27]);  # areas in frontal lobe
        frontal = self.cort_order[frontal] 
    
        # Make feedback connections target more inhibitory cells
        lambdai = np.ones((self.num_areas,self.num_areas)) - sln # for long-range connections

        def idx(*names):
            if names is None:
                return np.array([], dtype=int)
            if isinstance(names, str):
                names = [names]
            return np.where(np.isin(self.area_list, names))[0]
        self.idx = idx

        # We cap the inhibitory long-range strength to FEF from frontal areas.
        maxi = 0.4
        idx_mask = lambdai[frontal[0],frontal] > maxi
        lambdai[frontal[0],frontal[idx_mask]] = maxi
        idx_mask = lambdai[frontal[1],frontal] > maxi
        lambdai[frontal[1],frontal[idx_mask]] = maxi

        ######## PARAMETERS ########

        # Set up simulation parameters
        self.dt = 0.5 * 1e-3  # timestep [s]
        self.dt_rate = 10e-3  # store firing rate every 10 ms
        self.num_pops = 3

        # Time constants
        self.tau_NMDA = 60 * 1e-3  # s
        self.tau_GABA = 5 * 1e-3  # s
        self.tau_AMPA = 2 * 1e-3  # s
        self.tau_r = 2 * 1e-3  # s

        # Kinetic parameters of NMDA and GABA
        self.gamma = 1.282  # unitless
        self.gamma_i = 2.  # unitless

        # f-I curve parameters - E populations
        self.a_E = 135*np.ones(self.num_areas) # Hz/nA
        self.b_E = 54. # Hz
        self.d_E = 0.308  # s

        # f-I curve parameters - I populations
        self.g_I = 4.  # unitless
        self.c_1 = 615.  # Hz/nA
        self.c_0 = 177 # Hz
        self.r0_I = 5.5  # Hz

        if 'FI' in aging_mechanism:
            a_E_by_age = {
                        'Y': 135 * np.ones(self.num_areas),
                        'MA': np.linspace(135.5, 139.74557063887, self.num_areas),
                        'A': np.linspace(136, 142.24143888819532, self.num_areas),
                        }
            try:
                self.a_E = a_E_by_age[age] # Hz/nA

            except KeyError:
                raise ValueError(f"Unknown age: {age}")        

        # Local connectivity matrix strengths
        # Strength of connections from E cells
        self.g_E_self_min = 0.21  # nA
        J_max = J_max  # nA
        self.g_E_cross = 0.0107  # nA

        # Strength of connections from I cells
        self.g_E_I = -0.31  # nA
        self.g_I_I = -0.12  # nA

        # Excitatory input constant ("John Murray's term")
        self.J_0 = 0.2112  # nA - Matches value in John Murray's thesis for g_E_self - g_E_cross

        # Global coupling strength
        G = G  # unitless

        # Background inputs
        self.I0_E = 0.3294  # nA
        self.I0_I = 0.26  # nA

        ######## HIERARCHY ########        
        self.g_E_self = self.g_E_self_min + (J_max - self.g_E_self_min) * self.hierarchy 
        # Scale the E to I synapses to lie along the hierarachy (ensuring that the spontaneous activity fixed point is the same across areas)
        zeta = self.c_1 * self.tau_GABA * self.gamma_i / (self.g_I - self.g_I_I * self.tau_GABA * self.gamma_i * self.c_1)  # Eq. 14
        self.g_I_E = (self.J_0 - self.g_E_self - self.g_E_cross) / (2 * zeta * self.g_E_I)    # Eq. 17
        ####### LONG-RANGE CONNECTIONS ########

        # Long-range coupling strength
        mu_EE = G  
        Z = 2 * self.c_1 * self.tau_GABA * self.gamma_i * self.g_E_I / (self.c_1 * self.tau_GABA * self.gamma_i * self.g_I_I - self.g_I)    # constant to maintain EI balance; Eq. 21
        mu_IE = ALPHA * G / Z

        # Compress FLN (fraction of labelled neurons)
        if Jcrit == 'cutFB' and J_max > 0.4653:
            # Remove feedback (keep lower triangle only)
            fln = np.tril(fln)
        elif Jcrit == 'fullcutFB':
            fln = np.tril(fln)

        fln_squish = 1.2 * np.power(fln,0.3)    
        fln_rowtotal = np.sum(fln_squish,axis=1)        
        fln_squishnorm = fln_squish / np.maximum(fln_rowtotal[:, np.newaxis] , 0.1)

        # Make feedforward connections target more excitatory cells
        W_E = fln_squishnorm * sln
        W_I = fln_squishnorm * lambdai
                
        ####### CONNECTIVITY MATRIX ########

        # Compute effective connectivity matrix for excitatory connections
        LR_EE_gradient = self.g_E_self / np.max(self.g_E_self)
        W_E_eff = mu_EE * LR_EE_gradient[:, np.newaxis] * W_E 

        if 'spine' in aging_mechanism:
            D_by_age = {
                'Y': np.ones_like(self.hierarchy),
                'MA': 0.825 - 0.009 * np.arange(self.num_areas),
                'A':  0.874 - 0.017 * np.arange(self.num_areas),
            } 

            try:
                D = D_by_age[age]
            except KeyError:
                raise ValueError(f"Unknown age: {age}")

            self.g_E_self += (self.g_E_self - self.g_E_self_min) * (D-1) 
            self.D = D
        
        # Compute effective connectivity matrix for inhibitory connections
        LR_IE_gradient = self.g_I_E / np.max(self.g_I_E)
        W_I_eff = mu_IE * LR_IE_gradient[:, np.newaxis] * W_I 

        if 'longrange' in aging_mechanism:
            loss_by_age = {'Y': 0.0, 'MA': 0.10, 'A': 0.20}
            try:
                scale = 1.0 - loss_by_age[age]
            except KeyError:
                raise ValueError(f"Unknown age: {age}")

            # Each bundle: (source_areas, target_areas) — applied bidirectionally
            bundles = [
                # SLF II: parietal <-> frontal/prefrontal
                (idx(['7A','LIP','DP','7B']),
                idx(['46d','9/46d','8l','8m','8b','F2','F7'])),
                # Cingulum bundle
                (idx(['10','9/46v','8m','24c']),
                idx(['7m'])),
                # SLF III
                (idx(['7B','STPc','STPi']),
                idx(['F5','8l','9/46v','ProM'])),
                # Uncinate fasciculus
                (idx(['10']),
                idx(['STPr','PBr'])),
                # Fronto-cingulate
                (idx(['24c']),
                idx(['F1','F2','F7','8m'])),
            ]
            for src, tgt in bundles:
                # src -> tgt
                W_E_eff[np.ix_(tgt, src)] *= scale
                W_I_eff[np.ix_(tgt, src)] *= scale
                # tgt -> src (bidirectional)
                W_E_eff[np.ix_(src, tgt)] *= scale
                W_I_eff[np.ix_(src, tgt)] *= scale

            D_by_age = {
                'Y': np.ones_like(self.hierarchy),
                'MA': 0.825 - 0.009 * np.arange(self.num_areas),
                'A': 0.874 - 0.017 * np.arange(self.num_areas),
            }

            try:
                D = D_by_age[age]
            except KeyError:
                raise ValueError(f"Unknown age: {age}")

            # D = D[np.argsort(self.hierarchy)]
            W_E_eff *= D[:, np.newaxis]
            W_I_eff *= D[:, np.newaxis]

        np.fill_diagonal(W_E_eff, self.g_E_self)  # include self population weights in effective connectivity matrix
        np.fill_diagonal(W_I_eff, self.g_I_E)

        # Construct joint connectivity matrix for all 3 populations (E1, E2, and I)
        self.W_eff = np.zeros((self.num_areas*self.num_pops,self.num_areas*self.num_pops))
        self.W_eff[:self.num_areas,:self.num_areas] = W_E_eff  # excitatory to excitatory connections E1 -> E1
        self.W_eff[self.num_areas:2*self.num_areas,self.num_areas:2*self.num_areas] = W_E_eff  # excitatory to excitatory connections E2 -> E2
        np.fill_diagonal(self.W_eff[:self.num_areas,self.num_areas:2*self.num_areas], self.g_E_cross)  # excitatory to excitatory connections E1 -> E2
        np.fill_diagonal(self.W_eff[self.num_areas:2*self.num_areas,:self.num_areas], self.g_E_cross)  # excitatory to excitatory connections E2 -> E1
        self.W_eff[2*self.num_areas:,:self.num_areas] = W_I_eff  # excitatory to inhibitory connections E1 -> I
        self.W_eff[2*self.num_areas:,self.num_areas:2*self.num_areas] = W_I_eff  # excitatory to inhibitory connections E2 -> I
        np.fill_diagonal(self.W_eff[:self.num_areas,2*self.num_areas:], self.g_E_I)  # inhibitory to excitatory connections I -> E1
        np.fill_diagonal(self.W_eff[self.num_areas:2*self.num_areas,2*self.num_areas:], self.g_E_I)  # inhibitory to excitatory connections I -> E2
        np.fill_diagonal(self.W_eff[2*self.num_areas:,2*self.num_areas:], self.g_I_I)  # inhibitory to inhibitory connections I -> I
        pass

    def con_matrix(self, Mat = None):
        # Connectivity matrix for excitatory and inhibitory neurons
        if Mat is not None:
            self.W_eff = Mat
        plt.figure(figsize=(12, 8))
        W = self.W_eff
        k = self.num_areas
        
        labels = ["E1", "E2", "IN"]
        centers = [k/2, k + k/2, 2*k + k/2]

        def add_block_lines(ax):
            ax.axvline(k, linestyle=':', color='black', alpha=0.1)
            ax.axvline(2*k, linestyle=':', color='black', alpha=0.1)
            ax.axhline(k, linestyle=':', color='black', alpha=0.1)
            ax.axhline(2*k, linestyle=':', color='black', alpha=0.1)

        # --- First subplot ---
        ax1 = plt.subplot(2, 2, 1)
        im1 = ax1.imshow(W, cmap='RdBu', vmin=-np.max(W), vmax=np.max(W))
        plt.colorbar(im1, ax=ax1)
        ax1.set_title('Connectivity matrix W')

        ax1.set_xticks(centers)
        ax1.set_yticks(centers)
        ax1.set_xticklabels(labels)
        ax1.set_yticklabels(labels)
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top')
        add_block_lines(ax1)

        # --- Second subplot ---
        ax2 = plt.subplot(2, 2, 2)
        W_eff_ = W.copy()

        np.fill_diagonal(W_eff_, 0)
        np.fill_diagonal(W_eff_[:2*self.num_areas, self.num_areas:], 0)
        np.fill_diagonal(W_eff_[self.num_areas:, :2*self.num_areas], 0)
        np.fill_diagonal(W_eff_[:self.num_areas, 2*self.num_areas:], 0)
        np.fill_diagonal(W_eff_[2*self.num_areas:, :self.num_areas], 0)

        im2 = ax2.imshow(W_eff_, cmap='RdBu', vmin=-np.max(W_eff_), vmax=np.max(W_eff_))
        plt.colorbar(im2, ax=ax2)
        ax2.set_title('Connectivity matrix W (inter-area only)')

        ax2.set_xticks(centers)
        ax2.set_yticks(centers)
        ax2.set_xticklabels(labels)
        ax2.set_yticklabels(labels)
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')
        add_block_lines(ax2)
        plt.savefig('Fig/W.svg', dpi=150, bbox_inches='tight')
        plt.show()

    def runsim(self, stim=[4, 4.5, 0.3, 'V1'], distract=[3.5, 4, 0], inhi=[9, 9.5, 0.3, None],
            trial_length=10, lesion=[False, None], trials=1, save_fac=1,
            store_S=False, seed=None, dtype=np.float32):
        # Stim/distract.inhi: [start (s), end (s), strenght (nA), area(s) ] of stimulus or distractor or inhibition
        # Lesioned areas    [Bool perm [switch on and off], list lesioned area(s)]

        # You can run multiple trials (trials parameter) and varry the stim/dis/inhi strenght but none of the other parameters sadly
        # dtype: float32 for speed; pass np.float64 to reproduce the old precision (for validation)
        # store_S: self.S is only stored if True (set to None otherwise)
        # seed: makes the noise reproducible

        f = dtype
        k = self.num_areas
        k2 = 2 * k
        n = k * self.num_pops
        dt = self.dt

        # Trial parameters
        self.trial_length = trial_length  # trial length (s)
        self.num_iterations = int(self.trial_length / dt)
        # Lesioned areas    [Bool perm, lesioned area(s)]
        self.perm, self.lesioned_areas = lesion
        lesion_indexes = self.idx(self.lesioned_areas)
        les_E1, les_E2, les_I = lesion_indexes, lesion_indexes + k, lesion_indexes  # rows in r_E / S_NMDA / r_I / S_GABA

        self.trials = trials  # number of trials computed at the same time with same parameters
        ####### EXT INPUTS AND NOISE ########

        I0 = np.concatenate((self.I0_E * np.ones(k2), self.I0_I * np.ones(k)))
        I0 = np.ascontiguousarray(np.repeat(I0[:, None], self.trials, axis=1), dtype=f)  # (90, trials)

        # Let's apply external stimulation to some area populations E1 & E2
        # Stimulus and distractor
        self.stim_on, self.stim_off, self.stim_strength, self.stim_areas = stim
        self.distract_on, self.distract_off, self.distractor_strength = distract
        self.inhi_on, self.inhi_off, self.inhi_strength, self.inhi_areas = inhi

        # Locate where (area index) the stimulus/distraction/inhibition must be applied
        stim_idx = self.idx(self.stim_areas)
        dist_idx = stim_idx + k
        inhi_idx = self.idx(self.inhi_areas) + 2 * k

        # Locate when (time index) the stimulus/distraction/inhibition must be applied
        # Windows are stored as (on_step, off_step, rows, value); the loop starts at step 1.
        # Windows with off <= on or no target rows are skipped (same behaviour as the old if-checks).
        windows = []
        for on, off, rows, val in [(self.stim_on, self.stim_off, stim_idx, self.stim_strength),
                                (self.distract_on, self.distract_off, dist_idx, self.distractor_strength),
                                (self.inhi_on, self.inhi_off, inhi_idx, self.inhi_strength)]:
            on_i, off_i = max(int(on / dt), 1), int(off / dt)
            if on_i < off_i and len(rows) > 0:
                windows.append((on_i, off_i, rows, val))
        starts = {w[0]: [] for w in windows}
        stops = {w[1]: [] for w in windows}
        for on_i, off_i, rows, val in windows:
            starts[on_i].append((rows, val))
            stops[off_i].append(rows)

        # I_const = I0 + I_ext (one buffer). I0 is constant and I_ext only changes at window edges,
        # so I_const is only touched there instead of adding two arrays every step.
        I_const = I0.copy()

        # Let's set up the noise. We will model the noise as an Ornstein-Uhlenbeck process.
        rng = np.random.default_rng(seed)
        std_noise = 0.005  # nA   # Noise std dev
        noise_scale = float(np.sqrt(dt / self.tau_AMPA) * std_noise)  # scalar noise magnitude
        decay = float(1 - dt / self.tau_AMPA)                          # OU decay per step

        # ---- Constants hoisted out of the loop (python floats so float32 arrays stay float32) ----
        k_r = float(dt / self.tau_r)
        keep_r = 1.0 - k_r                                  # r += k_r*(target - r)  ->  r = keep_r*r + k_r*target
        keep_N = float(1 - dt / self.tau_NMDA)              # S_NMDA decay per step
        keep_G = float(1 - dt / self.tau_GABA)              # S_GABA decay per step
        gdt_N = float(self.gamma * dt)                      # gamma*dt folded into one constant
        gdt_G = float(self.gamma_i * dt)
        a2 = np.tile(self.a_E, 2)[:, None].astype(f)        # (60, 1), done once instead of every step
        b_E, d_E = float(self.b_E), float(self.d_E)
        cI_slope = float(self.c_1 / self.g_I)
        cI_off = float(self.r0_I - self.c_0 / self.g_I)
        W = np.ascontiguousarray(self.W_eff, dtype=f)

        ########### Timing #################

        # Preassign rate and synapse matrices
        r = np.zeros((n, self.trials), dtype=f)
        r_E = r[:k2]  # note that these slices of r are not copies but pointers to r
        r_I = r[k2:]
        S = np.zeros((n, self.trials), dtype=f)
        S_NMDA = S[:k2]  # note that these slices of S are not copies but pointers to S
        S_GABA = S[k2:]

        # Noise
        I_noise = np.zeros((n, self.trials), dtype=f)
        In_E = I_noise[:k2]  # only E rows are updated: inhibitory noise is always 0
        noise_buf = np.empty((k2, self.trials), dtype=f)

        # Create matrices in which we can store the currents
        I_total = np.zeros((n, self.trials), dtype=f)
        I_total_E = I_total[:k2]  # note that these slices of I_total are not copies but pointers to I_total
        I_total_I = I_total[k2:]

        # Scratch buffers (preallocated so the loop makes no temporaries)
        xn = np.empty((k2, self.trials), dtype=f)
        tmpE = np.empty((k2, self.trials), dtype=f)
        tmpI = np.empty((k, self.trials), dtype=f)
        tmpG = np.empty((k, self.trials), dtype=f)

        # Firing rate
        saving_counter = 1
        saving_factor = int(self.dt_rate // dt) * save_fac
        n_save = int(self.num_iterations // saving_factor)
        self.R = np.zeros((n_save, n, self.trials), dtype=np.float32)
        self.S = np.zeros((n_save, n, self.trials), dtype=np.float32) if store_S else None

        perm = self.perm
        for i_t in range(1, self.num_iterations):

            # Update noise - dims = num local pops x num areas (E rows only, no noise for inhibitory)
            rng.standard_normal(out=noise_buf, dtype=f)
            noise_buf *= noise_scale
            In_E *= decay
            In_E += noise_buf
            # Update external  current (only touched at the edges of the stimulus/distractor/inhibition windows)
            if i_t in stops:
                for rows in stops[i_t]:
                    I_const[rows] = I0[rows]
            if i_t in starts:
                for rows, val in starts[i_t]:
                    I_const[rows] = I0[rows] + val

            # Compute all the synaptic input using the joint connectivity matrix W_eff
            np.matmul(W, S, out=I_total)
            I_total += I_const   # I0 + I_ext
            I_total += I_noise

            # Update the firing rates of the two excitatory and the inhibitory populations
            # Excitatory: phi_E = x/(1-exp(-d x)) with x = a*I - b. Written with xn = -x:
            #   phi_E = xn / expm1(d*xn)   (stable for large |x|, no clip needed)
            np.multiply(a2, I_total_E, out=xn)
            np.subtract(b_E, xn, out=xn)          # xn = b - a*I = -x
            xn[xn == 0] = 1e-12                   # avoid 0/0 exactly at threshold (limit is 1/d)
            np.multiply(xn, d_E, out=tmpE)
            np.expm1(tmpE, out=tmpE)
            np.divide(xn, tmpE, out=tmpE)         # phi_E target
            tmpE *= k_r
            r_E *= keep_r
            r_E += tmpE                           # r_E = r_E + k_r*(phi_E - r_E)

            # Inhibitory: phi_I = max((1/g)*(c1*I - c0) + r0, 0)
            np.multiply(I_total_I, cI_slope, out=tmpI)
            tmpI += cI_off
            np.maximum(tmpI, 0, out=tmpI)         # phi_I target
            tmpI *= k_r
            r_I *= keep_r
            r_I += tmpI

            # Update the NMDA synapses: S += dt*(-S/tau + gamma*(1-S)*r_E)  (uses the updated r_E)
            np.subtract(1.0, S_NMDA, out=tmpE)
            tmpE *= r_E
            tmpE *= gdt_N
            S_NMDA *= keep_N
            S_NMDA += tmpE

            # Update the GABA synapses: S += dt*(-S/tau + gamma_i*r_I)  (uses the updated r_I)
            np.multiply(r_I, gdt_G, out=tmpG)
            S_GABA *= keep_G
            S_GABA += tmpG

            # Lesions
            if perm:                         # Silence all 6 rates and synapses
                r_E[les_E1] = 0
                r_E[les_E2] = 0
                r_I[les_I] = 0
                S_NMDA[les_E1] = 0
                S_NMDA[les_E2] = 0
                S_GABA[les_I] = 0

            # Store firing rates of all populations
            if i_t % saving_factor == 0:
                self.R[saving_counter, :] = r
                if store_S:
                    self.S[saving_counter, :] = S
                saving_counter += 1
        pass

    def plot_neural_activity(self, selection=None, rates=None, rates_info = [20, 4, 4.5], diff=False, paper_orientation=False, title = None):
        if rates is not None:
            self.R = rates
            self.dt_rate = self.R.shape[0]
            self.trials = self.R.shape[-1]
            self.trial_length, self.stim_on, self.stim_off = rates_info

        if selection==None:
            sel = np.arange(self.num_areas)
            R_E1 = self.R[:,:self.num_areas]        
            R_E2 = self.R[:,self.num_areas:2*self.num_areas]        
        else:
            sel = self.idx(selection)
            R_E1 = self.R[:,sel]        
            R_E2 = self.R[:,self.num_areas+sel]        

        # R_I = self.R[:,2*self.num_areas:]
        # t = np.arange(0, self.trial_length, self.dt_rate // self.save)
        t = np.linspace(0, self.trial_length, self.R.shape[0])

        fig = plt.figure(figsize=(16,3*len(sel)), dpi= 80, facecolor='w', edgecolor='k')
        plt.rcParams.update({'font.size': 12})


        for i in range(1,len(sel)+1):
            nu = 2 + 2 * paper_orientation
            ax = plt.subplot(len(sel),nu,i)
            ax.set_title(self.area_list[sel[i-1]])
            # Plot the rates for the E1 soma
            plt.subplots_adjust(hspace = 1)
            alp = max(0.2, 1 / (self.trials+1) ) if self.trials!=1 else 1
            if diff:
                plt.plot(t, R_E1[:,i-1] - R_E2[:,i-1],alpha=alp, color='tab:purple')
            else:
                plt.plot(t, R_E1[:,i-1],alpha=alp, color='tab:blue')
                plt.plot(t, R_E2[:,i-1],alpha=alp, color='tab:orange')
            
            # Plot the stimulation time
            max_R = max( np.max(R_E1[:,i-1]) , np.max(R_E2[:,i-1]) )
            plt.plot([self.stim_on,self.stim_off],[max_R , max_R],color='black',linewidth=5.0)
            
            # Place text above the black line
            # axes = plt.gca()
            if i==1:
                # axes.text(0.15, 1.1,'External stimulation to', transform=axes.transAxes, fontsize=10, verticalalignment='top')
                if selection!=None:
                    plt.ylabel('firing rate (Hz)')

            if not diff:
                legend_handles = [ mlines.Line2D([], [], color='tab:blue') ,
                                    mlines.Line2D([], [], color='tab:orange') ]
                plt.legend(legend_handles, ['E1', 'E2'])

            plt.xlabel('time (s)')
            if selection == None:
                plt.ylabel('firing rate (Hz)')
            # plt.ylim(0, 60) 
            # plt.savefig(f'Fig/hall_example_0.49_0.4.svg', dpi=150, bbox_inches='tight')
        fig.suptitle('Neural activity of selected areas')
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        if title != None:
            plt.savefig(f'Fig/{title}.svg', dpi=150, bbox_inches='tight')
        pass

    def firing_rates(self,rates=None, trial_length = None):
        if rates is not None:
            self.R = rates
            self.trials = self.R.shape[-1]
            self.trial_length = trial_length
        # plot parameters
        alp = max(0.1, 1 / self.trials)
        x = np.arange(1, self.num_areas + 1)

        # data
        R_E1 = self.R[:, :self.num_areas]
        delay_rate = np.mean(R_E1[-100:, :], axis=0)

        if self.trials == 1:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[12, 10])
            delay_rate = delay_rate.flatten()
            sort_idx = np.argsort(delay_rate)
            sorted_areas = [self.area_list[i] for i in sort_idx]
            # second plot
            ax2.scatter(x, delay_rate[sort_idx], s=75, color='tab:blue', alpha=alp)
            ax2.set_xlabel('Brain regions ordered by firing rate')
            ax2.set_xlim([0, self.num_areas + 1])
            ax2.set_xticks(range(1, self.num_areas + 1))
            ax2.set_xticklabels(sorted_areas, rotation=45)
            ax2.set_ylim([0, 50])
            ax2.set_ylabel('Mean firing rate during delay (Hz)')
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=[12, 5])
            x = np.broadcast_to(x[:, None], delay_rate.shape)
       

        # first plot
        ax1.scatter(x, delay_rate, s=75, color='tab:blue', alpha=alp)
        ax1.set_xlabel('Brain regions ordered by spine hierarchy')
        ax1.set_xlim([0, self.num_areas + 1])
        ax1.set_xticks(range(1, self.num_areas + 1))
        ax1.set_xticklabels(self.area_list, rotation=45)
        ax1.set_ylim([0, 50])
        ax1.set_ylabel('Mean firing rate during delay (Hz)')

        plt.rc('font', size=14)        
        fig.tight_layout()
        # fig.savefig('Fig/delay_activity_all_areas.svg')
        plt.show()
        pass 

    def raster(self, pops=1, diff=True):
        plt.figure(figsize=(15,3.8))
        maxR = np.max(abs(self.R).mean(axis=2))
        cmap = LinearSegmentedColormap.from_list("orange_blue", ["tab:orange", "white", "tab:blue"], N=256)

        if diff and pops==1:
            Rdiff = (self.R[:,:pops*self.num_areas] - self.R[:,pops*self.num_areas:2*pops*self.num_areas]).mean(axis=2).T
            plt.imshow(Rdiff, aspect="auto", origin="lower", cmap=cmap, vmax=maxR,vmin=-maxR)
            plt.yticks(np.arange(self.num_areas), self.area_list, size='xx-small')
        else:
            Rblock = self.R[:,:pops*self.num_areas].mean(axis=2).T
            plt.imshow(Rblock, aspect="auto", origin="lower", cmap=cmap, vmax=maxR,vmin=-maxR)
            ylabels = [['E1', 'E2' , 'I'][i//self.num_areas] if i % self.num_areas == 15 else '' for i in range(pops*self.num_areas)]
            plt.yticks(np.arange(0,pops*self.num_areas), labels=ylabels, size='xx-small')

        plt.axvline(int(self.stim_on / self.dt_rate), color='black', ls=':')
        plt.axvline(int(self.stim_off / self.dt_rate), color='black', ls=':')
        for i in range(1,pops):
            plt.axhline(i*self.num_areas-0.5, color='black', ls=':', alpha=0.2)
        plt.ylabel('Brain area')
        plt.xlabel('time (s)')

        plt.xticks(np.arange(0,self.R.shape[0]+100,100), labels=np.arange(0,self.trial_length+1,1))
        plt.colorbar(label='firing rate')


        plt.show()
        pass

    def brain3d(self, rates=None, all_areas=False):
        if rates is not None:
            self.R = rates
        k = self.num_areas
        perm = np.concatenate([self.cort_order, self.cort_order + k])  # E1 block, then E2 block
        R = self.R[:, :2*k][:, perm]           # (timestep, 60 cortical-order columns, trial)
        R = np.transpose(R, (2, 0, 1))         # (trial, timestep, areas)
        eng = matlab.engine.start_matlab("-nodesktop -nosplash -nodisplay")
        mrate = matlab.double(R.tolist())
        eng.brain3d(mrate, all_areas, nargout=0)
        eng.quit()
        pass

    def cortical_map(self, rates=None):
        if rates is not None:
            self.R = rates

        # Define lobe groupings
        regions = {
            'Occipital':   self.occipital,
            'Parietal':    self.parietal,
            'Temporal':    self.temporal,
            'Frontal':     self.frontal,
            'Prefrontal':  self.prefrontal,
        }

        # Compute mean firing rates over last 100 timesteps, averaged across trials
        R_E1 = self.R[-100:, :self.num_areas, :].mean(axis=0)           # (30, trials)
        R_E2 = self.R[-100:, self.num_areas:2*self.num_areas, :].mean(axis=0)  # (30, trials)
        active = np.abs(R_E1 - R_E2) > 5  # (30, trials)

        # Fraction of active areas per region
        labels = list(regions.keys())
        fracs = [active[lobe_idx].mean() for lobe_idx in regions.values()]

        # --- Radar setup ---
        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        fracs_plot  = fracs  + [fracs[0]]   # close the polygon
        angles_plot = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        ax.plot(angles_plot, fracs_plot, color='steelblue', linewidth=2, linestyle='solid')
        ax.fill(angles_plot, fracs_plot, color='steelblue', alpha=0.25)

        # Reference rings
        for r in [0.25, 0.5, 0.75, 1.0]:
            ax.plot(np.linspace(0, 2*np.pi, 200), [r]*200,
                    color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

        # Spoke labels with counts
        ax.set_thetagrids(np.degrees(angles), labels, fontsize=11)

        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8, color='gray')
        ax.set_rlabel_position(22.5)

        ax.set_title('Cortical activity', va='bottom', fontsize=13, pad=20)

        plt.tight_layout()
        # plt.show()
        pass

# mech = {'spine','FI'}

# kk = 0.4
# B = BuildModel(J_max=kk,Jcrit=None)
# B.con_matrix()
# B.runsim()
