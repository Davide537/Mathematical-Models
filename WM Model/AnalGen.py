import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from matplotlib.lines import Line2D
from scipy.optimize import brentq
from tqdm import tqdm
import BaseModel as bm


def phi_E(I, a_E, b_E, d_E):
    a_expanded = np.tile(a_E, 2)[:, None]   # shape (60, 1)
    return (a_expanded * I - b_E) / (1 - np.exp(np.clip(-d_E * (a_expanded * I - b_E), -700, 700)))

def phi_I(I, g_I, c_1, c_0, r0_I):
    return np.maximum((1 / g_I) * (c_1 * I - c_0) + r0_I, 0)

class Analytical_Search(bm.BuildModel):
    def __init__(self, J_max=0.42, G=0.48, age="Y", aging_mechanism=None, perc=0.4):
        super().__init__(J_max, G, age, aging_mechanism, perc)
        self.aging_mechanism = {aging_mechanism} if isinstance(aging_mechanism, str) else (aging_mechanism or set())        
        pass
    
    def _compute_J_IE(self, Js):
        zeta = self.c_1 * self.tau_GABA * self.gamma_i / (self.g_I - self.g_I_I * self.tau_GABA * self.gamma_i * self.c_1)
        return (self.J_0 - Js - self.g_E_cross) / (2 * zeta * self.g_E_I)

    def _area_params(self, Js, area):
        i = self.idx(area)
        J_IE = self._compute_J_IE(Js)
        if 'spine' in self.aging_mechanism:
            Js = Js + (Js - self.g_E_self_min) * (self.D[i].item() - 1)
        return self.a_E[i], J_IE, Js

    def _SC_quasi(self, SA, SB, Js, J_IE):
        # solve dS_C/dt = 0 for S_C given SA, SB held fixed (quasi-static elimination,
        # valid since tau_GABA << tau_NMDA)
        def f(SC):
            IC = J_IE*SA + J_IE*SB + self.g_I_I*SC + self.I0_I
            rC = phi_I(IC, self.g_I, self.c_1, self.c_0, self.r0_I)
            return self.tau_GABA * self.gamma_i * rC - SC
        return brentq(f, 0, 10, xtol=1e-10)

    def field(self, SA, SB, Js, area='9/46d'):
        i = self.idx(area)
        # reduced 2D vector field on the (S_A, S_B) plane
        J_IE = self._compute_J_IE(Js)
        SC = self._SC_quasi(SA, SB, Js, J_IE)

        IA = Js*SA + self.g_E_cross*SB + self.g_E_I*SC + self.I0_E
        IB = self.g_E_cross*SA + Js*SB + self.g_E_I*SC + self.I0_E
        rA = phi_E(IA, self.a_E[i], self.b_E, self.d_E)[0,0]
        rB = phi_E(IB, self.a_E[i], self.b_E, self.d_E)[1,0]

        FA = -SA/self.tau_NMDA + self.gamma*(1-SA)*rA
        FB = -SB/self.tau_NMDA + self.gamma*(1-SB)*rB
        return FA, FB

    def plot_phase_portrait(self, Js, ax=None, area='9/46d'):
        SA_vals = np.linspace(-0.1, 0.9, 30)
        SB_vals = np.linspace(-0.1, 0.9, 30)
        SA_grid, SB_grid = np.meshgrid(SA_vals, SB_vals)
        FA_grid = np.zeros_like(SA_grid)
        FB_grid = np.zeros_like(SA_grid)

        for i in range(30):
            for j in range(30):
                FA_grid[i,j], FB_grid[i,j] = self.field(SA_grid[i,j], SB_grid[i,j], Js, area)

        if ax is None:
            fig, ax = plt.subplots(figsize=(7,6))

        speed = np.sqrt(FA_grid**2 + FB_grid**2)
        strm = ax.streamplot(SA_grid, SB_grid, FA_grid, FB_grid, color=speed, cmap='viridis', density=1.3)
        plt.colorbar(strm.lines, ax=ax, label=r"$|\dot{S}|$")

        S, R, E = self.fixedpoint_computation(Js, area=area)
        for s, eigs in zip(S, E):
            SA_fp, SB_fp, SC_fp = s
            stable = np.all(eigs.real < 0)
            ax.plot(SA_fp, SB_fp, 'o',
                    color='green' if stable else 'red', markersize=7, markeredgecolor='k')

        ax.plot([-0.1,0.9],[-0.1,0.9],'k--',linewidth=0.7)
        ax.set_xlabel(r"$S_{E1}$"); ax.set_ylabel(r"$S_{E2}$")
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.set_title(fr"Phase portrait, $J_s={Js}$")
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                markeredgecolor='k', markersize=8, label='Stable fixed point'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                markeredgecolor='k', markersize=8, label='Saddle'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)  
        plt.savefig(f'Fig/phase_portrait_{Js}.svg', dpi=150, bbox_inches='tight')      
        return ax

    def fixedpoint_computation(self, Js, grid_pts=10, tol=1e-4, area='9/46d'):
        i = self.idx(area)
        J_IE = self._compute_J_IE(Js)

        if 'spine' in self.aging_mechanism:
            Js += (Js - self.g_E_self_min) * (self.D[i].item() - 1)

        # Assemble vectors and matrices
        I0 = np.array([self.I0_E, self.I0_E, self.I0_I])
        Tinv = np.diag((1/self.tau_NMDA, 1/self.tau_NMDA, 1/self.tau_GABA))
        J = np.array([[ Js,    self.g_E_cross,    self.g_E_I],
                      [ self.g_E_cross,    Js,    self.g_E_I],
                      [ J_IE,  J_IE,  self.g_I_I]])
        # Assemble vector functions
        def I_fn(S):
            return J@S + I0

        def Gamma(S):
            return np.array([self.gamma*(1 - S[0]), self.gamma*(1 - S[1]), self.gamma_i])

        def Phi(I):
            return np.array([phi_E(I[0], self.a_E[i], self.b_E, self.d_E)[0,0], 
                             phi_E(I[1], self.a_E[i], self.b_E, self.d_E)[1,0],
                             phi_I(I[2], self.g_I, self.c_1, self.c_0, self.r0_I) ])
        
        def phiE_prime(I,a,b,d):
            x = a*I - b
            e = np.exp(np.clip(-d*x, -700, 350))
            num = 1 - e - d*x*e
            den = (1 - e)**2
            return a*num/den

        def phiI_prime(I, gI, c1, c0, r0):
            return (c1/gI) if (c1*I - c0)/gI + r0 > 0 else 0.0

        def Phi_deriv(I):
            return np.hstack( (phiE_prime(I[0], self.a_E[i], self.b_E, self.d_E), 
                            phiE_prime(I[1], self.a_E[i], self.b_E, self.d_E) , 
                            phiI_prime(I[2], self.g_I, self.c_1, self.c_0, self.r0_I) ) )

        # The function to solve
        def F(S):
            return - Tinv @ S + Gamma(S) * Phi(I_fn(S))

        x0 = np.zeros(3)
        # Jacobian of F(S)
        def Jac(S):
            D_sat = np.diag( (self.gamma*Phi(I_fn(S))[0], self.gamma*Phi(I_fn(S))[1], 0)) 
            return -Tinv - D_sat + ( Gamma(S) * Phi_deriv(I_fn(S)) )[:, None]* J

        # Multi-start over a grid of initial conditions
        grid = np.linspace(0.01, 0.9, grid_pts)
        solutions = []; R = []; 
        eig = []
        for sa0 in grid:
            for sb0 in grid:
                sc0 = 0.15
                x0 = np.array([sa0, sb0, sc0])
                sol = scipy.optimize.root(F, x0, jac=Jac, method='hybr')
                if sol.success:
                    S = sol.x
                    if np.all(S > -1e-6) and np.all(S < 1.5):
                        if not any(np.max(np.abs(S - prev)) < tol for prev in solutions):
                            solutions.append(S); R.append(Phi(I_fn(S)))
                            eigval = np.linalg.eigvals(Jac(S))
                            eig.append(eigval)

        return solutions, R, eig

    def find_critical_Js(condition, area, Js_min=0.21, Js_start=0.8, Js_cap=100.0, tol=1e-4):
        # condition must switch False -> True as Js increases
        if condition(Js_min, area):
            return np.nan                      # already true at the lower bound
        lo, hi = Js_min, Js_start
        while not condition(hi, area):         # expand bracket
            lo, hi = hi, hi * 1.5
            if hi > Js_cap:
                return np.nan                  # never switches: genuinely absent
        while hi - lo > tol:                   # bisect inside the bracket
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if condition(mid, area) else (mid, hi)
        return round(hi, 4)
    
    def fixedpoint_computation_all(self, tol=1e-5):
        n = self.num_areas
        # Assemble vectors and matrices
        I0 = np.concatenate([
            np.full(n, self.I0_E),
            np.full(n, self.I0_E),
            np.full(n, self.I0_I),
        ])
        Tinv = np.diag( np.concatenate([
            np.full(n, 1 / self.tau_NMDA),
            np.full(n, 1 / self.tau_NMDA),
            np.full(n, 1 / self.tau_GABA),
        ]) )

        W = self.W_eff

        # Set up functions
        def I_fn(S):
            return W @ S + I0

        def Gamma(S):
            return np.array([self.gamma*(1 - S[:n]), self.gamma*(1 - S[n:2*n]), np.full(n, self.gamma_i)]).flatten()

        def phi_E_vec(I, a, b, d):
            a_expanded = np.tile(a, 2)
            x = a_expanded * I - b
            exponent = np.clip(-d * x, -700, 700)
            return x / (1 - np.exp(exponent))

        def Phi(I):
            return np.hstack((phi_E_vec(I[:2*n], self.a_E, self.b_E, self.d_E),
                            phi_I(I[2*n:], self.g_I, self.c_1, self.c_0, self.r0_I)))

        def phiE_prime(I, a, b, d):
            a_expanded = np.tile(a, 2)
            x = a_expanded*I - b
            exponent = np.clip(-d*x, -700, 350)
            e = np.exp(exponent)
            num = 1 - e - d*x*e
            den = (1 - e)**2
            return a_expanded*num/den

        def phiI_prime(I, gI, c1, c0, r0):
            x = (c1*I - c0)/gI + r0
            return np.where(x > 0, c1/gI, 0.0)

        def Phi_deriv(I):
            return np.hstack((phiE_prime(I[:2*n], self.a_E, self.b_E, self.d_E),
                            phiI_prime(I[2*n:], self.g_I, self.c_1, self.c_0, self.r0_I)))

        def F(S):
            return -Tinv @ S + Gamma(S) * Phi(I_fn(S))

        def Jac(S):
            D_sat = np.diag(np.hstack([self.gamma*Phi(I_fn(S))[:2*n], np.zeros(n)]))
            return -Tinv - D_sat + (Gamma(S) * Phi_deriv(I_fn(S)))[:, None] * W

        # Set up rootfinding
        

        seeds = []
        seeds.append(np.zeros(3*n))
        n_points = 10
        for i in range(1, n_points + 1):
            st = i * 4 / n_points

            self.runsim(trial_length=10, trials = 10, stim=[4, 4.5, st, 'V1'], store_S=True)
            S_cand = self.S[-100:].mean(axis=0)
            for S in S_cand.T:
                seeds.append(S)

            self.runsim(trial_length=10, trials = 10, stim=[4, 4.5, 0, 'V1'], distract=[4, 4.5, st], store_S=True)
            S_cand = self.S[-100:].mean(axis=0)
            for S in S_cand.T:
                seeds.append(S)

        S, R = [], []
        print(len(seeds))

        for x0 in tqdm(seeds):
            sol = scipy.optimize.root(F,x0, jac=Jac, method='hybr')
            if not sol.success:
                continue

            s_ori = sol.x
            s_symm = np.concatenate((s_ori[self.num_areas:2*self.num_areas],s_ori[:self.num_areas],s_ori[2*self.num_areas:]))
            for s in [s_ori, s_symm]:
                if any(np.linalg.norm(s - prev) / np.sqrt(len(s)) < tol for prev in S):
                    # print('s already included in set')
                    continue
    
                r = Phi(I_fn(s))
                eigval = np.linalg.eigvals(Jac(s))
                if np.any(eigval.real > 0): # unstable fixe point
                    # print('unstable')
                    continue
                S.append(s); R.append(r)

        return S, R

