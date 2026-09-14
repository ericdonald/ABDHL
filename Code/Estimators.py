"""""""""""
Estimators

Notes: Custom estimators for Poisson with instruments.
    
"""""""""""

import numpy as np
import pandas as pd
from scipy import optimize, stats
import Processing_Functions as gpf


class GLMWrap:
    def __init__(self, res, y=None, x=None, offset=None, fe=None, n_sectors=None):
        self.glm       = res
        self.y         = y
        self.x         = list(x) if x is not None else list(res.params.index)
        self.offset    = offset
        self.fe        = fe or []
        self.n_sectors = n_sectors
        self.params    = res.params
        self.bse       = res.bse
        self.tvalues   = res.tvalues
        self.pvalues   = res.pvalues
        self.nobs      = int(res.nobs)
        try:
            self.rsquared = res.pseudo_rsquared(kind='mcf')
        except Exception:
            self.rsquared = np.nan

    def conf_int(self, alpha=0.05):
        return self.glm.conf_int(alpha=alpha)

    def frame(self):
        ci = self.conf_int()
        rows = [v for v in self.x if v in self.params.index]
        return pd.DataFrame({
            'coef':  self.params[rows].round(4),
            'se':    self.bse[rows].round(4),
            'z':     self.tvalues[rows].round(2),
            'p':     self.pvalues[rows].round(4),
            'sig':   [gpf.get_stars(p) for p in self.pvalues[rows]],
            'ci_lo': ci.iloc[:, 0][rows].round(4),
            'ci_hi': ci.iloc[:, 1][rows].round(4),
        })

    def test(self, restriction):
        try:
            w = self.glm.wald_test(restriction, use_f=False, scalar=True)
            return float(np.squeeze(w.statistic)), float(np.squeeze(w.pvalue))
        except TypeError:
            w = self.glm.wald_test(restriction, use_f=False)
            return float(np.squeeze(w.statistic)), float(np.squeeze(w.pvalue))

    def __repr__(self):
        head = f'{self.y} ~ {" + ".join(self.x)}'
        spec = (f'PPML (Poisson pseudo-ML), offset log({self.offset}), '
                f'{" + ".join(self.fe) if self.fe else "no"} FE, '
                f'SE clustered by sector')
        info = [f'N = {self.nobs}']
        if self.n_sectors is not None:
            info.append(f'sectors = {self.n_sectors}')
        if np.isfinite(self.rsquared):
            info.append(f'pseudo R2 = {self.rsquared:.4f}')
        conv = getattr(self.glm, 'converged', None)
        if conv is None:
            conv = getattr(getattr(self.glm, 'mle_retvals', {}), 'get', lambda k, d: d)('converged', None)
        if conv is not None:
            info.append(f'converged = {conv}')

        out = [head, spec, '  |  '.join(info), '', self.frame().to_string()]

        pairs = [('up_G_pat_lag', 'down_G_pat_lag'), ('up_G_cite_lag', 'down_G_cite_lag'),
                 ('up_G_pat',     'down_G_pat'),     ('up_G_cite',     'down_G_cite')]
        for a, b in pairs:
            if a in self.params.index and b in self.params.index:
                try:
                    s1, p1 = self.test(f'{a} + {b} = 0')
                    s2, p2 = self.test(f'{a} = {b}')
                    out.append(f'\nH0: up + down = 0   chi2 = {s1:.3f}  p = {p1:.4f}   '
                               f'(complementarity implies > 0)')
                    out.append(f'H0: up = down       chi2 = {s2:.3f}  p = {p2:.4f}')
                except Exception as e:
                    out.append(f'\nWald tests unavailable: {e}')
        return '\n'.join(out)

    def full(self):
        return self.glm.summary()
    
    

class PoissonIVResults:
    "FE Poisson IV by GMM on Wooldridge's within-group multiplicative moment"

    def __init__(self, params, cov, x_cols, endog, instruments, y, off,
                 n, n_clusters, J_stat, J_df, J_p, converged, W_stage):
        self.params      = pd.Series(params, index=x_cols)
        self.cov         = pd.DataFrame(cov, index=x_cols, columns=x_cols)
        self.bse         = pd.Series(np.sqrt(np.diag(cov)), index=x_cols)
        self.tvalues     = self.params / self.bse
        self.pvalues     = pd.Series(2 * (1 - stats.norm.cdf(np.abs(self.tvalues))),
                                     index=x_cols)
        self.x           = list(x_cols)
        self.endog       = list(endog)
        self.instruments = list(instruments)
        self.y, self.offset = y, off
        self.nobs, self.n_clusters = n, n_clusters
        self.J, self.J_df, self.J_p = J_stat, J_df, J_p
        self.converged, self.W_stage = converged, W_stage
        self.rsquared = np.nan

    def conf_int(self, alpha=0.05):
         z = stats.norm.ppf(1 - alpha / 2)
         return pd.DataFrame({'lower': self.params - z * self.bse,
                              'upper': self.params + z * self.bse})

    def frame(self):
        ci = self.conf_int()
        return pd.DataFrame({
            'coef':  self.params.round(4), 'se': self.bse.round(4),
            'z':     self.tvalues.round(2), 'p': self.pvalues.round(4),
            'sig':   [gpf.get_stars(p) for p in self.pvalues],
            'ci_lo': ci['lower'].round(4), 'ci_hi': ci['upper'].round(4)})

    def test(self, a, b, mode='sum'):
        "Wald test of a+b=0 (mode='sum') or a-b=0 (mode='diff')"
        r = np.zeros(len(self.x))
        r[self.x.index(a)] = 1.0
        r[self.x.index(b)] = 1.0 if mode == 'sum' else -1.0
        val = float(r @ self.params.to_numpy())
        v   = float(r @ self.cov.to_numpy() @ r)
        w   = val**2 / max(v, 1e-14)
        return val, np.sqrt(max(v, 0)), w, 1 - stats.chi2.cdf(w, 1)

    def __repr__(self):
        out = [f'{self.y} ~ {" + ".join(self.x)}',
               f'FE Poisson IV (GMM, Wooldridge within-group moment), '
               f'offset log({self.offset})',
               f'endogenous: {", ".join(self.endog)}',
               f'instruments: {", ".join(self.instruments)}',
               f'N = {self.nobs}  |  sectors = {self.n_clusters}  |  '
               f'{self.W_stage}  |  converged = {self.converged}',
               '', self.frame().to_string()]
        if self.J_df > 0:
            out.append(f'\nHansen J = {self.J:.3f} on {self.J_df} df, '
                       f'p = {self.J_p:.4f}   (overidentification)')
        else:
            out.append('\nExactly identified: no overidentification test')
        for a, b in [('up_G_pat_lag', 'down_G_pat_lag'),
                     ('up_G_cite_lag', 'down_G_cite_lag')]:
            if a in self.x and b in self.x:
                v, se, w, p = self.test(a, b, 'sum')
                out.append(f'H0: up + down = 0   {v:+.3f} ({se:.3f})  '
                           f'chi2 = {w:.3f}  p = {p:.4f}')
                v, se, w, p = self.test(a, b, 'diff')
                out.append(f'H0: up  =  down     {v:+.3f} ({se:.3f})  '
                           f'chi2 = {w:.3f}  p = {p:.4f}')
        return '\n'.join(out)


def fit_poisson_iv(df, y_col, offset_col, x_cols, endog_cols, instrument_cols,
                   entity='BLS_Industry', time='period', time_fe=True,
                   two_step=True, maxiter=500, verbose=False):
    """
    Fixed-effects Poisson IV by GMM on Wooldridge's within-group multiplicative
    moment.
    """
    need = [y_col, offset_col] + list(x_cols) + list(instrument_cols)
    d = df.dropna(subset=need).copy()
    d = d[d[offset_col] > 0]

    keep_sec = d.groupby(entity)[y_col].transform('sum') > 0
    n_drop = int(d.loc[~keep_sec, entity].nunique())
    d = d[keep_sec].reset_index(drop=True)
    if n_drop:
        print(f'  fit_poisson_iv({y_col}): dropped {n_drop} sector(s) with no '
              f'positive outcome in any period.')

    exog_x = [c for c in x_cols if c not in endog_cols]

    parts = [d[list(x_cols)].astype(float)]
    if time_fe:
        parts.append(pd.get_dummies(d[time], prefix='per', drop_first=True, dtype=float))
    Wm = pd.concat(parts, axis=1)
    Wm.columns = [str(c) for c in Wm.columns]
    names = list(Wm.columns)

    zparts = [d[list(instrument_cols)].astype(float)]
    if exog_x:
        zparts.append(d[exog_x].astype(float))
    if time_fe:
        zparts.append(pd.get_dummies(d[time], prefix='zper', drop_first=True, dtype=float))
    Z = pd.concat(zparts, axis=1)
    Z.columns = [str(c) for c in Z.columns]

    Wraw = Wm.to_numpy(float)
    Zv   = Z.to_numpy(float)
    yv   = d[y_col].to_numpy(float)
    ov   = np.log(d[offset_col].to_numpy(float))
    codes, uniq = pd.factorize(d[entity].to_numpy())
    N, G_, K, L = len(yv), len(uniq), Wraw.shape[1], Zv.shape[1]

    if L < K:
        raise ValueError(f'under-identified: {L} moments for {K} parameters')

    # --- standardise regressors (and instruments) for conditioning -----------
    w_sd = Wraw.std(axis=0, ddof=0)
    w_sd = np.where(w_sd > 1e-12, w_sd, 1.0)
    Wv   = Wraw / w_sd
    z_sd = Zv.std(axis=0, ddof=0)
    z_sd = np.where(z_sd > 1e-12, z_sd, 1.0)
    Zs   = Zv / z_sd

    ysum = np.bincount(codes, weights=yv, minlength=G_)

    def pieces(theta):
        mu    = np.exp(np.clip(Wv @ theta + ov, -60, 60))
        musum = np.bincount(codes, weights=mu, minlength=G_)
        rho   = np.where(musum > 0, ysum / np.where(musum > 0, musum, 1.0), 0.0)
        return mu, musum, rho

    def resid(theta):
        mu, _, rho = pieces(theta)
        return yv - rho[codes] * mu

    def gbar(theta):
        return Zs.T @ resid(theta) / N

    def jac(theta):
        "analytic dg/dtheta"
        mu, musum, rho = pieces(theta)
        # mu-weighted within-sector mean of each column of W
        num = np.zeros((G_, K))
        for k in range(K):
            num[:, k] = np.bincount(codes, weights=mu * Wv[:, k], minlength=G_)
        wbar = num / np.where(musum[:, None] > 0, musum[:, None], 1.0)
        dU   = -(rho[codes] * mu)[:, None] * (Wv - wbar[codes])     # N x K
        return Zs.T @ dU / N

    def obj(theta, Wgt):
        g = gbar(theta)
        return float(g @ Wgt @ g)

    def cluster_omega(theta):
        u = resid(theta)
        Om = np.zeros((L, L))
        for gi in range(G_):
            m = codes == gi
            s = Zs[m].T @ u[m]
            Om += np.outer(s, s)
        return Om / N**2

    exact = (L == K)
    th0   = np.zeros(K)

    if exact:
        sol = optimize.root(gbar, th0, jac=jac, method='hybr',
                            options={'maxfev': maxiter * (K + 1), 'xtol': 1e-12})
        theta, conv, stage = sol.x, bool(sol.success), 'exactly identified (root)'
        if not conv:   # fall back to least squares on the moments
            ls = optimize.least_squares(gbar, th0, jac=jac, xtol=1e-14, ftol=1e-14,
                                        max_nfev=maxiter * 10)
            theta, conv, stage = ls.x, bool(ls.success), 'exactly identified (LS)'
        Wgt = np.eye(L)
    else:
        W1 = np.linalg.pinv(Zs.T @ Zs / N)
        r1 = optimize.minimize(obj, th0, args=(W1,), method='trust-constr',
                               jac=lambda t, Wg: 2 * jac(t).T @ Wg @ gbar(t),
                               options={'maxiter': maxiter, 'gtol': 1e-12})
        theta, conv, stage = r1.x, bool(r1.success), 'one-step GMM'
        Wgt = W1
        if two_step:
            W2 = np.linalg.pinv(cluster_omega(theta))
            r2 = optimize.minimize(obj, theta, args=(W2,), method='trust-constr',
                                   jac=lambda t, Wg: 2 * jac(t).T @ Wg @ gbar(t),
                                   options={'maxiter': maxiter, 'gtol': 1e-12})
            if r2.success or obj(r2.x, W2) < obj(theta, W2):
                theta, conv, stage = r2.x, bool(r2.success), 'two-step GMM'
            Wgt = W2

    g_norm = float(np.max(np.abs(gbar(theta))))
    if verbose or g_norm > 1e-6:
        print(f'  max|moment| at solution = {g_norm:.3e}'
              f'{"   <-- NOT SOLVED" if g_norm > 1e-6 else ""}')

    Gj   = jac(theta)
    Om   = cluster_omega(theta)
    GWG  = Gj.T @ Wgt @ Gj
    GWGi = np.linalg.pinv(GWG)
    cov  = GWGi @ (Gj.T @ Wgt @ Om @ Wgt @ Gj) @ GWGi
    K_eff = K + G_
    cov  *= (G_ / max(G_ - 1, 1)) * ((N - 1) / max(N - K_eff, 1))

    # map back from standardised to original scale
    theta_o = theta / w_sd
    cov_o   = cov / np.outer(w_sd, w_sd)

    g = gbar(theta)
    J_df = L - K
    if J_df > 0:
        J_stat = float(N * g @ np.linalg.pinv(Om) @ g)
        J_p = 1 - stats.chi2.cdf(J_stat, J_df)
    else:
        J_stat, J_p = np.nan, np.nan

    res = PoissonIVResults(theta_o, cov_o, names, endog_cols, instrument_cols,
                           y_col, offset_col, N, G_, J_stat, J_df, J_p,
                           conv, stage)
    res.moment_norm = g_norm
    res.jac_cond    = float(np.linalg.cond(Gj))
    return res
    
    
    
    