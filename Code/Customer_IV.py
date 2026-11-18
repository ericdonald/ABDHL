"""""""""""
Customer IV Module

Notes: Estimation of the customer instrumental variable for "Transition to Green Technology along the Supply Chain"
    (Section 1 and Table E1, Section 2 / Figure E1 / Table E2, Table E3 of the draft in Specifications/Customer_IV).

    Does a firm's innovation turn green when its customers' innovation does? A supplier's regressor is the link-weighted
    green share of its customers over the five years before the outcome year; the instrument replaces each customer's green
    share by the share it would have had if each of its technology classes had greened at the national rate (measured on
    patents of firms outside the sample). Links and class mixes are fixed in the base period FY1990-94.

    Data: Clean Data/Cust_*.pkl (Customer_Data.Customer_Data.Cleaner), Clean Data/BLS_Crosswalk.pkl, Clean Data/IO_Networks.pkl.
    Output: Specifications/Customer_IV/output/*.csv (every number in the draft); tables and figure are built from these by
    Customer_Tables.

"""""""""""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from multiprocessing import get_context
from pathlib import Path
import contextlib
import io
import Estimators as es

NRI = int(os.environ.get('NRI', 200))            # reshuffle draws (set NRI=20, NBOOT=19 for a quick test run)
NBOOT = int(os.environ.get('NBOOT', 199))        # supplier bootstrap draws for the control-function IV
KAPPA = 20.0                                     # shrinkage (pseudo-patents) toward the aggregate green share (Robustness_Lambda varies it)
SB = 0                                           # 1: supplier weights are also multiplied by the sector's 1990-94 green share (earlier version); 0: input share only
LAST = 2017                                      # last outcome / filing year
YEARS = np.arange(1976, LAST + 1)
LO, HI = 1990, 1994                              # base period: links (fiscal years) and class mixes (filing years)
SECS = np.arange(1, 94)                          # BLS sectors 1-93
HS = list(range(0, 21))                          # horizons of the timing analysis
CHI2_95 = stats.norm.ppf(0.975) ** 2
LEVELS = (68, 90, 95)





# ----------------------------------------------------------------------------------------------------------------------
# Estimators: fixed-effects Poisson, two-way demeaning, clustered OLS
# ----------------------------------------------------------------------------------------------------------------------



def fepois(df, y, x, entity, offset=None, time='period', cluster=None, maxit=100, tol=1e-10):
    cols = [y] + list(x) + [entity, time] + ([offset] if offset else []) + ([cluster] if cluster else [])
    d = df[list(dict.fromkeys(cols))].dropna()
    if offset:
        d = d[d[offset] > 0]
    d = d[d.groupby(entity)[y].transform('sum') > 0].reset_index(drop=True)
    T = pd.get_dummies(d[time], prefix='per', drop_first=True, dtype=float)
    X = np.column_stack([d[list(x)].to_numpy(float), T.to_numpy(float)])
    names = list(x) + list(T.columns)
    yv = d[y].to_numpy(float)
    off = np.log(d[offset].to_numpy(float)) if offset else np.zeros(len(d))
    f, _ = pd.factorize(d[entity])
    F = f.max() + 1
    ysum = np.bincount(f, yv, F)
    sd = X.std(0); sd[sd < 1e-12] = 1.0
    Xs = X / sd
    th = np.zeros(X.shape[1])

    def pieces(th):
        eta = Xs @ th + off
        eta = eta - np.bincount(f, eta, F)[f] / np.bincount(f, None, F)[f]   # numerically centred within entity
        mu = np.exp(np.clip(eta, -50, 50))
        p = mu / np.bincount(f, mu, F)[f]
        return p

    for it in range(maxit):
        p = pieces(th)
        yhat = ysum[f] * p
        g = Xs.T @ (yv - yhat)
        xbar = np.column_stack([np.bincount(f, p * Xs[:, k], F) for k in range(Xs.shape[1])])
        Xc = Xs - xbar[f]
        H = (Xc * yhat[:, None]).T @ Xc
        step = np.linalg.solve(H, g)
        th = th + step
        if np.max(np.abs(step)) < tol:
            break
    p = pieces(th); yhat = ysum[f] * p
    xbar = np.column_stack([np.bincount(f, p * Xs[:, k], F) for k in range(Xs.shape[1])])
    Xc = Xs - xbar[f]
    H = (Xc * yhat[:, None]).T @ Xc
    u = Xc * (yv - yhat)[:, None]
    cl = d[cluster if cluster else entity].to_numpy()
    cg, _ = pd.factorize(cl)
    G = cg.max() + 1
    U = np.zeros((G, X.shape[1]))
    np.add.at(U, cg, u)
    Hi = np.linalg.inv(H)
    N, K = len(d), X.shape[1] + F
    V = Hi @ (U.T @ U) @ Hi * (G / (G - 1)) * ((N - 1) / (N - K))
    b = th / sd
    se = np.sqrt(np.diag(V)) / sd
    out = pd.DataFrame({'coef': b, 'se': se}, index=names)
    out['p'] = 2 * (1 - stats.norm.cdf(np.abs(out.coef / out.se)))
    out.attrs.update(N=N, entities=F, clusters=G, iters=it + 1, V=V / np.outer(sd, sd))
    return out


def fp(d, y, x, entity, offset, time, cluster=None):
    "fepois; if its Newton step hits a singular Hessian, fall back to Poisson GLM with explicit dummies (same estimator)."
    try:
        return fepois(d, y, x, entity, offset=offset, time=time, cluster=cluster)
    except np.linalg.LinAlgError:
        import statsmodels.api as sm
        F = pd.get_dummies(d[entity].astype(str), dtype=float)
        T = pd.get_dummies(d[time].astype(str), dtype=float).iloc[:, 1:]
        M = pd.concat([d[x].reset_index(drop=True), F.reset_index(drop=True), T.reset_index(drop=True)], axis=1)
        cl = pd.factorize(d[cluster if cluster else entity])[0]
        m = sm.GLM(d[y].to_numpy(float), M, family=sm.families.Poisson(), offset=np.log(d[offset].to_numpy(float))).fit(
            maxiter=300, cov_type='cluster', cov_kwds={'groups': cl})
        return pd.DataFrame({'coef': m.params[x], 'se': m.bse[x], 'p': m.pvalues[x]})


def trim(d, fe):
    "Drop what the fixed effects make uninformative or separated, until nothing changes."
    for _ in range(20):
        n0 = len(d)
        d = d[d.groupby(fe).clean.transform('sum') > 0]
        d = d[d.groupby(fe).gvkey.transform('nunique') > 1]
        d = d[d.groupby('gvkey').clean.transform('sum') > 0]
        d = d[d.groupby('gvkey').E.transform('count') >= 2]
        if len(d) == n0:
            break
    return d.reset_index(drop=True)


def demean(df, cols, fe1, fe2, iters=200, tol=1e-10):
    "Two-way within transformation by alternating projections."
    A = df[cols].to_numpy(float).copy()
    g1, _ = pd.factorize(df[fe1]); g2, _ = pd.factorize(df[fe2])
    n1, n2 = np.bincount(g1), np.bincount(g2)
    for _ in range(iters):
        old = A.copy()
        A -= np.column_stack([np.bincount(g1, A[:, k]) / n1 for k in range(A.shape[1])])[g1]
        A -= np.column_stack([np.bincount(g2, A[:, k]) / n2 for k in range(A.shape[1])])[g2]
        if np.max(np.abs(A - old)) < tol:
            break
    return A


def ols_cluster(y, X, cl, ndf):
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ y
    u = y - X @ b
    cg, _ = pd.factorize(cl); G = cg.max() + 1
    S = np.zeros((G, X.shape[1])); np.add.at(S, cg, X * u[:, None])
    N = len(y)
    V = XtX @ (S.T @ S) @ XtX * G / (G - 1) * (N - 1) / max(N - ndf, 1)
    return b, np.sqrt(np.diag(V)), u


def ri_p(stat, sims):
    """Two-sided randomisation p from studentised statistics (|t| of each reshuffle vs the real |t|). Studentising makes the
    test robust to the occasional reshuffle whose Poisson fit does not converge; the earlier version centred raw
    coefficients on their mean, which a few exploding draws could distort (corrected 2026-11-05)."""
    sims = np.asarray(sims)
    sims = sims[np.isfinite(sims)]
    return (np.sum(np.abs(sims) >= abs(stat)) + 1) / (len(sims) + 1)


def estimate(d, z, ctrl, fe, ri=None):
    "Reduced form, OLS, first stage, control-function IV for one specification."
    cols = ['clean', 'noncl', 'x', z, 'gvkey', fe, 'E', 'sector', 'main_cust'] + ctrl
    d = d[list(dict.fromkeys(cols))].dropna().copy()
    d = trim(d, fe)
    if fe == 'sw':
        # firm effects absorb sector effects: pool each sector's first window into one base cell (same column space as
        # sector x window effects, without the collinearity)
        first = d.groupby('sector').E.transform('min')
        d['sw'] = np.where(d.E == first, '0_base', d.sector.astype(int).astype(str) + '_' + d.E.astype(str))
    nfe = d.gvkey.nunique() + d[fe].nunique()
    A = demean(d, ['x', z] + ctrl, 'gvkey', fe)
    xr, Zr = A[:, 0], A[:, 1:]
    b, se, v = ols_cluster(xr, Zr, d.gvkey.to_numpy(), nfe + Zr.shape[1])
    F = (b[0] / se[0]) ** 2
    d['vhat'] = v
    sdx = xr.std()
    rf = fp(d, 'clean', [z] + ctrl, 'gvkey', 'noncl', fe)
    rf_c = fp(d, 'clean', [z] + ctrl, 'gvkey', 'noncl', fe, cluster='main_cust')
    ols = fp(d, 'clean', ['x'] + ctrl, 'gvkey', 'noncl', fe)
    cf = fp(d, 'clean', ['x', 'vhat'] + ctrl, 'gvkey', 'noncl', fe)
    r = dict(N=len(d), suppliers=d.gvkey.nunique(), customers=d.main_cust.nunique(), F=F,
             fs_coef=b[0], rf=rf.at[z, 'coef'], rf_p=rf.at[z, 'p'], rf_p_maincust=rf_c.at[z, 'p'],
             ols_sd=100 * ols.at['x', 'coef'] * sdx, ols_p=ols.at['x', 'p'],
             iv_sd=100 * cf.at['x', 'coef'] * sdx)
    if ri is not None:
        r['rf_p_ri'] = ri(d, z, ctrl, fe)
    return r



# ----------------------------------------------------------------------------------------------------------------------
# Data: Clean Data tables -> shared context
# ----------------------------------------------------------------------------------------------------------------------

def load_inputs(D):
    "Clean Data tables used by the customer IV (see Customer_Data.Cleaner for how each is built)."
    D = Path(D)
    return dict(xwalk=pd.read_pickle(D / 'Cust_Xwalk.pkl'), pat=pd.read_pickle(D / 'Cust_Patents.pkl'),
                cy=pd.read_pickle(D / 'Cust_Class_Year.pkl'), firms=pd.read_pickle(D / 'Cust_Firms.pkl'),
                cs=pd.read_pickle(D / 'Cust_Segments.pkl'), fs=pd.read_pickle(D / 'Cust_Sales.pkl'),
                bls=pd.read_pickle(D / 'BLS_Crosswalk.pkl'), io=pd.read_pickle(D / 'IO_Networks.pkl'))


def bls_mapper(cw):
    "NAICS code -> BLS sector by longest prefix (BLS_Crosswalk)."
    pref = sorted(((p.strip().replace('.0', ''), int(r.BLS_Industry)) for _, r in cw.iterrows()
                   for p in str(r.NAICS_2022).split(',') if p.strip() and p.strip() != 'nan'), key=lambda x: -len(x[0]))
    def f(n):
        n = str(n).split('.')[0]
        return next((b for p, b in pref if n.startswith(p)), np.nan)
    return f


def firm_sectors(firms, cw):
    """BLS sector of every supplier-eligible firm: U.S.-incorporated, or incorporated abroad with most patents (1993-2017) having
    a U.S.-based first inventor (Medtronic, Eaton, Aptiv ...); sector from the firm's NAICS."""
    bmap = bls_mapper(cw)
    keep = firms.fic.eq('USA') | (firms.us_share > 0.5)
    g = firms[keep].dropna(subset=['naics'])
    return g.set_index('gvkey').naics.map(lambda n: bmap(int(n))).dropna().astype(int)


def sector_of_all(firms, cw):
    "BLS sector of every Compustat firm (customers included), from its first reported NAICS."
    bmap = bls_mapper(cw)
    g = firms.dropna(subset=['naics_any'])
    return g.set_index('gvkey').naics_any.map(lambda n: bmap(int(n))).dropna().astype(int)


def io_weights(io, year=1997):
    "Direct input-output coefficients A[buyer, supplier] of the BLS table, diagonal zeroed."
    S = np.array(io[year], dtype=float).copy()
    np.fill_diagonal(S, 0.0)
    return S


def leontief_weights(io, year=1997):
    "Leontief inverse (I - A)^-1 of the BLS table, A[buyer, supplier], diagonal zeroed (a sector's own total requirement excluded)."
    A = np.array(io[year], dtype=float)
    L = np.linalg.inv(np.eye(A.shape[0]) - A)
    np.fill_diagonal(L, 0.0)
    return L


def context(inp):
    "Everything shared across base periods and horizons."
    X = inp['xwalk']
    pt, sub = inp['pat']['pt'], inp['pat']['sub']
    pt = pt[pt.year.between(1976, LAST)]
    X = X[X.patent_id.isin(set(pt.patent_id))].copy()
    X['wg'] = 1 / X.groupby('patent_id').gvkey.transform('count')
    XP = X.merge(pt, on='patent_id')
    fy = XP.assign(c=XP.wg * XP.clean).groupby(['gvkey', 'year']).agg(clean=('c', 'sum'), all=('wg', 'sum'))
    Fc = fy.clean.unstack().reindex(columns=YEARS).fillna(0)
    Fa = fy['all'].unstack().reindex(columns=YEARS).fillna(0)
    Gprev = pd.DataFrame(shrunk_share(trailing(Fc.to_numpy(), -5, -1), trailing(Fa.to_numpy(), -5, -1)), index=Fc.index, columns=YEARS)
    cy = inp['cy']
    o = cy[~cy.inx]
    Oc = o.pivot_table(index='cpc_subclass', columns='year', values='clean', aggfunc='sum').reindex(columns=YEARS).fillna(0)
    Oa = o.pivot_table(index='cpc_subclass', columns='year', values='all', aggfunc='sum').reindex(columns=YEARS).fillna(0)
    classes = Oc.index
    gprev = pd.DataFrame(shrunk_share(trailing(Oc.to_numpy(), -5, -1), trailing(Oa.to_numpy(), -5, -1)), index=classes, columns=YEARS)
    gcurr = pd.DataFrame(shrunk_share(trailing(Oc.to_numpy(), -4, 0), trailing(Oa.to_numpy(), -4, 0)), index=classes, columns=YEARS)
    s = XP[['patent_id', 'gvkey', 'wg', 'year']].merge(sub, on='patent_id')
    s['m'] = s.wg * s.f
    sec = firm_sectors(inp['firms'], inp['bls'])
    cs, fs = inp['cs'], inp['fs']
    cs = cs.dropna(subset=['cust']).copy()
    cs['cust'] = cs.cust.astype(int)
    cs = cs[(cs.cust != cs.gvkey) & cs.gvkey.isin(sec.index)]
    return dict(Fc=Fc, Fa=Fa, Gprev=Gprev, classes=classes, gprev=gprev, gcurr=gcurr, s=s, sec=sec, cs=cs, fs=fs, Oc=Oc, Oa=Oa,
                allsec=sector_of_all(inp['firms'], inp['bls']), IO=io_weights(inp['io']), IOL=leontief_weights(inp['io']),
                IOY={y: io_weights(inp['io'], y) for y in range(1997, 2023) if y in inp['io']})




# ----------------------------------------------------------------------------------------------------------------------
# Supplier-year panel for one base period (Table E1 design)
# ----------------------------------------------------------------------------------------------------------------------



def trailing(mat, lag_lo, lag_hi):
    "rows x YEARS matrix -> sum over years t+lag_lo ... t+lag_hi for each t (columns YEARS)"
    out = np.zeros_like(mat)
    for j, t in enumerate(YEARS):
        cols = [k for k, y in enumerate(YEARS) if t + lag_lo <= y <= t + lag_hi]
        if cols:
            out[:, j] = mat[:, cols].sum(1)
    return out


def shrunk_share(Cm, Am):
    gbar = Cm.sum(0) / np.maximum(Am.sum(0), 1e-12)
    return (Cm + KAPPA * gbar[None, :]) / (Am + KAPPA)


def mixmat(ctx, lo, hi):
    s = ctx['s']
    m = s[s.year.between(lo, hi)].groupby(['gvkey', 'cpc_subclass']).m.sum()
    m = (m / m.groupby(level=0).transform('sum')).unstack().reindex(columns=ctx['classes']).fillna(0)
    return m[m.sum(1) > 0]


def base_panel(ctx, lo, hi, min_share=0.0, drop_sector=None, contemp=False):
    """Supplier-year panel for one base period; also returns the link matrix and customers' predicted paths.
    min_share: keep only links that are at least this share of the supplier's sales; drop_sector: remove customers of this BLS sector.
    contemp: the regressor X uses the links of the window t-5 ... t-1 (customers named in those fiscal years, sales shares of the window)
    instead of the fixed base-period links; the instrument keeps the base-period links and class mixes."""
    Fc, Fa, cs, fs = ctx['Fc'], ctx['Fa'], ctx['cs'], ctx['fs']
    cm = mixmat(ctx, lo, hi)
    custs = cm.index.to_numpy()
    ci = {c: k for k, c in enumerate(custs)}
    l = cs[cs.fy.between(lo, hi) & cs.cust.isin(custs)].groupby(['gvkey', 'cust']).salecs.sum().reset_index()
    sup = np.sort(l.gvkey.unique())
    si = {g: k for k, g in enumerate(sup)}
    W = np.zeros((len(sup), len(custs)))
    W[l.gvkey.map(si).to_numpy(), l.cust.map(ci).to_numpy()] = l.salecs.to_numpy()
    sale = fs[fs.fyear.between(lo, hi)].groupby('gvkey').sale.mean()
    W = W / sale.reindex(sup).to_numpy()[:, None]
    if min_share > 0:
        W = np.where(W >= min_share, W, 0.0)
    if drop_sector is not None:
        W[:, np.array([ctx['allsec'].get(int(c), -int(c)) == drop_sector for c in custs])] = 0.0
    ok = np.isfinite(W).all(1) & (W.sum(1) > 0)
    sup, W = sup[ok], W[ok]
    si = {g: k for k, g in enumerate(sup)}
    ty = YEARS[YEARS > hi]
    Zk = cm.to_numpy() @ ctx['gprev'][ty].to_numpy()
    Gk = ctx['Gprev'].reindex(index=custs, columns=ty)
    Gk = Gk.fillna(Gk.mean()).to_numpy()
    om = mixmat(ctx, lo, hi).reindex(sup)
    om = om.fillna(mixmat(ctx, 1976, hi).reindex(sup)).fillna(0)
    own = om.to_numpy() @ ctx['gcurr'][ty].to_numpy()
    has_own = om.sum(1).to_numpy() > 0
    Xs, Zs = W @ Gk, W @ Zk
    d = pd.DataFrame({'gvkey': np.repeat(sup, len(ty)), 'E': np.tile(ty, len(sup)), 'x': Xs.ravel(), 'z': Zs.ravel(),
                      'own': own.ravel(), 'mass': np.repeat(W.sum(1), len(ty)), 'main_cust': np.repeat(custs[W.argmax(1)], len(ty))})
    d = d[np.repeat(has_own, len(ty))]
    if contemp:
        xs = []
        Gp = ctx['Gprev']
        for t in ty:
            lt = cs[cs.fy.between(t - 5, t - 1) & cs.gvkey.isin(sup) & cs.cust.notna() & (cs.cust != cs.gvkey)].groupby(['gvkey', 'cust']).salecs.sum().reset_index()
            st = fs[fs.fyear.between(t - 5, t - 1)].groupby('gvkey').sale.mean()
            lt['w'] = lt.salecs / lt.gvkey.map(st)
            lt = lt[np.isfinite(lt.w)]
            col = Gp[t]
            lt['s'] = lt.cust.map(col).fillna(col.mean())
            xs.append(lt.assign(wx=lt.w * lt.s).groupby('gvkey').wx.sum().rename(t))
        Xc = pd.concat(xs, axis=1)
        d['x'] = [Xc.at[g, t] if (g in Xc.index and pd.notna(Xc.at[g, t])) else np.nan for g, t in zip(d.gvkey, d.E)]
        d = d.dropna(subset=['x'])
    d['clean'] = [Fc.at[g, t] if g in Fc.index else 0.0 for g, t in zip(d.gvkey, d.E)]
    d['noncl'] = [Fa.at[g, t] - Fc.at[g, t] if g in Fa.index else 0.0 for g, t in zip(d.gvkey, d.E)]
    d['sector'] = d.gvkey.map(ctx['sec'])
    d = d[d.noncl > 0].dropna(subset=['sector']).reset_index(drop=True)
    bins = sorted({(t - 1998) // 5 for t in ty})
    ctrl = ['own']
    for b in bins[1:]:
        d[f'mass_b{b}'] = d.mass * (((d.E - 1998) // 5) == b)
        ctrl.append(f'mass_b{b}')
    return dict(d=d, ctrl=ctrl, sup=sup, si=si, W=W, Zk=Zk, ty=ty, custs=custs)



# ----------------------------------------------------------------------------------------------------------------------
# Linear 2SLS inference: AKM / AKM0 shock-level tests (Adao, Kolesar and Morales 2019)
# ----------------------------------------------------------------------------------------------------------------------



def resid_w(M, Cw):
    "weighted residualisation: columns of M (already multiplied by sqrt(w)) on Cw (sqrt(w) * controls)"
    coef, *_ = np.linalg.lstsq(Cw, M, rcond=None)
    return M - Cw @ coef


def prep(d, tcol, ctrl, poisson_sample=True):
    d = d.copy()
    if poisson_sample:                       # same sample as the Poisson specification
        d = trim(d.dropna(subset=ctrl + ['x', 'z']), 'E')
    d['t'] = d[tcol]
    d['n'] = d.clean + d.noncl
    d['y'] = d.clean / d.n
    for t in sorted(d.t.unique())[1:]:
        d[f'mt_{t}'] = d.mass * (d.t == t)
    return d, ctrl + [f'mt_{t}' for t in sorted(d.t.unique())[1:]]


def akm_general(d, ctrl, links, shocks, tfe='t', clus=None, weight=True):
    "explore33.akm_firm with a chosen time-effect column and optional customer clusters (array over customers)."
    d = d.reset_index(drop=True)
    w = d.n.to_numpy(float) if weight else np.ones(len(d))
    sw = np.sqrt(w)
    Fd = pd.get_dummies(d.gvkey.astype(str), dtype=float).to_numpy()
    Td = pd.get_dummies(d[tfe].astype(str), dtype=float).to_numpy()[:, 1:]
    Cm = np.column_stack([Fd, Td, d[ctrl].to_numpy(float)])
    Cw = Cm * sw[:, None]
    yw, xw, zw = (resid_w((d[c].to_numpy(float) * sw)[:, None], Cw)[:, 0] for c in ['y', 'x', 'z'])
    den = zw @ xw
    b = (zw @ yw) / den
    ew = yw - b * xw
    times = sorted(d.t.unique())
    K = len(next(iter(shocks.values())))
    parts, gt = {}, {}
    for t in times:
        sup, W = links[t]
        si = {g: k for k, g in enumerate(sup)}
        m = (d.t == t).to_numpy()
        rows_ = d.loc[m, 'gvkey'].map(si).to_numpy()
        parts[t] = (m, rows_, W)
        ex_t = (w[m][:, None] * W[rows_]).sum(0)
        g = shocks[t]
        gt[t] = g - (np.sum(ex_t * g) / ex_t.sum() if ex_t.sum() > 0 else 0.0)
    cl = np.arange(K) if clus is None else pd.factorize(clus)[0]

    def R(e_w):
        acc = np.zeros(K)
        we = sw * e_w
        for t in times:
            m, rows_, W = parts[t]
            acc += gt[t] * (W[rows_].T @ we[m])
        return float(np.sum(np.bincount(cl, weights=acc) ** 2))
    se = np.sqrt(R(ew)) / abs(den)
    sdx = np.sqrt(np.sum(xw ** 2) / np.sum(w))
    t0 = (zw @ yw) / np.sqrt(R(yw))
    expo = sum((w[parts[t][0]][:, None] * parts[t][2][parts[t][1]]).sum(0) for t in times)
    sc = np.bincount(cl, weights=expo); sc = sc / sc.sum()
    cg, _ = pd.factorize(d.gvkey); G = cg.max() + 1
    v_c = np.sum(np.bincount(cg, weights=zw * ew) ** 2) / den ** 2 * G / (G - 1)
    fb, fse, _ = ols_cluster(xw, zw[:, None], d.gvkey.to_numpy(), Cm.shape[1] + 1)
    return dict(lin_pp=100 * b * sdx, lin_F=(fb[0] / fse[0]) ** 2, p_lin_supplier=2 * (1 - stats.norm.cdf(abs(b) / np.sqrt(v_c))),
                p_akm=2 * (1 - stats.norm.cdf(abs(b) / se)), p_akm0=2 * (1 - stats.norm.cdf(abs(t0))), eff_clusters=1 / np.sum(sc ** 2))


def lin_setup(d, ctrl, P, clus, extra=()):
    "Weighted residualisation and per-year link blocks shared by the linear estimators."
    d = d.reset_index(drop=True)
    w = d.n.to_numpy(float); sw = np.sqrt(w)
    Fd = pd.get_dummies(d.gvkey.astype(str), dtype=float).to_numpy()
    Td = pd.get_dummies(d.t.astype(str), dtype=float).to_numpy()[:, 1:]
    Cm = np.column_stack([Fd, Td, d[ctrl].to_numpy(float)])
    cols = ['y', 'x', 'z'] + list(extra)
    M = resid_w(d[cols].to_numpy(float) * sw[:, None], Cm * sw[:, None])
    R = {c: M[:, j] for j, c in enumerate(cols)}
    tix = {t: j for j, t in enumerate(P['ty'])}
    blocks = []
    for t in sorted(d.t.unique()):
        m = (d.t == t).to_numpy()
        Wt = P['W'][d.loc[m, 'gvkey'].map(P['si']).to_numpy()]
        blocks.append((m, Wt, tix[t]))
    cl = pd.factorize(clus)[0]
    return d, w, sw, Cm, R, blocks, cl


def recentred(Zk, blocks, w):
    "Partner shocks demeaned within year with exposure weights (Borusyak, Hull and Jaravel), per block."
    out = []
    for m, Wt, j in blocks:
        ex = (w[m][:, None] * Wt).sum(0)
        g = Zk[:, j]
        out.append(g - (np.sum(ex * g) / ex.sum() if ex.sum() > 0 else 0.0))
    return out


def score(v_w, sw, blocks, gts, cl):
    "Partner-group scores sum_{k in group} sum_t gtilde_kt sum_i W_ik(t) w_i e_i, for e_w = sqrt(w) e."
    acc = 0.0
    we = sw * v_w
    for (m, Wt, _), g in zip(blocks, gts):
        acc = acc + g * (Wt.T @ we[m])
    return np.bincount(cl, weights=acc)


def invert(a, c, A1, A2):
    "AKM0 95% set {b : (a - b c)^2 <= Q sum_g (A1_g - b A2_g)^2}: (lo, hi, kind); for two rays, (-inf, lo] and [hi, inf)."
    al, be, ga = c ** 2 - CHI2_95 * A2 @ A2, -2 * (a * c - CHI2_95 * A1 @ A2), a ** 2 - CHI2_95 * A1 @ A1
    disc = be ** 2 - 4 * al * ga
    if disc <= 0:
        return (-np.inf, np.inf, 'whole line') if al < 0 else (np.nan, np.nan, 'empty')
    r = np.sort([(-be - np.sqrt(disc)) / (2 * al), (-be + np.sqrt(disc)) / (2 * al)])
    return (r[0], r[1], 'bounded') if al > 0 else (r[0], r[1], 'two rays')


def akm0_iv(d, ctrl, P, clus):
    """Linear 2SLS and reduced form (patent-weighted), AKM0 p and AKM0 95% sets by exact inversion, firm-cluster SEs,
    shock-level and firm-cluster first stage.  Per SD as in Table E1: unweighted SD net of firm and year effects."""
    sdx, sdz = demean(d, ['x', 'z'], 'gvkey', 'E').std(0)
    d, w, sw, Cm, R, blocks, cl = lin_setup(d, ctrl, P, clus)
    yw, xw, zw = R['y'], R['x'], R['z']
    gts = recentred(P['Zk'], blocks, w)
    Ay, Ax, Az = (score(v, sw, blocks, gts, cl) for v in (yw, xw, zw))
    a, c = zw @ yw, zw @ xw
    b, rho = a / c, a / (zw @ zw)
    p0 = 2 * (1 - stats.norm.cdf(abs(a) / np.sqrt(np.sum(Ay ** 2))))     # same test for the IV and the reduced form
    lo_, hi_, kind = invert(a, c, Ay, Ax)
    rlo, rhi, rkind = invert(a, zw @ zw, Ay, Az)
    se_akm = np.sqrt(np.sum((Ay - b * Ax) ** 2)) / abs(c)
    fb, fse, _ = ols_cluster(xw, zw[:, None], d.gvkey.to_numpy(), Cm.shape[1] + 1)
    cg = pd.factorize(d.gvkey)[0]; G = cg.max() + 1
    clse = lambda num, den: np.sqrt(np.sum(np.bincount(cg, weights=num) ** 2) * G / (G - 1)) / abs(den)
    se_iv = clse(zw * (yw - b * xw), c)
    se_rf = clse(zw * (yw - rho * zw), zw @ zw)
    bo = (xw @ yw) / (xw @ xw)
    se_o = clse(xw * (yw - bo * xw), xw @ xw)
    expo = sum((w[m][:, None] * Wt).sum(0) for m, Wt, _ in blocks)
    sc = np.bincount(cl, weights=expo); sc = sc / sc.sum()
    k, kz = 100 * sdx, 100 * sdz
    return dict(lin_pp=k * b, lin_lo=k * lo_, lin_hi=k * hi_, lin_ci=kind, p_akm0=p0,
                lin_lo_firm=k * (b - 1.96 * se_iv), lin_hi_firm=k * (b + 1.96 * se_iv), p_lin_firm=2 * (1 - stats.norm.cdf(abs(b / se_iv))),
                p_akm=2 * (1 - stats.norm.cdf(abs(b) / se_akm)),
                rf_pp=kz * rho, rf_lo=kz * rlo, rf_hi=kz * rhi, rf_ci=rkind, rf_lo_firm=kz * (rho - 1.96 * se_rf),
                rf_hi_firm=kz * (rho + 1.96 * se_rf), lin_F_firm=(fb[0] / fse[0]) ** 2, lin_F_shock=c ** 2 / (Ax @ Ax),
                lin_noIV_pp=k * bo, lin_noIV_lo=k * (bo - 1.96 * se_o), lin_noIV_hi=k * (bo + 1.96 * se_o),
                eff_shock_groups=1 / np.sum(sc ** 2), N_lin=len(d))


def invert_q(a, c, A1, A2, q):
    "AKM0 set {b : (a - b c)^2 <= q sum_g (A1_g - b A2_g)^2} at critical value q (chi-square(1))."
    al, be, ga = c ** 2 - q * A2 @ A2, -2 * (a * c - q * A1 @ A2), a ** 2 - q * A1 @ A1
    disc = be ** 2 - 4 * al * ga
    if disc <= 0:
        return (-np.inf, np.inf, 'whole line') if al < 0 else (np.nan, np.nan, 'empty')
    r = np.sort([(-be - np.sqrt(disc)) / (2 * al), (-be + np.sqrt(disc)) / (2 * al)])
    return (r[0], r[1], 'bounded') if al > 0 else (r[0], r[1], 'two rays')


def bands(d, ctrl, P, clus):
    sdx, sdz = demean(d, ['x', 'z'], 'gvkey', 'E').std(0)
    d, w, sw, Cm, R, blocks, cl = lin_setup(d, ctrl, P, clus)
    yw, xw, zw = R['y'], R['x'], R['z']
    gts = recentred(P['Zk'], blocks, w)
    Ay, Ax, Az = (score(v, sw, blocks, gts, cl) for v in (yw, xw, zw))
    a, c, zz = zw @ yw, zw @ xw, zw @ zw
    b, rho = a / c, a / zz
    cg = pd.factorize(d.gvkey)[0]; G = cg.max() + 1
    clse = lambda num, den: np.sqrt(np.sum(np.bincount(cg, weights=num) ** 2) * G / (G - 1)) / abs(den)
    se_iv, se_rf = clse(zw * (yw - b * xw), c), clse(zw * (yw - rho * zw), zz)
    k, kz = 100 * sdx, 100 * sdz
    out = dict(sdx=sdx, sdz=sdz, lin_pp=k * b, rf_pp=kz * rho, p_akm0=2 * (1 - stats.norm.cdf(abs(a) / np.sqrt(Ay @ Ay))),
               p_lin_firm=2 * (1 - stats.norm.cdf(abs(b / se_iv))), p_rf_firm=2 * (1 - stats.norm.cdf(abs(rho / se_rf))),
               lin_F_firm=None, lin_F_shock=c ** 2 / (Ax @ Ax))
    # AKM (Adao, Kolesar and Morales 2019) shock-level standard errors, residuals at the estimate (Wald, not null-imposed)
    out['lin_se_akm'] = k * np.sqrt(np.sum((Ay - b * Ax) ** 2)) / abs(c)
    out['rf_se_akm'] = kz * np.sqrt(np.sum((Ay - rho * Az) ** 2)) / zz
    fb, fse, _ = ols_cluster(xw, zw[:, None], d.gvkey.to_numpy(), Cm.shape[1] + 1)
    out['lin_F_firm'] = (fb[0] / fse[0]) ** 2
    for lv in LEVELS:
        zc = stats.norm.ppf(0.5 + lv / 200)
        lo, hi, kind = invert_q(a, c, Ay, Ax, zc ** 2)
        rlo, rhi, rkind = invert_q(a, zz, Ay, Az, zc ** 2)
        out.update({f'lin_lo{lv}': k * lo, f'lin_hi{lv}': k * hi, f'lin_kind{lv}': kind,
                    f'rf_lo{lv}': kz * rlo, f'rf_hi{lv}': kz * rhi, f'rf_kind{lv}': rkind,
                    f'lin_lo{lv}_firm': k * (b - zc * se_iv), f'lin_hi{lv}_firm': k * (b + zc * se_iv),
                    f'rf_lo{lv}_firm': kz * (rho - zc * se_rf), f'rf_hi{lv}_firm': kz * (rho + zc * se_rf)})
    return out



# ----------------------------------------------------------------------------------------------------------------------
# Sector level: sectors 1-93 built from the same firms, links and class mixes
# ----------------------------------------------------------------------------------------------------------------------



def sector_setup(ctx):
    "Sector x year patents, shrunk shares, class mixes and the firm-built link matrix (customers' sales to sector j / sector j's sales)."
    sec, cs, fs = ctx['sec'], ctx['cs'], ctx['fs']
    allsec = ctx['allsec']
    Fc, Fa = ctx['Fc'], ctx['Fa']
    keep = Fc.index.intersection(sec.index)
    Sc = Fc.loc[keep].groupby(sec.reindex(keep).to_numpy()).sum().reindex(SECS).fillna(0)
    Sa = Fa.loc[keep].groupby(sec.reindex(keep).to_numpy()).sum().reindex(SECS).fillna(0)
    Gprev = pd.DataFrame(shrunk_share(trailing(Sc.to_numpy(), -5, -1), trailing(Sa.to_numpy(), -5, -1)), index=SECS, columns=YEARS)
    s = ctx['s'][ctx['s'].gvkey.isin(sec.index)].copy()
    s['sector'] = s.gvkey.map(sec)

    def smix(lo, hi):
        m = s[s.year.between(lo, hi)].groupby(['sector', 'cpc_subclass']).m.sum()
        m = (m / m.groupby(level=0).transform('sum')).unstack().reindex(columns=ctx['classes']).fillna(0)
        return m.reindex(SECS)

    def fa_matrix(lo, hi):                       # M[j, i] = sales of sector-i suppliers to sector-j customers / j's sales
        l = cs[cs.fy.between(lo, hi)].copy()
        l['si'] = l.gvkey.map(sec); l['sj'] = l.cust.map(allsec)
        l = l.dropna(subset=['si', 'sj'])
        l = l[l.si.isin(SECS) & l.sj.isin(SECS) & (l.si != l.sj)]
        flow = l.groupby(['sj', 'si']).salecs.sum()
        f = fs[fs.fyear.between(lo, hi)].groupby('gvkey').sale.mean()
        sales = f.groupby(allsec.reindex(f.index).to_numpy()).sum().reindex(SECS)
        M = np.zeros((len(SECS), len(SECS)))
        for (j, i), v in flow.items():
            if sales.get(j, 0) > 0:
                M[int(j) - 1, int(i) - 1] = v / sales[j]
        return M
    return dict(Sc=Sc, Sa=Sa, Gprev=Gprev, smix=smix, fa=fa_matrix)


def annual(ctx, Sc, Sa, Gprev, smix, fa_matrix, lo, hi):
    S = fa_matrix(lo, hi)
    np.fill_diagonal(S, 0.0)
    mix = smix(lo, hi)
    has = mix.sum(1).to_numpy() > 0
    ty = YEARS[YEARS > hi]
    Zk = np.nan_to_num(mix.to_numpy()) @ ctx['gprev'][ty].to_numpy()
    Zk[~has] = np.nanmean(Zk[has], axis=0)
    Gk = Gprev[ty].to_numpy()
    Wt = S.T * has[None, :]
    own = np.nan_to_num(mix.to_numpy()) @ ctx['gcurr'][ty].to_numpy()
    d = pd.DataFrame({'gvkey': np.repeat(SECS, len(ty)), 'E': np.tile(ty, len(SECS)), 'x': (Wt @ Gk).ravel(),
                      'z': (Wt @ Zk).ravel(), 'own': own.ravel(), 'mass': np.repeat(Wt.sum(1), len(ty))})
    d = d[np.repeat(has & (Wt.sum(1) > 0), len(ty))]
    d['clean'] = [Sc.at[i, t] for i, t in zip(d.gvkey, d.E)]
    d['noncl'] = [Sa.at[i, t] - Sc.at[i, t] for i, t in zip(d.gvkey, d.E)]
    d = d[d.noncl > 0].reset_index(drop=True)
    ctrl = ['own']
    for b in sorted({(t - 1998) // 5 for t in ty})[1:]:
        d[f'mass_b{b}'] = d.mass * (((d.E - 1998) // 5) == b)
        ctrl.append(f'mass_b{b}')
    return d, ctrl



# ----------------------------------------------------------------------------------------------------------------------
# Table E1 / E3: the nine columns (one regression per column)
#   Firm level (unit f, year t; firm and year effects): (1) Poisson, no IV  (2) OLS first stage  (3) Poisson reduced form
#   (4) Poisson control-function IV, supplier bootstrap  (5) linear 2SLS of the green share, weighted by patents.
#   Sector level: (6)-(8) as (1)-(3); (9) Poisson IV by GMM (Wooldridge's within moment), sector-clustered.
#   Coefficients are per within-SD of the regressor (SD after removing unit and year effects, estimation sample, unweighted).
# ----------------------------------------------------------------------------------------------------------------------



def pois_ll(y, mu):
    mu = np.maximum(mu, 1e-300)
    return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))


def pseudo_r2(d, y, off, mu):
    yv = d[y].to_numpy(float)
    on = d[off].to_numpy(float)
    mu0 = on * yv.sum() / on.sum()
    return 1 - pois_ll(yv, mu) / pois_ll(yv, mu0)


def fepois_fit(d, xcols, unit='gvkey', time='E', cluster=None):
    "fepois plus fitted means (rebuilt from the coefficients) for the pseudo R2."
    r = fepois(d, 'clean', xcols, unit, offset='noncl', time=time, cluster=cluster)
    assert r.attrs['N'] == len(d), 'fepois dropped rows: the sample is not the trimmed one'
    T = pd.get_dummies(d[time], prefix='per', drop_first=True, dtype=float)
    X = np.column_stack([d[xcols].to_numpy(float), T.to_numpy(float)])
    eta = X @ r.loc[list(xcols) + list(T.columns), 'coef'].to_numpy() + np.log(d.noncl.to_numpy(float))
    f, _ = pd.factorize(d[unit])
    eta = eta - np.bincount(f, eta)[f] / np.bincount(f)[f]
    e = np.exp(eta)
    mu = np.bincount(f, d.clean.to_numpy(float))[f] * e / np.bincount(f, e)[f]
    return r, pseudo_r2(d, 'clean', 'noncl', mu)


def within_sd(d, cols, unit='gvkey', time='E'):
    Ad = demean(d, cols, unit, time)
    return dict(zip(cols, Ad.std(0)))


def first_stage(d, ctrl, unit='gvkey', time='E'):
    Ad = demean(d, ['x', 'z'] + ctrl, unit, time)
    nfe = d[unit].nunique() + d[time].nunique()
    b, se, v = ols_cluster(Ad[:, 0], Ad[:, 1:], d[unit].to_numpy(), nfe + Ad.shape[1] - 1)
    return dict(zip(['z'] + ctrl, b)), dict(zip(['z'] + ctrl, se)), v, 1 - np.sum(v ** 2) / np.sum(Ad[:, 0] ** 2)


def rows_out(col, coefs, ses, sd, scale_y=1.0, sd_y=1.0):
    "coefficients and SEs per within-SD of each regressor (divided by sd_y for the first stage)."
    return [dict(column=col, term=t, coef=coefs[t] * sd[t] * scale_y / sd_y, se=ses[t] * sd[t] * scale_y / sd_y)
            for t in coefs.keys()]


def tsls(dl, cl, sd, base, col=5, unit='supplier'):
    dl = dl.reset_index(drop=True)
    w = dl.n.to_numpy(float); sw = np.sqrt(w)
    Fd = pd.get_dummies(dl.gvkey.astype(str), dtype=float).to_numpy()
    Td = pd.get_dummies(dl.t.astype(str), dtype=float).to_numpy()[:, 1:]
    Xo = dl[cl].to_numpy(float)
    X = np.column_stack([dl.x.to_numpy(float), Xo, Fd, Td]) * sw[:, None]
    Z = np.column_stack([dl.z.to_numpy(float), Xo, Fd, Td]) * sw[:, None]
    y = dl.y.to_numpy(float) * sw
    Xh = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    XhX = np.linalg.pinv(Xh.T @ Xh)
    b = XhX @ Xh.T @ y
    e = y - X @ b
    cg, _ = pd.factorize(dl.gvkey); G = cg.max() + 1
    Sg = np.zeros((G, X.shape[1])); np.add.at(Sg, cg, Xh * e[:, None])
    N, K = len(y), np.linalg.matrix_rank(X)
    V = XhX @ (Sg.T @ Sg) @ XhX * G / (G - 1) * (N - 1) / (N - K)
    se = np.sqrt(np.diag(V))
    # within R2 (structural residuals) and first-stage F (weighted, supplier cluster)
    D = np.column_stack([Fd, Td]) * sw[:, None]
    yd = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
    Zx = np.column_stack([dl.z.to_numpy(float), Xo, Fd, Td]) * sw[:, None]
    xw = dl.x.to_numpy(float) * sw
    bf = np.linalg.lstsq(Zx, xw, rcond=None)[0]
    uf = xw - Zx @ bf
    ZZi = np.linalg.pinv(Zx.T @ Zx)
    Sf = np.zeros((G, Zx.shape[1])); np.add.at(Sf, cg, Zx * uf[:, None])
    Vf = ZZi @ (Sf.T @ Sf) @ ZZi * G / (G - 1) * (N - 1) / (N - np.linalg.matrix_rank(Zx))
    rows = rows_out(col, {'x': b[0], 'own': b[1]}, {'x': se[0], 'own': se[1]}, sd, scale_y=100)
    return dict(column=col, model='2SLS, weighted by patents', dep='green share of patents', se_type=f'{unit} cluster',
                within_r2=1 - np.sum(e ** 2) / np.sum(yd ** 2), F=(bf[0] / np.sqrt(Vf[0, 0])) ** 2,
                mean_dep=100 * dl.clean.sum() / dl.n.sum(), rows=rows, **base)


def gmm_cluster_se(g, exog, m, z='zz'):
    "explore25.poisson_sandwich, returning the sector-clustered SE of every parameter and the fitted means."
    per = pd.get_dummies(g.period, prefix='per', drop_first=True, dtype=float)
    W = pd.concat([g[['down_x_lag'] + exog].astype(float), per], axis=1)
    W.columns = [str(c) for c in W.columns]
    th = m.params.reindex(W.columns).to_numpy(float)
    Wv = W.to_numpy(float)
    Z = np.column_stack([g[[z] + exog].to_numpy(float), per.to_numpy(float)])
    y = g.clean_pat_count.to_numpy(float); off = np.log(g.pat_count_nc.to_numpy(float))
    codes, uniq = pd.factorize(g.BLS_Industry.to_numpy())
    N, G_, K = len(y), len(uniq), Wv.shape[1]
    mu = np.exp(Wv @ th + off)
    musum = np.bincount(codes, weights=mu, minlength=G_); ysum = np.bincount(codes, weights=y, minlength=G_)
    rho = ysum / musum
    u = y - rho[codes] * mu
    wbar = np.column_stack([np.bincount(codes, weights=mu * Wv[:, k], minlength=G_) for k in range(K)]) / musum[:, None]
    dU = -(rho[codes] * mu)[:, None] * (Wv - wbar[codes])
    Gi = np.linalg.inv(Z.T @ dU / N)
    zbar = np.column_stack([np.bincount(codes, weights=mu * Z[:, k], minlength=G_) for k in range(Z.shape[1])]) / musum[:, None]
    sc = (Z - zbar[codes]) * u[:, None]
    adj = (N - 1) / max(N - K - G_, 1)
    Sg = np.vstack([sc[codes == q].sum(0) for q in range(G_)])
    Om = Sg.T @ Sg / N ** 2 * (G_ / (G_ - 1)) * adj
    se = np.sqrt(np.diag(Gi @ Om @ Gi.T))
    return pd.Series(se, index=W.columns), rho[codes] * mu


def poisson_iv(d, exog, z='zz'):
    with contextlib.redirect_stdout(io.StringIO()):
        return es.fit_poisson_iv(d, 'clean_pat_count', 'pat_count_nc', ['down_x_lag'] + exog, ['down_x_lag'], [z])


def cf_iv(d, ctrl, v, seed=11, nboot=None):
    """Control-function Poisson IV with a unit-block bootstrap (resample units with replacement, re-estimate the first stage and the Poisson
    step; seed 11, IQR / 1.349). d: panel with gvkey (unit), E (year), clean, noncl, x, z and ctrl; v: first-stage residuals. Returns the
    Poisson fit with the residual, the bootstrap standard errors of (x, own), the number of draws, pseudo R2."""
    d = d.assign(vhat=v)
    r, pr2 = fepois_fit(d, ['x', 'vhat'] + ctrl)
    rng = np.random.default_rng(seed)
    units = d.gvkey.unique(); grp = {g: ix for g, ix in d.groupby('gvkey').groups.items()}
    draws = []
    for _ in range(nboot or NBOOT):
        pick = rng.choice(units, len(units), replace=True)
        parts_ = []
        for k, g in enumerate(pick):
            p_ = d.loc[grp[g]].copy(); p_['gvkey'] = f'{g}_{k}'; parts_.append(p_)
        db = trim(pd.concat(parts_, ignore_index=True), 'E')
        try:
            Ab = demean(db, ['x', 'z'] + ctrl, 'gvkey', 'E')
            _, _, vb = ols_cluster(Ab[:, 0], Ab[:, 1:], db.gvkey.to_numpy(), 1)
            db['vhat'] = vb
            rb = fp(db, 'clean', ['x', 'vhat'] + ctrl, 'gvkey', 'noncl', 'E')
            draws.append([rb.at['x', 'coef'], rb.at['own', 'coef']])
        except Exception:
            pass
    q = np.percentile(np.array(draws), [25, 75], axis=0)
    return r, pd.Series((q[1] - q[0]) / 1.349, index=['x', 'own']), len(draws), pr2


def gmm_firm(d, ctrl, sd):
    """Poisson IV by GMM (Poisson fixed-effects moment conditions) with standard errors clustered by firm; the firm-level analog of the
    sector-level column 9. Returns coefficient and s.e. of x per within-standard-deviation, and the p-value."""
    g = d.rename(columns={'gvkey': 'BLS_Industry', 'E': 'period', 'clean': 'clean_pat_count', 'noncl': 'pat_count_nc', 'x': 'down_x_lag', 'z': 'zz'})
    m = poisson_iv(g, ctrl, 'zz')
    se, _ = gmm_cluster_se(g, ctrl, m)
    b, s = m.params['down_x_lag'], se['down_x_lag']
    return dict(coef=b * sd['x'], se=s * sd['x'], p=2 * (1 - stats.norm.cdf(abs(b / s))), converged=bool(m.converged))


def columns_firm(P, x_dep, se_unit):
    "Columns 1-5 of the table for one direction. P: panel dict (base_panel or io_panel). x_dep, se_unit: labels. Returns C, S, (dl, cl)."
    d, ctrl = P['d'], P['ctrl']
    d = trim(d[list(dict.fromkeys(['clean', 'noncl', 'x', 'z', 'gvkey', 'E', 'sector', 'main_cust'] + ctrl))].dropna(), 'E')
    sd = within_sd(d, ['x', 'z', 'own'])
    base = dict(level='firm', N=len(d), units=d.gvkey.nunique(), customers=d.main_cust.nunique())
    C, S = [], []
    # (1) no IV
    r, pr2 = fepois_fit(d, ['x'] + ctrl)
    C += rows_out(1, r.coef[['x', 'own']], r.se[['x', 'own']], sd)
    S.append(dict(column=1, model='Poisson', dep='green patents', se_type=f'{se_unit} cluster', pseudo_r2=pr2, mean_dep=d.clean.mean(), **base))
    # (2) first stage
    b, se, v, wr2 = first_stage(d, ctrl)
    C += rows_out(2, {k: b[k] for k in ['z', 'own']}, {k: se[k] for k in ['z', 'own']}, sd, sd_y=sd['x'])
    S.append(dict(column=2, model='OLS', dep=x_dep, se_type=f'{se_unit} cluster', within_r2=wr2,
                  F=(b['z'] / se['z']) ** 2, mean_dep=d.x.mean(), **base))
    # (3) reduced form
    r, pr2 = fepois_fit(d, ['z'] + ctrl)
    C += rows_out(3, r.coef[['z', 'own']], r.se[['z', 'own']], sd)
    S.append(dict(column=3, model='Poisson', dep='green patents', se_type=f'{se_unit} cluster', pseudo_r2=pr2, mean_dep=d.clean.mean(), **base))
    # (4) control-function IV, supplier bootstrap (seed 11, IQR / 1.349)
    d['vhat'] = v
    r, pr2 = fepois_fit(d, ['x', 'vhat'] + ctrl)
    rng = np.random.default_rng(11)
    firms = d.gvkey.unique(); grp = {g: ix for g, ix in d.groupby('gvkey').groups.items()}
    draws = []
    for _ in range(NBOOT):
        pick = rng.choice(firms, len(firms), replace=True)
        parts_ = []
        for k, g in enumerate(pick):
            p_ = d.loc[grp[g]].copy(); p_['gvkey'] = f'{g}_{k}'; parts_.append(p_)
        db = trim(pd.concat(parts_, ignore_index=True), 'E')
        try:
            Ab = demean(db, ['x', 'z'] + ctrl, 'gvkey', 'E')
            _, _, vb = ols_cluster(Ab[:, 0], Ab[:, 1:], db.gvkey.to_numpy(), 1)
            db['vhat'] = vb
            rb = fp(db, 'clean', ['x', 'vhat'] + ctrl, 'gvkey', 'noncl', 'E')
            draws.append([rb.at['x', 'coef'], rb.at['own', 'coef']])
        except Exception:
            pass
    q = np.percentile(np.array(draws), [25, 75], axis=0)
    bse = (q[1] - q[0]) / 1.349
    C += rows_out(4, r.coef[['x', 'own']], pd.Series(bse, index=['x', 'own']), sd)
    S.append(dict(column=4, model='Poisson IV (control function)', dep='green patents', se_type=f'{se_unit} bootstrap',
                  pseudo_r2=pr2, F=(b['z'] / se['z']) ** 2, boot_draws=len(draws),
                  p_x=2 * (1 - stats.norm.cdf(abs(r.at['x', 'coef'] / bse[0]))), mean_dep=d.clean.mean(), **base))
    # (5) linear 2SLS weighted by patents, supplier cluster (the AKM0 model); same sample as the Poisson columns
    dl, cl = prep(P['d'], 'E', ['own'])
    assert len(dl) == len(d)
    S.append(tsls(dl, cl, sd, base))
    C += S[-1].pop('rows')
    return C, S, (dl, cl)


def columns_sector(d, ctrl, x_dep):
    "Columns 6-9 (and the linear 2SLS, column 10) of the table. d, ctrl: trimmed sector-year panel."
    sd = within_sd(d, ['x', 'z', 'own'])
    base = dict(level='sector', N=len(d), units=d.gvkey.nunique())
    C, S = [], []
    r, pr2 = fepois_fit(d, ['x'] + ctrl)
    C += rows_out(6, r.coef[['x', 'own']], r.se[['x', 'own']], sd)
    S.append(dict(column=6, model='Poisson', dep='green patents', se_type='sector cluster', pseudo_r2=pr2, mean_dep=d.clean.mean(), **base))
    b, se, v, wr2 = first_stage(d, ctrl)
    C += rows_out(7, {k: b[k] for k in ['z', 'own']}, {k: se[k] for k in ['z', 'own']}, sd, sd_y=sd['x'])
    S.append(dict(column=7, model='OLS', dep=x_dep, se_type='sector cluster', within_r2=wr2,
                  F=(b['z'] / se['z']) ** 2, mean_dep=d.x.mean(), **base))
    r, pr2 = fepois_fit(d, ['z'] + ctrl)
    C += rows_out(8, r.coef[['z', 'own']], r.se[['z', 'own']], sd)
    S.append(dict(column=8, model='Poisson', dep='green patents', se_type='sector cluster', pseudo_r2=pr2, mean_dep=d.clean.mean(), **base))
    g = d.rename(columns={'gvkey': 'BLS_Industry', 'E': 'period', 'clean': 'clean_pat_count', 'noncl': 'pat_count_nc',
                          'x': 'down_x_lag', 'z': 'zz'})
    m = poisson_iv(g, ctrl, 'zz')
    sec_se, mu = gmm_cluster_se(g, ctrl, m)
    C += rows_out(9, {'x': m.params['down_x_lag'], 'own': m.params['own']}, {'x': sec_se['down_x_lag'], 'own': sec_se['own']}, sd)
    S.append(dict(column=9, model='Poisson IV (GMM)', dep='green patents', se_type='sector cluster',
                  pseudo_r2=pseudo_r2(d, 'clean', 'noncl', mu), F=(b['z'] / se['z']) ** 2, converged=bool(m.converged),
                  p_x=2 * (1 - stats.norm.cdf(abs(m.params['down_x_lag'] / sec_se['down_x_lag']))), mean_dep=d.clean.mean(), **base))
    # (11) control-function IV with a sector-block bootstrap (the sector analog of firm column 4)
    r, bse, nd, pr2 = cf_iv(d, ctrl, v)
    C += rows_out(11, r.coef[['x', 'own']], bse, sd)
    S.append(dict(column=11, model='Poisson IV (control function)', dep='green patents', se_type='sector bootstrap', pseudo_r2=pr2, F=(b['z'] / se['z']) ** 2,
                  boot_draws=nd, p_x=2 * (1 - stats.norm.cdf(abs(r.at['x', 'coef'] / bse['x']))), mean_dep=d.clean.mean(), **base))
    # (10) linear 2SLS weighted by patents, sector cluster (the sector analog of firm column 5); column 10 so that the numbering 6-9 is unchanged
    try:
        dl, cl = prep(d, 'E', ['own'])
        S.append(tsls(dl, cl, sd, base, col=10, unit='sector'))
        C += S[-1].pop('rows')
    except Exception as ex:
        print(f'sector 2SLS failed: {type(ex).__name__}: {ex}', flush=True)
    return C, S


def e1_sector_panel(ctx):
    "Table E1 sector panel (customers' sectors as partners, firm-built links), FY1990-94 base."
    Q = sector_setup(ctx)
    d, ctrl = annual(ctx, Q['Sc'], Q['Sa'], Q['Gprev'], Q['smix'], Q['fa'], LO, HI)
    return trim(d.dropna(subset=['x', 'z'] + ctrl).copy(), 'E'), ctrl


def e1_akm(ctx, P):
    "AKM0 for column 5 with customers' shocks grouped by customer firm and by customer sector (GM, Ford and Chrysler count as one)."
    links = {t: (P['sup'], P['W']) for t in P['ty']}
    shocks = {t: P['Zk'][:, j] for j, t in enumerate(P['ty'])}
    dl, cl = prep(P['d'], 'E', ['own'])
    clus = np.array([ctx['allsec'].get(int(c), -int(c)) for c in P['custs']])
    a_f = akm_general(dl, cl, links, shocks, tfe='t')
    a_s = akm_general(dl, cl, links, shocks, tfe='t', clus=clus)
    return dict(check='shocks clustered by customer sector', base=f'{LO}-{HI}',
                N=len(dl), lin_pp=a_s['lin_pp'], lin_F=a_s['lin_F'], p_akm0_firmclusters=a_f['p_akm0'],
                eff_firm_clusters=a_f['eff_clusters'], p_akm=a_s['p_akm'], p_akm0=a_s['p_akm0'], eff_clusters=a_s['eff_clusters'])




# ----------------------------------------------------------------------------------------------------------------------
# Inputs of Table E1 that no regression stores: size of supplier-clustered inference under reshuffles; matched share of sales
# ----------------------------------------------------------------------------------------------------------------------



def size_under_reshuffles(P):
    d, ctrl, si, W, Zk, ty, custs = P['d'], P['ctrl'], P['si'], P['W'], P['Zk'], P['ty'], P['custs']
    tix = {t: j for j, t in enumerate(ty)}
    rng = np.random.default_rng(2026)                       # as explore32
    perms = [rng.permutation(len(custs)) for _ in range(NRI)]
    store = {}

    def zvals(dd, perm):
        ZZ = W @ (Zk if perm is None else Zk[perm])
        return ZZ[dd.gvkey.map(si).to_numpy(), dd.E.map(tix).to_numpy()]

    def ri(dd, z, ctrl_, fe):                               # explore32's reshuffle, keeping every draw's |t|
        draws = [zvals(dd, p) for p in perms]
        dd = dd.copy()
        dd['mu'] = np.mean(draws, axis=0)
        dd[z] = zvals(dd, None)
        st = fp(dd, 'clean', [z, 'mu'] + ctrl_, 'gvkey', 'noncl', fe)
        t0 = st.at[z, 'coef'] / st.at[z, 'se']
        tt = []
        for zz in draws:
            dd[z] = zz
            try:
                rr = fp(dd, 'clean', [z, 'mu'] + ctrl_, 'gvkey', 'noncl', fe)
                tt.append(rr.at[z, 'coef'] / rr.at[z, 'se'])
            except Exception:
                pass
        store['tt'] = np.asarray(tt)
        return ri_p(t0, tt)
    r = estimate(d, 'z', ctrl, 'E', ri=ri)
    tt = store['tt'][np.isfinite(store['tt'])]
    return dict(N=r['N'], rf_p_ri=r['rf_p_ri'], reshuffle_draws=len(tt), reject_rate_supplier_cluster=float(np.mean(np.abs(tt) > 1.96)))


def matched_share(cs, lo, hi):
    "Share of disclosed COMPANY-customer sales matched to a Compustat firm in fiscal years lo-hi (cs: Clean Data/Cust_Segments)."
    x = cs[cs.fy.between(lo, hi)]
    return float(x.loc[x.cust.notna(), 'salecs'].sum() / x.salecs.sum())




# ----------------------------------------------------------------------------------------------------------------------
# Timing of both directions (Section 2, Figure E1, Table E2)
#   Horizon h: the partners' window is the five years t-h-4 ... t-h (regressor: partners' shrunk green share; instrument:
#   partners' base class mix x outside class green shares). h = 1 is Table E1.
#   cs  customers' greening -> supplier: Compustat customer links, Table E1's firms.
#   sc  suppliers' greening -> customer: green content of the firm's inputs (1997 BLS input share x the supplier sector's
#       1990-94 green share x its green share in the window), every patenting firm in an IO sector.
#   Each regression controls for the other direction's instrument and the firm's own predicted green share over the
#   partners' window. Linear 2SLS weighted by patents; shock-level (AKM) standard errors and exact AKM0 confidence sets.
# ----------------------------------------------------------------------------------------------------------------------



def window_tables(ctx, h):
    "Firm shrunk green share and outside class green share over the five years t-h-4 ... t-h (columns t)."
    tr = lambda M: trailing(M, -h - 4, -h)
    Gh = pd.DataFrame(shrunk_share(tr(ctx['Fc'].to_numpy()), tr(ctx['Fa'].to_numpy())), index=ctx['Fc'].index, columns=YEARS)
    Oc, Oa = ctx['Oc'], ctx['Oa']
    gh = pd.DataFrame(shrunk_share(tr(Oc.to_numpy()), tr(Oa.to_numpy())), index=Oc.index, columns=YEARS)
    return Gh, gh


def panel(ctx, h):
    "Supplier-year panel (Table E1 design) with the partners' window moved to t-h-4 ... t-h."
    c = dict(ctx)
    c['Gprev'], c['gprev'] = window_tables(ctx, h)
    return base_panel(c, LO, HI)


def reverse_panel(ctx, lo, hi):
    """Customer-year panel from the same Compustat segment links, inverted: the unit is a customer k that is named by suppliers with patents,
    its exposure is the green share of those recorded suppliers, weighted by their sales to k relative to k's sales (the analog of the
    sector-level matrix of the sector panel). Instrument: the suppliers' predicted shares (1990-94 class mixes x outside class shifts)."""
    Fc, Fa, cs, fs = ctx['Fc'], ctx['Fa'], ctx['cs'], ctx['fs']
    cm = mixmat(ctx, lo, hi)
    partners = cm.index.to_numpy()
    pi = {f: k for k, f in enumerate(partners)}
    l = cs[cs.fy.between(lo, hi) & cs.gvkey.isin(partners) & cs.cust.notna() & (cs.cust != cs.gvkey)].copy()
    l = l.groupby(['cust', 'gvkey']).salecs.sum().reset_index()
    units = np.sort(l.cust.unique())
    ui = {u: k for k, u in enumerate(units)}
    W = np.zeros((len(units), len(partners)))
    W[l.cust.map(ui).to_numpy(), l.gvkey.map(pi).to_numpy()] = l.salecs.to_numpy()
    sale = fs[fs.fyear.between(lo, hi)].groupby('gvkey').sale.mean()
    W = W / sale.reindex(units).to_numpy()[:, None]
    ok = np.isfinite(W).all(1) & (W.sum(1) > 0)
    units, W = units[ok], W[ok]
    ty = YEARS[YEARS > hi]
    Zk = cm.to_numpy() @ ctx['gprev'][ty].to_numpy()
    Gk = ctx['Gprev'].reindex(index=partners, columns=ty)
    Gk = Gk.fillna(Gk.mean()).to_numpy()
    om = mixmat(ctx, lo, hi).reindex(units)
    om = om.fillna(mixmat(ctx, 1976, hi).reindex(units)).fillna(0)
    own = om.to_numpy() @ ctx['gcurr'][ty].to_numpy()
    has_own = om.sum(1).to_numpy() > 0
    d = pd.DataFrame({'gvkey': np.repeat(units, len(ty)), 'E': np.tile(ty, len(units)), 'x': (W @ Gk).ravel(), 'z': (W @ Zk).ravel(),
                      'own': own.ravel(), 'mass': np.repeat(W.sum(1), len(ty)), 'main_cust': np.repeat(partners[W.argmax(1)], len(ty))})
    d = d[np.repeat(has_own, len(ty))]
    d['clean'] = [Fc.at[g, t] if g in Fc.index else 0.0 for g, t in zip(d.gvkey, d.E)]
    d['noncl'] = [Fa.at[g, t] - Fc.at[g, t] if g in Fa.index else 0.0 for g, t in zip(d.gvkey, d.E)]
    d['sector'] = d.gvkey.map(ctx['sec'])
    d = d[d.noncl > 0].dropna(subset=['sector']).reset_index(drop=True)
    ctrl = ['own']
    for b in sorted({(t - 1998) // 5 for t in ty})[1:]:
        d[f'mass_b{b}'] = d.mass * (((d.E - 1998) // 5) == b)
        ctrl.append(f'mass_b{b}')
    return dict(d=d, ctrl=ctrl, sup=units, si={u: k for k, u in enumerate(units)}, W=W, Zk=Zk, ty=ty, custs=partners)


def panel_reverse(ctx, h):
    c = dict(ctx)
    c['Gprev'], c['gprev'] = window_tables(ctx, h)
    return reverse_panel(c, LO, HI)


def years_for(h):
    "Outcome years whose partner window lies inside 1976-2017, after the base period."
    return (max(YEARS[0] + h + 4, HI + 1), min(YEARS[-1], YEARS[-1] + h))



def own_pred(ctx, units, lo, hi):
    om = mixmat(ctx, lo, hi).reindex(units)
    om = om.fillna(mixmat(ctx, 1976, hi).reindex(units)).fillna(0)
    return om


def joint_data(own, oth):
    d = own['d'].copy()
    o = oth['d'][['gvkey', 'E', 'z', 'mass']].rename(columns={'z': 'z_oth', 'mass': 'mass_oth'})
    d = d.merge(o, on=['gvkey', 'E'], how='left')
    d['has_oth'] = d.z_oth.notna()
    d[['z_oth', 'mass_oth']] = d[['z_oth', 'mass_oth']].fillna(0.0)
    extra = ['z_oth']
    for c in own['ctrl']:
        if c.startswith('mass_b'):
            b = int(c[6:])
            d[f'massoth_b{b}'] = d.mass_oth * (((d.E - 1998) // 5) == b)
            extra.append(f'massoth_b{b}')
    return d, extra


def shrink(C, A, k):
    "Shrink counts C of A toward the aggregate share (rows: units, columns: years)."
    gbar = C.sum(0) / np.maximum(A.sum(0), 1e-12)
    return (C + k * gbar[None, :]) / (A + k)


def measures(Pc, Pa, Oc, Oa, h):
    "Partners' shrunk green share G and the class shift g used by the instrument, over the window t-h-4 ... t-h (columns YEARS)."
    win = lambda M: trailing(M, -(h + 4), -h)
    return shrink(win(Pc), win(Pa), KAPPA), shrink(win(Oc.to_numpy()), win(Oa.to_numpy()), KAPPA)


def kappa_moments(ctx, min_den=10):
    """Method-of-moments shrinkage: if firms' true green shares have variance p(1-p)/(kappa+1) around the pooled share p (a beta
    distribution with concentration kappa), the variance of observed shares exceeds the binomial sampling variance by that amount.
    Cells: firm x non-overlapping five-year window (1995-99, ..., 2010-14) with at least min_den patents."""
    Fc, Fa = ctx['Fc'], ctx['Fa']
    c, n = [], []
    for y0 in range(1995, 2015, 5):
        cols = [y for y in YEARS if y0 <= y < y0 + 5]
        c.append(Fc[cols].sum(1).to_numpy()); n.append(Fa[cols].sum(1).to_numpy())
    c, n = np.concatenate(c), np.concatenate(n)
    ok = n >= min_den
    p = c[ok].sum() / n[ok].sum()
    v = np.var(c[ok] / n[ok], ddof=1)
    samp = np.mean(p * (1 - p) / n[ok])
    return float(p * (1 - p) / max(v - samp, 1e-12) - 1), int(ok.sum())


def quick_firm(ctx, P, boot=False):
    "Firm-level core of Table E1 for a robustness variant: first stage, reduced form, control-function IV (no bootstrap), 2SLS and AKM0."
    d, ctrl = P['d'], P['ctrl']
    d = trim(d[list(dict.fromkeys(['clean', 'noncl', 'x', 'z', 'gvkey', 'E', 'sector', 'main_cust'] + ctrl))].dropna(), 'E')
    sd = within_sd(d, ['x', 'z', 'own'])
    b, se, v, _ = first_stage(d, ctrl)
    d = d.assign(vhat=v)
    iv, _ = fepois_fit(d, ['x', 'vhat'] + ctrl)
    iv_se = iv_p = np.nan
    if boot:                                                  # supplier bootstrap standard error of the control-function IV (as Table 2, column 2)
        _, bse, _, _ = cf_iv(d.drop(columns='vhat'), ctrl, v)
        iv_se = bse['x'] * sd['x']
        iv_p = 2 * (1 - stats.norm.cdf(abs(iv.at['x', 'coef'] / bse['x'])))
    rf, _ = fepois_fit(d, ['z'] + ctrl)
    dl, cl = prep(P['d'], 'E', ['own'])
    t5 = tsls(dl, cl, sd, dict(level='firm', N=len(d), units=d.gvkey.nunique(), customers=d.main_cust.nunique()))
    lin = [r for r in t5.pop('rows') if r['term'] == 'x'][0]
    akm = e1_akm(ctx, P)
    return dict(N=len(d), units=d.gvkey.nunique(), F=(b['z'] / se['z']) ** 2, rf_p=2 * (1 - stats.norm.cdf(abs(rf.at['z', 'coef'] / rf.at['z', 'se']))),
                iv=iv.at['x', 'coef'] * sd['x'], iv_se=iv_se, iv_p=iv_p, lin_pp=lin['coef'], akm0=akm['p_akm0'], akm0_firm=akm['p_akm0_firmclusters'])


def outside_counts(ctx):
    "Outside patents (not matched to any Compustat firm): class x year clean and total counts."
    return ctx['Oc'], ctx['Oa']


def setup(ctx):
    "Sector-level pieces shared by the IO-based measures."
    return dict(ctx=ctx, Q=sector_setup(ctx), Oc=ctx['Oc'], Oa=ctx['Oa'])


def own_window(ctx, units, h):
    "Own predicted green share over the partners' window t-h-4 .. t-h (own 1990-94 class mix, else 1976-94)."
    _, gh = window_tables(ctx, h)
    om = own_pred(ctx, units, LO, HI)
    ow = om.to_numpy() @ gh.to_numpy()
    return pd.DataFrame({'gvkey': np.repeat(units, len(YEARS)), 'E': np.tile(YEARS, len(units)), 'own_w': ow.ravel()})



def io_panel(G_, h, down=False, net='A'):
    "Suppliers' IO green-content exposure for every patenting firm in an IO sector, with the AKM0 pieces (W, Zk, ty)."
    ctx, Q, Oc, Oa = G_['ctx'], G_['Q'], G_['Oc'], G_['Oa']
    Fc, Fa, sec = ctx['Fc'], ctx['Fa'], ctx['sec']
    contemp = net.endswith('c')                         # regressor weights from the table of the outcome year (the instrument keeps 1997)
    Aio = ctx['IO'] if net[0] == 'A' else ctx['IOL']    # direct input requirements, or the Leontief inverse
    mix = Q['smix'](LO, HI); has = mix.sum(1).to_numpy() > 0
    Pc, Pa = Q['Sc'].to_numpy(), Q['Sa'].to_numpy()
    units = np.array(sorted(set(Fc.index) & set(sec.index)))
    us = sec.reindex(units).to_numpy().astype(int)
    ok = np.isin(us, SECS); units, us = units[ok], us[ok]
    yb = (YEARS >= LO) & (YEARS <= HI)
    gbase = shrink(Pc[:, yb].sum(1, keepdims=True), Pa[:, yb].sum(1, keepdims=True), KAPPA)[:, 0]
    ty = YEARS[YEARS >= max(HI + 1, YEARS[0] + h + 4)]
    sel = np.isin(YEARS, ty)
    G, g = measures(Pc, Pa, Oc, Oa, h)
    Gs = G[:, sel]
    Zs = np.nan_to_num(mix.to_numpy()) @ g[:, sel]; Zs[~has] = np.nanmean(Zs[has], axis=0)
    Wraw = Aio[SECS - 1][:, us - 1].T if down else Aio[us - 1][:, SECS - 1]       # down: customers j of the firm's sector, weight A[j, i]
    W = Wraw * ((has * gbase) if SB else has.astype(float))[None, :]
    om = own_pred(ctx, units, LO, HI)
    keep = om.sum(1).to_numpy() > 0
    units, W, om, us = units[keep], W[keep], om[keep], us[keep]
    own = om.to_numpy() @ ctx['gcurr'][ty].to_numpy()
    Xm = W @ Gs
    if contemp:
        mask = ((has * gbase) if SB else has.astype(float))[None, :]
        for k, t in enumerate(ty):
            My = ctx['IOY'][max(1997, int(t))]
            Wr = My[SECS - 1][:, us - 1].T if down else My[us - 1][:, SECS - 1]
            Xm[:, k] = (Wr * mask) @ Gs[:, k]
    d = pd.DataFrame({'gvkey': np.repeat(units, len(ty)), 'E': np.tile(ty, len(units)),
                      'x': Xm.ravel(), 'z': (W @ Zs).ravel(), 'mass': np.repeat(W.sum(1), len(ty)),
                      'own': own.ravel(), 'main_cust': np.repeat(np.asarray(SECS)[W.argmax(1)], len(ty))})
    Fcr, Far = Fc.reindex(units), Fa.reindex(units)
    d['clean'] = [Fcr.at[u, t] for u, t in zip(d.gvkey, d.E)]
    d['noncl'] = [Far.at[u, t] - Fcr.at[u, t] for u, t in zip(d.gvkey, d.E)]
    d['sector'] = d.gvkey.map(sec)
    d = d[d.noncl > 0].dropna(subset=['sector']).reset_index(drop=True)
    bins = sorted({(t - 1998) // 5 for t in ty})
    ctrl = ['own']
    for b in bins[1:]:
        d[f'mass_b{b}'] = d.mass * (((d.E - 1998) // 5) == b)
        ctrl.append(f'mass_b{b}')
    return dict(d=d, ctrl=ctrl, sup=units, si={u: k for k, u in enumerate(units)}, W=W, Zk=Zs, ty=ty,
                custs=np.asarray(SECS), Xk=Gs)


def other(P):
    return {'d': P['d'][['gvkey', 'E', 'z', 'mass']]}


def sector_io_panel(G_, h, down=False, net='A'):
    """Sector-level counterpart of io_panel: the units are the BLS sectors (patents summed over each sector's sample firms), the partners
    are the other sectors, weighted by the input-output table (net A direct, L Leontief inverse; row i: suppliers of i, column i: customers
    of i; diagonal excluded). Same instrument, window and controls as the firm-level panels."""
    ctx, Q, Oc, Oa = G_['ctx'], G_['Q'], G_['Oc'], G_['Oa']
    Aio = ctx['IO'] if net[0] == 'A' else ctx['IOL']
    secs = np.asarray(SECS)
    mix = Q['smix'](LO, HI); has = mix.sum(1).to_numpy() > 0
    Pc, Pa = Q['Sc'].to_numpy(), Q['Sa'].to_numpy()
    yb = (YEARS >= LO) & (YEARS <= HI)
    gbase = shrink(Pc[:, yb].sum(1, keepdims=True), Pa[:, yb].sum(1, keepdims=True), KAPPA)[:, 0]
    ty = YEARS[YEARS >= max(HI + 1, YEARS[0] + h + 4)]
    sel = np.isin(YEARS, ty)
    Gm, g = measures(Pc, Pa, Oc, Oa, h)
    Gs = Gm[:, sel]
    Zs = np.nan_to_num(mix.to_numpy()) @ g[:, sel]; Zs[~has] = np.nanmean(Zs[has], axis=0)
    base = Aio[secs - 1][:, secs - 1]
    W = (base.T if down else base) * ((has * gbase) if SB else has.astype(float))[None, :]
    np.fill_diagonal(W, 0.0)
    units = secs[has & (W.sum(1) > 0)]
    W = W[has & (W.sum(1) > 0)]
    mu = np.nan_to_num(mix.reindex(units).to_numpy())
    own = mu @ ctx['gcurr'][ty].to_numpy()
    iu = np.searchsorted(secs, units)
    d = pd.DataFrame({'gvkey': np.repeat(units, len(ty)), 'E': np.tile(ty, len(units)), 'x': (W @ Gs).ravel(), 'z': (W @ Zs).ravel(),
                      'mass': np.repeat(W.sum(1), len(ty)), 'own': own.ravel(), 'clean': Pc[iu][:, sel].ravel(), 'noncl': (Pa - Pc)[iu][:, sel].ravel(),
                      'main_cust': np.repeat(secs[W.argmax(1)], len(ty))})
    d = d[d.noncl > 0].reset_index(drop=True)
    ctrl = ['own']
    for b in sorted({(t - 1998) // 5 for t in ty})[1:]:
        d[f'mass_b{b}'] = d.mass * (((d.E - 1998) // 5) == b)
        ctrl.append(f'mass_b{b}')
    return dict(d=d, ctrl=ctrl, sup=units, si={u: k for k, u in enumerate(units)}, W=W, Zk=Zs, ty=ty, custs=secs, Xk=Gs, mix=mix)


def own_window_sector(ctx, G_, units, h):
    "Sectors' own predicted green share over the partners' window t-h-4 ... t-h."
    _, gh = window_tables(ctx, h)
    om = np.nan_to_num(G_['Q']['smix'](LO, HI).reindex(units).to_numpy()) @ gh.to_numpy()
    return pd.DataFrame({'gvkey': np.repeat(units, len(YEARS)), 'E': np.tile(YEARS, len(units)), 'own_w': om.ravel()})


_STATE = {}


def timing_job(job):
    "One regression of the timing figure: direction 'cs' (customers -> supplier) or 'sc' (suppliers -> customer) at horizon h."
    direction, h = job[0], job[1]
    net = job[2] if len(job) > 2 else 'A'
    restrict = len(job) > 3 and job[3]
    G = _STATE['G']; ctx = G['ctx']; allsec = ctx['allsec']
    sector_level = direction in ('sci', 'ssi')
    if direction == 'cr':                  # suppliers' greening -> customer from the inverted Compustat segment links
        P, Q = panel_reverse(ctx, h), other(panel(ctx, h))
        clus = np.array([allsec.get(int(c), -int(c)) for c in P['custs']])       # suppliers' shocks grouped by supplier sector
    elif sector_level:                       # sector-level units, sector-level input-output network in both directions
        P = sector_io_panel(G, h, down=(direction == 'sci'), net=net)
        Q = other(sector_io_panel(G, h, down=(direction == 'ssi'), net=net))
        clus = P['custs']
    elif direction in ('ci', 'si'):          # both directions from the sector-level input-output network (net: A direct, L Leontief inverse)
        P = io_panel(G, h, down=(direction == 'ci'), net=net)
        Q = other(io_panel(G, h, down=(direction == 'si'), net=net))
        clus = P['custs']                                                        # one shock group per partner sector
    else:
        Pd = panel(ctx, h)
        if direction == 'cs':
            P, Q = Pd, other(io_panel(G, h))
            clus = np.array([allsec.get(int(c), -int(c)) for c in P['custs']])      # customers' shocks grouped by customer sector
        else:
            P, Q = io_panel(G, h), other(Pd)
            clus = P['custs']                                                        # one shock group per supplier sector
    lo, hi = years_for(h)
    d, extra = joint_data(P, Q)
    if len(job) > 3 and job[3]:            # restrict to the suppliers with Compustat customer links (the sample of the customers -> supplier direction)
        d = d[d.gvkey.isin(set(base_panel(ctx, LO, HI)['sup']))]
    d = d[d.E.between(lo, hi)].merge(own_window_sector(ctx, G, P['sup'], h) if sector_level else own_window(ctx, P['sup'], h), on=['gvkey', 'E'], how='left').reset_index(drop=True)
    r = dict(direction=direction, h=h, net=net, restricted=bool(restrict), years=f'{lo}-{hi}')
    dl, cl = prep(d, 'E', ['own', 'own_w', 'z_oth'])
    for t in sorted(dl.t.unique())[1:]:
        dl[f'mot_{t}'] = dl.mass_oth * (dl.t == t)
    cl = cl + [f'mot_{t}' for t in sorted(dl.t.unique())[1:] if dl[f'mot_{t}'].abs().sum() > 0]
    r.update(bands(dl, cl, P, clus))
    r.update(N_lin=len(dl), units_lin=dl.gvkey.nunique(), share_firms_with_oth=float(dl.groupby('gvkey').has_oth.first().mean()))
    print(f"[{direction}{net if len(job) > 2 else ''} h={h:2d}] N {r['N_lin']} firms {r['units_lin']} | 2SLS {r['lin_pp']:7.2f} AKM0 p {r['p_akm0']:.3f} "
          f"90% [{r['lin_lo90']:.2f},{r['lin_hi90']:.2f}] {r['lin_kind90']} 68% [{r['lin_lo68']:.2f},{r['lin_hi68']:.2f}] "
          f"{r['lin_kind68']} | RF {r['rf_pp']:.3f} 90% [{r['rf_lo90']:.3f},{r['rf_hi90']:.3f}] | F firm {r['lin_F_firm']:.1f} "
          f"shock t2 {r['lin_F_shock']:.2f}", flush=True)
    return r


def coverage(ctx):
    """How much of a customer's purchases the recorded Compustat suppliers cover (base period): their sales to the customer / the
    customer's sales, and relative to the customer's total intermediate input share in the IO table."""
    cs, fs, sec = ctx['cs'], ctx['fs'], ctx['sec']
    sale = fs[fs.fyear.between(LO, HI)].groupby('gvkey').sale.mean()
    l = cs[cs.fy.between(LO, HI)].groupby(['cust', 'fy']).salecs.sum().groupby('cust').mean()
    cov = (l / sale.reindex(l.index)).replace([np.inf, -np.inf], np.nan).dropna()
    us = sec.reindex(cov.index).dropna().astype(int)
    us = us[us.isin(SECS)]
    io_in = pd.Series(ctx['IO'][us.to_numpy() - 1][:, SECS - 1].sum(1), index=us.index)
    return dict(window=f'{LO}-{HI}', customers=len(cov), median_sales_share=cov.median(), p75=cov.quantile(.75), p90=cov.quantile(.9),
                io_input_share_median=io_in.median(), median_pct_inputs=100 * (cov.reindex(io_in.index) / io_in).median())


# ----------------------------------------------------------------------------------------------------------------------
# Table E3: Table E1's nine columns for the other direction (suppliers' greening -> customer) at h = 1
# ----------------------------------------------------------------------------------------------------------------------
def e3_sector_panel(G_):
    "Outcome sectors of the IO table; partners = the other IO sectors, weighted by input share x 1990-94 green share."
    ctx, Q, Oc, Oa = G_['ctx'], G_['Q'], G_['Oc'], G_['Oa']
    Aio = ctx['IO']
    secs = np.asarray(SECS)
    mix = Q['smix'](LO, HI); has = mix.sum(1).to_numpy() > 0
    assert list(mix.columns) == list(ctx['gcurr'].index)
    Pc, Pa = Q['Sc'].to_numpy(), Q['Sa'].to_numpy()
    assert len(Pc) == len(secs) == len(mix)
    yb = (YEARS >= LO) & (YEARS <= HI)
    gbase = shrink(Pc[:, yb].sum(1, keepdims=True), Pa[:, yb].sum(1, keepdims=True), KAPPA)[:, 0]
    ty = YEARS[YEARS >= HI + 1]
    sel = np.isin(YEARS, ty)
    Gm, g = measures(Pc, Pa, Oc, Oa, 1)
    Gs = Gm[:, sel]
    Zs = np.nan_to_num(mix.to_numpy()) @ g[:, sel]; Zs[~has] = np.nanmean(Zs[has], axis=0)
    W = Aio[secs - 1][:, secs - 1] * ((has * gbase) if SB else has.astype(float))[None, :]
    np.fill_diagonal(W, 0.0)                                       # a sector's own patents are not its suppliers'
    own = np.nan_to_num(mix.to_numpy()) @ ctx['gcurr'][ty].to_numpy()
    d = pd.DataFrame({'gvkey': np.repeat(secs, len(ty)), 'E': np.tile(ty, len(secs)), 'x': (W @ Gs).ravel(),
                      'z': (W @ Zs).ravel(), 'own': own.ravel(), 'mass': np.repeat(W.sum(1), len(ty)),
                      'clean': Pc[:, sel].ravel(), 'noncl': (Pa - Pc)[:, sel].ravel()})
    d = d[np.repeat(has & (W.sum(1) > 0), len(ty))]
    d = d[d.noncl > 0].reset_index(drop=True)
    ctrl = ['own']
    for b in sorted({(t - 1998) // 5 for t in ty})[1:]:
        d[f'mass_b{b}'] = d.mass * (((d.E - 1998) // 5) == b)
        ctrl.append(f'mass_b{b}')
    return trim(d.dropna(subset=['x', 'z'] + ctrl).copy(), 'E'), ctrl


# ----------------------------------------------------------------------------------------------------------------------
# Workflow
# ----------------------------------------------------------------------------------------------------------------------
class Customer_IV:
    "Customer IV workflow: Table E1 (customers' greening -> supplier), Table E3 (the other direction), Section 2 timing."

    def __init__(self, P):
        "P: Processor instance (project directory)."
        self.Directory = P.Directory
        self.Clean = self.Directory / 'Clean Data'
        self.Root = self.Directory / 'Specifications' / 'Customer_IV'
        self.Out = self.Root / 'output'
        self.Out.mkdir(parents=True, exist_ok=True)
        self._inp = self._ctx = self._G = None

    def inputs(self):
        if self._inp is None:
            self._inp = load_inputs(self.Clean)
        return self._inp

    def ctx(self):
        if self._ctx is None:
            self._ctx = context(self.inputs())
        return self._ctx

    def G(self):
        if self._G is None:
            self._G = setup(self.ctx())
        return self._G

    def Table_E1(self):
        """
        Table E1: every column of the customers' greening -> supplier table (annual outcomes 1995-2017, links and class mixes
        fixed FY1990-94), AKM0 with customers' shocks grouped by sector, and the two inputs no regression stores.

        Output: Specifications/Customer_IV/output/table_e1_{coefs,stats,akm,inputs}.csv
        """
        ctx = self.ctx()
        P = base_panel(ctx, LO, HI)
        C1, S1 = columns_sector(*e1_sector_panel(ctx), 'customers green share X')
        C0, S0, _ = columns_firm(P, 'customers green share X', 'supplier')
        C = pd.DataFrame(C0 + C1)
        C['p'] = 2 * (1 - stats.norm.cdf(np.abs(C.coef / C.se)))
        C.to_csv(self.Out / 'table_e1_coefs.csv', index=False)
        pd.DataFrame(S0 + S1).to_csv(self.Out / 'table_e1_stats.csv', index=False)
        pd.DataFrame([e1_akm(ctx, P)]).to_csv(self.Out / 'table_e1_akm.csv', index=False)
        r = size_under_reshuffles(P)
        r.update(base=f'{LO}-{HI}', matched_share_base=matched_share(self.inputs()['cs'], LO, HI),
                 matched_share_1990_1997=matched_share(self.inputs()['cs'], 1990, 1997))
        pd.DataFrame([r]).to_csv(self.Out / 'table_e1_inputs.csv', index=False)

    def Robustness_Lambda(self, lambdas=(1, 5, 10, 20, 50, 100)):
        """
        Customer IV (Table E1 columns) for alternative shrinkage lambda, plus the method-of-moments value of lambda from the firms'
        own green shares. Lambda enters the customers' shares s_kt, the class shifts g_ct and the base-period shares.

        Output: Specifications/Customer_IV/output/robust_lambda.csv
        """
        global KAPPA
        keep = KAPPA
        ctx0 = self.ctx()
        mom, ncell = kappa_moments(ctx0)
        print(f'method-of-moments lambda: {mom:.1f} ({ncell} firm-windows with >= 10 patents)', flush=True)
        rows = []
        try:
            for lam in list(lambdas) + [round(mom, 1)]:
                KAPPA = float(lam)
                ctx = context(self.inputs())
                P = base_panel(ctx, LO, HI)
                C0, S0, _ = columns_firm(P, 'customers green share X', 'supplier')
                C1, S1 = columns_sector(*e1_sector_panel(ctx), 'customers green share X')
                C = pd.DataFrame(C0 + C1); S = pd.DataFrame(S0 + S1).set_index('column')
                C['p'] = 2 * (1 - stats.norm.cdf(np.abs(C.coef / C.se)))
                g = lambda col, term, f: float(C[(C.column == col) & (C.term == term)][f].iloc[0])
                akm = e1_akm(ctx, P)
                rows.append(dict(lam=lam, mom=bool(lam == round(mom, 1) and lam not in lambdas), firm_ols=g(1, 'x', 'coef'), firm_F=S.loc[2, 'F'],
                                 firm_rf=g(3, 'z', 'coef'), firm_rf_p=g(3, 'z', 'p'), firm_iv=g(4, 'x', 'coef'), firm_iv_se=g(4, 'x', 'se'),
                                 firm_lin_pp=g(5, 'x', 'coef'), firm_lin_F=akm['lin_F'],
                                 akm0_sector=akm['p_akm0'], akm0_firm=akm['p_akm0_firmclusters'],
                                 sec_F=S.loc[7, 'F'], sec_rf=g(8, 'z', 'coef'), sec_rf_se=g(8, 'z', 'se'), sec_rf_p=g(8, 'z', 'p'), sec_iv=g(11, 'x', 'coef'), sec_iv_se=g(11, 'x', 'se'), sec_iv_p=S.loc[11, 'p_x'],
                                 N_firm=int(S.loc[1, 'N']), N_sec=int(S.loc[6, 'N'])))
                print(rows[-1], flush=True)
        finally:
            KAPPA = keep
        pd.DataFrame(rows).to_csv(self.Out / 'robust_lambda.csv', index=False)

    def Robustness_Other(self):
        """
        Customer IV firm level (Table E1 columns 2-5) under other choices: base period, last outcome year, minimum link size, and
        dropping each large customer sector in turn. Shrinkage lambda is 20 as in the baseline.

        Output: Specifications/Customer_IV/output/robust_other.csv, robust_loo.csv
        """
        ctx = self.ctx()
        rows = []
        def run(label, group, **kw):
            lo, hi = kw.pop('base', (LO, HI)); last = kw.pop('last', None)
            P = base_panel(ctx, lo, hi, **kw)
            if last is not None:
                dd = P['d'][P['d'].E <= last]
                P = dict(P, d=dd, ctrl=[c for c in P['ctrl'] if dd[c].abs().sum() > 0])
            r = quick_firm(ctx, P, boot=True); r.update(label=label, group=group)
            rows.append(r); print(r, flush=True)
            return P
        P0 = run('Baseline (links and class mixes 1990-94)', 'base')
        # alternative estimator for the Poisson IV: GMM with the Poisson fixed-effects moments, standard errors clustered by firm
        d0 = trim(P0['d'][list(dict.fromkeys(['clean', 'noncl', 'x', 'z', 'gvkey', 'E', 'sector', 'main_cust'] + P0['ctrl']))].dropna(), 'E')
        gm = gmm_firm(d0, P0['ctrl'], within_sd(d0, ['x', 'z', 'own']))
        rg = dict(rows[0]); rg.update(label='Poisson IV by GMM instead of the control function', group='estimator', iv=gm['coef'], iv_se=gm['se'], iv_p=gm['p'], gmm_converged=gm['converged'])
        rows.append(rg); print(rg, flush=True)
        run('Base period 1985-89', 'base', base=(1985, 1989))
        run('Base period 1990-97', 'base', base=(1990, 1997))
        run('Outcome years to 2014', 'sample', last=2014)
        for ms in (0.05, 0.10, 0.20):
            run(f'Links of at least {int(ms * 100)} percent of the supplier\'s sales', 'links', min_share=ms)
        run('Regressor: links of the window t-5 ... t-1 (instrument: base links)', 'network', contemp=True)
        inp = self.inputs()
        # outside-patent class shifts
        cy = inp['cy']
        def with_cy(cy2):
            c2 = context(dict(inp, cy=cy2)); return c2
        def run_ctx(c2, label, group):
            P = base_panel(c2, LO, HI)
            r = quick_firm(c2, P, boot=True); r.update(label=label, group=group); rows.append(r); print(r, flush=True)
        run_ctx(with_cy(cy.groupby(['cpc_subclass', 'year'], as_index=False)[['clean', 'all']].sum().assign(inx=False)),
                'Shifts from all patents (sample firms included)', 'shifts')
        c3 = context(inp)
        Oc3, Oa3 = c3['Oc'], c3['Oa']
        c3['gprev'] = pd.DataFrame(shrunk_share(trailing(Oc3.to_numpy(), -10, -6), trailing(Oa3.to_numpy(), -10, -6)), index=c3['classes'], columns=YEARS)
        run_ctx(c3, 'Shifts measured over t-10 ... t-6', 'shifts')
        sub_cls = lambda x: x.assign(cpc_subclass=x.cpc_subclass.astype(str).str[:3])
        cy3 = sub_cls(cy).groupby(['cpc_subclass', 'year', 'inx'], as_index=False)[['clean', 'all']].sum()
        pat3 = dict(inp['pat']); sb = sub_cls(pat3['sub']).groupby(['patent_id', 'cpc_subclass'], as_index=False).f.sum(); pat3['sub'] = sb
        run_ctx(context(dict(inp, cy=cy3, pat=pat3)), 'Three-character CPC classes', 'shifts')
        pd.DataFrame(rows).to_csv(self.Out / 'robust_other.csv', index=False)
        # leave one customer sector out: the sectors carrying the most link weight
        allsec = ctx['allsec']
        w = pd.Series(P0['W'].sum(0), index=[allsec.get(int(c), -int(c)) for c in P0['custs']]).groupby(level=0).sum().sort_values(ascending=False)
        loo = []
        for sct in [x for x in w.index if x > 0][:15]:
            try:
                P = base_panel(ctx, LO, HI, drop_sector=sct)
                r = quick_firm(ctx, P); r.update(dropped_sector=int(sct), link_weight=float(w[sct]))
                loo.append(r); print('LOO', sct, r['lin_pp'], r['akm0'], flush=True)
            except Exception as ex:
                print('LOO failed', sct, ex)
        pd.DataFrame(loo).to_csv(self.Out / 'robust_loo.csv', index=False)

    def Table_E3(self):
        """
        Table E3: Table E1's nine columns for suppliers' greening -> customer (green content of inputs, 1997 BLS table), h = 1.

        Output: Specifications/Customer_IV/output/table_e3_{coefs,stats,akm}.csv
        """
        G = self.G()
        P = io_panel(G, 1)
        C1, S1 = columns_sector(*e3_sector_panel(G), 'suppliers green content X')
        C0, S0, (dl, cl) = columns_firm(P, 'suppliers green content X', 'firm')
        a = akm0_iv(dl, cl, P, P['custs'])
        c5 = [c for c in C0 if c['column'] == 5 and c['term'] == 'x'][0]['coef']
        assert abs(a['lin_pp'] - c5) < 1e-6 * max(1, abs(c5)), (a['lin_pp'], c5)
        akm = dict(N=a['N_lin'], lin_pp=a['lin_pp'], p_akm0=a['p_akm0'], eff_shocks=a['eff_shock_groups'],
                   lin_F_firm=a['lin_F_firm'], lin_F_shock=a['lin_F_shock'], rf_pp=a['rf_pp'], n_shock_groups=len(np.unique(P['custs'])))
        C = pd.DataFrame(C0 + C1)
        C['p'] = 2 * (1 - stats.norm.cdf(np.abs(C.coef / C.se)))
        C.to_csv(self.Out / 'table_e3_coefs.csv', index=False)
        pd.DataFrame(S0 + S1).to_csv(self.Out / 'table_e3_stats.csv', index=False)
        pd.DataFrame([akm]).to_csv(self.Out / 'table_e3_akm.csv', index=False)

    def Timing_IO(self, nets=('A', 'L')):
        """
        Timing of both directions with the same sector-level input-output network in both: customers' greening -> supplier ('ci': weight
        A[j, i], j the customer sectors of the firm's sector i) and suppliers' greening -> customer ('si': weight A[i, j]), for the direct
        coefficients (net A) and the Leontief inverse (net L), diagonal excluded. Instruments, controls and inference as in Timing.

        Output: Specifications/Customer_IV/output/timing_io_{A,L}.csv
        """
        _STATE['G'] = self.G()
        nw = int(os.environ.get('NW', 7))
        for net in nets:
            jobs = [(dr, h, net) for dr in ('ci', 'si') for h in HS]
            if nw > 1:
                with get_context('fork').Pool(nw) as pool:
                    rows = pool.map(timing_job, jobs, chunksize=1)
            else:
                rows = [timing_job(j) for j in jobs]
            pd.DataFrame(rows).to_csv(self.Out / f'timing_io_{net}.csv', index=False)

    def Timing_IO_linked(self, net='A'):
        """
        As Timing_IO for the input-output network, but only for the suppliers with Compustat customer links, the sample of the
        customers -> supplier direction (separates the sample from the network).

        Output: Specifications/Customer_IV/output/timing_io_<net>_linked.csv
        """
        _STATE['G'] = self.G()
        nw = int(os.environ.get('NW', 7))
        jobs = [(dr, h, net, True) for dr in ('ci', 'si') for h in HS]
        with get_context('fork').Pool(nw) as pool:
            rows = pool.map(timing_job, jobs, chunksize=1)
        pd.DataFrame(rows).to_csv(self.Out / f'timing_io_{net}_linked.csv', index=False)

    def Timing_Sector(self, nets=('A', 'L')):
        """
        Timing of both directions with sectors as the units and the sector-level input-output network in both directions
        ('sci': customer sectors' greening -> sector, 'ssi': supplier sectors' greening -> sector). Controls, instruments and inference as in Timing.

        Output: Specifications/Customer_IV/output/timing_sector_{A,L}.csv
        """
        _STATE['G'] = self.G()
        nw = int(os.environ.get('NW', 7))
        for net in nets:
            jobs = [(dr, h, net) for dr in ('sci', 'ssi') for h in HS]
            with get_context('fork').Pool(nw) as pool:
                rows = pool.map(timing_job, jobs, chunksize=1)
            pd.DataFrame(rows).to_csv(self.Out / f'timing_sector_{net}.csv', index=False)

    def Timing_Reverse(self):
        """
        Suppliers' greening -> customer with the Compustat segment links inverted: the customer's exposure is the green share of the suppliers
        that name it as a customer. Same horizons, instrument construction, controls and inference as Timing.

        Output: Specifications/Customer_IV/output/timing_reverse.csv
        """
        _STATE['G'] = self.G()
        nw = int(os.environ.get('NW', 7))
        jobs = [('cr', h) for h in HS]
        with get_context('fork').Pool(nw) as pool:
            rows = pool.map(timing_job, jobs, chunksize=1)
        pd.DataFrame(rows).to_csv(self.Out / 'timing_reverse.csv', index=False)

    def Timing(self):
        """
        Section 2: both directions at horizons h = 0-20 (about 15 minutes with NW=7 workers; NW=1 runs serially), and the
        coverage of customers' purchases by recorded Compustat suppliers.

        Output: Specifications/Customer_IV/output/timing_both.csv, timing_coverage.csv
        """
        _STATE['G'] = self.G()
        jobs = [(dr, h) for dr in ('cs', 'sc') for h in HS]
        nw = int(os.environ.get('NW', 7))
        if nw > 1:
            with get_context('fork').Pool(nw) as pool:
                rows = pool.map(timing_job, jobs, chunksize=1)
        else:
            rows = [timing_job(j) for j in jobs]
        pd.DataFrame(rows).to_csv(self.Out / 'timing_both.csv', index=False)
        pd.DataFrame([coverage(self.ctx())]).to_csv(self.Out / 'timing_coverage.csv', index=False)
