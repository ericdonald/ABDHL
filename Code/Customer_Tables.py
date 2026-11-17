"""""""""""
Customer Tables Module

Notes: Builds the tables and figure of the customer-IV draft (Specifications/Customer_IV/draft_customer_iv_section.tex) from the saved
    outputs of Customer_IV. No estimation, and no number is typed: every number quoted in the text is a macro (\\civ..., \\tm...,
    \\siv...) written here.

    Table E1  customers' greening -> supplier, nine columns           Results/Tables/Customer_IV.tex, Customer_IV_macros.tex
    Table E3  the same for suppliers' greening -> customer            Results/Tables/Supplier_IV.tex, Supplier_IV_macros.tex
    Table E2  timing of both directions (Section 2)                   Results/Tables/Timing_both.tex, Timing_macros.tex
    Figure E1 timing of both directions                               Results/Figures/timing_both.pdf

"""""""""""

import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = '1990-1994'
COLS = range(1, 10)
HS = [0, 1, 2, 4, 6, 8, 12, 16, 20]
FMIN = 10.0
BLUE, ORANGE, INK, INK2, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'
FIG_COLS = [('cs', 'Customers\u2019 greening \u2192 supplier\n(travels upstream; Table E1)', BLUE),
            ('sc', 'Suppliers\u2019 greening \u2192 customer\n(travels downstream)', ORANGE)]



CLAIMS = []


def claim(cond_fn, msg):
    """A statement of the draft's text that the generated numbers must support. If it fails the draft's wording needs another look: the
    failure is printed and kept in CLAIMS (the tables and macros are still written)."""
    try:
        ok = bool(cond_fn())
    except Exception as ex:
        ok, msg = False, f'{msg} ({type(ex).__name__}: {ex})'
    if not ok:
        CLAIMS.append(msg)
        print(f'TEXT CHECK FAILED: {msg}', flush=True)


def p3(x):
    return '$<$0.001' if x < 0.0005 else f'{x:.3f}'


def star(p):
    "Significance stars from the p-value shown in brackets (or from coefficient / s.e. where no p-value is shown)."
    return '^{***}' if p < 0.01 else ('^{**}' if p < 0.05 else ('^{*}' if p < 0.10 else ''))


def n(x):
    return f'{int(round(x)):,}'


def g3(x):
    return f'{x:.3g}' if abs(x) < 0.1 else f'{x:.2f}'


def tabular(C, S, xlabel, se_row, akm_rows, unit_label, partner_label):
    "Table E1's nine-column tabular for one direction (C, S: explore39-format coefficient and column tables)."
    def cf(col, term, k='coef'):
        r = C[(C.column == col) & (C.term == term)]
        return float(r[k].iloc[0]) if len(r) else np.nan

    def cell_row(label, term, with_p):
        c, se, pp = [], [], []
        for k in COLS:
            v = cf(k, term)
            if np.isnan(v):
                c.append(''); se.append(''); pp.append('')
            else:
                pk = S.at[k, 'p_x'] if (term == 'x' and k in (4, 9)) else cf(k, term, 'p')
                if not with_p:                                   # no bracketed p-value: normal p from coefficient / s.e.
                    pk = 2 * (1 - stats.norm.cdf(abs(v / cf(k, term, 'se'))))
                c.append(f'${v:.3f}{star(pk)}$'); se.append(f'({cf(k, term, "se"):.3f})')
                pp.append(f'[{p3(pk)}]')
        out = [f'{label} & ' + ' & '.join(c) + ' \\\\', ' & ' + ' & '.join(se) + ' \\\\']
        if with_p:
            out.append(' & ' + ' & '.join(pp) + ' \\\\')
        return out

    def line(label, vals):
        return f'{label} & ' + ' & '.join(vals) + ' \\\\'

    firm = [k <= 5 for k in COLS]
    yes = lambda m: ['Yes' if b else 'No' for b in m]
    L = ['\\begin{tabular}{l*{9}{c}}', '\\toprule',
         ' & \\multicolumn{5}{c}{Firm level} & \\multicolumn{4}{c}{Sector level} \\\\',
         '\\cmidrule(lr){2-6}\\cmidrule(lr){7-10}',
         line('', [f'({k})' for k in COLS]),
         line('Dependent variable', ['$P^{g}$', '$X$', '$P^{g}$', '$P^{g}$', '$P^{g}/P$', '$P^{g}$', '$X$', '$P^{g}$', '$P^{g}$']),
         line('Estimator', ['Poisson', 'OLS', 'Poisson', 'Poisson', '2SLS', 'Poisson', 'OLS', 'Poisson', 'Poisson']),
         line('Role', ['No IV', 'First', 'Reduced', 'IV', 'IV', 'No IV', 'First', 'Reduced', 'IV']),
         line('', ['', 'stage', 'form', '', '', '', 'stage', 'form', '']),
         '\\midrule']
    L += cell_row(xlabel, 'x', True)
    L += cell_row('Instrument, $Z$', 'z', True)
    L += cell_row('Own predicted green share, $O$', 'own', False)
    L += ['\\midrule',
          line('Standard errors', se_row),
          *[line(lab, ['', '', '', '', val, '', '', '', '']) for lab, val in akm_rows],
          line('First-stage F', ['', '', '', f'{S.at[4, "F"]:.1f}', f'{S.at[5, "F"]:.1f}', '', '', '', f'{S.at[9, "F"]:.1f}']),
          '\\midrule',
          line('Link intensity $\\times$ period', ['Yes'] * 9),
          line('Firm effects', yes(firm)),
          line('Sector effects', yes([not b for b in firm])),
          line('Year effects', ['Yes'] * 9),
          line('Sector $\\times$ year effects', ['No'] * 9),
          '\\midrule',
          line('Pseudo $R^{2}$', [f'{S.at[k, "pseudo_r2"]:.3f}' if pd.notna(S.at[k, 'pseudo_r2']) else '' for k in COLS]),
          line('Within $R^{2}$', [f'{S.at[k, "within_r2"]:.3f}' if pd.notna(S.at[k, 'within_r2']) else '' for k in COLS]),
          line('Mean dep.\\ variable', [g3(S.at[k, 'mean_dep']) for k in COLS]),
          line('Observations', [n(S.at[k, 'N']) for k in COLS]),
          line(unit_label, [n(S.at[k, 'units']) for k in COLS]),
          line(partner_label, [n(S.at[k, 'customers']) if k <= 5 else '' for k in COLS]),
          '\\bottomrule', '\\end{tabular}']
    return L


def customer_table(OUT, TAB):
    "Table E1 (customers' greening -> supplier) and the macros for every number quoted in the text."
    C = pd.read_csv(OUT / 'table_e1_coefs.csv')
    S = pd.read_csv(OUT / 'table_e1_stats.csv').set_index('column')
    a = pd.read_csv(OUT / 'table_e1_akm.csv').iloc[0]
    x = pd.read_csv(OUT / 'table_e1_inputs.csv').iloc[0]

    def cf(col, term, k='coef'):
        r = C[(C.column == col) & (C.term == term)]
        return float(r[k].iloc[0]) if len(r) else np.nan

    # the nine columns must be one sample (the AKM0 and reshuffle runs use the Poisson sample of column 1)
    assert int(a.N) == int(S.at[1, 'N']) == int(x.N) == int(S.at[5, 'N'])
    assert abs(S.at[5, 'F'] - a.lin_F) <= 1e-3 * max(1, abs(a.lin_F)), 'column 5 first-stage F differs from the AKM0 run'

    L = tabular(C, S, "Customers' green share, $X$",
                ['Supplier', 'Supplier', 'Supplier', 'Bootstrap', 'Supplier', 'Sector', 'Sector', 'Sector', 'Sector'],
                [('AKM0 p, sector-grouped shocks', p3(a.p_akm0)), ('AKM0 p, customer shocks', p3(a.p_akm0_firmclusters)),
                 ('Effective no. of shocks', f'{a.eff_clusters:.1f}')],
                'Suppliers / sectors', 'Main customers')
    TAB.mkdir(parents=True, exist_ok=True)
    head = '% Generated by Customer_Tables.customer_table from output/table_e1_*.csv. Do not edit by hand.\n'
    (TAB / 'Customer_IV.tex').write_text(head + '\n'.join(L) + '\n')

    pct = lambda b: f'{100 * (np.exp(b) - 1):.0f}'
    mac = {
        'civRejectRate': f'{100 * x.reject_rate_supplier_cluster:.0f}',
        'civReshuffleDraws': n(x.reshuffle_draws),
        'civMatchedShare': f'{100 * x.matched_share_base:.0f}',
        'civFirmOLS': f'{cf(1, "x"):.2f}', 'civFirmOLSpct': pct(cf(1, 'x')),
        'civFirmFS': f'{cf(2, "z"):.2f}', 'civFirmF': f'{S.at[2, "F"]:.1f}',
        'civFirmRF': f'{cf(3, "z"):.2f}', 'civFirmRFp': p3(cf(3, 'z', 'p')),
        'civFirmIV': f'{cf(4, "x"):.2f}', 'civFirmIVpct': pct(cf(4, 'x')), 'civFirmIVp': p3(S.at[4, 'p_x']),
        'civFirmLinPP': f'{cf(5, "x"):.2f}', 'civFirmLinF': f'{S.at[5, "F"]:.1f}',
        'civFirmAKMsec': p3(a.p_akm0), 'civFirmAKMfirm': p3(a.p_akm0_firmclusters), 'civEffShocks': f'{a.eff_clusters:.0f}',
        'civNsup': n(S.at[1, 'units']), 'civNsupyr': n(S.at[1, 'N']), 'civNcust': n(S.at[1, 'customers']),
        'civSecOLS': f'{cf(6, "x"):.3f}', 'civSecOLSpct': f'{100 * (np.exp(cf(6, "x")) - 1):.1f}',
        'civSecF': f'{S.at[7, "F"]:.1f}', 'civSecRFp': p3(cf(8, 'z', 'p')),
        'civSecIV': f'{cf(9, "x"):.3f}', 'civSecIVpct': f'{100 * (np.exp(cf(9, "x")) - 1):.1f}', 'civSecIVp': p3(S.at[9, 'p_x']),
        'civNsec': n(S.at[6, 'units']), 'civNsecyr': n(S.at[6, 'N']),
    }
    (TAB / 'Customer_IV_macros.tex').write_text(head + '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in mac.items()) + '\n')



def supplier_table(OUT, TAB):
    "Table E3: Table E1's nine columns for suppliers' greening -> customer (Customer_IV.Table_E3)."
    if not (OUT / 'table_e3_coefs.csv').exists():
        return
    C = pd.read_csv(OUT / 'table_e3_coefs.csv')
    S = pd.read_csv(OUT / 'table_e3_stats.csv').set_index('column')
    a = pd.read_csv(OUT / 'table_e3_akm.csv').iloc[0]
    if str(S.at[9, 'converged']) != 'True':                        # GMM did not converge: leave column 9 blank (see the table notes)
        C = C[C.column != 9]

    def cf(col, term, k='coef'):
        r = C[(C.column == col) & (C.term == term)]
        return float(r[k].iloc[0]) if len(r) else np.nan
    assert abs(a.lin_pp - cf(5, 'x')) < 1e-6 * max(1, abs(cf(5, 'x'))) and int(a.N) == int(S.at[5, 'N']) == int(S.at[1, 'N'])
    L = tabular(C, S, "Suppliers' green content, $X$",
                ['Customer', 'Customer', 'Customer', 'Bootstrap', 'Customer', 'Sector', 'Sector', 'Sector', 'Sector'],
                [('AKM0 p, supplier-sector shocks', p3(a.p_akm0)), ('Effective no. of shocks', f'{a.eff_shocks:.1f}')],
                'Customers / sectors', 'Main supplier sectors')
    head = '% Generated by Customer_Tables.supplier_table from output/table_e3_*.csv. Do not edit by hand.\n'
    (TAB / 'Supplier_IV.tex').write_text(head + '\n'.join(L) + '\n')
    pv = lambda k: S.at[k, 'p_x'] if k in (4, 9) else cf(k, 'x', 'p')
    # text: no-IV association insignificant at both levels; unweighted first stages weak; column 5's first stage negative
    claim(lambda: pv(1) > 0.10 and pv(6) > 0.10 and S.at[2, 'F'] < 10 and S.at[7, 'F'] < 10 and a.rf_pp / a.lin_pp < 0, 'Table E3: no-IV association insignificant, unweighted first stages weak and column 5 first stage negative')
    assert abs(S.at[5, 'F'] - a.lin_F_firm) < 1e-6 * a.lin_F_firm
    t1 = pd.read_csv(OUT / 'timing_both.csv').query("direction == 'sc' and h == 1").iloc[0]
    assert int(t1.units_lin) == int(S.at[1, 'units']) and int(t1.N_lin) == int(S.at[1, 'N'])   # text: Figure E1's firms at h = 1
    mac = {
        'sivFirmOLS': f'{cf(1, "x"):.2f}', 'sivFirmOLSp': p3(pv(1)),
        'sivFirmFS': f'{cf(2, "z"):.2f}', 'sivFirmF': f'{S.at[2, "F"]:.1f}',
        'sivFirmRF': f'{cf(3, "z"):.2f}', 'sivFirmRFp': p3(cf(3, 'z', 'p')),
        'sivFirmIV': f'{cf(4, "x"):.2f}', 'sivFirmIVp': p3(pv(4)),
        'sivFirmLinPP': f'{cf(5, "x"):.2f}', 'sivFirmLinp': p3(pv(5)), 'sivFirmAKM': p3(a.p_akm0),
        'sivEffShocks': f'{a.eff_shocks:.0f}', 'sivShockT': f'{a.lin_F_shock:.1f}',
        'sivFirmLinFS': f'{a.rf_pp / a.lin_pp:.2f}', 'sivFirmLinF': f'{a.lin_F_firm:.1f}',
        'sivNfirm': n(S.at[1, 'units']), 'sivNfirmyr': n(S.at[1, 'N']),
        'sivSecOLS': f'{cf(6, "x"):.3f}', 'sivSecOLSp': p3(pv(6)), 'sivSecFS': f'{cf(7, "z"):.2f}', 'sivSecF': f'{S.at[7, "F"]:.1f}',
        'sivSecRF': f'{cf(8, "z"):.3f}', 'sivSecRFp': p3(cf(8, 'z', 'p')), 'sivSecIV': f'{cf(9, "x"):.3f}', 'sivSecIVp': p3(pv(9)),
        'sivNsec': n(S.at[6, 'units']), 'sivNsecyr': n(S.at[6, 'N']),
    }
    mac = {k: ('$-$' + v[1:] if v.startswith('-') else v) for k, v in mac.items()}
    (TAB / 'Supplier_IV_macros.tex').write_text(head + '\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in mac.items()) + '\n')


HEAD_T = '% Generated by Customer_Tables.timing_table from output/timing_both.csv and output/timing_coverage.csv. Do not edit by hand.\n'


def fm(x, d):
    s = f'{x:.{d}f}'
    if float(s) == 0:
        s = s.lstrip('-')
    return s.replace('-', '$-$')


def star_z(b, se):
    z = abs(b / se)
    return '^{***}' if z > 2.576 else ('^{**}' if z > 1.96 else ('^{*}' if z > 1.645 else ''))


def pv(p):
    return '$<$0.001' if p < 0.0005 else f'{p:.3f}'


def timing_table(OUT, TAB):
    "Table E2 (timing of both directions) and the macros for every number quoted in Section 2."
    d = pd.read_csv(OUT / 'timing_both.csv')
    cov = pd.read_csv(OUT / 'timing_coverage.csv').iloc[0]
    Q = {dr: d[d.direction == dr].set_index('h').copy() for dr in ('cs', 'sc')}
    for q in Q.values():
        q['fs'] = q.rf_pp / q.lin_pp
        q['ok'] = (q.lin_F_firm >= FMIN) & (q.fs > 0)
        q['p_lin'] = 2 * (1 - stats.norm.cdf(np.abs(q.lin_pp / q.lin_se_akm)))
    L = [HEAD_T, rf'\begin{{tabular}}{{l*{{{len(HS)}}}{{c}}}}', r'\toprule',
         r" & \multicolumn{" + str(len(HS)) + r"}{c}{Horizon $h$ (years from the end of the partners' window to the outcome year)} \\",
         r'\cmidrule(lr){2-' + str(len(HS) + 1) + '}', ' & ' + ' & '.join(f'$h={h}$' for h in HS) + r' \\', r'\midrule']
    for dr, title in [('cs', "A. Customers' greening $\\rightarrow$ supplier (Compustat customer links)"),
                      ('sc', "B. Suppliers' greening $\\rightarrow$ customer (input--output green content of inputs)")]:
        q = Q[dr]
        lin = [f"${fm(q.loc[h, 'lin_pp'], 2).replace('$', '')}{star_z(q.loc[h, 'lin_pp'], q.loc[h, 'lin_se_akm'])}$" if q.loc[h, 'ok'] else '--' for h in HS]
        lse = [f"({fm(q.loc[h, 'lin_se_akm'], 2)})" if q.loc[h, 'ok'] else '' for h in HS]
        rf = [f"${fm(q.loc[h, 'rf_pp'], 2).replace('$', '')}{star_z(q.loc[h, 'rf_pp'], q.loc[h, 'rf_se_akm'])}$" for h in HS]
        rse = [f"({fm(q.loc[h, 'rf_se_akm'], 2)})" for h in HS]
        L += [rf'\multicolumn{{{len(HS) + 1}}}{{l}}{{\textit{{{title}}}}} \\', r'\addlinespace[2pt]',
              r'\quad 2SLS (pp per SD of $X$) & ' + ' & '.join(lin) + r' \\', r' & ' + ' & '.join(lse) + r' \\',
              r'\quad Reduced form (pp per SD of $Z$) & ' + ' & '.join(rf) + r' \\', r' & ' + ' & '.join(rse) + r' \\',
              r'\quad First stage (SD of $X$ per SD of $Z$) & ' + ' & '.join(fm(q.loc[h, 'fs'], 2) for h in HS) + r' \\',
              r'\quad First-stage F & ' + ' & '.join(f"{q.loc[h, 'lin_F_firm']:.1f}" for h in HS) + r' \\',
              r'\quad AKM0 p-value, zero effect & ' + ' & '.join(pv(q.loc[h, 'p_akm0']) for h in HS) + r' \\',
              r'\quad Firm-years & ' + ' & '.join(f"{int(q.loc[h, 'N_lin']):,}" for h in HS) + r' \\',
              r'\quad Firms & ' + ' & '.join(f"{int(q.loc[h, 'units_lin'])}" for h in HS) + r' \\', r'\addlinespace[3pt]']
    L += [r'\bottomrule', r'\end{tabular}']
    (TAB / 'Timing_both.tex').write_text('\n'.join(L) + '\n')

    cs, sc = Q['cs'], Q['sc']
    H = range(0, 21)
    contig = lambda q, cond: max(h for h in H if all(cond(q, k) for k in range(0, h + 1)))
    cs_hi = contig(cs, lambda q, k: q.loc[k, 'ok'] and q.loc[k, 'p_lin'] < 0.10)
    cs_akm0_hi = contig(cs, lambda q, k: q.loc[k, 'p_akm0'] < 0.10)
    cs_shown = [h for h in H if cs.loc[h, 'ok']]
    sc_shown = [h for h in H if sc.loc[h, 'ok']]
    sc_blank = [h for h in H if not sc.loc[h, 'ok']]
    cs_blank = [h for h in H if not cs.loc[h, 'ok']]
    sc_neg = [h for h in sc_blank if sc.loc[h, 'fs'] < 0]               # hidden: first stage of the wrong sign
    sc_weak = [h for h in sc_blank if sc.loc[h, 'lin_F_firm'] < FMIN]    # hidden: first-stage F below 10
    sc_sig = [h for h in sc_shown if sc.loc[h, 'p_lin'] < 0.10]
    sc_akm = [h for h in sc_shown if sc.loc[h, 'p_akm0'] < 0.10]
    # text: each series is one contiguous run of horizons (customers from h = 0, suppliers to h = 20)
    claim(lambda: cs_blank == list(range(min(cs_blank), 21)), f'customer side: horizons without a usable first stage are not one run to h = 20: {cs_blank}')
    claim(lambda: sc_blank == list(range(0, max(sc_blank) + 1)), f'supplier side: horizons without a usable first stage are not one run from h = 0: {sc_blank}')
    claim(lambda: all(cs.loc[h, 'fs'] > 0 for h in cs_blank), 'customer side: hidden horizons are not only those with a weak first stage')
    claim(lambda: set(sc_neg) | set(sc_weak) == set(sc_blank), 'supplier side: hidden horizons are not only negative or weak first stages')
    sgn = [sc.loc[h, 'lin_pp'] < 0 for h in sc_shown]
    sc_neg_hi = max(h for h in sc_shown if sc.loc[h, 'lin_pp'] < 0)
    # text: supplier side negative from the first reported horizon to sc_neg_hi, rising toward zero, positive after
    claim(lambda: sgn == [h <= sc_neg_hi for h in sc_shown] and np.all(np.diff(sc.loc[min(sc_shown):sc_neg_hi, 'lin_pp']) > 0), 'supplier side: estimates are not negative then rising toward zero then positive')
    # text: where AKM0 rejects zero on the supplier side, its 90% set is two rays, so it contains both signs
    claim(lambda: all(sc.loc[h, 'lin_kind90'] == 'two rays' for h in sc_akm), 'supplier side: where AKM0 rejects zero its 90% set is not always two rays')
    def runs(xs):                                                        # e.g. [5, 18, 19] -> 5 and 18--19 (math mode)
        if not xs:
            return 'none'
        bl = np.split(np.array(xs), np.where(np.diff(xs) > 1)[0] + 1)
        r = [f'{b[0]}' if len(b) == 1 else f'{b[0]}\\text{{--}}{b[-1]}' for b in bl]
        return r[0] if len(r) == 1 else ', '.join(r[:-1]) + '\\text{ and }' + r[-1]
    rng = lambda xs: (f'{min(xs)}' if min(xs) == max(xs) else f'{min(xs)}\\text{{--}}{max(xs)}') if xs else 'none'   # used in math mode
    claim(lambda: all(cs.loc[h, 'lin_pp'] > 0 for h in cs_shown), 'customer side: estimates are not positive at every reported horizon')
    M = {
        'tmCSlinOne': f"{cs.loc[1, 'lin_pp']:.2f}", 'tmCSseOne': f"{cs.loc[1, 'lin_se_akm']:.2f}",
        'tmCSpOne': f"{cs.loc[1, 'p_akm0']:.3f}",
        'tmCSlinFour': f"{cs.loc[4, 'lin_pp']:.2f}", 'tmCSlinEight': f"{cs.loc[8, 'lin_pp']:.2f}",
        'tmCSlinSixteen': f"{cs.loc[16, 'lin_pp']:.2f}",
        'tmCSsigHi': str(cs_hi), 'tmCSakmHi': str(cs_akm0_hi), 'tmCSshownHi': str(max(cs_shown)), 'tmCSblankLo': str(min(cs_blank)),
        'tmCSFone': f"{cs.loc[1, 'lin_F_firm']:.1f}", 'tmCSshockT': f"{cs.loc[1, 'lin_F_shock']:.1f}",
        'tmCSfirms': str(int(cs.loc[1, 'units_lin'])), 'tmSCfirms': str(int(sc.loc[1, 'units_lin'])),
        'tmSCblank': rng(sc_blank), 'tmSCneg': rng(sc_neg), 'tmSCweak': rng(sc_weak), 'tmSCshownLo': str(min(sc_shown)),
        'tmSCakmSig': runs(sc_akm), 'tmSCsig': runs(sc_sig), 'tmSCnegHi': str(sc_neg_hi),
        'tmSClinSixteen': f"{sc.loc[16, 'lin_pp']:.2f}", 'tmSCseSixteen': f"{sc.loc[16, 'lin_se_akm']:.2f}",
        'tmSCpSixteen': f"{sc.loc[16, 'p_akm0']:.3f}",
        'tmCovSales': f'{100 * round(float(cov.median_sales_share), 4):.2f}', 'tmCovInputs': f'{cov.median_pct_inputs:.1f}',
    }
    M = {k: ('$-$' + v[1:] if v.startswith('-') else v) for k, v in M.items()}
    (TAB / 'Timing_macros.tex').write_text(HEAD_T + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in M.items()))


def timing_figure(OUT, FIG, src='timing_both.csv', cols=None, name='timing_both', vline='cs'):
    "Figure E1: both directions, 2SLS impulse responses on one horizon axis and one vertical scale (shock-level 68 / 90 percent bands)."
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = pd.read_csv(OUT / src).sort_values('h')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True, sharey=True)
    for ax, (dr, title, col) in zip(axes, cols or FIG_COLS):
        q = d[d.direction == dr]
        h = q.h.to_numpy()
        ok = ((q.lin_F_firm >= FMIN) & (q.rf_pp / q.lin_pp > 0)).to_numpy()
        if src == 'timing_both.csv':
            claim(lambda: np.all(np.diff(np.flatnonzero(ok)) == 1), f'figure: {dr} horizons shown are not one contiguous run: {list(h[ok])}')
        b = np.where(ok, q.lin_pp, np.nan)
        se = np.where(ok, q.lin_se_akm, np.nan)
        for zc, al in [(1.645, 0.16), (1.0, 0.34)]:
            ax.fill_between(h, b - zc * se, b + zc * se, color=col, alpha=al, lw=0)
        ax.plot(h, b, color=col, lw=2)
        ax.plot(h[ok], b[ok], ls='none', marker='o', ms=5, color=col, zorder=3)
        ax.axhline(0, color=INK2, lw=0.7)
        if dr == vline:                                                 # h = 1 is Table E1 (customer direction only)
            ax.axvline(1, color=INK2, lw=0.8, ls=':')
        ax.set_title(title, fontsize=10.5, color=INK, loc='left')
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_xlim(-0.6, 20.6)
        ax.set_xticks(range(0, 21, 2))
        ax.tick_params(labelsize=8, colors=INK2)
        for s_ in ['top', 'right']:
            ax.spines[s_].set_visible(False)
        for s_ in ['left', 'bottom']:
            ax.spines[s_].set_color('#c9c8c3')
    axes[0].set_ylabel('2SLS: pp of green share per SD', fontsize=8.8, color=INK2)
    fig.supxlabel('horizon h: years from the end of the partners’ five-year window to the outcome year' + (' (h = 1 is Table E1)' if vline else ''),
                  fontsize=8.8, color=INK2)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f'{name}.png', dpi=170)
    fig.savefig(FIG / f'{name}.pdf')


IO_COLS = {'A': [('ci', 'Customers\u2019 greening \u2192 supplier\n(input-output table, direct coefficients)', BLUE),
                 ('si', 'Suppliers\u2019 greening \u2192 customer\n(input-output table, direct coefficients)', ORANGE)],
           'Ac': [('ci', 'Customers\u2019 greening \u2192 supplier\n(input-output table of the outcome year, direct)', BLUE),
                  ('si', 'Suppliers\u2019 greening \u2192 customer\n(input-output table of the outcome year, direct)', ORANGE)],
           'L': [('ci', 'Customers\u2019 greening \u2192 supplier\n(input-output table, Leontief inverse)', BLUE),
                 ('si', 'Suppliers\u2019 greening \u2192 customer\n(input-output table, Leontief inverse)', ORANGE)]}


def io_timing(OUT, TAB, FIG):
    "Timing of both directions with the sector-level input-output network in both (timing_io_{A,L}.csv): figures and text macros."
    mac = {}
    for net, suf in (('A', ''), ('L', ''), ('A', '_linked'), ('Ac', '')):
        f = OUT / f'timing_io_{net}{suf}.csv'
        if not f.exists():
            continue
        cols = IO_COLS[net] if not suf else [(dr, t.replace('(input-output table, direct coefficients)', '(input-output table, direct coefficients;\nsuppliers with Compustat customer links only)'), c) for dr, t, c in IO_COLS[net]]
        timing_figure(OUT, FIG, src=f.name, cols=cols, name=f'timing_io_{net}{suf}', vline=None)
        net = net + ('lk' if suf else '')
        d = pd.read_csv(f)
        for dr, tag in (('ci', 'Ci'), ('si', 'Si')):
            q = d[d.direction == dr].set_index('h')
            ok = (q.lin_F_firm >= FMIN) & (q.rf_pp / q.lin_pp > 0)
            sh = [h for h in q.index if ok[h]]
            k = f'io{net}{tag}'
            mac[k + 'Lo'] = str(min(sh)) if sh else 'none'
            mac[k + 'Hi'] = str(max(sh)) if sh else 'none'
            if 1 in q.index:
                mac[k + 'One'] = f'{q.at[1, "lin_pp"]:.2f}'
            mac[k + 'Four'] = f'{q.at[4, "lin_pp"]:.2f}'
            mac[k + 'PFour'] = p3(q.at[4, 'p_akm0'])
            mac[k + 'FFour'] = f'{q.at[4, "lin_F_firm"]:.1f}'
            mac[k + 'AKMmin'] = p3(q.loc[sh, 'p_akm0'].min()) if sh else 'none'
            mac[k + 'AKMsig'] = runs_text([h for h in sh if q.at[h, 'p_akm0'] < 0.10])
            mac[k + 'Eff'] = f'{q.loc[sh, "lin_F_shock"].median():.1f}' if sh else 'none'
            mac[k + 'Firms'] = n(q.at[4, 'units_lin'])
            mac[k + 'PosHi'] = str(max([h for h in sh if q.at[h, 'lin_pp'] > 0], default=-1)) if sh else 'none'
            mac[k + 'Shown'] = str(len(sh))
            mac[k + 'MinB'] = f'{q.loc[sh, "lin_pp"].min():.2f}' if sh else 'none'
            mac[k + 'MaxB'] = f'{q.loc[sh, "lin_pp"].max():.2f}' if sh else 'none'
    if mac:
        (TAB / 'Timing_IO_macros.tex').write_text('% Generated by Customer_Tables.io_timing. Do not edit by hand.\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in mac.items()))


def sector_timing(OUT, TAB, FIG):
    "Timing of both directions with sectors as the units and the sector-level input-output network (timing_sector_{A,L}.csv): figures and macros."
    mac = {}
    for net, nm in (('A', 'direct coefficients'), ('L', 'Leontief inverse')):
        f = OUT / f'timing_sector_{net}.csv'
        if not f.exists():
            continue
        cols = [('sci', f'Customer sectors\u2019 greening \u2192 sector\n(sector-level input-output table, {nm})', BLUE),
                ('ssi', f'Supplier sectors\u2019 greening \u2192 sector\n(sector-level input-output table, {nm})', ORANGE)]
        timing_figure(OUT, FIG, src=f.name, cols=cols, name=f'timing_sector_{net}', vline=None)
        d = pd.read_csv(f)
        for dr, tag in (('sci', 'Ci'), ('ssi', 'Si')):
            q = d[d.direction == dr].set_index('h')
            ok = (q.lin_F_firm >= FMIN) & (q.rf_pp / q.lin_pp > 0)
            sh = [h for h in q.index if ok[h]]
            k = f'sec{net}{tag}'
            mac[k + 'Lo'] = str(min(sh)) if sh else 'none'
            mac[k + 'Hi'] = str(max(sh)) if sh else 'none'
            for h_, w_ in ((1, 'One'), (4, 'Four'), (8, 'Eight')):
                mac[k + w_] = f'{q.at[h_, "lin_pp"]:.2f}'
            mac[k + 'PFour'] = p3(q.at[4, 'p_akm0']); mac[k + 'FFour'] = f'{q.at[4, "lin_F_firm"]:.1f}'
            mac[k + 'AKMmin'] = p3(q.loc[sh, 'p_akm0'].min()) if sh else 'none'
            mac[k + 'AKMsig'] = runs_text([h for h in sh if q.at[h, 'p_akm0'] < 0.10])
            mac[k + 'Units'] = n(q.at[4, 'units_lin']); mac[k + 'Shown'] = str(len(sh))
            mac[k + 'MinB'] = f'{q.loc[sh, "lin_pp"].min():.2f}' if sh else 'none'
            mac[k + 'MaxB'] = f'{q.loc[sh, "lin_pp"].max():.2f}' if sh else 'none'
    if mac:
        (TAB / 'Timing_Sector_macros.tex').write_text('% Generated by Customer_Tables.sector_timing. Do not edit by hand.\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in mac.items()))


def runs_text(xs):
    if not xs:
        return 'none'
    bl = np.split(np.array(xs), np.where(np.diff(xs) > 1)[0] + 1)
    r = [f'{b[0]}' if len(b) == 1 else f'{b[0]}\\text{{--}}{b[-1]}' for b in bl]
    return r[0] if len(r) == 1 else ', '.join(r[:-1]) + '\\text{ and }' + r[-1]


def robust_tables(OUT, TAB):
    "Appendix tables of robustness checks of the customer IV (firm and sector level), from robust_lambda.csv, robust_other.csv, robust_loo.csv."
    out = []
    if (OUT / 'robust_lambda.csv').exists():
        r = pd.read_csv(OUT / 'robust_lambda.csv').sort_values('lam')
        L = [r'\begin{tabular}{lccccccc}', r'\toprule',
             r' & \multicolumn{4}{c}{Firm level} & \multicolumn{3}{c}{Sector level} \\', r'\cmidrule(lr){2-5}\cmidrule(lr){6-8}',
             r'$\lambda$ & First-stage $F$ & IV & 2SLS (pp) & AKM0 $p$ & First-stage $F$ & IV & $p$ \\', r'\midrule']
        for _, x in r.iterrows():
            lab = f'{x.lam:g}' + (r' (moments)' if x.mom else (r' (baseline)' if x.lam == 20 else ''))
            L.append(f'{lab} & {x.firm_F:.1f} & {x.firm_iv:.2f} ({x.firm_iv_se:.2f}) & {x.firm_lin_pp:.2f} & {p3(x.akm0_sector)} & '
                     f'{x.sec_F:.1f} & {x.sec_iv:.3f} ({x.sec_iv_se:.3f}) & {p3(x.sec_iv_p)} \\\\')
        L += [r'\bottomrule', r'\end{tabular}']
        (TAB / 'Robust_Lambda.tex').write_text('\n'.join(L) + '\n')
        mac = {'rlLamMom': f'{r[r.mom].lam.iloc[0]:g}', 'rlIVmin': f'{r.firm_iv.min():.2f}', 'rlIVmax': f'{r.firm_iv.max():.2f}',
               'rlAKMmin': p3(r.akm0_sector.min()), 'rlAKMmax': p3(r.akm0_sector.max()),
               'rlSecPmax': p3(r.sec_iv_p.max()), 'rlLinMin': f'{r.firm_lin_pp.min():.2f}', 'rlLinMax': f'{r.firm_lin_pp.max():.2f}'}
        out.append(mac)
    if (OUT / 'robust_other.csv').exists():
        r = pd.read_csv(OUT / 'robust_other.csv')
        r = r[r.label != 'Outcome years to 2010']                   # not reported
        L = [r'\begin{tabular}{lcccccc}', r'\toprule',
             r' & Suppliers & Firm-years & First-stage $F$ & IV & 2SLS (pp) & AKM0 $p$ \\', r'\midrule']
        for grp, title in [('base', 'Base period'), ('sample', 'Outcome years'), ('links', 'Link size'), ('network', 'Network in the regressor'), ('shifts', 'Class shifts in the instrument')]:
            L.append(r'\multicolumn{7}{l}{\textit{' + title + r'}} \\')
            for _, x in r[r.group == grp].iterrows():
                lab = x.label.replace('Base period ', '').replace('Outcome years to ', 'Through ').replace('Baseline (links and class mixes 1990-94)', '1990--94 (baseline)')
                lab = lab.replace('1985-89', '1985--89').replace('1990-97', '1990--97').replace("Links of at least ", 'At least ').replace(" percent of the supplier's sales", r'\% of sales')
                lab = {'Regressor: links of the window t-5 ... t-1 (instrument: base links)': r'Links of window $t-5,\dots,t-1$ (instrument: 1990--94 links)',
                       'Shifts from all patents (sample firms included)': 'All patents (sample firms included)',
                       'Shifts measured over t-10 ... t-6': r'Shifts over $t-10,\dots,t-6$',
                       'Three-character CPC classes': 'Three-character CPC classes'}.get(x.label, lab)
                L.append(f'{lab} & {n(x.units)} & {n(x.N)} & {x.F:.1f} & {x.iv:.2f} & {x.lin_pp:.2f} & {p3(x.akm0)} \\\\')
        nt = r[r.group == 'network'].iloc[0] if (r.group == 'network').any() else None
        if nt is not None:
            out.append({'rlNetIV': f'{nt.iv:.2f}', 'rlNetLin': f'{nt.lin_pp:.2f}', 'rlNetF': f'{nt.F:.1f}', 'rlNetP': p3(nt.akm0), 'rlNetUnits': n(nt.units), 'rlNetN': n(nt.N)})
        sh = r[r.group == 'shifts']
        if len(sh):
            out.append({'rlShIVmin': f'{sh.iv.min():.2f}', 'rlShIVmax': f'{sh.iv.max():.2f}', 'rlShLinMin': f'{sh.lin_pp.min():.2f}', 'rlShLinMax': f'{sh.lin_pp.max():.2f}',
                        'rlShPmin': p3(sh.akm0.min()), 'rlShPmax': p3(sh.akm0.max()), 'rlShFmin': f'{sh.F.min():.1f}'})
        if (OUT / 'robust_loo.csv').exists():
            q = pd.read_csv(OUT / 'robust_loo.csv')
            rng2 = lambda v, f: (f.format(v.min()) if f.format(v.min()) == f.format(v.max()) else f.format(v.min()) + '--' + f.format(v.max()))
            L.append(f'Dropping one of the {len(q)} customer sectors with the most link weight (range) & {rng2(q.units, "{:,.0f}")} & {rng2(q.N, "{:,.0f}")} & '
                     f'{rng2(q.F, "{:.1f}")} & {rng2(q.iv, "{:.2f}")} & {rng2(q.lin_pp, "{:.2f}")} & {rng2(q.akm0, "{:.3f}")} \\\\')
            out.append({'rlLooN': str(len(q)), 'rlLooMin': f'{q.lin_pp.min():.2f}', 'rlLooMax': f'{q.lin_pp.max():.2f}',
                        'rlLooPmax': p3(q.akm0.max()), 'rlLooPsig': str(int((q.akm0 < 0.10).sum())), 'rlLooFmin': f'{q.F.min():.1f}'})
        L += [r'\bottomrule', r'\end{tabular}']
        (TAB / 'Robust_Other.tex').write_text('\n'.join(L) + '\n')
    mac = {k: v for d in out for k, v in d.items()}
    if mac:
        (TAB / 'Robust_macros.tex').write_text('% Generated by Customer_Tables.robust_tables. Do not edit by hand.\n'
                                               + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in mac.items()))


class Customer_Tables:
    "Tables and figure of the customer-IV draft, built from Specifications/Customer_IV/output."

    def __init__(self, P):
        self.Root = P.Directory / 'Specifications' / 'Customer_IV'
        self.Out = self.Root / 'output'
        self.Tab = P.Directory / 'Results' / 'Tables'
        self.Fig = P.Directory / 'Results' / 'Figures'

    def Tables(self):
        """
        Build Tables E1, E3, E2 and Figure E1 from the saved outputs (each only if its outputs exist).

        Output: Results/Tables/{Customer_IV,Supplier_IV,Timing_both}.tex and *_macros.tex, Results/Figures/timing_both.pdf
        """
        self.Tab.mkdir(parents=True, exist_ok=True)
        if (self.Out / 'table_e1_coefs.csv').exists():
            customer_table(self.Out, self.Tab)
        if (self.Out / 'table_e3_coefs.csv').exists() and (self.Out / 'timing_both.csv').exists():
            supplier_table(self.Out, self.Tab)
        if (self.Out / 'timing_both.csv').exists() and (self.Out / 'timing_coverage.csv').exists():
            timing_table(self.Out, self.Tab)
            timing_figure(self.Out, self.Fig)
        robust_tables(self.Out, self.Tab)
        io_timing(self.Out, self.Tab, self.Fig)
        sector_timing(self.Out, self.Tab, self.Fig)
        if CLAIMS:
            print(f'{len(CLAIMS)} statement(s) of the draft are not supported by these numbers:'); [print('  -', c) for c in CLAIMS]

    def Check_Reproduction(self, rtol=1e-6):
        """
        Compare the saved outputs with the reference run of the draft's authors (Specifications/Customer_IV/reference/): for every
        numeric column the two files share, the largest relative deviation (absolute where the reference is near zero). Prints
        one line per file and returns the table of deviations; a column above rtol is listed.
        """
        ref = self.Root / 'reference'
        rows = []
        for f in sorted(ref.glob('*.csv')):
            new = self.Out / f.name
            if not new.exists():
                rows.append(dict(file=f.name, columns=0, worst=np.nan, above_tol=['missing output']))
                continue
            a, b = pd.read_csv(f), pd.read_csv(new)
            if len(a) != len(b):
                rows.append(dict(file=f.name, columns=0, worst=np.nan, above_tol=[f'rows {len(a)} vs {len(b)}']))
                continue
            dev = {}
            for c in a.columns.intersection(b.columns):
                if a[c].dtype.kind in 'fiu' and b[c].dtype.kind in 'fiu':
                    x, y = a[c].to_numpy(float), b[c].to_numpy(float)
                    ok = np.isfinite(x) & np.isfinite(y)
                    same_na = (np.isnan(x) == np.isnan(y)).all() and (np.isposinf(x) == np.isposinf(y)).all()
                    d = np.abs(x[ok] - y[ok]) / np.maximum(np.abs(x[ok]), 1e-3)
                    dev[c] = float(d.max()) if len(d) and same_na else (0.0 if same_na else np.inf)
            rows.append(dict(file=f.name, columns=len(dev), worst=max(dev.values()) if dev else 0.0,
                             above_tol=[c for c, v in dev.items() if v > rtol]))
        out = pd.DataFrame(rows)
        for r in out.itertuples():
            print(f"{r.file:28s} {r.columns:3d} numeric columns, largest relative deviation {r.worst:.2e}" +
                  (f"; above {rtol:g}: {', '.join(r.above_tol)}" if r.above_tol else ''))
        return out
