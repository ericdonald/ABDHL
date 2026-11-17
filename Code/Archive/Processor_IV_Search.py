"""""""""""
Archive: instrument-search methods of Processor (tentatively dropped)

Notes: The government-patent, R&D-price and knowledge-spillover instruments (batches 2-4, 6, 7, 9), their first-stage and shock-relevance
    diagnostics, and the builders of their shocks. They were superseded by the customer IV (Customer_IV.py). Kept verbatim as methods of
    a stand-alone class; nothing imports this file.

    Restore:  git checkout pre-customer-iv-prune -- Code/Processor.py Code/Executor.py     (the full code before the customer IV was added)
          or  copy the methods below back into class Processor (they only use self.Directory, other Processor methods and Estimators).
    Data:     the Cleaner blocks that wrote the files these methods read (Spill_Net*, Tech_Pat*, Gov_CPC*, Ind_Pat_Shares_*, RD price files, ...) are in
              Archive/Cleaner_IV_Search.txt.

"""""""""""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
import Processing_Functions as gpf
import Estimators as es


class Processor_IV_Search:
    "Archived instrument-search methods of Processor"


    def Instruments(self, sfx='', asinh_pat=False, spill=True, type_gov=False):
        """""
        Create Series of Greenification Shocks

        sfx: '' = 4-digit CPC subclasses (baseline); '_c3' = 3-digit CPC classes. For '_c3' only the government and
        spillover shocks are built (the R&D-price shock does not depend on the technology class).

        Output: Clean Data/Govt_Shocks.pkl
                Clean Data/KS_Shocks.pkl
                Clean Data/RD_Firm_Panel.pkl  (zero-stage estimation sample)
                Clean Data/RD_Shocks.pkl
        """""
        
        # ----------------------------------------------------------------

        # Build instrument dataframes.

        # ----------------------------------------------------------------
        
        gov_cpc_df = pd.read_pickle(f'{self.Directory}/Clean Data/Gov_CPC{sfx}.pkl')
        ind_pat_shares_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_Pat_Shares_Pre{sfx}.pkl')

        
        # ------------------------ #
        # Government Patent Shocks #
        # ------------------------ #
        if asinh_pat:   # government patenting as asinh(patents) by class-year; only the government shock is rebuilt
            gov_cpc_df = gov_cpc_df.copy()
            gov_cpc_df[['gov_pat_count', 'gov_pat_cites']] = np.arcsinh(gov_cpc_df[['gov_pat_count', 'gov_pat_cites']])
        govt_shocks_df = pd.merge(gov_cpc_df,
                                    ind_pat_shares_df,
                                    on='cpc_subclass',
                                    how='inner'
                                    )
        
        frames = []
        for ty in ['clean', 'dirty', 'gen']:
            shock = govt_shocks_df.copy()
            shock = shock[shock['type'] == ty]
            if type_gov:   # the government series of the same type as the recipient type
                shock['gov_pat_count'], shock['gov_pat_cites'] = shock[f'gov_{ty}_count'], shock[f'gov_{ty}_cites']
            
            shock['weighted_pat_govt'] = shock['cpc_pat_share'] * shock['gov_pat_count']
            shock['pat_govt_shock'] = shock.groupby(['BLS_Industry', 'year'])['weighted_pat_govt'].transform('sum')
            
            shock['weighted_cite_govt'] = shock['cpc_cite_share'] * shock['gov_pat_cites']
            shock['cite_govt_shock'] = shock.groupby(['BLS_Industry', 'year'])['weighted_cite_govt'].transform('sum')
            
            shock = shock[['BLS_Industry', 'year', 'type',
                           'pat_govt_shock', 'cite_govt_shock']].drop_duplicates()

            frames.append(shock)
                
        govt_shocks_df = pd.concat(frames, ignore_index=True)
        
        govt_shocks_wide = govt_shocks_df.pivot(index=['BLS_Industry', 'year'], columns=['type'], 
                                              values=['pat_govt_shock', 'cite_govt_shock'])

        govt_shocks_wide.columns = [f'{v}_{t}' for v, t in govt_shocks_wide.columns]
        govt_shocks_df = govt_shocks_wide.reset_index()
        
        govt_shocks_df.to_pickle(f"{self.Directory}/Clean Data/Govt_Shocks{sfx}{'_Type' if type_gov else ''}{'_asinh' if asinh_pat else ''}.pkl")
        if asinh_pat or type_gov:
            return
        
        
        if spill:
            self.Spillover_Shocks(sfx)
        if sfx:
            return

        IV_year_start = 1980
        
        state_rdp_df = pd.read_pickle(f'{self.Directory}/Clean Data/state_rd_price.pkl')
        PV_inventor_location_df = pd.read_pickle(f'{self.Directory}/Clean Data/Inventor_Locations.pkl')
        pat_firms_df = pd.read_pickle(f'{self.Directory}/Clean Data/Pat_Firms.pkl')
        pat_firms_df = pat_firms_df[pat_firms_df['year'] >= IV_year_start]
        # Keep only the inventor rows that can merge (memory); the inner merge below is unchanged by this
        PV_inventor_location_df = PV_inventor_location_df.loc[
            PV_inventor_location_df['patent_id'].isin(pat_firms_df['patent_id']),
            ['patent_id', 'inventor_id', 'state_fips']]
        
        
        # ------------------------ #
        # State R&D Price Exposure #
        # ------------------------ #
        
        # Firm Inventor Distribution
        firm_inv_df = pd.merge(pat_firms_df,
                                PV_inventor_location_df,
                                on='patent_id',
                                how='inner'
                                )
        
        firm_inv_df = firm_inv_df.drop_duplicates(
            subset=['patent_id', 'gvkey', 'inventor_id', 'state_fips'])
        firm_inv_df = firm_inv_df.dropna(subset=['inventor_id', 'state_fips'])
        
        firm_inv_df['pat_authors'] = firm_inv_df.groupby(['patent_id', 'gvkey'])['inventor_id'].transform('count')
        
        firm_inv_df['pat_weight'] = firm_inv_df['split_weight'] / firm_inv_df['pat_authors']
        firm_inv_df['pat_weight_clean'] = firm_inv_df['clean'] * firm_inv_df['pat_weight']
        firm_inv_df['pat_weight_dirty'] = firm_inv_df['dirty'] * firm_inv_df['pat_weight']
        
        firm_inv_df['cite_weight'] = firm_inv_df['norm_cites'] * firm_inv_df['split_weight'] / firm_inv_df['pat_authors']
        firm_inv_df['cite_weight_clean'] = firm_inv_df['clean'] * firm_inv_df['cite_weight']
        firm_inv_df['cite_weight_dirty'] = firm_inv_df['dirty'] * firm_inv_df['cite_weight']
        
        w_cols = ['pat_weight', 'pat_weight_clean', 'pat_weight_dirty', 'cite_weight', 'cite_weight_clean', 'cite_weight_dirty']
        
        fsy = (firm_inv_df.groupby(['gvkey', 'state_fips', 'year'], as_index=False)[w_cols]
                          .sum())
 
        window = []
        for d in range(-4, 6):
            tmp = fsy.copy()
            tmp['year'] = tmp['year'] + d
            window.append(tmp)
 
        fsy_win = (pd.concat(window, ignore_index=True)
                     .groupby(['gvkey', 'state_fips', 'year'], as_index=False)
                     .agg(**{c: (c, 'sum') for c in w_cols}))
  
        firm_tot = (fsy_win.groupby(['gvkey', 'year'], as_index=False)[w_cols]
                           .sum()
                           .rename(columns={c: f'{c}_tot' for c in w_cols}))
        fsy_win = fsy_win.merge(firm_tot, on=['gvkey', 'year'], how='left')
        
        share_map = {
            'firm_fips_pat_share':        ('pat_weight',        'pat_weight_tot'),
            'firm_fips_pat_share_clean':  ('pat_weight_clean',  'pat_weight_clean_tot'),
            'firm_fips_pat_share_dirty':  ('pat_weight_dirty',  'pat_weight_dirty_tot'),
            'firm_fips_cite_share':       ('cite_weight',       'cite_weight_tot'),
            'firm_fips_cite_share_clean': ('cite_weight_clean', 'cite_weight_clean_tot'),
            'firm_fips_cite_share_dirty': ('cite_weight_dirty', 'cite_weight_dirty_tot'),
        }
        for out_col, (num, den) in share_map.items():
            fsy_win[out_col] = fsy_win[num] / fsy_win[den].where(fsy_win[den] > 0)
 
        firm_inv_df = fsy_win[['gvkey', 'state_fips', 'year']
                              + list(share_map)].copy()

        
        # Firm Exposure
        firm_inv_df = pd.merge(state_rdp_df,
                                firm_inv_df,
                                on=['year', 'state_fips'],
                                how='inner'
                                )
        
        firm_inv_cols = []
        for ty in ['', '_clean', '_dirty']:
            firm_inv_df['weighted_rho_pats' + ty] = firm_inv_df['firm_fips_pat_share' + ty] * firm_inv_df['rho_h']
            firm_inv_df['weighted_rho_cites'+ ty] = firm_inv_df['firm_fips_cite_share'+ ty] * firm_inv_df['rho_h']
            
            firm_inv_df['E_rho_pats'+ ty] = firm_inv_df.groupby(['gvkey', 'year'])['weighted_rho_pats'+ ty].transform('sum')
            firm_inv_df['E_rho_cites'+ ty] = firm_inv_df.groupby(['gvkey', 'year'])['weighted_rho_cites'+ ty].transform('sum')
            
            firm_inv_cols.append('E_rho_pats' + ty)
            firm_inv_cols.append('E_rho_cites' + ty)
        
        firm_inv_df = firm_inv_df[['gvkey', 'year'] + firm_inv_cols].drop_duplicates()
        
        
        # Firm Patenting
        firm_pats_df = pat_firms_df.copy()
        
        firm_pats_df['pat_weight'] = firm_pats_df['split_weight']
        firm_pats_df['pat_weight_clean'] = firm_pats_df['clean'] * firm_pats_df['pat_weight']
        firm_pats_df['pat_weight_dirty'] = firm_pats_df['dirty'] * firm_pats_df['pat_weight']
        
        firm_pats_df['cite_weight'] = firm_pats_df['norm_cites'] * firm_pats_df['split_weight']
        firm_pats_df['cite_weight_clean'] = firm_pats_df['clean'] * firm_pats_df['cite_weight']
        firm_pats_df['cite_weight_dirty'] = firm_pats_df['dirty'] * firm_pats_df['cite_weight']
        
        
        firm_pat_cols = []
        for ty in ['', '_clean', '_dirty']:
            firm_pats_df['pat_count'+ ty] = firm_pats_df.groupby(['gvkey', 'year'])['pat_weight'+ ty].transform('sum')
            firm_pats_df['pat_cites'+ ty] = firm_pats_df.groupby(['gvkey', 'year'])['cite_weight'+ ty].transform('sum')
            
            firm_pat_cols.append('pat_count' + ty)
            firm_pat_cols.append('pat_cites' + ty)
        
        firm_pats_df = firm_pats_df[['gvkey', 'BLS_Industry', 'year'] + firm_pat_cols].drop_duplicates()
        
        
        # Zero Stage Regressions
        firm_pat_wide_df = pd.merge(firm_pats_df,
                                firm_inv_df,
                                on=['gvkey', 'year'],
                                how='inner'
                                )
        
        type_map = {
            'general': {'pat_count':  'pat_count',
                        'pat_cites':  'pat_cites',
                        'E_rho_pats': 'E_rho_pats',
                        'E_rho_cites':'E_rho_cites'},
            'clean':   {'pat_count':  'pat_count_clean',
                        'pat_cites':  'pat_cites_clean',
                        'E_rho_pats': 'E_rho_pats_clean',
                        'E_rho_cites':'E_rho_cites_clean'},
            'dirty':   {'pat_count':  'pat_count_dirty',
                        'pat_cites':  'pat_cites_dirty',
                        'E_rho_pats': 'E_rho_pats_dirty',
                        'E_rho_cites':'E_rho_cites_dirty'},
        }
 
        # Two versions: all firms, and only firms that ever have clean patents (suffix _cf)
        _clean_firms = set(firm_pat_wide_df.loc[firm_pat_wide_df['pat_count_clean'] > 0, 'gvkey'])
        for _v, _vsfx in [('all', ''), ('cf', '_cf')]:
            _wide = firm_pat_wide_df if _v == 'all' else firm_pat_wide_df[firm_pat_wide_df['gvkey'].isin(_clean_firms)]
            id_cols = ['gvkey', 'BLS_Industry', 'year']
            frames  = []
            for tname, cmap in type_map.items():
                missing = [c for c in cmap.values() if c not in _wide.columns]
                if missing:
                    raise KeyError(f'type "{tname}" needs columns {missing}')
                sub = _wide[id_cols + list(cmap.values())].copy()
                sub = sub.rename(columns={v: k for k, v in cmap.items()})
                sub['type'] = tname
                frames.append(sub)
 
            firm_pat_panel_df = pd.concat(frames, ignore_index=True)
                
            for c in ['pat_count', 'pat_cites', 'E_rho_pats', 'E_rho_cites']:
                firm_pat_panel_df[f'ln_{c}'] = np.log(
                    firm_pat_panel_df[c].where(firm_pat_panel_df[c] > 0))
        
            firm_pat_panel_df['entity'] = (firm_pat_panel_df['gvkey'].astype(str) + '_'
                                              + firm_pat_panel_df['type'])
            firm_pat_panel_df = firm_pat_panel_df.set_index(['entity','year']).sort_index()
            firm_pat_panel_df = firm_pat_panel_df.dropna(subset=['ln_pat_count', 'ln_pat_cites', 'ln_E_rho_pats', 'ln_E_rho_cites'])
            firm_pat_panel_df.reset_index().to_pickle(f'{self.Directory}/Clean Data/RD_Firm_Panel{_vsfx}.pkl')
    
            m_pats = gpf.run_reg(firm_pat_panel_df['ln_pat_count'], firm_pat_panel_df['ln_E_rho_pats'], 'panel')
            m_cites = gpf.run_reg(firm_pat_panel_df['ln_pat_cites'], firm_pat_panel_df['ln_E_rho_cites'], 'panel')
        
            firm_pat_panel_df['pat_count_hat'] = np.exp(m_pats.predict().fitted_values)
            firm_pat_panel_df['pat_cites_hat'] = np.exp(m_cites.predict().fitted_values)
            firm_pat_panel_df = firm_pat_panel_df.reset_index()
        
            hat_cols = ['pat_count_hat', 'pat_cites_hat']
 
            firm_pat_panel_df[hat_cols] = firm_pat_panel_df[hat_cols].fillna(0.0)
 
            ind_hat_df = (firm_pat_panel_df
                          .pivot_table(index=['BLS_Industry', 'year'],
                                       columns='type',
                                       values=hat_cols,
                                       aggfunc='sum',
                                       fill_value=0.0))
            ind_hat_df.columns = [f'{v}__{t}' for v, t in ind_hat_df.columns]
            ind_hat_df = ind_hat_df.reset_index()
 
            RD_shocks_df = ind_hat_df.rename(columns={
                'pat_count_hat__general': 'pat_count_hat_gen',
                'pat_count_hat__clean':   'pat_count_clean_hat',
                'pat_count_hat__dirty':   'pat_count_dirty_hat',
                'pat_cites_hat__general': 'pat_cites_hat_gen',
                'pat_cites_hat__clean':   'pat_cites_clean_hat',
                'pat_cites_hat__dirty':   'pat_cites_dirty_hat'})
        
            RD_shocks_df.to_pickle(f'{self.Directory}/Clean Data/RD_Shocks{_vsfx}.pkl')


    def RD_ZeroStage(self, batch):
        """""
        Zero-stage regressions behind the R&D-price shock: log firm patenting on log state R&D-price exposure,
        with entity (firm x type) and year effects, SEs clustered by entity. Whole sample (general, clean and dirty
        firm-type panels stacked, as used to build the instrument) and each type on its own. Not used to build
        instruments except the whole-sample fit.

        Output: Results/Batches/<batch>/zero_stage.csv, table_zero_stage.tex
        """""
        outdir = self.Directory / 'Results' / 'Batches' / batch
        outdir.mkdir(parents=True, exist_ok=True)
        panels_rd = {'': pd.read_pickle(f'{self.Directory}/Clean Data/RD_Firm_Panel.pkl').set_index(['entity', 'year']).sort_index()}
        cf_path = Path(f'{self.Directory}/Clean Data/RD_Firm_Panel_cf.pkl')
        if cf_path.exists():
            panels_rd['cf'] = pd.read_pickle(cf_path).set_index(['entity', 'year']).sort_index()
        samples = [('all', 'Whole sample', None, ''), ('clean', 'Clean only', 'clean', ''),
                   ('dirty', 'Dirty only', 'dirty', ''), ('general', 'General only', 'general', '')]
        if 'cf' in panels_rd:   # firms that ever have clean patents (all three types)
            samples += [('cf_all', 'Clean firms: whole sample', None, 'cf'), ('cf_clean', 'Clean firms: clean only', 'clean', 'cf'),
                        ('cf_general', 'Clean firms: general only', 'general', 'cf')]
        rows = []
        for meas, y, x in [('pat', 'ln_pat_count', 'ln_E_rho_pats'), ('cite', 'ln_pat_cites', 'ln_E_rho_cites')]:
            for key, lab, ty, pk in samples:
                panel = panels_rd[pk]
                d = panel if ty is None else panel[panel['type'] == ty]
                m = gpf.run_reg(d[y], d[[x]], 'panel')
                rows.append(dict(measure=meas, sample=key, label=lab, coef=float(m.params[x]), se=float(m.std_errors[x]),
                                 p=float(m.pvalues[x]), t=float(m.tstats[x]), N=int(m.nobs), n_entities=int(m.entity_info['total']),
                                 n_firms=int(d['gvkey'].nunique()), r2_within=float(m.rsquared)))
        out = pd.DataFrame(rows)
        out.to_csv(outdir / 'zero_stage.csv', index=False)
        def write_table(smp, fname):
            ns = len(smp)
            lines = [r'\begin{tabular}{l' + 'c' * ns + '}', r'\toprule',
                     ' & ' + ' & '.join(f'({i + 1})' for i in range(ns)) + r' \\',
                     ' & ' + ' & '.join(l for _, l, _, _ in smp) + r' \\', r'\midrule']
            for meas, title in [('pat', 'Dependent variable: log patent count'), ('cite', 'Dependent variable: log normalised citations')]:
                sub = out[out['measure'] == meas].set_index('sample').loc[[k for k, _, _, _ in smp]]
                lines += [r'\multicolumn{%d}{l}{\textit{' % (ns + 1) + title + r'}} \\',
                          'Log state R\\&D-price exposure & ' + ' & '.join(f"{r.coef:.3f}{gpf.get_stars(r.p)}" for r in sub.itertuples()) + r' \\',
                          ' & ' + ' & '.join(f'({r.se:.3f})' for r in sub.itertuples()) + r' \\',
                          '$t$-statistic & ' + ' & '.join(f'{r.t:.2f}' for r in sub.itertuples()) + r' \\',
                          'Observations & ' + ' & '.join(f'{r.N:,}' for r in sub.itertuples()) + r' \\',
                          'Firm-type entities & ' + ' & '.join(f'{r.n_entities:,}' for r in sub.itertuples()) + r' \\',
                          'Firms & ' + ' & '.join(f'{r.n_firms:,}' for r in sub.itertuples()) + r' \\',
                          '$R^2$ (effects removed) & ' + ' & '.join(f'{r.r2_within:.4f}' for r in sub.itertuples()) + r' \\', r'\midrule']
            lines += ['Fixed effects & ' + ' & '.join(['entity+year'] * ns) + r' \\', r'\bottomrule', r'\end{tabular}']
            with open(outdir / fname, 'w') as f:
                f.write('\n'.join(lines) + '\n')

        write_table([x for x in samples if x[3] == ''], 'table_zero_stage.tex')
        cf = [x for x in samples if x[3] == 'cf']
        if cf:   # clean-only (unchanged by the restriction) next to the same regressions on firms with clean patents
            ref = [('clean', 'Reference: all firms, clean only', 'clean', '')]
            write_table(ref + [x for x in cf if x[0] in ('cf_all', 'cf_general')], 'table_zero_stage_cf.tex')
        print(out.round(4).to_string())
        return out


    def KS_Shares(self, batch, BLS_year_start, Year_end, bin_len=5, sfx='_cg'):
        """""
        Composition of the spillovers each recipient type receives: for recipient type r in {clean, dirty, gen} the share
        of its received spillovers that originates in source type {clean, dirty, general} (the nine KS_Shocks columns).
        Pooled shares sum the shocks over manufacturing industries and years; sector-average shares are the mean of
        the industry-level shares (industry-years with no spillovers received are skipped).

        Output: Results/Batches/<batch>/ks_shares.csv, table_ks_shares.tex, table_ks_shares_bins.tex
        """""
        outdir = self.Directory / 'Results' / 'Batches' / batch
        outdir.mkdir(parents=True, exist_ok=True)
        ks = pd.read_pickle(f'{self.Directory}/Clean Data/KS_Shocks{sfx}.pkl')
        manu = np.arange(self.manu_cols[0], self.manu_cols[1] + 1)
        ks = ks[ks['BLS_Industry'].isin(manu) & (ks['year'] > BLS_year_start - bin_len) & (ks['year'] <= Year_end)].copy()
        ks['period'] = Year_end - ((Year_end - ks['year']) // bin_len) * bin_len
        if sfx.startswith('_cg'):
            recips, srcs = ['clean', 'gen'], ['c', 'g']
        else:
            recips, srcs = ['clean', 'dirty', 'gen'], ['c', 'd', 'g']
        names = {'clean': 'Clean', 'dirty': 'Dirty', 'gen': 'General'}
        src_names = {k: v for k, v in {'c': 'clean', 'd': 'dirty', 'g': 'general'}.items() if k in srcs}
        ns = len(srcs)
        rows = []
        for meas, c in [('pat', 'cnt'), ('cite', 'cit')]:
            for r in recips:
                cols = [f'spill_shock_{c}_{r}_{x}' for x in srcs]
                v = ks[['BLS_Industry', 'period'] + cols].copy()
                v[cols] = v[cols].fillna(0.0)
                v.columns = ['BLS_Industry', 'period'] + srcs
                for lab, d in [('all', v)] + [(int(p), g) for p, g in v.groupby('period')]:
                    tot = d[srcs].sum()
                    pooled = tot / tot.sum()
                    ind = d.groupby('BLS_Industry')[srcs].sum()
                    ind = ind[ind.sum(axis=1) > 0]
                    avg = ind.div(ind.sum(axis=1), axis=0).mean()
                    for x in srcs:
                        rows.append(dict(measure=meas, recipient=r, period=lab, source=src_names[x], pooled=pooled[x],
                                         sector_avg=avg[x], received=tot.sum()))
        out = pd.DataFrame(rows)
        out.to_csv(outdir / 'ks_shares.csv', index=False)
        hdr = [r'\begin{tabular}{l' + 'c' * (2 * ns) + '}', r'\toprule',
               r' & \multicolumn{%d}{c}{Pooled} & \multicolumn{%d}{c}{Sector average} \\' % (ns, ns),
               r'\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}' % (ns + 1, ns + 2, 2 * ns + 1),
               'Recipient & ' + ' & '.join(['From ' + v for v in src_names.values()] * 2) + r' \\', r'\midrule']
        lines = list(hdr)
        for meas, title in [('pat', 'Patent counts'), ('cite', 'Citation-weighted')]:
            lines.append(r'\multicolumn{%d}{l}{\textit{' % (2 * ns + 1) + title + r'}} \\')
            for r in recips:
                d = out[(out['measure'] == meas) & (out['recipient'] == r) & (out['period'] == 'all')].set_index('source')
                lines.append(names[r] + ' & ' + ' & '.join(f"{d.loc[x, 'pooled']:.3f}" for x in src_names.values()) + ' & '
                             + ' & '.join(f"{d.loc[x, 'sector_avg']:.3f}" for x in src_names.values()) + r' \\')
            lines.append(r'\midrule')
        lines[-1] = r'\bottomrule'; lines.append(r'\end{tabular}')
        with open(outdir / 'table_ks_shares.tex', 'w') as f:
            f.write('\n'.join(lines) + '\n')
        periods = sorted(p for p in out['period'].unique() if p != 'all')
        lines = [r'\begin{tabular}{ll' + 'c' * len(periods) + '}', r'\toprule',
                 'Recipient & Source & ' + ' & '.join(str(p) for p in periods) + r' \\', r'\midrule']
        for meas, title in [('pat', 'Patent counts'), ('cite', 'Citation-weighted')]:
            lines.append(r'\multicolumn{%d}{l}{\textit{%s (pooled shares)}} \\' % (len(periods) + 2, title))
            for r in recips:
                for x in src_names.values():
                    d = out[(out['measure'] == meas) & (out['recipient'] == r) & (out['source'] == x)].set_index('period')
                    lines.append((names[r] if x == 'clean' else '') + f' & {x} & ' + ' & '.join(f"{d.loc[p, 'pooled']:.3f}" for p in periods) + r' \\')
            lines.append(r'\midrule')
        lines[-1] = r'\bottomrule'; lines.append(r'\end{tabular}')
        with open(outdir / 'table_ks_shares_bins.tex', 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(out[out['period'] == 'all'].round(3).to_string())
        return out


    def _sparse_spill(self, sfx, asinh_pat=False, what='shocks', diag=False):
        """""
        Sparse-matrix version of Spillover_Shocks / KS_Recv for large technology sets (CPC main groups).
        what='shocks': industry x year receipts by recipient type and source type (same output as Spillover_Shocks);
        what='recv': class x type x year receipts (same output as KS_Recv).
        """""
        import scipy.sparse as sps
        D = f'{self.Directory}/Clean Data/'
        sn = pd.read_pickle(D + f"Spill_Net{sfx}{'_Diag' if diag else ''}.pkl")
        tp = pd.read_pickle(D + f'Tech_Pat{sfx}.pkl')
        if asinh_pat:
            tp[['pat_count', 'pat_cites']] = np.arcsinh(tp[['pat_count', 'pat_cites']])
        sh = None
        if what == 'shocks':
            sh = pd.read_pickle(D + f'Ind_Pat_Shares_Pre{sfx}.pkl')
            sh = sh[sh['type'] != 'all']
            sh = sh.assign(tech=sh['cpc_subclass'] + '_' + sh['type'].str[:1])
        sn = sn.dropna(subset=['S_ijk'])
        techs = sorted(set(sn['i']) | set(sn['j']) | set(tp['tech']) | (set(sh['tech']) if sh is not None else set()))
        tid = {t: n for n, t in enumerate(techs)}
        T = len(techs)
        y0, y1 = int(tp['year'].min()), int(tp['year'].max()) + 10
        Y = y1 - y0 + 1
        Pc, Pq = np.zeros((T, Y)), np.zeros((T, Y))
        ti, yi = tp['tech'].map(tid).to_numpy(), tp['year'].to_numpy(int) - y0
        Pc[ti, yi], Pq[ti, yi] = tp['pat_count'].to_numpy(), tp['pat_cites'].to_numpy()
        src_letter = np.array([t[-1] for t in techs])
        lags = sorted(sn['k'].unique())
        S = {}
        for l in lags:
            sub = sn[sn['k'] == l]
            S[l] = sps.csr_matrix((sub['S_ijk'].to_numpy(), (sub['i'].map(tid).to_numpy(), sub['j'].map(tid).to_numpy())), shape=(T, T))

        def shift(M, l):
            out = np.zeros_like(M); l = int(l)
            out[:, l:] = M[:, :Y - l]
            return out

        if what == 'recv':
            Rc, Rq = np.zeros((T, Y)), np.zeros((T, Y))
            for l in lags:
                Rc += shift(S[l] @ Pc, l); Rq += shift(S[l] @ Pq, l)
            r = []
            for arr, nm in [(Rc, 'cnt'), (Rq, 'cit')]:
                nz = np.nonzero(arr)
                r.append(pd.DataFrame({'i': np.array(techs)[nz[0]], 'year': nz[1] + y0, nm: arr[nz]}))
            r = r[0].merge(r[1], on=['i', 'year'], how='outer').fillna(0.0)
            r['cpc_subclass'] = r['i'].str[:-2]
            r['type'] = r['i'].str[-1].map({'c': 'clean', 'd': 'dirty', 'g': 'gen'})
            wide = r.pivot(index=['cpc_subclass', 'year'], columns='type', values=['cnt', 'cit']).fillna(0.0)
            wide.columns = [f'recv_{v}_{t}' for v, t in wide.columns]
            return wide.reset_index()

        inds = sorted(sh['BLS_Industry'].unique())
        iid = {b: n for n, b in enumerate(inds)}
        frames = []
        for typ in sorted(sh['type'].unique()):
            sub = sh[sh['type'] == typ]
            res = {}
            for wcol, P, nm in [('cpc_pat_share', Pc, 'cnt'), ('cpc_cite_share', Pq, 'cit')]:
                s_ = sub[sub[wcol] > 0]
                W = sps.csr_matrix((s_[wcol].to_numpy(), (s_['BLS_Industry'].map(iid).to_numpy(), s_['tech'].map(tid).to_numpy())),
                                   shape=(len(inds), T))
                for src in sorted(set(src_letter)):
                    cols = np.where(src_letter == src)[0]
                    acc = np.zeros((len(inds), Y))
                    for l in lags:
                        L = (W @ S[l])[:, cols]
                        acc += shift(np.asarray(L @ P[cols, :]), l)
                    res[(nm, src)] = acc
            for src in sorted(set(src_letter)):
                for nm in ['cnt', 'cit']:
                    arr = res[(nm, src)]
                    nz = np.nonzero(arr)
                    frames.append(pd.DataFrame({'BLS_Industry': np.array(inds)[nz[0]], 'year': nz[1] + y0,
                                                f'spill_shock_{nm}_{typ}_{src}': arr[nz]}))
        out = frames[0]
        for f_ in frames[1:]:
            out = out.merge(f_, on=['BLS_Industry', 'year'], how='outer')
        return out


    def Spillover_Shocks(self, sfx='', asinh_pat=False, diag=False):
        """""
        Knowledge-spillover shocks received by each industry and recipient type, by source type and year.
        sfx selects the files: '' (clean/dirty/general, 4-digit), '_c3' (3-digit), '_cg' (clean/general only, 4-digit),
        '_cg_c3' (clean/general only, 3-digit).

        diag=True uses the network that keeps the own-technology term (Spill_Net*_Diag.pkl) and writes KS_Shocks{sfx}_Diag.pkl.

        Output: Clean Data/KS_Shocks{sfx}.pkl
        """""
        dsfx = '_Diag' if diag else ''
        if '_mg' in sfx:   # main groups: sparse matrices
            KS_df = self._sparse_spill(sfx, asinh_pat, 'shocks', diag)
            KS_df.to_pickle(f"{self.Directory}/Clean Data/KS_Shocks{sfx}{dsfx}{'_asinh' if asinh_pat else ''}.pkl")
            return KS_df
        spill_net_df = pd.read_pickle(f'{self.Directory}/Clean Data/Spill_Net{sfx}{dsfx}.pkl')
        tech_pat_df = pd.read_pickle(f'{self.Directory}/Clean Data/Tech_Pat{sfx}.pkl')
        if asinh_pat:
            tech_pat_df[['pat_count', 'pat_cites']] = np.arcsinh(tech_pat_df[['pat_count', 'pat_cites']])
        ind_pat_shares_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_Pat_Shares_Pre{sfx}.pkl')
        ind_pat_shares_df = ind_pat_shares_df[ind_pat_shares_df['type'] != 'all']
        
        ind_pat_shares_df['tech'] = ind_pat_shares_df['cpc_subclass'] + '_' + ind_pat_shares_df['type'].str[:1]
        
        KS_df = pd.merge(spill_net_df,
                        ind_pat_shares_df.rename(columns={'tech': 'i'}), 
                        on='i', how='inner')
        
        KS_df['BLS_tech_loading_cnt'] = KS_df['cpc_pat_share'] * KS_df['S_ijk']
        KS_df['BLS_tech_loading_cit'] = KS_df['cpc_cite_share'] * KS_df['S_ijk']
                
        KS_df['total_BLS_tech_loading_cnt'] = KS_df.groupby(['BLS_Industry', 'type', 'j', 'k'])['BLS_tech_loading_cnt'].transform('sum')
        KS_df['total_BLS_tech_loading_cit'] = KS_df.groupby(['BLS_Industry', 'type', 'j', 'k'])['BLS_tech_loading_cit'].transform('sum')
        #Diagonal already removed
        KS_df = KS_df[['BLS_Industry', 'type', 'j', 'k', 'total_BLS_tech_loading_cnt', 'total_BLS_tech_loading_cit']].drop_duplicates()
        
        KS_df = KS_df.merge(tech_pat_df.rename(columns={'tech': 'j', 'year': 'spill_send_year'}), 
                            on='j', how='inner')
        
        KS_df['year'] = KS_df['spill_send_year'] + KS_df['k']
        KS_df['src_type'] = KS_df['j'].str[-1]
        
        KS_df['BLS_tech_receipt_cnt'] = KS_df['total_BLS_tech_loading_cnt'] * KS_df['pat_count']
        KS_df['BLS_tech_receipt_cit'] = KS_df['total_BLS_tech_loading_cit'] * KS_df['pat_cites']
        
        KS_df['spill_shock_cnt'] = KS_df.groupby(['BLS_Industry', 'type', 'src_type', 'year'])['BLS_tech_receipt_cnt'].transform('sum')
        KS_df['spill_shock_cit'] = KS_df.groupby(['BLS_Industry', 'type', 'src_type', 'year'])['BLS_tech_receipt_cit'].transform('sum')
        
        KS_df = KS_df[['BLS_Industry', 'type', 'src_type', 'year', 'spill_shock_cnt', 'spill_shock_cit']].drop_duplicates()
        
        KS_wide = KS_df.pivot(index=['BLS_Industry', 'year'], columns=['type', 'src_type'], 
                       values=['spill_shock_cnt', 'spill_shock_cit'])

        KS_wide.columns = [f'{v}_{t}_{s}' for v, t, s in KS_wide.columns]
        KS_df = KS_wide.reset_index()
        
        KS_df.to_pickle(f"{self.Directory}/Clean Data/KS_Shocks{sfx}{dsfx}{'_asinh' if asinh_pat else ''}.pkl")
        return KS_df


    def KS_CleanAgg(self, sfx='_cg', asinh_pat=False):
        """""
        Spillovers received by all clean patents (one aggregate recipient), by year:
        R_y = sum_k sum_j S_clean,j,k * Pat_j,y-k, with S from the clean-citing-patent network of the pre-sample window.

        Output: Clean Data/KS_CleanAgg{sfx}.pkl  (year, recv_cnt, recv_cit)
        """""
        D = f'{self.Directory}/Clean Data/'
        sn = pd.read_pickle(D + f'Spill_Clean_Net{sfx}.pkl')
        tp = pd.read_pickle(D + f'Tech_Pat{sfx}.pkl').rename(columns={'tech': 'j', 'year': 'send_year'})
        if asinh_pat:
            tp[['pat_count', 'pat_cites']] = np.arcsinh(tp[['pat_count', 'pat_cites']])
        m = sn.merge(tp, on='j', how='inner')
        m['year'] = m['send_year'] + m['k'].astype(int)
        m['recv_cnt'] = m['S_jk'] * m['pat_count']
        m['recv_cit'] = m['S_jk'] * m['pat_cites']
        out = m.groupby('year', as_index=False)[['recv_cnt', 'recv_cit']].sum()
        out.to_pickle(D + f"KS_CleanAgg{sfx}{'_asinh' if asinh_pat else ''}.pkl")
        return out


    def KS_Recv(self, sfx='', asinh_pat=False, diag=False):
        """""
        Spillovers received by recipient technology class, type and year: for each recipient (class, type) the
        citation-share-weighted patenting of its source technologies, recv(i, y) = sum_j S_ij,k * Pat_j,y-k.
        The class-level ratio recv(clean) / recv(clean + dirty + gen) is formed in UDG_Instrument.

        Output: Clean Data/KS_Recv.pkl  (columns cpc_subclass, year, recv_{cnt,cit}_{clean,dirty,gen})
        """""
        D = f'{self.Directory}/Clean Data/'
        dsfx = '_Diag' if diag else ''
        if '_mg' in sfx:   # main groups: sparse matrices
            wide = self._sparse_spill(sfx, asinh_pat, 'recv', diag)
            wide.to_pickle(D + f"KS_Recv{sfx}{dsfx}{'_asinh' if asinh_pat else ''}.pkl")
            return wide
        sn = pd.read_pickle(D + f'Spill_Net{sfx}{dsfx}.pkl')
        tp = pd.read_pickle(D + f'Tech_Pat{sfx}.pkl').rename(columns={'tech': 'j', 'year': 'send_year'})
        if asinh_pat:   # source patenting as asinh(patents), so that large technologies do not dominate
            tp[['pat_count', 'pat_cites']] = np.arcsinh(tp[['pat_count', 'pat_cites']])
        parts = []
        for k, sub in sn.groupby('k'):
            m = sub[['i', 'j', 'S_ijk']].merge(tp, on='j', how='inner')
            m['year'] = m['send_year'] + int(k)
            m['cnt'] = m['S_ijk'] * m['pat_count']
            m['cit'] = m['S_ijk'] * m['pat_cites']
            parts.append(m.groupby(['i', 'year'], as_index=False)[['cnt', 'cit']].sum())
            del m
        r = pd.concat(parts, ignore_index=True).groupby(['i', 'year'], as_index=False)[['cnt', 'cit']].sum()
        r['cpc_subclass'] = r['i'].str[:-2]
        r['type'] = r['i'].str[-1].map({'c': 'clean', 'd': 'dirty', 'g': 'gen'})
        wide = r.pivot(index=['cpc_subclass', 'year'], columns='type', values=['cnt', 'cit']).fillna(0.0)
        wide.columns = [f'recv_{v}_{t}' for v, t in wide.columns]
        wide = wide.reset_index()
        wide.to_pickle(D + f"KS_Recv{sfx}{dsfx}{'_asinh' if asinh_pat else ''}.pkl")
        return wide


    def UDG_OwnFirstStage(self, batch, BLS_year_start, Year_end, srcs=('gov', 'ks', 'kswt', 'rd')):
        """""
        Diagnostic first stages, before and after the network aggregation. For each instrument source and measure:
        OLS of the lagged own-sector clean share on the lagged own-sector instrument, and of the lagged network (net)
        clean share on the lagged network instrument, with sector and period effects and sector-clustered SEs.
        The sample is the G-baseline estimation sample (positive non-clean offset, sectors with a clean outcome), so the
        network rows are the first stages shown in the instrumented batches.

        Output: Results/Batches/<batch>/own_first_stage.csv, table_own_first_stage.tex
        """""
        outdir = self.Directory / 'Results' / 'Batches' / batch
        outdir.mkdir(parents=True, exist_ok=True)
        reg_df, _ = self.UDG_Panel(BLS_year_start, Year_end, 5, ivs=tuple(srcs))
        rows = []
        for meas, suf in [('pat', 'count'), ('cite', 'cites')]:
            for src in srcs:
                for lvl in ['own', 'net']:
                    y, z = f'{lvl}_G_{meas}_lag', f'{lvl}_Z{src}_{meas}_lag'
                    fs, n = self.UDG_FirstStage(reg_df, [y], [z], [], f'clean_pat_{suf}', f'pat_{suf}_nc')
                    r = fs[lvl]
                    rows.append(dict(measure=meas, source=src, level=lvl, coef=r['coef'], se=r['se'], p=r['p'],
                                     F=r['F'], pF=r['pF'], pr2=r['pr2'], N=n))
        out = pd.DataFrame(rows)
        out.to_csv(outdir / 'own_first_stage.csv', index=False)
        names = {'gov': 'Government patents', 'ks': 'Spillovers, all sources', 'kswt': 'Spillovers, within type',
                 'rd': 'State R\\&D prices', 'rdcf': 'R\\&D prices, clean firms', 'govcl': 'Gov. clean patenting (aggregate)', 'ksagg': 'Clean spillovers (aggregate)', 'govr': 'Gov. clean / gov. all ratio', 'govrn': 'Gov. ratio, renormalised', 'ksr': 'Spillover clean / all ratio', 'ksrn': 'Spillover ratio, renormalised', 'gov_c3': 'Government patents (3-digit)', 'ks_c3': 'Spillovers (3-digit)'}
        lines = [r'\begin{tabular}{llcccccc}', r'\toprule',
                 r' & & \multicolumn{3}{c}{Patent counts} & \multicolumn{3}{c}{Citations} \\',
                 r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
                 'Instrument & Level & Coefficient (SE) & $F$ & Partial $R^2$' + ' & Coefficient (SE) & $F$ & Partial $R^2$' + r' \\', r'\midrule']
        for src in srcs:
            for lvl, lname in [('own', 'Own sector'), ('net', 'Network (net)')]:
                cells = []
                for meas in ['pat', 'cite']:
                    r = out[(out['measure'] == meas) & (out['source'] == src) & (out['level'] == lvl)].iloc[0]
                    cells += [f"{r.coef:.3f}{gpf.get_stars(r.p)} ({r.se:.3f})", f"{r.F:.1f}{gpf.get_stars(r.pF)}", f"{r.pr2:.3f}"]
                lines.append((names.get(src, src) if lvl == 'own' else '') + f' & {lname} & ' + ' & '.join(cells) + r' \\')
            lines.append(r'\midrule')
        lines[-1] = r'\bottomrule'; lines.append(r'\end{tabular}')
        with open(outdir / 'table_own_first_stage.tex', 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(out.round(3).to_string())
        return out


    def Shock_Relevance(self, batch, BLS_year_start, Year_end, bin_len=5, last_outcome=2017, sfx='', cur=False):
        """""
        Does the industry-type shock predict patenting of that type in that industry, before any shares or network?
        Level regressions for (industry i, type k, bin t): PPML of the patents of type k in industry i and bin t on
        asinh of the shock Z^k_it (same bin, or lagged one bin), with industry-by-type and type-by-bin effects, standard
        errors clustered by industry. Government shock: Z^k = sum over classes of the industry's pre-sample type-k class
        shares times government patenting (types clean / dirty / general). Spillover shock: spillovers received by the
        industry's type-k technologies from all source types (types clean / general). Placebo: the shock of the other type.

        cur=True replaces the industry's pre-sample class distribution by its distribution in the year of the shock (Ind_Pat_Shares_Cur*.pkl),
        which is mechanically related to current patenting and is shown only to see whether the correlations are sensible.

        Output: Results/Batches/<batch>/shock_relevance.csv, table_shock_relevance.tex
        """""
        outdir = self.Directory / 'Results' / 'Batches' / batch
        outdir.mkdir(parents=True, exist_ok=True)
        D = f'{self.Directory}/Clean Data/'
        pf = pd.read_pickle(D + 'Pat_Firms.pkl')[['patent_id', 'year', 'BLS_Industry', 'clean', 'dirty']].drop_duplicates()
        pf['w'] = 1.0 / pf.groupby('patent_id')['BLS_Industry'].transform('count')
        pf['n_clean'] = pf['w'] * pf['clean']
        pf['n_dirty'] = pf['w'] * pf['dirty']
        pf['n_gen3'] = pf['w'] * (1 - pf['clean']) * (1 - pf['dirty'])
        pf['n_gen2'] = pf['w'] * (1 - pf['clean'])
        ends = list(range(BLS_year_start, Year_end + 1, bin_len))
        inds = sorted(pd.read_pickle(D + 'Ind_Pat.pkl')['BLS_Industry'].unique())
        inds = [i for i in inds if self.manu_cols[0] <= i <= self.manu_cols[1]]

        def binned(df, cols, year='year'):
            out = []
            for e in ends:
                w = df[(df[year] > e - bin_len) & (df[year] <= e)]
                g = w.groupby('BLS_Industry')[cols].sum().reindex(inds).fillna(0.0)
                g['period'] = e
                out.append(g.reset_index())
            return pd.concat(out, ignore_index=True)

        pat = binned(pf, ['n_clean', 'n_dirty', 'n_gen3', 'n_gen2'])
        if cur:   # contemporaneous class distribution of the industry's type-k patents
            shc = pd.read_pickle(D + f'Ind_Pat_Shares_Cur{sfx}.pkl')
            g_ = pd.read_pickle(D + f'Gov_CPC{sfx}.pkl')[['cpc_subclass', 'year', 'gov_pat_count']]
            m_ = shc[shc['type'].isin(['clean', 'dirty', 'gen'])].merge(g_, on=['cpc_subclass', 'year'])
            m_['v'] = m_['cpc_pat_share'] * m_['gov_pat_count']
            gov = m_.groupby(['BLS_Industry', 'year', 'type'])['v'].sum().unstack('type').fillna(0.0)
            gov.columns = [f'pat_govt_shock_{c}' for c in gov.columns]
            gov = gov.reset_index()
            shg = pd.read_pickle(D + f'Ind_Pat_Shares_Cur_cg{sfx}.pkl')
            kr_ = pd.read_pickle(D + f'KS_Recv_cg{sfx}.pkl')
            ks = None
            for t_, c_ in [('clean', 'z_clean'), ('gen', 'z_gen')]:
                m_ = shg[shg['type'] == t_].merge(kr_[['cpc_subclass', 'year', f'recv_cnt_{t_}']], on=['cpc_subclass', 'year'])
                m_[c_] = m_['cpc_pat_share'] * m_[f'recv_cnt_{t_}']
                z_ = m_.groupby(['BLS_Industry', 'year'], as_index=False)[c_].sum()
                ks = z_ if ks is None else ks.merge(z_, on=['BLS_Industry', 'year'], how='outer')
            ks = ks.fillna(0.0)
            del shc, shg, m_
        else:
            gov = pd.read_pickle(D + f'Govt_Shocks{sfx}.pkl')
            ks = pd.read_pickle(D + f'KS_Shocks_cg{sfx}.pkl')
            ks['z_clean'] = ks[[c for c in ks.columns if c.startswith('spill_shock_cnt_clean_')]].fillna(0.0).sum(axis=1)
            ks['z_gen'] = ks[[c for c in ks.columns if c.startswith('spill_shock_cnt_gen_')]].fillna(0.0).sum(axis=1)
        gov = binned(gov, [c for c in gov.columns if c.startswith('pat_govt_shock_')])
        ks = binned(ks, ['z_clean', 'z_gen'])

        def long_panel(types, ycols, zcols):
            frames = []
            for k, yc, zc in zip(types, ycols, zcols):
                d = pat[['BLS_Industry', 'period', yc]].rename(columns={yc: 'y'}).merge(
                    zcols_df[['BLS_Industry', 'period', zc]].rename(columns={zc: 'z'}), on=['BLS_Industry', 'period'])
                d['type'] = k
                frames.append(d)
            return pd.concat(frames, ignore_index=True)

        def lag(d, c):
            l = d[['BLS_Industry', 'period', 'type', c]].copy()
            l['period'] += bin_len
            return d.merge(l.rename(columns={c: c + '_lag'}), on=['BLS_Industry', 'period', 'type'], how='left')

        def fit(d, zc, types):
            d = d[d['type'].isin(types)].dropna(subset=['y', zc]).copy()
            d = d[d['period'] <= last_outcome] if last_outcome else d
            d['grp'] = d['BLS_Industry'].astype(str) + '_' + d['type']
            d = d[d.groupby('grp')['y'].transform('sum') > 0]
            d['tp'] = d['type'] + '_' + d['period'].astype(str)
            x = np.arcsinh(d[zc].astype(float))
            X = pd.concat([pd.Series(1.0, index=d.index, name='const'), x.rename('z'),
                           pd.get_dummies(d['grp'], prefix='g', drop_first=True, dtype=float),
                           pd.get_dummies(d['tp'], prefix='t', drop_first=True, dtype=float)], axis=1)
            X.columns = [str(c) for c in X.columns]
            y = d['y'].astype(float)
            fam = sm.families.Poisson()
            try:
                m = sm.GLM(y, X, family=fam).fit(cov_type='cluster', cov_kwds={'groups': d['BLS_Industry'].to_numpy()}, maxiter=200)
            except ValueError:   # IRLS diverged: start from a constant-only fit with the log of the mean
                start = np.zeros(X.shape[1]); start[0] = np.log(y.mean())
                m = sm.GLM(y, X, family=fam).fit(start_params=start, method='bfgs', maxiter=2000,
                                                 cov_type='cluster', cov_kwds={'groups': d['BLS_Industry'].to_numpy()})
            return m.params['z'], m.bse['z'], m.pvalues['z'], int(m.nobs), d['BLS_Industry'].nunique()

        rows = []
        for name, types, ycols, zdf, zcols, plc in [
                ('Government patents', ['clean', 'dirty', 'general'], ['n_clean', 'n_dirty', 'n_gen3'], gov,
                 ['pat_govt_shock_clean', 'pat_govt_shock_dirty', 'pat_govt_shock_gen'], {'clean': 'general', 'dirty': 'clean', 'general': 'dirty'}),
                ('Spillovers received', ['clean', 'general'], ['n_clean', 'n_gen2'], ks, ['z_clean', 'z_gen'], {'clean': 'general', 'general': 'clean'})]:
            zcols_df = zdf
            panel = long_panel(types, ycols, zcols)
            panel = lag(panel, 'z')
            # placebo shock: shock of another type for the same industry and bin
            zmap = dict(zip(types, zcols))
            oth = zdf.set_index(['BLS_Industry', 'period'])
            panel['z_other'] = [oth.loc[(i, t), zmap[plc[k]]] if (i, t) in oth.index else np.nan
                                for i, t, k in zip(panel['BLS_Industry'], panel['period'], panel['type'])]
            panel = lag(panel, 'z_other')
            for lab, ty, zc in [('All types, same bin', types, 'z'), ('All types, previous bin', types, 'z_lag'),
                                ('All types, shock of another type (placebo)', types, 'z_other'),
                                ('All types, previous-bin shock of another type (placebo)', types, 'z_other_lag')] + \
                               [(f'{t.capitalize()} only, same bin', [t], 'z') for t in types] + \
                               [(f'{t.capitalize()} only, previous bin', [t], 'z_lag') for t in types]:
                b, se, p, n, ns = fit(panel, zc, ty)
                rows.append(dict(shock=name, spec=lab, coef=b, se=se, p=p, N=n, n_industries=ns))
        out = pd.DataFrame(rows)
        out.to_csv(outdir / 'shock_relevance.csv', index=False)
        labs = list(dict.fromkeys(out['spec']))
        shocks = list(dict.fromkeys(out['shock']))
        lines = [r'\begin{tabular}{l' + 'c' * len(shocks) + '}', r'\toprule', ' & ' + ' & '.join(shocks) + r' \\', r'\midrule']
        for lab in labs:
            cs, ss = [], []
            for sh in shocks:
                r = out[(out['shock'] == sh) & (out['spec'] == lab)]
                if len(r):
                    r = r.iloc[0]; cs.append(f"{r.coef:.3f}{gpf.get_stars(r.p)}"); ss.append(f"({r.se:.3f})")
                else:
                    cs.append(''); ss.append('')
            lines += [lab + ' & ' + ' & '.join(cs) + r' \\', ' & ' + ' & '.join(ss) + r' \\']
        lines += [r'\midrule', 'Observations (first row) & ' + ' & '.join(str(int(out[(out['shock'] == sh)].iloc[0].N)) for sh in shocks) + r' \\',
                  r'\bottomrule', r'\end{tabular}']
        with open(outdir / 'table_shock_relevance.tex', 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(out.round(3).to_string())
        return out


    def Shock_Relevance_Type(self, batch, BLS_year_start, bin_len=5, first=1993, last=2017, sfx=''):
        """""
        Gut check at the level of the patent type (no industries): does the national type-k shock predict national
        patenting of type k? Z^k_y = sum over classes s of w^k_s * X_{s,y}, with X the government patenting of class s in
        year y, or the spillovers received by the type-k technologies of class s (all sources), and w^k_s the national
        distribution of type-k patents over classes, either from the pre-sample window (1983-1992) or from the same year
        (contemporaneous, which is mechanically biased upward). The outcome is the number of patents of type k matched to
        firms, summed over industries. PPML with type effects, annual data, standard errors clustered by year.
        Government shock: types clean / dirty / general; spillovers: clean / general.

        The shock is also entered lagged by one bin (bin_len years), as in the main specifications.

        Output: Results/Batches/<batch>/shock_relevance_type.csv, table_shock_relevance_type.tex
        """""
        outdir = self.Directory / 'Results' / 'Batches' / batch
        outdir.mkdir(parents=True, exist_ok=True)
        D = f'{self.Directory}/Clean Data/'
        pf = pd.read_pickle(D + 'Pat_Firms.pkl')[['patent_id', 'year', 'BLS_Industry', 'clean', 'dirty']].drop_duplicates()
        pf['w'] = 1.0 / pf.groupby('patent_id')['BLS_Industry'].transform('count')
        pf['clean_n'], pf['dirty_n'] = pf['w'] * pf['clean'], pf['w'] * pf['dirty']
        pf['gen3_n'], pf['gen2_n'] = pf['w'] * (1 - pf['clean']) * (1 - pf['dirty']), pf['w'] * (1 - pf['clean'])
        out_n = pf.groupby('year')[['clean_n', 'dirty_n', 'gen3_n', 'gen2_n']].sum()
        years = list(range(first, last + 1))
        pre = (BLS_year_start - bin_len - 10 + 1, BLS_year_start - bin_len)

        def tech_matrix(tp):
            "classes x years matrix of patents per type: dict letter -> DataFrame(class x year)"
            tp = tp.assign(cls=tp['tech'].str[:-2], let=tp['tech'].str[-1])
            return {l: g.pivot_table(index='cls', columns='year', values='pat_count', aggfunc='sum').fillna(0.0) for l, g in tp.groupby('let')}

        def shock(P, X, contemporaneous):
            "P: class x year patents of the type; X: class x year shock input; returns Series over years"
            cls = P.index.intersection(X.index)
            P, X = P.loc[cls], X.reindex(cls).fillna(0.0)
            if contemporaneous:
                share = P / P.sum(axis=0).where(P.sum(axis=0) > 0)
                return (share.fillna(0.0) * X.reindex(columns=P.columns).fillna(0.0)).sum(axis=0)
            w = P.loc[:, pre[0]:pre[1]].sum(axis=1)
            w = w / w.sum()
            return (X.reindex(columns=P.columns).fillna(0.0).mul(w, axis=0)).sum(axis=0)

        gov = pd.read_pickle(D + f'Gov_CPC{sfx}.pkl').pivot_table(index='cpc_subclass', columns='year', values='gov_pat_count', aggfunc='sum').fillna(0.0)
        rv = pd.read_pickle(D + f'KS_Recv_cg{sfx}.pkl')
        rec = {l: rv.pivot_table(index='cpc_subclass', columns='year', values=f'recv_cnt_{t}', aggfunc='sum').fillna(0.0)
               for l, t in [('c', 'clean'), ('g', 'gen')]}
        tp2 = tech_matrix(pd.read_pickle(D + f'Tech_Pat_cg{sfx}.pkl'))
        if '_mg' in sfx:   # six-digit CPC: only the clean / general types exist, also for the government shock
            gov_types = {'c': ('clean_n', tp2['c'], gov), 'g': ('gen2_n', tp2['g'], gov)}
        else:
            tp3 = tech_matrix(pd.read_pickle(D + f'Tech_Pat{sfx}.pkl'))
            gov_types = {'c': ('clean_n', tp3['c'], gov), 'd': ('dirty_n', tp3['d'], gov), 'g': ('gen3_n', tp3['g'], gov)}
        specs = [('Government patents', gov_types),
                 ('Spillovers received', {'c': ('clean_n', tp2['c'], rec['c']), 'g': ('gen2_n', tp2['g'], rec['g'])})]
        rows = []
        for name, types in specs:
            for dist, lag in [(a, b) for b in [0, bin_len] for a in ['pre-sample', 'contemporaneous']]:
                timing = 'same year' if lag == 0 else 'previous bin'
                panel = []
                for l, (ycol, P, X) in types.items():
                    z = shock(P, X, dist == 'contemporaneous')
                    d = pd.DataFrame({'year': years, 'type': l, 'y': out_n.loc[years, ycol].to_numpy(),
                                      'z': np.arcsinh(z.reindex([y - lag for y in years]).to_numpy())})
                    panel.append(d)
                d = pd.concat(panel, ignore_index=True)
                for fe in ['type effects', 'type and year effects', 'type effects and type-specific trends']:
                    parts = [pd.Series(1.0, index=d.index, name='const'), d['z'].rename('z'), pd.get_dummies(d['type'], prefix='k', drop_first=True, dtype=float)]
                    if fe == 'type and year effects':
                        parts.append(pd.get_dummies(d['year'], prefix='y', drop_first=True, dtype=float))
                    if fe == 'type effects and type-specific trends':
                        tr = pd.concat([((d['type'] == l) * (d['year'] - first)).rename(f't_{l}') for l in types], axis=1)
                        parts.append(tr)
                    X_ = pd.concat(parts, axis=1)
                    m = sm.GLM(d['y'], X_, family=sm.families.Poisson()).fit(cov_type='cluster', cov_kwds={'groups': d['year'].to_numpy()})
                    rows.append(dict(shock=name, distribution=dist, timing=timing, fe=fe, coef=m.params['z'], se=m.bse['z'], p=m.pvalues['z'], N=int(m.nobs)))
        out = pd.DataFrame(rows)
        out.to_csv(outdir / 'shock_relevance_type.csv', index=False)
        cols = list(dict.fromkeys(zip(out['shock'], out['distribution'], out['timing'])))
        lines = [r'\begin{tabular}{l' + 'c' * len(cols) + '}', r'\toprule',
                 ' & ' + ' & '.join(f'\\multicolumn{{1}}{{c}}{{{a}}}' for a, _, _ in cols) + r' \\',
                 ' & ' + ' & '.join(b for _, b, _ in cols) + r' \\',
                 ' & ' + ' & '.join(t for _, _, t in cols) + r' \\', r'\midrule']
        for fe in list(dict.fromkeys(out['fe'])):
            cs, ss = [], []
            for a, b, t in cols:
                r = out[(out['shock'] == a) & (out['distribution'] == b) & (out['timing'] == t) & (out['fe'] == fe)].iloc[0]
                cs.append(f"{r.coef:.3f}{gpf.get_stars(r.p)}"); ss.append(f"({r.se:.3f})")
            lines += [fe.capitalize() + ' & ' + ' & '.join(cs) + r' \\', ' & ' + ' & '.join(ss) + r' \\']
        lines += [r'\midrule', 'Observations (first row) & ' + ' & '.join(str(int(out[(out['shock'] == a) & (out['distribution'] == b) & (out['timing'] == t)].iloc[0].N)) for a, b, t in cols) + r' \\',
                  r'\bottomrule', r'\end{tabular}']
        with open(outdir / 'table_shock_relevance_type.tex', 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(out.round(3).to_string())
        return out


    def New_IV_Variants(self, path, kind):
        """""
        Summary of an instrument variant against its baseline (first-stage F of the net instrument, second-stage estimate).
        kind='diag': spillover instrument with the own-technology term (Batch 4, 4-digit and main groups);
        kind='type': government instrument with type-specific series (Batch 2).
        """""
        reg = pd.read_csv(self.Directory / 'Results' / 'registry.csv').set_index('spec_id')
        if kind == 'diag':
            b, rows = 'b4', [('4-digit CPC, term removed (baseline)', 'base'), ('4-digit CPC, term kept', 'diag'),
                             ('Six-digit CPC, term removed', 'mg'), ('Six-digit CPC, term kept', 'diagmg')]
        elif kind == 'direct':
            rows = [('Government patents, 4-digit CPC', 'b2', 'base'), ('Government patents, 4-digit CPC, direct network', 'b2', 'ds'),
                    ('Government patents, six-digit CPC', 'b2', 'mg'), ('Government patents, six-digit CPC, direct network', 'b2', 'dsmg'),
                    ('Spillovers, 4-digit CPC', 'b4', 'base'), ('Spillovers, 4-digit CPC, direct network', 'b4', 'ds'),
                    ('Spillovers, six-digit CPC', 'b4', 'mg'), ('Spillovers, six-digit CPC, direct network', 'b4', 'dsmg')]
            b = None
        elif kind == 'cur':
            rows = [('Government patents, 4-digit CPC, pre-sample distribution', 'b2', 'base'), ('Government patents, 4-digit CPC, contemporaneous distribution', 'b9', 'gov'),
                    ('Government patents, six-digit CPC, pre-sample distribution', 'b2', 'mg'), ('Government patents, six-digit CPC, contemporaneous distribution', 'b9', 'govmg'),
                    ('Spillovers, 4-digit CPC, pre-sample distribution', 'b4', 'base'), ('Spillovers, 4-digit CPC, contemporaneous distribution', 'b9', 'ks'),
                    ('Spillovers, six-digit CPC, pre-sample distribution', 'b4', 'mg'), ('Spillovers, six-digit CPC, contemporaneous distribution', 'b9', 'ksmg')]
            b = None
        elif kind == 'curds':
            rows = [('Government patents, 4-digit CPC, pre-sample distribution', 'b2', 'ds'), ('Government patents, 4-digit CPC, contemporaneous distribution', 'b9', 'govds'),
                    ('Government patents, six-digit CPC, pre-sample distribution', 'b2', 'dsmg'), ('Government patents, six-digit CPC, contemporaneous distribution', 'b9', 'govmgds'),
                    ('Spillovers, 4-digit CPC, pre-sample distribution', 'b4', 'ds'), ('Spillovers, 4-digit CPC, contemporaneous distribution', 'b9', 'ksds'),
                    ('Spillovers, six-digit CPC, pre-sample distribution', 'b4', 'dsmg'), ('Spillovers, six-digit CPC, contemporaneous distribution', 'b9', 'ksmgds')]
            b = None
        elif kind == 'gov6':
            b, rows = 'b2', [('4-digit CPC, baseline', 'base'), ('4-digit CPC, instrument lagged two bins', 'lag2'),
                             ('Six-digit CPC, baseline', 'mg'), ('Six-digit CPC, instrument lagged two bins', 'mglag2')]
        else:
            b, rows = 'b2', [('4-digit CPC, all types (baseline)', 'base'), ('4-digit CPC, recipient type', 'typ'),
                             ('Six-digit CPC, all types', 'mg'), ('Six-digit CPC, recipient type', 'typmg')]
        def cell(i):
            if i not in reg.index:
                return '', ''
            r = reg.loc[i]
            if not r.converged or pd.isna(r.coef_net):
                return f"{r.fs_F_net:.1f}", 'not conv.'
            return f"{r.fs_F_net:.1f}", f"{r.coef_net:.1f}{gpf.get_stars(r.p_net)} ({r.se_net:.1f})"
        lines = [r'\begin{tabular}{lcccc}', r'\toprule', r' & \multicolumn{2}{c}{Patent counts} & \multicolumn{2}{c}{Citations} \\',
                 r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}', r' & First-stage $F$ & Estimate (SE) & First-stage $F$ & Estimate (SE) \\', r'\midrule']
        for r_ in rows:
            lab, s_ = r_[0], r_[-1]
            bb = r_[1] if kind in ('direct', 'cur', 'curds') else b
            c1, c2 = cell(f'{bb}_pat_{s_}'), cell(f'{bb}_cite_{s_}')
            lines.append(f'{lab} & {c1[0]} & {c1[1]} & {c2[0]} & {c2[1]} \\\\')
        lines += [r'\bottomrule', r'\end{tabular}']
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')


    def Instruments_Cur(self, sfx=''):
        """""
        Government and spillover shocks built with the industry's contemporaneous class distribution (the distribution of its type-k
        patents in the year of the shock) instead of the pre-sample distribution: Z^k_{iy} = sum_s w^k_{is,y} X^k_{s,y}. This is
        mechanically related to current patenting and is used only to see whether the shocks predict the clean share when the weights
        move with the industry's own patenting. Counts only (the contemporaneous citation shares are not saved).
        sfx: '' = 4-digit subclasses (government: clean / dirty / general types), '_mg' = six-digit CPC (clean / general types).

        Output: Clean Data/Govt_Shocks{sfx}_Cur.pkl, Clean Data/KS_Shocks_cg{sfx}_Cur.pkl
        """""
        D = f'{self.Directory}/Clean Data/'
        gov = pd.read_pickle(D + f'Gov_CPC{sfx}.pkl')[['cpc_subclass', 'year', 'gov_pat_count', 'gov_pat_cites']]
        sh = pd.read_pickle(D + ('Ind_Pat_Shares_Cur.pkl' if not sfx else f'Ind_Pat_Shares_Cur_cg{sfx}.pkl'))
        has_cite = 'cpc_cite_share' in sh.columns   # citation shares need the Cleaner version that saves them
        sh = sh[sh['type'].isin(['clean', 'gen'])].merge(gov, on=['cpc_subclass', 'year'])
        sh['v'] = sh['cpc_pat_share'] * sh['gov_pat_count']
        sh['vc'] = sh['cpc_cite_share'] * sh['gov_pat_cites'] if has_cite else np.nan
        g = sh.groupby(['BLS_Industry', 'year', 'type'])[['v', 'vc']].sum(min_count=1).unstack('type')
        g.columns = [('pat' if a == 'v' else 'cite') + f'_govt_shock_{t}' for a, t in g.columns]
        g.reset_index().to_pickle(D + f'Govt_Shocks{sfx}_Cur.pkl')
        shg = pd.read_pickle(D + f'Ind_Pat_Shares_Cur_cg{sfx}.pkl')
        rv = pd.read_pickle(D + f'KS_Recv_cg{sfx}.pkl')
        ks = None
        for t_ in ['clean', 'gen']:
            m_ = shg[shg['type'] == t_].merge(rv[['cpc_subclass', 'year', f'recv_cnt_{t_}', f'recv_cit_{t_}']], on=['cpc_subclass', 'year'])
            m_['z'] = m_['cpc_pat_share'] * m_[f'recv_cnt_{t_}']
            m_['zc'] = m_['cpc_cite_share'] * m_[f'recv_cit_{t_}'] if 'cpc_cite_share' in m_.columns else np.nan
            z_ = (m_.groupby(['BLS_Industry', 'year'])[['z', 'zc']].sum(min_count=1).reset_index()
                    .rename(columns={'z': f'spill_shock_cnt_{t_}_all', 'zc': f'spill_shock_cit_{t_}_all'}))
            ks = z_ if ks is None else ks.merge(z_, on=['BLS_Industry', 'year'], how='outer')
        ks.to_pickle(D + f'KS_Shocks_cg{sfx}_Cur.pkl')


    def Shock_Relevance_Class(self, batch, BLS_year_start, bin_len=5, first=1993, last=2017, sfx=''):
        """""
        Type-level gut check with a finer type: the type is the 3-digit CPC class (about a hundred types instead of clean / general).
        For a class c, Z^c_y = sum over the units s of c (4-digit subclasses, or six-digit CPC codes with sfx='_mg') of w^c_s * X_{s,y},
        with X the government patenting of unit s in year y, or the spillovers received by unit s (clean and general recipients added),
        and w^c_s the national distribution of the class's patents over its units, either from the pre-sample window or from the same year
        (contemporaneous, biased upward). The outcome is the number of patents of class c matched to firms (summed over industries).
        PPML with class effects; annual data; standard errors clustered by class. The government shock enters from one bin earlier,
        the spillover shock from the same year (as in Shock_Relevance_Type).

        Output: Results/Batches/<batch>/shock_relevance_class.csv, table_shock_relevance_class.tex
        """""
        outdir = self.Directory / 'Results' / 'Batches' / batch
        outdir.mkdir(parents=True, exist_ok=True)
        D = f'{self.Directory}/Clean Data/'
        cur = pd.read_pickle(D + 'Ind_Pat_Shares_Cur_c3.pkl')
        cur = cur[cur['type'] == 'all']
        out_n = cur.groupby(['cpc_subclass', 'year'])['cpc_pat_count'].sum().unstack('year').fillna(0.0)
        tp = pd.read_pickle(D + f'Tech_Pat_cg{sfx}.pkl')
        tp = tp.assign(unit=tp['tech'].str[:-2])
        P = tp.groupby(['unit', 'year'])['pat_count'].sum().unstack('year').fillna(0.0)
        cls = pd.Series([u[:3] for u in P.index], index=P.index)
        gov = pd.read_pickle(D + f'Gov_CPC{sfx}.pkl').pivot_table(index='cpc_subclass', columns='year', values='gov_pat_count', aggfunc='sum').fillna(0.0)
        rv = pd.read_pickle(D + f'KS_Recv_cg{sfx}.pkl')
        rv = rv.assign(recv=rv['recv_cnt_clean'] + rv['recv_cnt_gen'])
        rec = rv.pivot_table(index='cpc_subclass', columns='year', values='recv', aggfunc='sum').fillna(0.0)
        pre = (BLS_year_start - bin_len - 10 + 1, BLS_year_start - bin_len)
        years = list(range(first, last + 1))

        def shock(X, contemporaneous):
            Xa = X.reindex(index=P.index, columns=P.columns).fillna(0.0)
            if contemporaneous:
                tot = P.groupby(cls).transform('sum')
                W = (P / tot.where(tot > 0)).fillna(0.0)
            else:
                wpre = P.loc[:, pre[0]:pre[1]].sum(axis=1)
                tot = wpre.groupby(cls).transform('sum')
                w = (wpre / tot.where(tot > 0)).fillna(0.0)
                W = pd.DataFrame(np.outer(w.to_numpy(), np.ones(P.shape[1])), index=P.index, columns=P.columns)
            return (W * Xa).groupby(cls).sum()

        rows = []
        for name, X, lag_ in [('Government patents', gov, bin_len), ('Spillovers received', rec, 0)]:
            for dist in ['pre-sample', 'contemporaneous']:
                Z = shock(X, dist == 'contemporaneous')
                panel = []
                for c in Z.index.intersection(out_n.index):
                    y = out_n.loc[c].reindex(years).fillna(0.0).to_numpy()
                    if y.sum() <= 0:
                        continue
                    z = np.arcsinh(Z.loc[c].reindex([t - lag_ for t in years]).to_numpy())
                    panel.append(pd.DataFrame({'year': years, 'type': c, 'y': y, 'z': z}))
                d = pd.concat(panel, ignore_index=True).dropna()
                for fe in ['type effects', 'type and year effects', 'type effects and type-specific trends']:
                    parts = [pd.Series(1.0, index=d.index, name='const'), d['z'].rename('z'), pd.get_dummies(d['type'], prefix='k', drop_first=True, dtype=float)]
                    if fe == 'type and year effects':
                        parts.append(pd.get_dummies(d['year'], prefix='y', drop_first=True, dtype=float))
                    if fe == 'type effects and type-specific trends':
                        tr = pd.get_dummies(d['type'], prefix='t', dtype=float).mul(d['year'] - first, axis=0)
                        parts.append(tr)
                    X_ = pd.concat(parts, axis=1)
                    X_.columns = [str(c) for c in X_.columns]
                    try:
                        m = sm.GLM(d['y'], X_, family=sm.families.Poisson()).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d['type'])[0]}, maxiter=200)
                    except ValueError:
                        start = np.zeros(X_.shape[1]); start[0] = np.log(d['y'].mean())
                        m = sm.GLM(d['y'], X_, family=sm.families.Poisson()).fit(start_params=start, method='bfgs', maxiter=2000,
                                                                              cov_type='cluster', cov_kwds={'groups': pd.factorize(d['type'])[0]})
                    rows.append(dict(shock=name, distribution=dist, timing='previous bin' if lag_ else 'same year', fe=fe,
                                     coef=m.params['z'], se=m.bse['z'], p=m.pvalues['z'], N=int(m.nobs), n_types=d['type'].nunique()))
        out = pd.DataFrame(rows)
        out.to_csv(outdir / 'shock_relevance_class.csv', index=False)
        cols = list(dict.fromkeys(zip(out['shock'], out['distribution'])))
        lines = [r'\begin{tabular}{l' + 'c' * len(cols) + '}', r'\toprule',
                 ' & ' + ' & '.join(a for a, _ in cols) + r' \\', ' & ' + ' & '.join(b for _, b in cols) + r' \\', r'\midrule']
        for fe in list(dict.fromkeys(out['fe'])):
            cs, ss = [], []
            for a, b in cols:
                r = out[(out['shock'] == a) & (out['distribution'] == b) & (out['fe'] == fe)].iloc[0]
                cs.append(f"{r.coef:.3f}{gpf.get_stars(r.p)}"); ss.append(f"({r.se:.3f})")
            lines += [fe.capitalize() + ' & ' + ' & '.join(cs) + r' \\', ' & ' + ' & '.join(ss) + r' \\']
        lines += [r'\midrule', 'Observations (first row) & ' + ' & '.join(str(int(out[(out['shock'] == a) & (out['distribution'] == b)].iloc[0].N)) for a, b in cols) + r' \\',
                  'Types (3-digit classes) & ' + ' & '.join(str(int(out[(out['shock'] == a) & (out['distribution'] == b)].iloc[0].n_types)) for a, b in cols) + r' \\',
                  r'\bottomrule', r'\end{tabular}']
        with open(outdir / 'table_shock_relevance_class.tex', 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(out.round(3).to_string())
        return out


    def Shock_Relevance_Summary(self, path, sfx=''):
        """""
        Compact table for the "New material" section. The government shock enters lagged one bin (as in the instrumented
        regressions); the spillover shock enters from the same bin, because its lag is built into the shock. Four panels:
        industry-type panel and national type-level panel, each with the pre-sample and the contemporaneous class distribution.
        """""
        D = self.Directory / 'Results' / 'Batches'
        a = pd.read_csv(D / f'diag_shock_relevance{sfx}' / 'shock_relevance.csv')
        ac = pd.read_csv(D / f'diag_shock_relevance_cur{sfx}' / 'shock_relevance.csv')
        b = pd.read_csv(D / f'diag_shock_relevance_type{sfx}' / 'shock_relevance_type.csv')
        shocks = ['Government patents (previous bin)', 'Spillovers received (same bin)']
        gname, sname = 'Government patents', 'Spillovers received'
        lines = [r'\begin{tabular}{lcc}', r'\toprule', ' & ' + ' & '.join(shocks) + r' \\']
        def cell(df, mask):
            r = df[mask]
            return (f"{r.iloc[0].coef:.3f}{gpf.get_stars(r.iloc[0].p)}", f"({r.iloc[0].se:.3f})") if len(r) else ('', '')
        ind_rows = [('Shock', 'All types, previous bin', 'All types, same bin'),
                    ('Shock of another type (placebo)', 'All types, previous-bin shock of another type (placebo)', 'All types, shock of another type (placebo)'),
                    ('Clean patents only', 'Clean only, previous bin', 'Clean only, same bin')]
        for title, df in [('Industry-type panel, pre-sample class distribution', a), ('Industry-type panel, contemporaneous class distribution', ac)]:
            lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{' + title + r'}} \\']
            for lab, sg, ss in ind_rows:
                c = [cell(df, (df['shock'] == gname) & (df['spec'] == sg)), cell(df, (df['shock'] == sname) & (df['spec'] == ss))]
                lines += [lab + ' & ' + ' & '.join(x[0] for x in c) + r' \\', ' & ' + ' & '.join(x[1] for x in c) + r' \\']
        for title, dist in [('Type-level panel (annual), pre-sample class distribution', 'pre-sample'),
                            ('Type-level panel (annual), contemporaneous class distribution', 'contemporaneous')]:
            lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{' + title + r'}} \\']
            for fe in ['type effects', 'type and year effects', 'type effects and type-specific trends']:
                c = [cell(b, (b['shock'] == gname) & (b['distribution'] == dist) & (b['timing'] == 'previous bin') & (b['fe'] == fe)),
                     cell(b, (b['shock'] == sname) & (b['distribution'] == dist) & (b['timing'] == 'same year') & (b['fe'] == fe))]
                lines += [fe.capitalize() + ' & ' + ' & '.join(x[0] for x in c) + r' \\', ' & ' + ' & '.join(x[1] for x in c) + r' \\']
        bc = pd.read_csv(D / f'diag_shock_relevance_class{sfx}' / 'shock_relevance_class.csv')
        for title, dist in [('Type = 3-digit CPC class (annual), pre-sample class distribution', 'pre-sample'),
                            ('Type = 3-digit CPC class (annual), contemporaneous class distribution', 'contemporaneous')]:
            lines += [r'\midrule', r'\multicolumn{3}{l}{\textit{' + title + r'}} \\']
            for fe in ['type effects', 'type and year effects', 'type effects and type-specific trends']:
                c = [cell(bc, (bc['shock'] == gname) & (bc['distribution'] == dist) & (bc['fe'] == fe)),
                     cell(bc, (bc['shock'] == sname) & (bc['distribution'] == dist) & (bc['fe'] == fe))]
                lines += [fe.capitalize() + ' & ' + ' & '.join(x[0] for x in c) + r' \\', ' & ' + ' & '.join(x[1] for x in c) + r' \\']
        lines += [r'\bottomrule', r'\end{tabular}']
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')


    def UDG_InstrumentSummary(self, path, instruments):
        """""
        One row per instrument: network first-stage F and second-stage estimate for the baseline net specification,
        with the network-mass x period controls and with fixed first-bin network weights. Counts and citations panels.
        instruments: list of (label, id prefix).
        """""
        reg = pd.read_csv(self.Directory / 'Results' / 'registry.csv').set_index('spec_id')
        tex = lambda t: str(t).replace('_', r'\_')
        variants = [('', 'Baseline'), ('mtrend', 'Network-mass $\\times$ period'), ('fixw', 'Fixed weights')]
        lines = [r'\begin{tabular}{l' + 'c' * 6 + '}', r'\toprule',
                 r' & \multicolumn{3}{c}{First-stage $F$} & \multicolumn{3}{c}{Second-stage estimate (SE)} \\',
                 r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
                 'Instrument & ' + ' & '.join(v[1] for v in variants) * 1 + ' & ' + ' & '.join(v[1] for v in variants) + r' \\', r'\midrule']
        for meas, title in [('pat', 'Patent counts'), ('cite', 'Citation-weighted patents')]:
            lines.append(r'\multicolumn{7}{l}{\textit{' + title + r'}} \\')
            first = instruments[0][1]
            refs = []
            for key in ('noiv', 'noivmt', 'noivfw'):
                sid = f'{first}_{meas}_{key}'
                r = reg.loc[sid]
                refs.append(f"{r['coef_net']:.2f}{gpf.get_stars(r['p_net'])} ({r['se_net']:.2f})")
            lines.append('No instrument (PPML-type estimate) & & & & ' + ' & '.join(refs) + r' \\')
            for label, tag in instruments:
                fs, ss = [], []
                for key, _ in variants:
                    sid = f"{tag}_{meas}_{key or 'base'}"
                    if sid not in reg.index:
                        fs.append(''); ss.append(''); continue
                    r = reg.loc[sid]
                    fs.append(f"{r['fs_F_net']:.1f}{gpf.get_stars(r['fs_pF_net'])}" if np.isfinite(r['fs_F_net']) else '')
                    if r['converged'] == 0:
                        ss.append('not conv.')
                    else:
                        ss.append(f"{r['coef_net']:.1f}{gpf.get_stars(r['p_net'])} ({r['se_net']:.1f})")
                lines.append(tex(label) + ' & ' + ' & '.join(fs) + ' & ' + ' & '.join(ss) + r' \\')
            lines.append(r'\midrule')
        lines[-1] = r'\bottomrule'
        lines.append(r'\end{tabular}')
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
