"""""""""""
Executor

Notes: This file executes the code for "Transition to Green Technology along the Supply Chain".

    Data build (slow; cluster for the Cleaner), then the specification batches documented in
    Specifications/spec_description.tex. Every batch is a list of spec dicts (see make_*_specs) run by Processor.UDG_Run.
    The customer IV (Customer_Data, Customer_IV, Customer_Tables; Specifications/Customer_IV, Results/Tables, Results/Figures) comes last.
    The government / R&D-price / spillover instrument batches were archived (Code/Archive; git tag pre-customer-iv-prune).

"""""""""""

import Processor as p
import Customer_IV as civ       # customer IV: Tables E1 and E3, timing (Section 2)
import Customer_Tables as ctb   # tables and figure of Specifications/Customer_IV/draft_customer_iv_section.tex


# ----------------------------------------------------------------

# Define project objects.

# ----------------------------------------------------------------

P = p.Processor()
BLS_year_start, Year_start, Year_mid, Year_end = (1997, 2012, 2017, 2022)
bin_len = 5


# ----------------------------------------------------------------

# Build data (comment out once built).

# ----------------------------------------------------------------

API = 0   # set to 1 for a new API download
#P.Cleaner(BLS_year_start, Year_start, Year_end, API)      # cluster only

# The Cleaner also downloads (API = 1, WRDS login in .keys) and cleans the Compustat data and the customer-IV patent tables (see WRDS.py, Customer_Data.py).


P.IO_Change(Year_start, Year_mid, Year_end)      # input-output change figures (Results/Figures/IO_Change_*)
P.Cap_Citations()                                # citation weights capped at the 99th percentile (Ind_Pat_Cap.pkl), from Pat_Firms
P.Firm_Entry()                                  # firms' first clean patents (Ind_Pat_Entry.pkl), from Pat_Firms


# ----------------------------------------------------------------

# Customer IV (Sections 1-2 and Tables E1-E3 of Specifications/Customer_IV/draft_customer_iv_section.tex).
# Needs Clean Data/Cust_*.pkl (written by the Cleaner). Table_E1 and Table_E3 take minutes; Timing about 15 minutes with NW=7 workers.
# NRI / NBOOT set the number of reshuffle and bootstrap draws (defaults 200 / 199, as reported); NRI=20 NBOOT=19 NW=1 for a quick test.

# ----------------------------------------------------------------

if (P.Directory / 'Clean Data' / 'Cust_Segments.pkl').exists():
    CI = civ.Customer_IV(P)
    CI.Table_E1()       # customers' greening -> supplier, nine columns
    CI.Table_E3()       # suppliers' greening -> customer, nine columns
    CI.Timing()         # both directions at horizons 0-20
    CT = ctb.Customer_Tables(P)
    CT.Tables()         # tables, macros and Figure E1 for the draft
    CT.Check_Reproduction()   # compare with the reference run of the draft's authors (Specifications/Customer_IV/reference)


# ----------------------------------------------------------------

# Specification sets. Each spec overrides Processor.UDG_BASE (see CLAUDE.md for the baseline).

# ----------------------------------------------------------------

from pathlib import Path


C5 = (P.Directory / 'Clean Data' / 'Ind_Pat_Cite5.pkl').exists()   # five-year citation weights: needs the Cleaner's Ind_Pat_Cite5.pkl


def make_specs(tag, share, extra_offsets=()):
    "Reduced-form (PPML) permutation set around the baseline, for patents and citations."
    specs = []
    for m in ['pat', 'cite']:
        suf = 'count' if m == 'pat' else 'cites'
        def add(name, label, **kw):
            specs.append(dict(id=f'{tag}_{m}_{name}', short=name, measure=m, share=share, label=label, **kw))
        plc = 'P' if share == 'G' else 'PD'
        plc_lab = 'dirty/all' if share == 'G' else 'dirty/(clean+dirty)'
        # A: structure and controls
        add('base',     'Baseline: no own lag', panel='A')
        add('updown',   'Separate up/down', direction='updown', panel='A')
        add('ownlag',   'Own lagged share', own_lag=True, panel='A')
        add('wsum',     'Control: sum of weights, net', controls=['W_w'], panel='A')
        add('wsumud',   'Control: sum of weights, up/down', controls=['W_w'], direction='updown', panel='A')
        # B: construction of the regressor
        add('norm',     'Normalised weights, net', normalise=True, panel='B')
        add('normud',   'Normalised weights, up/down', normalise=True, direction='updown', panel='B')
        add('bin3',     'Three-year bins', bin_len=3, panel='B')
        add('noshr',    'No kappa shrinkage, net', shrink=False, panel='B')
        add('noshrud',  'No kappa shrinkage, up/down', shrink=False, direction='updown', panel='B')
        # C: offset
        add('offall',   'Offset: all patents, net', offset=f'pat_{suf}', panel='C')
        add('offallud', 'Offset: all patents, up/down', offset=f'pat_{suf}', direction='updown', panel='C')
        for name, col in extra_offsets:
            add(f'{name}',   f'Offset: {name}, net', offset=f'{col}_{suf}', panel='C')
            add(f'{name}ud', f'Offset: {name}, up/down', offset=f'{col}_{suf}', direction='updown', panel='C')
        # D: placebos and emissions
        add('plc',      f'Placebo: {plc_lab}, net', rhs=plc, panel='D')
        add('plcud',    f'Placebo: {plc_lab}, up/down', rhs=plc, direction='updown', panel='D')
        add('plcw',     f'Placebo: {plc_lab} + sum of weights, net', rhs=plc, controls=['W_w'], panel='D')
        add('plcwud',   f'Placebo: {plc_lab} + sum of weights, up/down', rhs=plc, controls=['W_w'], direction='updown', panel='D')
        add('em',       'Emissions reduction, net', rhs='E', entity_fe=False, panel='D')
        add('emud',     'Emissions reduction, up/down', rhs='E', entity_fe=False, direction='updown', panel='D')
        # E: annual outcome; F: narrow clean definition; G: network-mass controls; H: fixed network weights
        add('annual',   'Annual outcome, regressor from bin ending 5 years earlier, net', annual=True, panel='E')
        add('annualud', 'Annual outcome, regressor from bin ending 5 years earlier, up/down', annual=True, direction='updown', panel='E')
        add('y02p',     'Narrow clean definition (Y02P only), net', clean='y02p', panel='F')
        add('y02pud',   'Narrow clean definition (Y02P only), up/down', clean='y02p', direction='updown', panel='F')
        add('mtrend',   'Network-mass x period controls, net', mass_trend=True, panel='G')
        add('mtrendud', 'Network-mass x period controls, up/down', mass_trend=True, direction='updown', panel='G')
        add('fixw',     'Fixed first-bin network weights, net', fixed_w=True, panel='H')
        add('fixwud',   'Fixed first-bin network weights, up/down', fixed_w=True, direction='updown', panel='H')
        add('fixwmt',   'Fixed weights + network-mass x period controls, net', fixed_w=True, mass_trend=True, panel='H')
        add('freeoff',   'Offset coefficient estimated, net', free_offset=True, panel='I')
        add('freeoffud', 'Offset coefficient estimated, up/down', free_offset=True, direction='updown', panel='I')
        # J: treatment of partners without patents (M1), last outcome bin (M2), direct IO weights (M3), capped citations (M4)
        fm = dict(fill_missing=True)
        m12 = dict(fill_missing=True, max_period=2017)
        add('m1',       'Partners without patents at the aggregate share (M1), net', panel='J', **fm)
        add('m1ud',     'Partners without patents at the aggregate share (M1), up/down', direction='updown', panel='J', **fm)
        add('m2',       'Outcome bins up to 2017 (M2), net', max_period=2017, panel='J')
        add('m2ud',     'Outcome bins up to 2017 (M2), up/down', max_period=2017, direction='updown', panel='J')
        add('m12',      'M1 + M2, net', panel='J', **m12)
        add('m12ud',    'M1 + M2, up/down', direction='updown', panel='J', **m12)
        add('ds',       'Direct IO coefficients (M3), net', direct_s=True, panel='J')
        add('dsud',     'Direct IO coefficients (M3), up/down', direct_s=True, direction='updown', panel='J')
        add('m12ds',    'M1 + M2 + M3, net', direct_s=True, panel='J', **m12)
        add('m12dsud',  'M1 + M2 + M3, up/down', direct_s=True, direction='updown', panel='J', **m12)
        add('m12plc',   f'M1 + M2, placebo {plc_lab}, net', rhs=plc, panel='J', **m12)
        add('m12dsplc', f'M1 + M2 + M3, placebo {plc_lab}, net', rhs=plc, direct_s=True, panel='J', **m12)
        add('m12drop68',   'M1 + M2, motor vehicles not estimated, net', drop_sectors=(68,), panel='J', **m12)
        add('m12dsdrop68', 'M1 + M2 + M3, motor vehicles not estimated, net', drop_sectors=(68,), direct_s=True, panel='J', **m12)
        if m == 'cite' and C5:
            add('c5',       'Citation weights from the five years after filing, net', clean='cite5', panel='K')
            add('c5ud',     'Citation weights from the five years after filing, up/down', clean='cite5', direction='updown', panel='K')
            add('m12c5',    'M1 + M2 + five-year citation weights, net', clean='cite5', panel='K', **m12)
            add('m12dsc5',  'M1 + M2 + M3 + five-year citation weights, net', clean='cite5', direct_s=True, panel='K', **m12)
        if m == 'cite':
            add('cap',       'Citation weights capped at the 99th percentile (M4), net', clean='cap', panel='J')
            add('capud',     'Citation weights capped (M4), up/down', clean='cap', direction='updown', panel='J')
            add('m12cap',    'M1 + M2 + M4, net', clean='cap', panel='J', **m12)
            add('m12dscap',  'M1 + M2 + M3 + M4, net', clean='cap', direct_s=True, panel='J', **m12)
    return specs


def make_asinh_specs(tag):
    "Levels specification: OLS of asinh(clean) on the network sums of winsorised asinh(clean)."
    specs = []
    for m in ['pat', 'cite']:
        def add(name, label, **kw):
            specs.append(dict(id=f'{tag}_{m}_{name}', short=name, measure=m, share='A', label=label, **kw))
        add('base',     'Baseline: no own lag', panel='A')
        add('updown',   'Separate up/down', direction='updown', panel='A')
        add('ownlag',   'Own lagged level', own_lag=True, panel='A')
        add('wsum',     'Control: sum of weights, net', controls=['W_w'], panel='A')
        add('wsumud',   'Control: sum of weights, up/down', controls=['W_w'], direction='updown', panel='A')
        add('norm',     'Normalised weights, net', normalise=True, panel='B')
        add('normud',   'Normalised weights, up/down', normalise=True, direction='updown', panel='B')
        add('bin3',     'Three-year bins', bin_len=3, panel='B')
        add('annual',   'Annual outcome, net', annual=True, panel='C')
        add('annualud', 'Annual outcome, up/down', annual=True, direction='updown', panel='C')
    return specs


def make_entry_specs(tag):
    "Extensive-margin adoption: firms' first clean patents over firms at risk (Processor.Firm_Entry), counts only."
    specs = []
    def add(name, label, **kw):
        specs.append(dict(id=f'{tag}_pat_{name}', short=name, measure='pat', share='G', clean='entry', label=label, panel='A', **kw))
    add('base',     'Entry hazard: baseline')
    add('updown',   'Separate up/down', direction='updown')
    add('ownlag',   'Own lagged hazard', own_lag=True)
    add('wsum',     'Control: sum of weights, net', controls=['W_w'])
    add('norm',     'Normalised weights, net', normalise=True)
    add('noshr',    'No kappa shrinkage, net', shrink=False)
    add('bin3',     'Three-year bins', bin_len=3)
    add('annual',   'Annual outcome, net', annual=True)
    add('mtrend',   'Network-mass x period controls, net', mass_trend=True)
    add('mtrendud', 'Network-mass x period controls, up/down', mass_trend=True, direction='updown')
    add('fixw',     'Fixed first-bin network weights, net', fixed_w=True)
    add('fixwud',   'Fixed first-bin network weights, up/down', fixed_w=True, direction='updown')
    return specs


# ----------------------------------------------------------------

# Run the batches. Batch folders: Specifications/Batches/<batch>; registry: Specifications/registry.csv.

# ----------------------------------------------------------------

# Batch 0: G = clean / all patents (and the counts-versus-citations diagnostics)
batch = 'batch0_base'
specs = make_specs('b0', 'G')
specs += [
    dict(id='b0_x_cnt_own',   short='counts~counts', table='diag_cross', measure='pat',  direction='updown'),
    dict(id='b0_x_cnt_cite',  short='counts~cites',  table='diag_cross', measure='pat',  direction='updown', rhs_meas='cite'),
    dict(id='b0_x_cite_own',  short='cites~cites',   table='diag_cross', measure='cite', direction='updown'),
    dict(id='b0_x_cite_cnt',  short='cites~counts',  table='diag_cross', measure='cite', direction='updown', rhs_meas='pat'),
    dict(id='b0_c_none',  short='none',          table='diag_ctrl', measure='cite', direction='updown'),
    dict(id='b0_c_tot',   short='+ patents',     table='diag_ctrl', measure='cite', direction='updown', controls=['T_pat']),
    dict(id='b0_c_cpp',   short='+ cites/patent', table='diag_ctrl', measure='cite', direction='updown', controls=['Q_cite']),
    dict(id='b0_c_both',  short='+ both',        table='diag_ctrl', measure='cite', direction='updown', controls=['T_pat', 'Q_cite']),
]
P.UDG_Run(batch, specs, BLS_year_start, Year_end)
P.UDG_Corr(batch, BLS_year_start, Year_end)

# Batch 1: D = clean / (clean + dirty), offset = dirty
P.UDG_Run('batch1_D', make_specs('b1', 'D', extra_offsets=[('offcd', 'clim_pat')]), BLS_year_start, Year_end)
# Batch 10: annual outcome with direct input-output coefficients, up/down (the fourth column of the main-text reduced-form table)
P.UDG_Run('batch10_annual', [dict(id='b10_pat_annualdsud', short='annualdsud', measure='pat', share='G', panel='A',
                                  label='Annual outcome, direct IO coefficients (M3), up/down', annual=True, direct_s=True, direction='updown')],
          BLS_year_start, Year_end)
P.UpDown_Table(P.Directory / 'Results' / 'Tables' / 'Reduced_Form_UpDown.tex')
P.UpDown_Robust(P.Directory / 'Results' / 'Tables' / 'Reduced_Form_Robust.tex')   # appendix: reduced-form robustness   # main-text table: upstream and downstream, bins and annual


# Batch 5: asinh(clean patents) on network asinh(clean patents)
P.UDG_Run('batch5_asinh', make_asinh_specs('b5'), BLS_year_start, Year_end)


# Batch 8: adoption as a firm's first clean patent (entry hazard)
P.UDG_Run('batch8_entry', make_entry_specs('b8'), BLS_year_start, Year_end)


# ----------------------------------------------------------------

# Diagnostics and summary tables for the main text (Specifications/Summary).

# ----------------------------------------------------------------

P.Summary_Stats(BLS_year_start, Year_end, bin_len)   # Results/Tables/Summary_Stats.tex, Top_Network_Sectors.tex
P.IO_Validation('diag_io_validation', BLS_year_start, Year_end, bin_len)   # does clean patenting predict changes in input shares?
P.Old_Benchmark('diag_old_benchmark', BLS_year_start, Year_start, Year_mid, Year_end, bin_len)   # patent regressions of the submitted version (branch Old)
P.Match_Coverage('diag_match_coverage', BLS_year_start, Year_end, bin_len)   # share of patents matched to firms

cols = [('$G$, counts', 'b0', 'pat'), ('$G$, citations', 'b0', 'cite'), ('$D$, counts', 'b1', 'pat'), ('$D$, citations', 'b1', 'cite')]
rows = [('Baseline (net)', 'base', 'net'),
        ('Up/down: upstream', 'updown', 'up'), ('Up/down: downstream', 'updown', 'down'),
        ('\\quad $p$-value: upstream = downstream', 'updown', 'eq'),
        ('Own lagged share added', 'ownlag', 'net'),
        ('Weight-sum control (net)', 'wsum', 'net'),
        ('Normalised weights (net)', 'norm', 'net'),
        ('No $\\kappa$ shrinkage (net)', 'noshr', 'net'),
        ('Three-year bins (net)', 'bin3', 'net'),
        ('Annual outcome (net)', 'annual', 'net'),
        ('Clean = Y02P only (net)', 'y02p', 'net'),
        ('Network-mass $\\times$ period (net)', 'mtrend', 'net'),
        ('Network-mass $\\times$ period: upstream', 'mtrendud', 'up'), ('Network-mass $\\times$ period: downstream', 'mtrendud', 'down'),
        ('\\quad $p$-value: upstream = downstream', 'mtrendud', 'eq'),
        ('Fixed first-bin weights (net)', 'fixw', 'net'),
        ('Fixed first-bin weights: upstream', 'fixwud', 'up'), ('Fixed first-bin weights: downstream', 'fixwud', 'down'),
        ('\\quad $p$-value: upstream = downstream', 'fixwud', 'eq')]
(P.Directory / 'Specifications' / 'Summary').mkdir(parents=True, exist_ok=True)
P.UDG_Summary(P.Directory / 'Specifications' / 'Summary' / 'table_ppml_summary.tex', rows, cols)
rows2 = [('Baseline (net)', 'base', 'net'),
         ('Partners without patents at the aggregate share (M1)', 'm1', 'net'),
         ('Outcome bins up to 2017 (M2)', 'm2', 'net'),
         ('M1 + M2', 'm12', 'net'),
         ('\\quad up/down: upstream', 'm12ud', 'up'), ('\\quad up/down: downstream', 'm12ud', 'down'),
         ('\\quad $p$-value: upstream = downstream', 'm12ud', 'eq'),
         ('Direct IO coefficients (M3)', 'ds', 'net'),
         ('M1 + M2 + M3', 'm12ds', 'net'),
         ('\\quad up/down: upstream', 'm12dsud', 'up'), ('\\quad up/down: downstream', 'm12dsud', 'down'),
         ('\\quad $p$-value: upstream = downstream', 'm12dsud', 'eq'),
         ('M1 + M2, dirty-share placebo', 'm12plc', 'net'),
         ('M1 + M2 + M3, dirty-share placebo', 'm12dsplc', 'net'),
         ('M1 + M2, motor vehicles not estimated', 'm12drop68', 'net'),
         ('M1 + M2 + M3, motor vehicles not estimated', 'm12dsdrop68', 'net'),
         ('Capped citation weights (M4)', 'cap', 'net'),
         ('M1 + M2 + M4', 'm12cap', 'net'),
         ('M1 + M2 + M3 + M4', 'm12dscap', 'net')]
P.UDG_Summary(P.Directory / 'Specifications' / 'Summary' / 'table_ppml_summary_alt.tex', rows2, cols)
# compact table for the "New material" section of the document: M1-M4 for G (counts, citations) and D (counts)
cols_n = [('$G$, counts', 'b0', 'pat'), ('$G$, citations', 'b0', 'cite'), ('$D$, counts', 'b1', 'pat')]
rows_n = [('Baseline (net)', 'base', 'net'),
          ('M1: partners without patents at the aggregate share', 'm1', 'net'),
          ('M2: outcome bins up to 2017', 'm2', 'net'),
          ('M1 + M2', 'm12', 'net'),
          ('M3: direct input--output coefficients', 'ds', 'net'),
          ('M1 + M2 + M3', 'm12ds', 'net'),
          ('M4: capped citation weights', 'cap', 'net'),
          ('M1 + M2 + M3 + M4', 'm12dscap', 'net')]
P.UDG_Summary(P.Directory / 'Specifications' / 'Summary' / 'table_new_m14.tex', rows_n, cols_n)
P.Linkage_Comparison(P.Directory / 'Specifications' / 'Summary' / 'table_new_linkage.tex')   # needs Specifications/Batches/diag_linkage/ (pre-union registry and coverage)
P.Cite5_Summary(P.Directory / 'Specifications' / 'Summary' / 'table_new_cite5.tex')
cols_a = [('asinh clean patents, counts', 'b5', 'pat'), ('asinh clean patents, citations', 'b5', 'cite')]
rows_a = [('Baseline (net)', 'base', 'net'), ('Up/down: upstream', 'updown', 'up'), ('Up/down: downstream', 'updown', 'down'),
          ('Own lagged level added: network', 'ownlag', 'net'), ('Own lagged level added: own lag', 'ownlag', 'lag'),
          ('Weight-sum control (net)', 'wsum', 'net'), ('Normalised weights (net)', 'norm', 'net'),
          ('Three-year bins (net)', 'bin3', 'net'), ('Annual outcome (net)', 'annual', 'net')]
P.UDG_Summary(P.Directory / 'Specifications' / 'Summary' / 'table_asinh_summary.tex', rows_a, cols_a)


# ----------------------- #
# Record Package Versions #
# ----------------------- #
packages = ["linearmodels", "matplotlib", "numpy", "openpyxl", "pandas", "psycopg2", "rapidfuzz", "scipy", "statsmodels"]
P.write_package_versions(packages)
P.Theory_Check()   # which specifications are consistent with the theory (notes under the tables, scorecard)
