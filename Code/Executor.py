"""""""""""
Executor

Notes: This file executes the code for "Transition to Green Technology along the Supply Chain".

    Data build (slow; cluster for the Cleaner), then the specification batches documented in
    Results/spec_description.tex. Every batch is a list of spec dicts (see make_*_specs) run by Processor.UDG_Run.

"""""""""""

import Processor as p


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
#P.Cleaner(BLS_year_start, Year_start, Year_end, bin_len, API)      # cluster only

# Industry shocks: government, R&D prices (all firms and firms with clean patents), spillovers (clean / dirty / general)
#P.Instruments()
#P.Instruments('_c3')                      # 3-digit CPC classes (government and spillovers)
#P.Instruments('', asinh_pat=True)         # government shock with patent counts as asinh
# Spillovers on two patent types (clean / general), 4- and 3-digit classes, and their asinh versions
#for _s in ['_cg', '_cg_c3']:
#    P.Spillover_Shocks(_s); P.KS_Recv(_s); P.KS_CleanAgg(_s)
#P.Spillover_Shocks('_cg', asinh_pat=True); P.KS_Recv('_cg', asinh_pat=True); P.KS_CleanAgg('_cg', asinh_pat=True)

#P.Spillover_Shocks('_cg', diag=True); P.KS_Recv('_cg', diag=True); P.Spillover_Shocks('_cg_mg', diag=True)   # own-technology term kept
#P.Instruments('', spill=False, type_gov=True); P.Instruments('_mg', spill=False, type_gov=True)   # type-specific government series
#P.Instruments_Cur(); P.Instruments_Cur('_mg')   # shocks with the contemporaneous class distribution
#P.Instruments('_mg', spill=False)         # government shock at six-digit CPC (needs the Cleaner's *_mg files)
#P.Spillover_Shocks('_cg_mg'); P.KS_Recv('_cg_mg')   # spillover shocks and class-level receipts, main groups (sparse)
#P.IO_Change(Year_start, Year_mid, Year_end)     # input-output change graphs
#P.Cap_Citations()                                # citation weights capped at the 99th percentile (Ind_Pat_Cap.pkl), from Pat_Firms
#P.Firm_Entry()                                  # firms' first clean patents (Ind_Pat_Entry.pkl), from Pat_Firms


# ----------------------------------------------------------------

# Specification sets. Each spec overrides Processor.UDG_BASE (see CLAUDE.md for the baseline).

# ----------------------------------------------------------------

# six-digit CPC are used only if the Cleaner files and the shocks have been built (see the build section)
from pathlib import Path
DIAG = (P.Directory / 'Clean Data' / 'KS_Shocks_cg_Diag.pkl').exists()   # spillovers with the own-technology term
TYP = (P.Directory / 'Clean Data' / 'Govt_Shocks_Type.pkl').exists()     # type-specific government series
MG = all((P.Directory / 'Clean Data' / f).exists() for f in ['Govt_Shocks_mg.pkl', 'KS_Shocks_cg_mg.pkl', 'KS_Recv_cg_mg.pkl'])


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


def make_iv_specs(tag, src, asinh=False, c3=False, extras=(), mg=False):
    """IV permutation set around the baseline. The network G is instrumented by the instrument src (see
    Processor.UDG_Instrument). asinh: add the version with patent levels as asinh in the shock; c3: add 3-digit
    classes; extras: names of source-specific specs ('lag' for the government instrument, 'within' for spillovers)."""
    specs = []
    for m in ['pat', 'cite']:
        suf = 'count' if m == 'pat' else 'cites'
        def add(name, label, iv=src, **kw):
            specs.append(dict(id=f'{tag}_{m}_{name}', short=name, measure=m, share='G', iv=iv, label=label, **kw))
        add('noiv',     'Same estimator, no instrument', iv='none', panel='A')
        add('base',     'Baseline IV: no own lag', panel='A')
        add('updown',   'Separate up/down', direction='updown', panel='A')
        add('ownlag',   'Own lagged share', own_lag=True, panel='A')
        add('wsum',     'Control: sum of weights, net', controls=['W_w'], panel='A')
        add('wsumud',   'Control: sum of weights, up/down', controls=['W_w'], direction='updown', panel='A')
        add('norm',     'Normalised weights, net', normalise=True, panel='B')
        add('normud',   'Normalised weights, up/down', normalise=True, direction='updown', panel='B')
        add('bin3',     'Three-year bins', bin_len=3, panel='B')
        add('noshr',    'No kappa shrinkage, net', shrink=False, panel='B')
        add('noshrud',  'No kappa shrinkage, up/down', shrink=False, direction='updown', panel='B')
        add('offall',   'Offset: all patents, net', offset=f'pat_{suf}', panel='C')
        add('offallud', 'Offset: all patents, up/down', offset=f'pat_{suf}', direction='updown', panel='C')
        add('wins',     'Instrument winsorised at 5/95, net', iv_wins=0.05, panel='D')
        add('winsud',   'Instrument winsorised at 5/95, up/down', iv_wins=0.05, direction='updown', panel='D')
        if asinh:
            add('asinh',   'Patent levels in the shock as asinh, net', iv=f'{src}_asinh', panel='D')
            add('asinhud', 'Patent levels in the shock as asinh, up/down', iv=f'{src}_asinh', direction='updown', panel='D')
        if c3:
            add('c3',      'Technology classes at 3 digits, net', iv=f'{src}_c3', panel='D')
            add('c3ud',    'Technology classes at 3 digits, up/down', iv=f'{src}_c3', direction='updown', panel='D')
        if mg:
            add('mg',      'Technology classes: six-digit CPC, net', iv=f'{src}_mg', panel='D')
            add('mgud',    'Technology classes: six-digit CPC, up/down', iv=f'{src}_mg', direction='updown', panel='D')
        if 'lag' in extras:
            add('lag2',    'Instrument lagged two bins, net', iv_lags=(2,), panel='D')
            add('lag2ud',  'Instrument lagged two bins, up/down', iv_lags=(2,), direction='updown', panel='D')
            add('lag12',   'Instruments lagged one and two bins, net', iv_lags=(1, 2), panel='D')
            add('lag12ud', 'Instruments lagged one and two bins, up/down', iv_lags=(1, 2), direction='updown', panel='D')
        if 'within' in extras:
            add('wt',       'Within-type spillovers (clean from clean / general from general), net', iv='kswt', panel='D')
            add('wtud',     'Within-type spillovers, up/down', iv='kswt', direction='updown', panel='D')
            add('wtwins',   'Within-type spillovers, winsorised at 5/95, net', iv='kswt', iv_wins=0.05, panel='D')
            add('wtwinsud', 'Within-type spillovers, winsorised at 5/95, up/down', iv='kswt', iv_wins=0.05, direction='updown', panel='D')
        if 'direct' in extras:   # direct input-output coefficients: first-bin direct network in the instrument, the regressor's network changes by bin
            add('noivds',  'No instrument, direct input-output coefficients', iv='none', direct_s=True, panel='D')
            add('ds',      'Direct input-output coefficients, net', direct_s=True, panel='D')
            add('dsud',    'Direct input-output coefficients, up/down', direct_s=True, direction='updown', panel='D')
            if mg:
                add('dsmg',    'Direct input-output coefficients, six-digit CPC, net', iv=f'{src}_mg', direct_s=True, panel='D')
                add('dsmgud',  'Direct input-output coefficients, six-digit CPC, up/down', iv=f'{src}_mg', direct_s=True, direction='updown', panel='D')
        if 'lag' in extras and mg:   # six-digit government instrument: the double lag
            add('mglag2',     'Six-digit CPC, instrument lagged two bins, net', iv=f'{src}_mg', iv_lags=(2,), panel='D')
            add('mglag2ud',   'Six-digit CPC, instrument lagged two bins, up/down', iv=f'{src}_mg', iv_lags=(2,), direction='updown', panel='D')
        if 'diag' in extras:
            add('diag',     'Spillover network with the own-technology term, net', iv=f'{src}_diag', panel='D')
            add('diagud',   'Spillover network with the own-technology term, up/down', iv=f'{src}_diag', direction='updown', panel='D')
            if mg:
                add('diagmg',   'Own-technology term, six-digit CPC, net', iv=f'{src}_mg_diag', panel='D')
                add('diagmgud', 'Own-technology term, six-digit CPC, up/down', iv=f'{src}_mg_diag', direction='updown', panel='D')
        if 'type' in extras:
            add('typ',      'Type-specific government series, net', iv=f'{src}_type', panel='D')
            add('typud',    'Type-specific government series, up/down', iv=f'{src}_type', direction='updown', panel='D')
            if mg:
                add('typmg',    'Type-specific government series, six-digit CPC, net', iv=f'{src}_mg_type', panel='D')
                add('typmgud',  'Type-specific government series, six-digit CPC, up/down', iv=f'{src}_mg_type', direction='updown', panel='D')
        add('noivmt',   'No instrument, network-mass x period controls', iv='none', mass_trend=True, panel='E')
        add('mtrend',   'Network-mass x period controls, net', mass_trend=True, panel='E')
        add('mtrendud', 'Network-mass x period controls, up/down', mass_trend=True, direction='updown', panel='E')
        add('noivfw',   'No instrument, fixed first-bin network weights', iv='none', fixed_w=True, panel='F')
        add('fixw',     'Fixed first-bin network weights, net', fixed_w=True, panel='F')
        add('fixwud',   'Fixed first-bin network weights, up/down', fixed_w=True, direction='updown', panel='F')
    return specs


def make_cur_specs(tag, mg=False):
    "Instruments built with the industry's contemporaneous class distribution (counts and citations): government and spillovers, 4-digit and six-digit CPC."
    specs = []
    for name, src, label in [('gov', 'gov_cur', 'Government patents, contemporaneous distribution'), ('ks', 'ks_cur', 'Spillovers, contemporaneous distribution')] + \
                             ([('govmg', 'gov_mg_cur', 'Government patents, six-digit CPC, contemporaneous distribution'),
                               ('ksmg', 'ks_mg_cur', 'Spillovers, six-digit CPC, contemporaneous distribution')] if mg else []):
        for m in ['pat', 'cite']:
            specs.append(dict(id=f'{tag}_{m}_{name}', short=name, measure=m, share='G', iv=src, label=label, panel='A'))
            specs.append(dict(id=f'{tag}_{m}_{name}ds', short=f'{name}ds', measure=m, share='G', iv=src, label=label + ', direct network', panel='B', direct_s=True))
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


def renormalised(tag, src_n):
    "Limited set with the class-level ratio renormalised over classes where it is defined (ratio instruments)."
    out = []
    for sp in make_iv_specs(tag, src_n, c3=True):
        if sp['short'] in ('base', 'updown', 'c3', 'c3ud'):
            nm = 'rn' + sp['short']
            out.append({**sp, 'id': sp['id'].rsplit('_', 1)[0] + '_' + nm, 'short': nm, 'panel': 'D',
                        'label': 'Renormalised over classes with a ratio: ' + sp['label']})
    return out


# ----------------------------------------------------------------

# Run the batches. Batch folders: Results/Batches/<batch>; registry: Results/registry.csv.

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

# Batch 2: instrument = government patents
P.UDG_Run('batch2_gov', make_iv_specs('b2', 'gov', asinh=True, c3=True, extras=('lag', 'direct') + (('type',) if TYP else ()), mg=MG), BLS_year_start, Year_end)

# Batch 3: instrument = state R&D prices; zero stage; and on firms with clean patents
P.UDG_Run('batch3_rd', make_iv_specs('b3', 'rd'), BLS_year_start, Year_end)
P.UDG_Run('batch3_rdcf', make_iv_specs('b3c', 'rdcf'), BLS_year_start, Year_end)
P.RD_ZeroStage('batch3_rd')

# Batch 4: instrument = knowledge spillovers (two patent types: clean and general)
P.KS_Shares('batch4_ks', BLS_year_start, Year_end, bin_len)
P.UDG_Run('batch4_ks', make_iv_specs('b4', 'ks', asinh=True, c3=True, extras=('within', 'direct') + (('diag',) if DIAG else ()), mg=MG), BLS_year_start, Year_end)

# Batch 5: asinh(clean patents) on network asinh(clean patents)
P.UDG_Run('batch5_asinh', make_asinh_specs('b5'), BLS_year_start, Year_end)

# Batch 6: industry class distribution x class-level ratio (government clean / all; spillovers clean / general)
for batch, src, srcn, tag in [('batch6_govr', 'govr', 'govrn', 'b6g'), ('batch6_ksr', 'ksr', 'ksrn', 'b6k')]:
    specs = make_iv_specs(tag, src, asinh=(src == 'ksr'), c3=True, mg=MG) + renormalised(tag, srcn)
    P.UDG_Run(batch, specs, BLS_year_start, Year_end)

# Batch 7: aggregate clean shock x sector pre-sample clean share
for batch, src, tag in [('batch7_govcl', 'govcl', 'b7g'), ('batch7_ksagg', 'ksagg', 'b7k')]:
    P.UDG_Run(batch, make_iv_specs(tag, src, asinh=True), BLS_year_start, Year_end)


# Batch 8: adoption as a firm's first clean patent (entry hazard)
P.UDG_Run('batch8_entry', make_entry_specs('b8'), BLS_year_start, Year_end)
if (P.Directory / 'Clean Data' / 'Govt_Shocks_Cur.pkl').exists():
    P.UDG_Run('batch9_cur', make_cur_specs('b9', mg=MG), BLS_year_start, Year_end)   # contemporaneous class distribution in the instruments


# ----------------------------------------------------------------

# Diagnostics and summary tables for the main text (Results/Summary).

# ----------------------------------------------------------------

P.IO_Validation('diag_io_validation', BLS_year_start, Year_end, bin_len)   # does clean patenting predict changes in input shares?
P.Old_Benchmark('diag_old_benchmark', BLS_year_start, Year_start, Year_mid, Year_end, bin_len)   # patent regressions of the submitted version (branch Old)
P.Shock_Relevance('diag_shock_relevance', BLS_year_start, Year_end, bin_len)
if MG:
    P.Shock_Relevance('diag_shock_relevance_mg', BLS_year_start, Year_end, bin_len, sfx='_mg')   # does the industry-type shock predict patenting of that type?
P.Shock_Relevance('diag_shock_relevance_cur', BLS_year_start, Year_end, bin_len, cur=True)   # contemporaneous class distribution
if MG:
    P.Shock_Relevance('diag_shock_relevance_cur_mg', BLS_year_start, Year_end, bin_len, sfx='_mg', cur=True)
P.Match_Coverage('diag_match_coverage', BLS_year_start, Year_end, bin_len)   # share of patents matched to firms
P.Shock_Relevance_Type('diag_shock_relevance_type', BLS_year_start, bin_len)   # type-level gut check
P.Shock_Relevance_Class('diag_shock_relevance_class', BLS_year_start, bin_len)   # type = 3-digit CPC class
if MG:
    P.Shock_Relevance_Type('diag_shock_relevance_type_mg', BLS_year_start, bin_len, sfx='_mg')   # six-digit CPC
    P.Shock_Relevance_Class('diag_shock_relevance_class_mg', BLS_year_start, bin_len, sfx='_mg')
P.UDG_OwnFirstStage('diag_first_stage', BLS_year_start, Year_end, ['gov', 'rd', 'rdcf', 'ks', 'kswt', 'govr', 'ksr', 'govcl', 'ksagg'])

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
(P.Directory / 'Results' / 'Summary').mkdir(parents=True, exist_ok=True)
P.UDG_Summary(P.Directory / 'Results' / 'Summary' / 'table_ppml_summary.tex', rows, cols)
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
P.UDG_Summary(P.Directory / 'Results' / 'Summary' / 'table_ppml_summary_alt.tex', rows2, cols)
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
P.UDG_Summary(P.Directory / 'Results' / 'Summary' / 'table_new_m14.tex', rows_n, cols_n)
P.Shock_Relevance_Summary(P.Directory / 'Results' / 'Summary' / 'table_new_relevance.tex')
if MG:
    P.Shock_Relevance_Summary(P.Directory / 'Results' / 'Summary' / 'table_new_relevance_mg.tex', sfx='_mg')   # six-digit CPC version
P.Linkage_Comparison(P.Directory / 'Results' / 'Summary' / 'table_new_linkage.tex')   # needs Results/Batches/diag_linkage/ (pre-union registry and coverage)
P.New_IV_Variants(P.Directory / 'Results' / 'Summary' / 'table_new_diag.tex', 'diag')
P.New_IV_Variants(P.Directory / 'Results' / 'Summary' / 'table_new_type.tex', 'type')
P.New_IV_Variants(P.Directory / 'Results' / 'Summary' / 'table_new_gov6.tex', 'gov6')
P.New_IV_Variants(P.Directory / 'Results' / 'Summary' / 'table_new_direct.tex', 'direct')
P.New_IV_Variants(P.Directory / 'Results' / 'Summary' / 'table_new_cur.tex', 'cur')
P.New_IV_Variants(P.Directory / 'Results' / 'Summary' / 'table_new_curds.tex', 'curds')
P.Cite5_Summary(P.Directory / 'Results' / 'Summary' / 'table_new_cite5.tex')
cols_a = [('asinh clean patents, counts', 'b5', 'pat'), ('asinh clean patents, citations', 'b5', 'cite')]
rows_a = [('Baseline (net)', 'base', 'net'), ('Up/down: upstream', 'updown', 'up'), ('Up/down: downstream', 'updown', 'down'),
          ('Own lagged level added: network', 'ownlag', 'net'), ('Own lagged level added: own lag', 'ownlag', 'lag'),
          ('Weight-sum control (net)', 'wsum', 'net'), ('Normalised weights (net)', 'norm', 'net'),
          ('Three-year bins (net)', 'bin3', 'net'), ('Annual outcome (net)', 'annual', 'net')]
P.UDG_Summary(P.Directory / 'Results' / 'Summary' / 'table_asinh_summary.tex', rows_a, cols_a)
P.UDG_InstrumentSummary(P.Directory / 'Results' / 'Summary' / 'table_iv_summary.tex',
                        [('Government patents (B2)', 'b2'), ('State R\\&D prices, all firms (B3)', 'b3'),
                         ('State R\\&D prices, clean firms (B3)', 'b3c'), ('Spillovers, levels (B4)', 'b4'),
                         ('Government ratio (B6)', 'b6g'), ('Spillover ratio (B6)', 'b6k'),
                         ('Aggregate government clean patenting (B7)', 'b7g'), ('Aggregate clean spillovers (B7)', 'b7k')])


# ----------------------- #
# Record Package Versions #
# ----------------------- #
packages = ["linearmodels", "matplotlib", "numpy", "openpyxl", "pandas", "scipy", "statsmodels"]
P.write_package_versions(packages)
P.Theory_Check()   # which specifications are consistent with the theory (notes under the tables, scorecard)
