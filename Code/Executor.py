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

#P.IO_Change(Year_start, Year_mid, Year_end)     # input-output change graphs


# ----------------------------------------------------------------

# Specification sets. Each spec overrides Processor.UDG_BASE (see CLAUDE.md for the baseline).

# ----------------------------------------------------------------

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
    return specs


def make_iv_specs(tag, src, asinh=False, c3=False, extras=()):
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
        add('noivmt',   'No instrument, network-mass x period controls', iv='none', mass_trend=True, panel='E')
        add('mtrend',   'Network-mass x period controls, net', mass_trend=True, panel='E')
        add('mtrendud', 'Network-mass x period controls, up/down', mass_trend=True, direction='updown', panel='E')
        add('noivfw',   'No instrument, fixed first-bin network weights', iv='none', fixed_w=True, panel='F')
        add('fixw',     'Fixed first-bin network weights, net', fixed_w=True, panel='F')
        add('fixwud',   'Fixed first-bin network weights, up/down', fixed_w=True, direction='updown', panel='F')
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
P.UDG_Run('batch2_gov', make_iv_specs('b2', 'gov', asinh=True, c3=True, extras=('lag',)), BLS_year_start, Year_end)

# Batch 3: instrument = state R&D prices; zero stage; and on firms with clean patents
P.UDG_Run('batch3_rd', make_iv_specs('b3', 'rd'), BLS_year_start, Year_end)
P.UDG_Run('batch3_rdcf', make_iv_specs('b3c', 'rdcf'), BLS_year_start, Year_end)
P.RD_ZeroStage('batch3_rd')

# Batch 4: instrument = knowledge spillovers (two patent types: clean and general)
P.KS_Shares('batch4_ks', BLS_year_start, Year_end, bin_len)
P.UDG_Run('batch4_ks', make_iv_specs('b4', 'ks', asinh=True, c3=True, extras=('within',)), BLS_year_start, Year_end)

# Batch 5: asinh(clean patents) on network asinh(clean patents)
P.UDG_Run('batch5_asinh', make_asinh_specs('b5'), BLS_year_start, Year_end)

# Batch 6: industry class distribution x class-level ratio (government clean / all; spillovers clean / general)
for batch, src, srcn, tag in [('batch6_govr', 'govr', 'govrn', 'b6g'), ('batch6_ksr', 'ksr', 'ksrn', 'b6k')]:
    specs = make_iv_specs(tag, src, asinh=(src == 'ksr'), c3=True) + renormalised(tag, srcn)
    P.UDG_Run(batch, specs, BLS_year_start, Year_end)

# Batch 7: aggregate clean shock x sector pre-sample clean share
for batch, src, tag in [('batch7_govcl', 'govcl', 'b7g'), ('batch7_ksagg', 'ksagg', 'b7k')]:
    P.UDG_Run(batch, make_iv_specs(tag, src, asinh=True), BLS_year_start, Year_end)


# ----------------------------------------------------------------

# Diagnostics and summary tables for the main text (Results/Summary).

# ----------------------------------------------------------------

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
