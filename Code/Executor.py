"""""""""""
Executor

Notes: This file executes the code for "Transition to Green Technology along the Supply Chain".
    
"""""""""""

import Processor as p


# ----------------------------------------------------------------

# Define project objects.

# ----------------------------------------------------------------

P = p.Processor()
BLS_year_start, Year_start, Year_mid, Year_end = (1997, 2012, 2017, 2022)
bin_len = 5

# ----------------------------------------------------------------

# Run project methods.

# ----------------------------------------------------------------

# ---------- #
# Clean Data #
# ---------- #
API = 0
#Set to 1 for new API download

#P.Cleaner(BLS_year_start, Year_start, Year_end, bin_len, API)


# ----------------- #
# Build Instruments #
# ----------------- #
#P.Instruments()


# ---------------- #
# IO Change Graphs #
# ---------------- #
#P.IO_Change(Year_start, Year_mid, Year_end)


# --------------------- #
# Directional Incentive #
# --------------------- #
#P.Up_Down_Green(BLS_year_start, Year_end, bin_len)   # legacy; superseded by UDG_Run below


# ------------------------------ #
# Specification batches (UDG_Run) #
# ------------------------------ #
# Each spec overrides Processor.UDG_BASE (see CLAUDE.md for the baseline).
def make_specs(tag, share, extra_offsets=()):
    "The standard permutation set around the baseline, for patents and citations."
    specs = []
    for m in ['pat', 'cite']:
        suf = 'count' if m == 'pat' else 'cites'
        off_all = f'pat_{suf}'
        def add(name, label, **kw):
            specs.append(dict(id=f'{tag}_{m}_{name}', short=name, measure=m, share=share, label=label, **kw))
        add('base',     'Baseline: no own lag', panel='A')
        add('updown',   'Separate up/down', direction='updown', panel='A')
        add('norm',     'Normalised weights, net', normalise=True, panel='B')
        add('normud',   'Normalised weights, up/down', normalise=True, direction='updown', panel='B')
        add('bin3',     'Three-year bins', bin_len=3, panel='B')
        plc = 'P' if share == 'G' else 'PD'
        plc_lab = 'dirty/all' if share == 'G' else 'dirty/(clean+dirty)'
        add('plc',      f'Placebo: {plc_lab}, net', rhs=plc, panel='D')
        add('plcud',    f'Placebo: {plc_lab}, up/down', rhs=plc, direction='updown', panel='D')
        add('plcw',     f'Placebo: {plc_lab} + sum of weights, net', rhs=plc, controls=['W_w'], panel='D')
        add('plcwud',   f'Placebo: {plc_lab} + sum of weights, up/down', rhs=plc, controls=['W_w'], direction='updown', panel='D')
        add('em',       'Emissions reduction, net', rhs='E', entity_fe=False, panel='D')
        add('emud',     'Emissions reduction, up/down', rhs='E', entity_fe=False, direction='updown', panel='D')
        add('ownlag',   'Own lagged share', own_lag=True, panel='A')
        add('noshr',    'No kappa shrinkage, net', shrink=False, panel='B')
        add('noshrud',  'No kappa shrinkage, up/down', shrink=False, direction='updown', panel='B')
        add('wsum',     'Control: sum of weights, net', controls=['W_w'], panel='A')
        add('wsumud',   'Control: sum of weights, up/down', controls=['W_w'], direction='updown', panel='A')
        add('offall',   'Offset: all patents, net', offset=off_all, panel='C')
        add('offallud', 'Offset: all patents, up/down', offset=off_all, direction='updown', panel='C')
        for name, col in extra_offsets:
            add(f'{name}',   f'Offset: {name}, net', offset=f'{col}_{suf}', panel='C')
            add(f'{name}ud', f'Offset: {name}, up/down', offset=f'{col}_{suf}', direction='updown', panel='C')
    return specs


# ---------------------------------------------- #
# Batch 0: G = clean / all patents               #
# ---------------------------------------------- #
batch = 'batch0_base'
specs = make_specs('b0', 'G')
# Diagnostics on the opposite up/down pattern in counts vs citations
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


# ------------------------------------------------------ #
# Batch 1: D = clean / (clean + dirty), offset = dirty    #
# ------------------------------------------------------ #
batch = 'batch1_D'
specs = make_specs('b1', 'D', extra_offsets=[('offcd', 'clim_pat')])
P.UDG_Run(batch, specs, BLS_year_start, Year_end)


# ----------------------- #
# Record Package Versions #
# ----------------------- #
packages = ["linearmodels", "matplotlib", "numpy", "openpyxl", "pandas", "scipy", "statsmodels"]
P.write_package_versions(packages)




