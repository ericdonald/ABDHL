"""""""""""
Archive: instrument-search sections of Executor (tentatively dropped)

Notes: Spec builders (make_iv_specs, make_cur_specs, renormalised), the batches 2-4, 6, 7 and 9 that ran them through Processor.UDG_Run, the
    shock-relevance / first-stage diagnostics and the instrument summary tables, as they stood in Executor.py. Kept as text, in the original
    order; nothing imports this file. The methods they call are in Archive/Processor_IV_Search.py.

    Restore:  git checkout pre-customer-iv-prune -- Code/Processor.py Code/Executor.py

"""""""""""

# flags (Executor): DIAG, TYP, MG = presence of the Cleaner's own-technology, type-specific and six-digit files

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

# six-digit CPC are used only if the Cleaner files and the shocks have been built (see the build section)

DIAG = (P.Directory / 'Clean Data' / 'KS_Shocks_cg_Diag.pkl').exists()   # spillovers with the own-technology term

TYP = (P.Directory / 'Clean Data' / 'Govt_Shocks_Type.pkl').exists()     # type-specific government series

MG = all((P.Directory / 'Clean Data' / f).exists() for f in ['Govt_Shocks_mg.pkl', 'KS_Shocks_cg_mg.pkl', 'KS_Recv_cg_mg.pkl'])

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

def renormalised(tag, src_n):
    "Limited set with the class-level ratio renormalised over classes where it is defined (ratio instruments)."
    out = []
    for sp in make_iv_specs(tag, src_n, c3=True):
        if sp['short'] in ('base', 'updown', 'c3', 'c3ud'):
            nm = 'rn' + sp['short']
            out.append({**sp, 'id': sp['id'].rsplit('_', 1)[0] + '_' + nm, 'short': nm, 'panel': 'D',
                        'label': 'Renormalised over classes with a ratio: ' + sp['label']})
    return out

# Batch 2: instrument = government patents
P.UDG_Run('batch2_gov', make_iv_specs('b2', 'gov', asinh=True, c3=True, extras=('lag', 'direct') + (('type',) if TYP else ()), mg=MG), BLS_year_start, Year_end)

# Batch 3: instrument = state R&D prices; zero stage; and on firms with clean patents
P.UDG_Run('batch3_rd', make_iv_specs('b3', 'rd'), BLS_year_start, Year_end)

P.UDG_Run('batch3_rdcf', make_iv_specs('b3c', 'rdcf'), BLS_year_start, Year_end)

P.RD_ZeroStage('batch3_rd')

# Batch 4: instrument = knowledge spillovers (two patent types: clean and general)
P.KS_Shares('batch4_ks', BLS_year_start, Year_end, bin_len)

P.UDG_Run('batch4_ks', make_iv_specs('b4', 'ks', asinh=True, c3=True, extras=('within', 'direct') + (('diag',) if DIAG else ()), mg=MG), BLS_year_start, Year_end)

# Batch 6: industry class distribution x class-level ratio (government clean / all; spillovers clean / general)
for batch, src, srcn, tag in [('batch6_govr', 'govr', 'govrn', 'b6g'), ('batch6_ksr', 'ksr', 'ksrn', 'b6k')]:
    specs = make_iv_specs(tag, src, asinh=(src == 'ksr'), c3=True, mg=MG) + renormalised(tag, srcn)
    P.UDG_Run(batch, specs, BLS_year_start, Year_end)

# Batch 7: aggregate clean shock x sector pre-sample clean share
for batch, src, tag in [('batch7_govcl', 'govcl', 'b7g'), ('batch7_ksagg', 'ksagg', 'b7k')]:
    P.UDG_Run(batch, make_iv_specs(tag, src, asinh=True), BLS_year_start, Year_end)

if (P.Directory / 'Clean Data' / 'Govt_Shocks_Cur.pkl').exists():
    P.UDG_Run('batch9_cur', make_cur_specs('b9', mg=MG), BLS_year_start, Year_end)   # contemporaneous class distribution in the instruments

P.Shock_Relevance('diag_shock_relevance', BLS_year_start, Year_end, bin_len)

if MG:
    P.Shock_Relevance('diag_shock_relevance_mg', BLS_year_start, Year_end, bin_len, sfx='_mg')   # does the industry-type shock predict patenting of that type?

P.Shock_Relevance('diag_shock_relevance_cur', BLS_year_start, Year_end, bin_len, cur=True)   # contemporaneous class distribution

if MG:
    P.Shock_Relevance('diag_shock_relevance_cur_mg', BLS_year_start, Year_end, bin_len, sfx='_mg', cur=True)

P.Shock_Relevance_Type('diag_shock_relevance_type', BLS_year_start, bin_len)   # type-level gut check

P.Shock_Relevance_Class('diag_shock_relevance_class', BLS_year_start, bin_len)   # type = 3-digit CPC class

if MG:
    P.Shock_Relevance_Type('diag_shock_relevance_type_mg', BLS_year_start, bin_len, sfx='_mg')   # six-digit CPC
    P.Shock_Relevance_Class('diag_shock_relevance_class_mg', BLS_year_start, bin_len, sfx='_mg')

P.UDG_OwnFirstStage('diag_first_stage', BLS_year_start, Year_end, ['gov', 'rd', 'rdcf', 'ks', 'kswt', 'govr', 'ksr', 'govcl', 'ksagg'])

P.Shock_Relevance_Summary(P.Directory / 'Results' / 'Summary' / 'table_new_relevance.tex')

if MG:
    P.Shock_Relevance_Summary(P.Directory / 'Results' / 'Summary' / 'table_new_relevance_mg.tex', sfx='_mg')   # six-digit CPC version

P.New_IV_Variants(P.Directory / 'Results' / 'Summary' / 'table_new_diag.tex', 'diag')

P.New_IV_Variants(P.Directory / 'Results' / 'Summary' / 'table_new_type.tex', 'type')

P.New_IV_Variants(P.Directory / 'Results' / 'Summary' / 'table_new_gov6.tex', 'gov6')

P.New_IV_Variants(P.Directory / 'Results' / 'Summary' / 'table_new_direct.tex', 'direct')

P.New_IV_Variants(P.Directory / 'Results' / 'Summary' / 'table_new_cur.tex', 'cur')

P.New_IV_Variants(P.Directory / 'Results' / 'Summary' / 'table_new_curds.tex', 'curds')

P.UDG_InstrumentSummary(P.Directory / 'Results' / 'Summary' / 'table_iv_summary.tex',
                        [('Government patents (B2)', 'b2'), ('State R\\&D prices, all firms (B3)', 'b3'),
                         ('State R\\&D prices, clean firms (B3)', 'b3c'), ('Spillovers, levels (B4)', 'b4'),
                         ('Government ratio (B6)', 'b6g'), ('Spillover ratio (B6)', 'b6k'),
                         ('Aggregate government clean patenting (B7)', 'b7g'), ('Aggregate clean spillovers (B7)', 'b7k')])
