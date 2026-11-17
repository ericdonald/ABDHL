"""""""""""
Customer Data Module

Notes: Compustat customer data and patent tables -> Clean Data tables used by the customer IV (Customer_IV). Called by Processor.Cleaner:
    Collect  (API = 1) downloads the customer data through the WRDS module and saves it in Raw Data/Compustat_Customers/
    Clean    Raw Data -> Clean Data/Cust_*.pkl (large: Cleaner runs on the computing cluster)

"""""""""""

import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path





# ----------------------------------------------------------------------------------------------------------------------
# Customer-name matching: Compustat segment customers are named as they appeared in the filing ("GEN ELEC", "INTL BUS MA")
# ----------------------------------------------------------------------------------------------------------------------



ABBR = {'MTR': 'MOTOR', 'MTRS': 'MOTOR', 'MOTORS': 'MOTOR', 'GEN': 'GENERAL', 'GENL': 'GENERAL', 'INTL': 'INTERNATIONAL',
        'INTERNATL': 'INTERNATIONAL', 'MFG': 'MANUFACTURING', 'ELEC': 'ELECTRIC', 'ELECTR': 'ELECTRIC', 'AMER': 'AMERICAN',
        'AMERN': 'AMERICAN', 'NATL': 'NATIONAL', 'SYS': 'SYSTEMS', 'COMMUN': 'COMMUNICATIONS', 'COMM': 'COMMUNICATIONS',
        'PETE': 'PETROLEUM', 'PETRO': 'PETROLEUM', 'PHARM': 'PHARMACEUTICALS', 'PHARMACEUTICAL': 'PHARMACEUTICALS',
        'LAB': 'LABORATORIES', 'LABS': 'LABORATORIES', 'IND': 'INDUSTRIES', 'INDS': 'INDUSTRIES', 'INDUSTRY': 'INDUSTRIES',
        'SVCS': 'SERVICES', 'SVC': 'SERVICES', 'INSTR': 'INSTRUMENTS', 'INSTRUMENT': 'INSTRUMENTS', 'TECH': 'TECHNOLOGIES',
        'TECHNOLOGY': 'TECHNOLOGIES', 'TEL': 'TELEPHONE', 'TELE': 'TELEPHONE', 'CHEM': 'CHEMICAL', 'CHEMICALS': 'CHEMICAL',
        'STL': 'STEEL', 'AIRCRAFT': 'AIRCRAFT', 'AUTO': 'AUTOMOTIVE', 'AUTOMOTIVE': 'AUTOMOTIVE', 'BROS': 'BROTHERS',
        'MACH': 'MACHINES', 'MACHINE': 'MACHINES', 'BUS': 'BUSINESS', 'CORPN': 'CORP', 'STORE': 'STORES', 'DEPT': 'DEPARTMENT',
        'PROD': 'PRODUCTS', 'PRODS': 'PRODUCTS', 'PRODUCT': 'PRODUCTS', 'ENTRPR': 'ENTERPRISES', 'ENTERPRISE': 'ENTERPRISES',
        'HLDGS': 'HOLDINGS', 'HLDG': 'HOLDINGS', 'GRP': 'GROUP', 'PWR': 'POWER', 'UTILS': 'UTILITIES', 'RES': 'RESOURCES',
        'TRANS': 'TRANSPORTATION', 'COMPUTERS': 'COMPUTER', 'COMPTR': 'COMPUTER', 'ENGR': 'ENGINEERING', 'ENERGY': 'ENERGY'}


NORM_DROP = {'INC', 'CORP', 'CORPORATION', 'CO', 'COMPANY', 'LTD', 'LIMITED', 'LLC', 'PLC', 'LP', 'AG', 'SA', 'NV', 'SE', 'AB',
        'ASA', 'KK', 'THE', 'INCORPORATED', 'HOLDINGS', 'HOLDING', 'GROUP', 'OF', 'AND', 'SPA', 'BV', 'GMBH', 'OYJ', 'NEW', 'OLD',
        'CONSOLIDATED', 'CP', 'TR', 'TRUST', 'USA', 'US', 'ADR'}


JUNK = re.compile(r'^\s*\d+\s+CUSTOMERS?\b|NOT REPORTED|^CUSTOMER\b|^CUSTOMERS\b|^\s*\d+\s+(MAJOR|LARGEST|SIGNIFICANT)|'
                  r'^(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\s+CUSTOMERS|^UNNAMED|^UNDISCLOSED|^VARIOUS|^OTHER\b|'
                  r'^DISTRIBUTOR\b|^ALL OTHER', re.I)


def norm(s):
    s = str(s).upper().replace('&', ' AND ')
    s = re.split(r'\s+-|\(', s)[0]                     # drop ' -ADR', ' -CL A', '(...)' tails (not WAL-MART)
    s = re.sub(r"[^A-Z0-9 ]", ' ', s.replace("'", '').replace('.', ''))
    toks = [ABBR.get(t, t) for t in s.split()]
    toks = [t for t in toks if t not in NORM_DROP]
    return ' '.join(toks)


ALIAS_G = [(r'HEWLETT[- ]?P|^HP INC', 5606), (r'^GEN(ERAL|L)?\.? ?ELEC', 5047),
           (r'INTL BUS|INTERNATIONAL BUSINESS MACH|^IBM\b', 6066), (r'COMPAQ', 3282), (r'MOTOROLA', 7585), (r'^DELL\b', 14489),
           (r'GOOGLE|ALPHABET', 160329), (r'DAIMLER ?-?CHRYSLER|^CHRYSLER CORP|^CHRYSLER$', 3022), (r'FIAT', 15172),
           (r'PHIL(IP)? ?MOR|ALTRIA', 8543), (r'MCDONNEL', 7155), (r'WEST(ER)?N DIG', 11399), (r'PROCT(E)?R', 8762),
           (r'TOYOTA', 19661), (r'HONDA', 5691), (r'NISSAN', 19113), (r'VISTEON', 136648), (r'LUCENT', 62599),
           (r'NORTEL', 7980), (r'HONEYWELL', 1300), (r'UNITED TECH|PRATT ?& ?WHIT', 10983), (r'^SIEMENS', 19349),
           (r'^SONY', 9818), (r'NAVISTAR', 6081), (r'CATERPILLAR', 2817), (r'DEERE', 3835), (r'TESLA', 184996),
           (r'RESEARCH IN MOTION|BLACKBERRY', 65986), (r'^SPRINT', 10984), (r'NIPPON TEL|^NTT\b', 7908),
           (r'WAL.?MART', 11259), (r'WALGREEN', 11264), (r'^CVS\b', 7241), (r'^SEARS', 9563), (r'EXXON', 4503),
           (r'^GEN(ERAL)?\.? ?M(O)?T(O)?R', 5073), (r'^FORD M', 4839), (r'DELPHI', 118122)]


TOK_DROP = {'CO', 'CORP', 'INC', 'LTD', 'THE', 'CORPORATION', 'COMPANY', 'COS', 'PLC', 'LLC', 'LP', 'GROUP', 'HOLDINGS',
        'HLDGS', 'INTERNATIONAL', 'INTL', 'AND'}


def toks(name, keep_intl=True):
    t = norm(name).split()
    drop = TOK_DROP - ({'INTERNATIONAL', 'INTL'} if keep_intl else set())
    return [x for x in t if x not in drop] or t


def subseq(a, b):
    "a is an in-order abbreviation of b with the same first letter"
    if not a or not b or a[0] != b[0]:
        return False
    it = iter(b)
    return all(ch in it for ch in a)


def name_to_gvkey(fs):
    "Normalised Compustat name -> gvkey (the largest-sales gvkey among those sharing a name)."
    nm = fs.groupby('gvkey').agg(conm=('conm', 'last'), sale=('sale', 'sum')).reset_index()
    nm['key'] = nm.conm.map(norm)
    nm = nm[nm.key.str.len() >= 2].sort_values('sale', ascending=False).drop_duplicates('key')
    return nm.set_index('key').gvkey


def match_customer_gvkeys(c, k2g):
    from rapidfuzz import process, fuzz
    names = c.cnms.dropna().unique()
    names = [n for n in names if not JUNK.search(n)]
    keys = pd.Series({n: norm(n) for n in names})
    out = keys.map(k2g)
    nospace = pd.Series(k2g.values, index=k2g.index.str.replace(' ', ''))
    nospace = nospace[~nospace.index.duplicated()]
    m2 = keys.str.replace(' ', '').map(nospace)
    out = out.fillna(m2)
    block = {}
    for k in k2g.index:
        block.setdefault(k.split()[0], []).append(k)
    for n in out.index[out.isna()]:
        k = keys[n]
        cand = block.get(k.split()[0], []) if k else []
        if cand:
            best = process.extractOne(k, cand, scorer=fuzz.token_sort_ratio, score_cutoff=92)
            if best:
                out[n] = k2g[best[0]]
    return out.dropna().astype(int)


def name_table(fs, hist, company):
    "Names of every firm: dated CRSP names (valid between lo and hi) and current / legal / Compustat names."
    h = hist.copy()
    h['lo'] = pd.to_datetime(h.namedt).dt.year
    h['hi'] = pd.to_datetime(h.nameenddt).dt.year
    h = h.rename(columns={'comnam': 'name'})[['gvkey', 'name', 'lo', 'hi']]
    c = company
    cur = pd.concat([c[['gvkey', 'conm']].rename(columns={'conm': 'name'}), c[['gvkey', 'conml']].rename(columns={'conml': 'name'}),
                     fs[['gvkey', 'conm']].astype({'gvkey': str}).rename(columns={'conm': 'name'})]).dropna().drop_duplicates()
    cur['lo'], cur['hi'] = 1900, 2100
    h['dated'], cur['dated'] = True, False
    t = pd.concat([h, cur], ignore_index=True).dropna(subset=['name'])
    t['gvkey'] = t.gvkey.astype(int)
    t['key'] = t.name.map(lambda n: ' '.join(toks(n)))
    t = t[t.key.str.len() >= 2]
    size = fs.groupby('gvkey').sale.max()
    t['size'] = t.gvkey.map(size).fillna(0)
    return t


def match(cs, fs, hist, company):
    "Match customer names to gvkeys: exact (dated), exact (current), abbreviation, fuzzy, then a hand list that overrides."
    t = name_table(fs, hist, company)
    pairs = cs[['cnms', 'fy']].dropna().drop_duplicates()
    pairs = pairs[~pairs.cnms.map(lambda x: bool(JUNK.search(x)))]
    pairs['key'] = pairs.cnms.map(lambda n: ' '.join(toks(n)))
    out, how = {}, {}
    # 1-2: exact keys
    ex = pairs.merge(t, on='key', how='inner')
    ex['valid'] = ex.lo.le(ex.fy) & ex.hi.ge(ex.fy)
    ex = ex.sort_values(['valid', 'dated', 'size'], ascending=False).drop_duplicates(['cnms', 'fy'])
    for r in ex.itertuples():
        out[(r.cnms, r.fy)] = r.gvkey; how[(r.cnms, r.fy)] = 'exact dated' if r.valid and r.dated else 'exact'
    # 3: abbreviation match on what is left
    rest = pairs[[(a, b) not in out for a, b in zip(pairs.cnms, pairs.fy)]]
    tt = t.assign(tk=t.key.str.split())
    block = {}
    for i, tk in zip(tt.index, tt.tk):
        block.setdefault(tk[0][0] + (tk[0][1] if len(tk[0]) > 1 else ''), []).append(i)
    T = tt[['gvkey', 'lo', 'hi', 'dated', 'size', 'tk']].to_dict('index')
    cand_of = {}
    for key in rest.key.unique():                      # candidates computed once per distinct name
        sk = key.split()
        if not sk or (len(sk) == 1 and len(sk[0]) < 5):
            continue
        hits = []
        for i in block.get(sk[0][0] + (sk[0][1] if len(sk[0]) > 1 else ''), []):
            c = T[i]['tk']
            if len(c) >= len(sk) and all(subseq(x, y) for x, y in zip(sk, c)):
                hits.append((T[i]['lo'], T[i]['hi'], T[i]['size'], T[i]['gvkey']))
        if hits:
            cand_of[key] = hits
    for cn, fy, key in zip(rest.cnms, rest.fy, rest.key):
        hits = cand_of.get(key)
        if hits:
            best = max(hits, key=lambda h: (h[0] <= fy <= h[1], h[2]))
            out[(cn, fy)] = best[3]; how[(cn, fy)] = 'abbreviation'
    m = pd.Series(out)
    m.index.names = ['cnms', 'fy']
    cs = cs.set_index(['cnms', 'fy'])
    cs['cust'] = m.reindex(cs.index).to_numpy()
    cs['how'] = pd.Series(how).reindex(cs.index).to_numpy()
    cs = cs.reset_index()
    # 4: explore30's fuzzy match for names still unmatched; hand list overrides
    left = cs.cust.isna()
    fz = match_customer_gvkeys(cs[left], name_to_gvkey(fs))
    cs.loc[left, 'cust'] = cs.loc[left, 'cnms'].map(fz)
    cs.loc[left & cs.cust.notna(), 'how'] = 'fuzzy'
    names = pd.Series(cs.cnms.dropna().unique())
    for pat, gv in ALIAS_G:
        hit = set(names[names.str.upper().map(lambda x: bool(re.search(pat, x)))])
        sel = cs.cnms.isin(hit)
        cs.loc[sel, 'cust'] = gv; cs.loc[sel, 'how'] = 'hand list'
    return cs



# ----------------------------------------------------------------------------------------------------------------------
# Patents, firms, customer links
# ----------------------------------------------------------------------------------------------------------------------
def patent_tables(raw, cpc, xw, clean_codes, ice_codes, last=2017):
    """Green flag, filing year and non-Y subclass split of every patent filed 1976-last.
    cpc: the PatentsView CPC table (Raw Data/CPC.pkl, as loaded by the Cleaner). Returns (class_year, crosswalk_patents):
      class_year  class x filing year x in-crosswalk: green and total patents (a patent is split equally over its non-Y subclasses)
      crosswalk   dict(pt=patent_id, green flag, filing year; sub=patent_id, subclass, split weight) for patents in the crosswalk
    A patent is green if any of its CPC codes is a clean code, except the internal-combustion groups."""
    app = pd.read_pickle(raw / 'applications.pkl')[['patent_id', 'year']].drop_duplicates('patent_id')
    app = app[app.year.between(1976, last)]
    cpc = cpc[['patent_id', 'cpc_class', 'cpc_subclass', 'cpc_group']]
    cpc = cpc[cpc.patent_id.isin(set(app.patent_id))].copy()
    codes, ice = set(clean_codes), set(ice_codes)
    g = cpc.cpc_group.astype(str)
    cpc['cl'] = (cpc.cpc_class.isin(codes) | cpc.cpc_subclass.isin(codes) | g.isin(codes) | g.str[:5].isin(codes)
                 | g.str[:6].isin(codes))
    cpc['ice'] = g.isin(ice)
    fl = cpc.groupby('patent_id')[['cl', 'ice']].max()
    clean = (fl.cl.astype(int) - fl.ice.astype(int)).clip(lower=0).rename('clean')
    sub = cpc.loc[~cpc.cpc_subclass.astype(str).str.startswith('Y'), ['patent_id', 'cpc_subclass']].drop_duplicates()
    sub['f'] = 1 / sub.groupby('patent_id').cpc_subclass.transform('count')
    ids = set(xw.patent_id)
    pt = pd.DataFrame({'clean': clean}).join(app.set_index('patent_id'), how='inner').reset_index()
    pt = pt[pt.patent_id.isin(ids)]
    cw_sub = sub[sub.patent_id.isin(ids)].copy()
    sub = sub.join(clean, on='patent_id').merge(app, on='patent_id')
    sub['inx'] = sub.patent_id.isin(ids)
    sub['c'] = sub.f * sub.clean
    cy = sub.groupby(['cpc_subclass', 'year', 'inx']).agg(clean=('c', 'sum'), all=('f', 'sum')).reset_index()
    return cy, dict(pt=pt, sub=cw_sub)


def firm_table(raw, clean, xw, pt):
    """One row per Compustat gvkey: country of incorporation (fic), first reported NAICS (fiscal years up to 2022 and any year), and
    the share of the firm's crosswalk patents filed 1993-2017 whose first inventor is U.S.-based (inventors in Inventor_Locations are
    U.S.-located only, so a patent missing from it has a foreign first inventor)."""
    comp = pd.read_csv(raw / 'compustat.csv', low_memory=False, usecols=['gvkey', 'fyear', 'fic', 'naics'])
    comp['naics'] = pd.to_numeric(comp.naics, errors='coerce')
    assert comp.dropna(subset=['fic']).groupby('gvkey').fic.nunique().max() == 1, 'fic varies within gvkey'
    late = comp[comp.fyear <= 2022]
    firms = pd.DataFrame({'fic': comp.dropna(subset=['fic']).drop_duplicates('gvkey').set_index('gvkey').fic})
    firms['naics'] = late.dropna(subset=['naics']).drop_duplicates('gvkey').set_index('gvkey').naics
    firms['naics_any'] = comp.dropna(subset=['gvkey', 'naics']).drop_duplicates('gvkey').set_index('gvkey').naics
    IL = pd.read_pickle(clean / 'Inventor_Locations.pkl')[['patent_id', 'inventor_sequence']]
    first_us = set(IL.loc[IL.inventor_sequence == 0, 'patent_id'])
    del IL
    XP = xw.merge(pt, on='patent_id')
    us = XP[XP.year.between(1993, 2017)].assign(us=lambda d: d.patent_id.isin(first_us)).groupby('gvkey').us.mean()
    firms['us_share'] = us
    return firms.reset_index()


def load_customers(pulls):
    "Customer segments (COMPANY customers with positive sales) and firm sales, fiscal years 1976-2022."
    cs = pd.concat([pd.read_csv(pulls / 'customer_segments_1976_1989.csv', dtype={'gvkey': str}),
                    pd.read_csv(pulls / 'customer_segments.csv', dtype={'gvkey': str})], ignore_index=True)
    cs = cs[(cs.ctype == 'COMPANY') & (cs.salecs > 0)].copy()
    cs['gvkey'] = cs.gvkey.astype(int)
    cs['fy'] = pd.to_datetime(cs.srcdate).dt.year
    cs = cs.drop_duplicates(['gvkey', 'srcdate', 'cnms', 'salecs'])
    fs = pd.concat([pd.read_csv(pulls / 'funda_sales_1976_1989.csv', dtype={'gvkey': str}),
                    pd.read_csv(pulls / 'funda_sales.csv', dtype={'gvkey': str})], ignore_index=True)
    fs['gvkey'] = fs.gvkey.astype(int)
    fs = fs.drop_duplicates(['gvkey', 'fyear'])
    return cs, fs


# ----------------------------------------------------------------------------------------------------------------------
# Workflow (called by Processor.Cleaner)
# ----------------------------------------------------------------------------------------------------------------------
SEGMENT_FILES = {'customer_segments_1976_1989.csv': ('1976-01-01', '1989-12-31'), 'customer_segments.csv': ('1990-01-01', '2022-12-31')}
FUNDA_FILES = {'funda_sales_1976_1989.csv': (1976, 1989), 'funda_sales.csv': (1990, 2022)}


class Customer_Data:
    "Compustat customer data and patent tables for the customer IV"

    def __init__(self, P):
        "P: Processor instance (project directory, CPC code lists)."
        self.P = P
        self.Raw = P.Directory / 'Raw Data'
        self.CleanDir = P.Directory / 'Clean Data'
        self.Pulls = self.Raw / 'Compustat_Customers'

    def Collect(self, API, wrds=None):
        """
        With API = 1, download the Compustat customer data through the WRDS module and save it in Raw Data/Compustat_Customers/; with
        API = 0 the saved files are used by Clean.

        Output: Raw Data/Compustat_Customers/customer_segments_1976_1989.csv, customer_segments.csv, funda_sales_1976_1989.csv,
                funda_sales.csv, company_names.csv, historical_names.csv
        """
        if API != 1:
            return
        self.Pulls.mkdir(parents=True, exist_ok=True)
        for name, (d0, d1) in SEGMENT_FILES.items():
            wrds.Customer_Segments(d0, d1).to_csv(self.Pulls / name, index=False)
        for name, (y0, y1) in FUNDA_FILES.items():
            wrds.Fundamentals_Sales(y0, y1).to_csv(self.Pulls / name, index=False)
        wrds.Company_Names().to_csv(self.Pulls / 'company_names.csv', index=False)
        wrds.Historical_Names().to_csv(self.Pulls / 'historical_names.csv', index=False)

    def Clean(self, xwalk, cpc):
        """
        Clean the customer-IV data. Called by Processor.Cleaner once the patent-firm crosswalk and the CPC table are built; also reads
        Raw Data/applications.pkl and compustat.csv, and Clean Data/Inventor_Locations.pkl.

        xwalk: the Cleaner's union of the three patent-firm crosswalks (patent_id, gvkey), before any Compustat filter.
        cpc:   PatentsView CPC codes (Raw Data/CPC.pkl).

        Output: Clean Data/Cust_Xwalk.pkl          patent_id, gvkey (integer): the crosswalk union
                Clean Data/Cust_Patents.pkl        dict(pt, sub): green flag, filing year and subclass split of crosswalk patents
                Clean Data/Cust_Class_Year.pkl     class x filing year x in-crosswalk: green and total patents (outside patents give the shifts)
                Clean Data/Cust_Firms.pkl          gvkey: fic, NAICS, share of patents with a U.S.-based first inventor
                Clean Data/Cust_Segments.pkl       customer segments with the customer's gvkey (cust) and how it was matched
                Clean Data/Cust_Sales.pkl          gvkey, fiscal year, sales
        """
        xw = xwalk[['patent_id', 'gvkey']].dropna().copy()
        xw['gvkey'] = pd.to_numeric(xw.gvkey).astype(int)
        xw = xw.drop_duplicates()
        print(f'crosswalk: {xw.patent_id.nunique():,} patents, {xw.gvkey.nunique():,} gvkeys', flush=True)
        cy, cwp = patent_tables(self.Raw, cpc, xw, self.P.CPC_classes['clean'], self.P.ICE_classes)
        xw.to_pickle(self.CleanDir / 'Cust_Xwalk.pkl')
        pd.to_pickle(cwp, self.CleanDir / 'Cust_Patents.pkl')
        cy.to_pickle(self.CleanDir / 'Cust_Class_Year.pkl')
        firm_table(self.Raw, self.CleanDir, xw, cwp['pt']).to_pickle(self.CleanDir / 'Cust_Firms.pkl')
        cs, fs = load_customers(self.Pulls)
        hist = pd.read_csv(self.Pulls / 'historical_names.csv', dtype={'gvkey': str})
        company = pd.read_csv(self.Pulls / 'company_names.csv', dtype={'gvkey': str})
        cs = match(cs, fs, hist, company)
        for lo, hi in [(1976, 1989), (1990, 1997), (1998, 2017)]:
            x = cs[cs.fy.between(lo, hi)]
            m = x.cust.notna()
            print(f'FY{lo}-{hi}: customer sales matched {x.loc[m, "salecs"].sum() / x.salecs.sum():.0%} (rows {m.mean():.0%})', flush=True)
        cs['cust'] = cs.cust.astype('Int64')
        cs[['gvkey', 'fy', 'cnms', 'salecs', 'cust', 'how']].to_pickle(self.CleanDir / 'Cust_Segments.pkl')
        fs[['gvkey', 'fyear', 'sale']].to_pickle(self.CleanDir / 'Cust_Sales.pkl')
