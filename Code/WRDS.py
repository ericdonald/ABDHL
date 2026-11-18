"""""""""""
WRDS Module

Notes: All Wharton Research Data Services (WRDS) downloads of the project, behind one connection. Processor.Cleaner calls these when
    API = 1 and saves the result in Raw Data (as for the other API sources); with API = 0 the saved files are read.

    Credentials: the .keys file in the repository root (the same file as the USPTO key), two lines
        WRDS_USERNAME = XX
        WRDS_PASSWORD = XX
    Connection: WRDS is a PostgreSQL server (wrds-pgdata.wharton.upenn.edu, port 9737, database wrds; each WRDS library is a schema), reached
    here with psycopg2 directly. The wrds package is a thin wrapper around the same connection but pins pandas below 2.3, so it is not used.
    WRDS requires two-factor authentication (Duo): each new connection asks for a push approval on the account's registered device, so the
    Cleaner connects once, on first use, and reuses the connection.

    Compustat                 Fundamentals Annual (comp.funda), the variables the project uses, all years                    -> Raw Data/compustat.csv
    Customer_Segments         Compustat customer segments, one pull per date range                             -> Raw Data/Compustat_Customers/
    Fundamentals_Sales        annual sales, company name, NAICS of every firm, one pull per fiscal-year range  -> Raw Data/Compustat_Customers/
    Company_Names             current and legal names of every gvkey (comp.company)                            -> Raw Data/Compustat_Customers/
    Historical_Names          dated CRSP stock names linked to gvkey (CRSP/Compustat Merged link)              -> Raw Data/Compustat_Customers/

"""""""""""

import sys
import numpy as np
import pandas as pd


HOST, PORT, DBNAME = 'wrds-pgdata.wharton.upenn.edu', 9737, 'wrds'
SEG_WANT = ['gvkey', 'datadate', 'srcdate', 'cid', 'cnms', 'ctype', 'salecs', 'gareac', 'gareat', 'sid', 'stype']
COMPUSTAT_COLS = ['costat', 'curcd', 'datafmt', 'indfmt', 'consol', 'gvkey', 'datadate', 'fic', 'naics', 'sic', 'final', 'fyear',
                  'at', 'ppent', 'sale', 'xrd', 'capxv', 'emp']          # the variables of the file used so far (all 949 would not fit in memory)
CS_NEED = ['fic', 'final', 'at', 'ppent', 'emp', 'capxv', 'sale', 'xrd']     # Compustat variables the Cleaner uses (besides gvkey, fyear, naics)


def prefer_usd(df):
    """One row per firm-year: where Compustat reports a firm-year in both USD and CAD (or other currencies), keep the USD row; firms that
    report only in another currency are kept as they are. Sorted by gvkey and date, so the file order does not depend on the server."""
    out = df.assign(_other=(df.curcd != 'USD')).sort_values('_other', kind='stable').drop_duplicates(['gvkey', 'fyear']).drop(columns='_other')
    return out.sort_values(['gvkey', 'datadate']).reset_index(drop=True)


class WRDS:
    "WRDS connection (opened on first use) and the project's queries"

    def __init__(self, username=None, password=None):
        "username, password: WRDS_USERNAME and WRDS_PASSWORD of the .keys file."
        if not username or not password:
            sys.exit('API = 1 needs WRDS_USERNAME and WRDS_PASSWORD in the .keys file (see README).')
        self.username, self.password = username, password
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            import psycopg2
            print('Waiting for WRDS two-factor authentication: approve the Duo push on your registered device.', flush=True)
            self._conn = psycopg2.connect(host=HOST, port=PORT, dbname=DBNAME, user=self.username, password=self.password, sslmode='require')
            self._conn.set_session(readonly=True, autocommit=True)                    # approve the Duo push when asked
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def sql(self, query, params=None, date_cols=()):
        "Run a query; DataFrame with numeric columns as floats and date_cols as datetimes."
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            names = [d[0] for d in cur.description]
            df = pd.DataFrame.from_records(cur.fetchall(), columns=names, coerce_float=True)
        for c in date_cols:
            df[c] = pd.to_datetime(df[c])
        return df

    def _columns(self, lib, table):
        q = "select column_name from information_schema.columns where table_schema = %s and table_name = %s"
        return [c.lower() for c in self.sql(q, (lib, table)).column_name]

    def _libraries(self):
        return set(self.sql("select schema_name from information_schema.schemata").schema_name)

    def _tables(self, lib):
        return set(self.sql("select table_name from information_schema.tables where table_schema = %s", (lib,)).table_name)

    # ---------------------------------------------------------------------------------------------------------------- Compustat
    def Compustat(self):
        """Fundamentals Annual, all years, with the web query's screens: industrial format only (the web form's "FS" unchecked), standard
        data format, domestic population source, consolidated. Variables: COMPUSTAT_COLS (the Cleaner needs fic, final, at, ppent, emp, capxv,
        sale, xrd and naics; all 949 variables of comp.funda would be several GB). Where a firm-year is reported in several currencies the
        USD row is kept (prefer_usd); firms reporting only in another currency stay. Variables that comp.funda lacks (e.g. NAICS, SIC) come
        from the header table comp.company."""
        f_cols, c_cols = set(self._columns('comp', 'funda')), set(self._columns('comp', 'company'))
        from_f = [c for c in COMPUSTAT_COLS if c in f_cols]
        from_c = [c for c in COMPUSTAT_COLS if c not in f_cols and c in c_cols]
        missing = [c for c in COMPUSTAT_COLS if c not in from_f + from_c]
        if missing:
            sys.exit(f'Compustat variables not found in comp.funda or comp.company: {missing}')
        fund = self.sql(f"select {', '.join(from_f)} from comp.funda where indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'",
                        date_cols=['datadate'])
        n0 = len(fund)
        fund = prefer_usd(fund)
        out = fund
        if from_c:
            out = fund.merge(self.sql(f"select gvkey, {', '.join(from_c)} from comp.company"), on='gvkey', how='left')
        out['gvkey'] = pd.to_numeric(out.gvkey).astype(int)                         # as in the file the Cleaner reads
        out = out[COMPUSTAT_COLS]
        dup = int(out.duplicated(['gvkey', 'fyear']).sum())
        return out

    # ---------------------------------------------------------------------------------------------------------- customer links
    def Customer_Segments(self, d0, d1):
        "Customer segments reported between the dates d0 and d1 ('YYYY-MM-DD')."
        libs = self._libraries()
        found = None
        for lib in ['comp_segments_hist_daily', 'comp_segments_hist', 'compsegd', 'compseg']:
            if lib not in libs:
                continue
            cust = [t for t in sorted(self._tables(lib)) if 'cust' in t.lower()]
            for pref in ['wrds_seg_customer', 'seg_customer']:
                if pref in cust:
                    found = (lib, pref)
                    break
            if found is None and cust:
                found = (lib, cust[0])
            if found:
                break
        if found is None:
            sys.exit('Could not find a customer-segment table. Libraries you can access: '
                     + ', '.join(sorted(l for l in libs if 'seg' in l.lower())))
        lib, table = found
        cols = self._columns(lib, table)
        keep = [c for c in SEG_WANT if c in cols] or cols
        date_col = 'datadate' if 'datadate' in cols else 'srcdate'
        df = self.sql(f"select {', '.join(keep)} from {lib}.{table} where {date_col} between '{d0}' and '{d1}'",
                             date_cols=[c for c in ['datadate', 'srcdate'] if c in keep])
        df['gvkey'] = pd.to_numeric(df.gvkey).astype(int)
        return df

    def Fundamentals_Sales(self, y0, y1):
        "Annual sales, company name and header NAICS / SIC of every firm, fiscal years y0-y1."
        have = set(self._columns('comp', 'funda'))
        hist = [c for c in ['naicsh', 'sich'] if c in have]
        fund = self.sql(f"""
            select gvkey, fyear, datadate, conm, sale, curcd{''.join(', ' + c for c in hist)}
            from comp.funda
            where indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C' and fyear between {y0} and {y1}
        """, date_cols=['datadate'])
        fund = prefer_usd(fund)
        have_c = set(self._columns('comp', 'company'))
        hdr = [c for c in ['naics', 'sic'] if c in have_c]
        if hdr:
            fund = fund.merge(self.sql(f"select gvkey, {', '.join(hdr)} from comp.company"), on='gvkey', how='left')
        fund['gvkey'] = pd.to_numeric(fund.gvkey).astype(int)
        return fund

    def Company_Names(self):
        "Current and legal name of every gvkey."
        have = set(self._columns('comp', 'company'))
        keep = [c for c in ['gvkey', 'conm', 'conml'] if c in have]
        return self.sql(f"select {', '.join(keep)} from comp.company")

    def Historical_Names(self):
        """Dated CRSP stock names linked to gvkey through the CRSP/Compustat Merged link. Segment filings name customers as of the filing
        year ("EXXON CORP", "GEN ELEC"), so dated names match many more of them than current names. Empty if the account has no CRSP."""
        empty = pd.DataFrame(columns=['gvkey', 'comnam', 'namedt', 'nameenddt'])
        if 'crsp' not in self._libraries():
            return empty
        tables = self._tables('crsp')
        link = next((t for t in ['ccmxpf_lnkhist', 'ccmxpf_linktable'] if t in tables), None)
        if 'stocknames' not in tables or link is None:
            return empty
        return self.sql(f"""
            select distinct l.gvkey, s.comnam, s.namedt, s.nameenddt
            from crsp.stocknames s join crsp.{link} l on s.permno = l.lpermno
            where l.linktype in ('LU', 'LC') and l.linkprim in ('P', 'C')
              and s.nameenddt >= l.linkdt and (l.linkenddt is null or s.namedt <= l.linkenddt)
        """, date_cols=['namedt', 'nameenddt'])
