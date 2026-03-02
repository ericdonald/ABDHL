"""""""""""
Processing Functions

Notes: Functions that accomplish basic processing for the project.
    
"""""""""""

import numpy as np
import pandas as pd
import requests, zipfile, io



def Extract_PatentsView(Table):
    "Download and Extract a PatentsView Bulk Table"
    
    url = f"https://s3.amazonaws.com/data.patentsview.org/download/{Table}.tsv.zip"

    r = requests.get(url, stream=True)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        tsv_name = f"{Table}.tsv"
        with z.open(tsv_name) as f:
            df = pd.read_csv(f, sep="\t", low_memory=False)
    
    return df
    
    

def clean_naics_str(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s



def split_comma_list(s):
    if pd.isna(s):
        return []
    parts = [p.strip() for p in str(s).split(',')]
    return [p for p in parts if p]



def children_in_universe(prefix, universe):
    if pd.isna(prefix):
        return []
    prefix = str(prefix)
    return [code for code in universe if code.startswith(prefix)]



def expand_bls_row_to_6(row, naics2022_6_universe):
    prefix = row['naics_prefix']
    return children_in_universe(prefix, naics2022_6_universe)



def map_naics2012_to_2022_6(code2012, naics2012_6_universe, NAICS_2012_2017_df, NAICS_2017_2022_df):
    code2012 = clean_naics_str(code2012)
    if pd.isna(code2012):
        return set()

    # Step 1: find 2012 6-digit children
    children_2012 = children_in_universe(code2012, naics2012_6_universe)
    if not children_2012:
        return set()

    # Step 2: 2012 → 2017 (6-digit)
    cw_12_subset = NAICS_2012_2017_df[NAICS_2012_2017_df['NAICS_2012'].isin(children_2012)]
    naics2017_6 = cw_12_subset['NAICS_2017'].dropna().unique()

    if len(naics2017_6) == 0:
        return set()

    # Step 3: 2017 → 2022 (6-digit)
    cw_17_subset = NAICS_2017_2022_df[NAICS_2017_2022_df['NAICS_2017'].isin(naics2017_6)]
    naics2022_6 = cw_17_subset['NAICS_2022'].dropna().unique()

    return set(naics2022_6)





