"""""""""""
Processor Module

Notes: This file defines a class for processing the workflow of "Transition to Green Technology along the Supply Chain".

"""""""""""

import numpy as np
import pandas as pd
import io, sys
from pathlib import Path
import requests as api
import importlib.metadata as md



class Processor:
    "Object Processing Workflow"
    
    def __init__(self):
        "Initialize Processor Object"
        
        self.Directory = Path(__file__).resolve().parent.parent
        self.CO2e = {'Carbon dioxide': 1,
                     'Methane': 28,
                     'Nitrous oxide': 273,
                     'Carbon tetrafluoride': 7390,
                     'Hexafluoroethane': 12200,
                     'HFC-125': 3500,
                     'HFC-134a': 1430,
                     'HFC-143a': 4470,
                     'HFC-23': 14800,
                     'HFC-236fa': 9810,
                     'HFC-32': 675,
                     'Nitrogen trifluoride': 17200,
                     'Perfluorocyclobutane': 10300,
                     'Perfluoropropane': 8830,
                     'Sulfur hexafluoride': 22800
                     }

        
        
    def IO_Change(self, Year_start, Year_end):
        """""
        Plot of Changes in IO Network from Decarbonization
    
        Output: Results/core_versions.txt
        """""
        
        # ----------------------------------------------------------------

        # Unpack data sets.

        # ----------------------------------------------------------------

        # ------------ #
        # BLS IO Table #
        # ------------ #
        USE_start_df = pd.read_excel(f'{self.Directory}/Raw Data/REAL_USE.xlsx', sheet_name=f"{Year_start}")
        MAKE_start_df = pd.read_excel(f'{self.Directory}/Raw Data/REAL_MAKE.xlsx', sheet_name=f"{Year_start}")
        
        USE_end_df = pd.read_excel(f'{self.Directory}/Raw Data/REAL_USE.xlsx', sheet_name=f"{Year_end}")
        MAKE_end_df = pd.read_excel(f'{self.Directory}/Raw Data/REAL_MAKE.xlsx', sheet_name=f"{Year_end}")
        #NAICS 2022
        
        BLS_Crosswalk_df = pd.read_excel(f'{self.Directory}/Raw Data/BLS_Crosswalk.xlsx', sheet_name="Stubs")
        BLS_Crosswalk_df['BLS_Industry'] = BLS_Crosswalk_df['Sector Number']
        
        # ----------------------- #
        # EPA Emissions by Sector #
        # ----------------------- #
        EPA_url = "https://pasteur.epa.gov/uploads/10.23719/1529805/GHGs_by_Detailed_Sector_US_2012-2020.xlsx"

        EPA_df = pd.read_excel(EPA_url, sheet_name="Main")    
        
        EPA_df = EPA_df[EPA_df['Flowable'].isin(self.CO2e.keys())].copy()
        EPA_df['GWP'] = EPA_df['Flowable'].map(self.CO2e)
        EPA_df['FlowAmount_CO2e'] = EPA_df['FlowAmount'] * EPA_df['GWP']
        
        EPA_df['CO2e'] = EPA_df.groupby(['Sector', 'Year'])['FlowAmount_CO2e'].transform("sum")
        EPA_df = EPA_df[['Sector', 'Year', 'CO2e']].drop_duplicates()
        EPA_df = EPA_df[EPA_df["Year"].isin([Year_start, Year_end])]
        #NAICS 2012


        # ---------------- #
        # NAICS Crosswalks #
        # ---------------- #
        NAICS_2012_2017_url = "https://www.census.gov/naics/concordances/2012_to_2017_NAICS.xlsx"
        NAICS_2017_2022_url = "https://www.census.gov/naics/concordances/2017_to_2022_NAICS.xlsx"
        headers = {"User-Agent": "Mozilla/5.0"}

        r = api.get(NAICS_2012_2017_url, headers=headers)
        NAICS_2012_2017_df = pd.read_excel(io.BytesIO(r.content), skiprows=2)
        
        r = api.get(NAICS_2017_2022_url, headers=headers)
        NAICS_2017_2022_df = pd.read_excel(io.BytesIO(r.content), skiprows=2)


        # ----------------------------------------------------------------

        # Build industry IO matrix.

        # ----------------------------------------------------------------
        
        # --------------------------------------- #
        # Industry x Commodity Expenditure Shares #
        # --------------------------------------- #
        
        U_start_rev = USE_start_df.iloc[:, 1:-3].to_numpy()
        ind_Y_start = np.sum(U_start_rev, 0)
        B_start = (U_start_rev[:-3,:] @ np.diag(ind_Y_start**(-1))).T
        
        U_end_rev = USE_end_df.iloc[:, 1:-3].to_numpy()
        ind_Y_end = np.sum(U_end_rev, 0)
        B_end = (U_end_rev[:-3,:] @ np.diag(ind_Y_end**(-1))).T

        
        # -------------------------------------- #
        # Commodity x Industry Production Shares #
        # -------------------------------------- #
        
        M_start_rev = MAKE_start_df.iloc[:, 1:].to_numpy()
        com_Y_start = np.sum(M_start_rev, 0)[:-2]
        A_start = (M_start_rev[:-2,:-2] @ np.diag(com_Y_start**(-1))).T
        
        M_end_rev = MAKE_end_df.iloc[:, 1:].to_numpy()
        com_Y_end = np.sum(M_end_rev, 0)[:-2]
        A_end = (M_end_rev[:-2,:-2] @ np.diag(com_Y_end**(-1))).T
        
        
        # ------------------- #
        # Input-Output Matrix #
        # ------------------- #
        
        IO_start = B_start @ A_start
        IO_end = B_end @ A_end
        J = IO_start.shape[0]
        
        diff = np.abs(IO_end - IO_start)
        tv_by_industry = 0.5 * diff.sum(axis=1)
        
        IO_df = pd.DataFrame({
            "BLS_Industry": np.arange(1, J+1),
            "TV_distance": tv_by_industry})
        
        df_VA_start = pd.DataFrame({"BLS_Industry": np.arange(1, J+1),
                                    "Value_Added": ind_Y_start,
                                    "Year": Year_start})

        df_VA_end = pd.DataFrame({"BLS_Industry": np.arange(1, J+1),
                                  "Value_Added": ind_Y_end,
                                  "Year": Year_end})

        VA_panel = pd.concat([df_VA_start, df_VA_end], ignore_index=True)
        
        
        # ----------------------------------------------------------------

        # Make unified NAICS mapping.

        # ----------------------------------------------------------------
        
        # ---------------- #
        # Helper Functions #
        # ---------------- #
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
        
        
        def expand_bls_row_to_6(row):
            prefix = row['naics_prefix']
            return children_in_universe(prefix, naics2022_6_universe)
        
        
        def map_naics2012_to_2022_6(code2012):
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
        
        # -------------------- #
        # Clean for Comparison #
        # -------------------- #
        EPA_df['EPA_Sector'] = EPA_df['Sector'].apply(clean_naics_str)
        
        NAICS_2012_2017_df['NAICS_2012'] = NAICS_2012_2017_df['2012 NAICS Code'].apply(clean_naics_str)
        NAICS_2012_2017_df['NAICS_2017'] = NAICS_2012_2017_df['2017 NAICS Code'].apply(clean_naics_str)
        
        NAICS_2017_2022_df['NAICS_2017'] = NAICS_2017_2022_df['2017 NAICS Code'].apply(clean_naics_str)
        NAICS_2017_2022_df['NAICS_2022'] = NAICS_2017_2022_df['2022 NAICS Code'].apply(clean_naics_str)

        BLS_Crosswalk_df['NAICS_2022'] = BLS_Crosswalk_df['NAICS_2022'].apply(clean_naics_str)
        
        naics2012_6_universe = sorted(NAICS_2012_2017_df['NAICS_2012'].dropna().unique())
        naics2017_6_universe = sorted(NAICS_2012_2017_df['NAICS_2017'].dropna().unique())
        naics2022_6_universe = sorted(NAICS_2017_2022_df['NAICS_2022'].dropna().unique())
        
        
        # ---------- #
        # Expand BLS #
        # ---------- #
        BLS_long = (BLS_Crosswalk_df
                    .assign(naics_code_list=lambda x: x['NAICS_2022'].apply(split_comma_list))
                    .explode('naics_code_list')
                    .rename(columns={'naics_code_list': 'naics_prefix'}))
        
        BLS_long['naics_prefix'] = BLS_long['naics_prefix'].apply(clean_naics_str)
        
       
        bls_expanded_rows = []
        for _, row in BLS_long.iterrows():
            bls_id = row['BLS_Industry']  # adjust column name
            children = expand_bls_row_to_6(row)
            for c in children:
                bls_expanded_rows.append((bls_id, c))
        
        bls_naics2022_6 = pd.DataFrame(bls_expanded_rows,
                                       columns=['BLS_Industry', 'naics2022_6'])
        
        bls_naics2022_6 = (bls_naics2022_6
                                .merge(IO_df['BLS_Industry'], on='BLS_Industry', how='inner')
                                .drop_duplicates())
        
        
        # ---------- #
        # Expand EPA #
        # ---------- #
        EPA_Sectors = EPA_df['EPA_Sector'].dropna().unique()

        epa_mapping_rows = []
        for s in EPA_Sectors:
            mapped_2022_6 = map_naics2012_to_2022_6(s)
            for c in mapped_2022_6:
                epa_mapping_rows.append((s, c))
        
        epa_naics2022_6 = pd.DataFrame(epa_mapping_rows,
                                       columns=['EPA_Sector', 'naics2022_6'])
        
        epa_naics2022_6 = epa_naics2022_6.drop_duplicates()
        
        
        # ------------------------------ #
        # Crosswalk & Allocate Emissions #
        # ------------------------------ #
        EPA_BLS_Crosswalk = (epa_naics2022_6
                                .merge(bls_naics2022_6, on='naics2022_6', how='inner')
                                .drop_duplicates())

        IO_df = IO_df.merge(EPA_BLS_Crosswalk[['EPA_Sector', 'BLS_Industry']],
                                    on='BLS_Industry',
                                    how='inner')
        
        IO_df = IO_df.merge(EPA_df,
                            on='EPA_Sector',
                            how='inner')
        
        IO_df = IO_df.merge(VA_panel,
                            on=['BLS_Industry', 'Year'],
                            how='inner')
        
        IO_df['CO2e_denom'] = IO_df.groupby(['EPA_Sector', 'Year'])['Value_Added'].transform("sum")
        IO_df['CO2e_integrand'] = IO_df['Value_Added'] * IO_df['CO2e'] / IO_df['CO2e_denom']
        IO_df['CO2e_Industry'] = IO_df.groupby(['BLS_Industry', 'Year'])['CO2e_integrand'].transform("sum")
        IO_df['CO2e_intensity_Industry'] = IO_df['CO2e_Industry'] / IO_df['Value_Added']
        
        IO_df = IO_df[['BLS_Industry', 'Year', 'TV_distance', 'CO2e_intensity_Industry']].drop_duplicates()
        
        IO_wide_df = IO_df.pivot(index="BLS_Industry", columns="Year", values='CO2e_intensity_Industry')
        IO_wide_df["dlog_CO2e_inten"] = np.log(IO_wide_df[Year_end]) - np.log(IO_wide_df[Year_start])
        
        reg_df = pd.merge(IO_df[['BLS_Industry', 'TV_distance']].drop_duplicates(),
                          IO_wide_df[['BLS_Industry', 'dlog_CO2e_inten']].drop_duplicates(),
                          on='BLS_Industry')
        

        # ----------------------------------------------------------------

        # Run regressions and graph.

        # ----------------------------------------------------------------
 
        
    def write_package_versions(self, packages):
        """""
        Table of Package Versions
    
        Output: Results/core_versions.txt
        """""
        
        filename=f'{self.Directory}/Results/core_versions.txt'
        
        
        # ---------------- #
        # Collect Packages #
        # ---------------- #
        rows = []
        for pkg in packages:
            ver = md.version(pkg)
            rows.append((pkg, ver))
    
    
        # ----------- #
        # Write Table #
        # ----------- #
        print(sys.version)
        
        with open(filename, "w") as f:
            f.write("| Package | Version |\n")
            f.write("|---------|---------|\n")
            for pkg, ver in rows:
                f.write(f"| {pkg} | {ver} |\n")



