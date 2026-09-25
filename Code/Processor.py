"""""""""""
Processor Module

Notes: This file defines a class for processing the workflow of "Transition to Green Technology along the Supply Chain".

"""""""""""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import io, sys
from datetime import datetime
from pathlib import Path
import requests as api
import importlib.metadata as md
import Processing_Functions as gpf
import Estimators as es



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
        self.ICE_classes = ["Y02T10/10", "Y02T10/12", "Y02T10/20", "Y02T10/30", "Y02T10/40"]
        self.CPC_classes = {("clean"): ["Y02E", "Y02P", "Y02T", "B60L"],
                            ("dirty"): ["F22", "F23", "F27", "C10J", "F01K", "F02C", "F02G",
                                        "B01J8/20", "B01J8/22", "B01J8/24", "B01J8/26", "B01J8/28", "B01J8/30",
                                        "F02B", "F02D", "F02F", "F02M", "F02N", "F02P", "Y02T10/12", "Y02T10/40"]}
        self.manu_cols = [1, 93]
        self.fossil_cols = [7-1, 8-1]#, 12-1] #Exclude electricity as well
        
        keys_path = self.Directory / ".keys"
        keys = {}
        with open(keys_path) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    keys[k.strip()] = v.strip()
        
        self.USPTO_API = keys.get("USPTO_API")

        
        
    def Cleaner(self, BLS_year_start, Year_start, Year_end, API):
        """""
        Clean Data
        
        Output: Clean Data/IO_Networks.pkl
                Raw Data/EPA.pkl
                Raw Data/NAICS.pkl
                Clean Data/BLS_Crosswalk.pkl
                Clean Data/Ind_CO2.pkl
                Clean Data/Ind_CO2_full.pkl
                Raw Data/assignee.pkl
                Raw Data/CPC.pkl
                Raw Data/applications.pkl
                Raw Data/citations.pkl
                Raw Data/Patent_Inventors.pkl
                Raw Data/Patent_Locations.pkl
                Clean Data/Inventor_Locations.pkl
                Clean Data/state_rd_price.pkl
                Clean Data/Pat_Firms.pkl
                Clean Data/Ind_Pat.pkl
                Clean Data/Ind_Pat_full.pkl
        """""
        
        # ----------------------------------------------------------------

        # Unpack data sets.

        # ----------------------------------------------------------------
        
        # ------------ #
        # BLS IO Table #
        # ------------ #
        def compute_IO(year):
            USE_df  = pd.read_excel(f'{self.Directory}/Raw Data/REAL_USE.xlsx',  sheet_name=f"{year}")
            MAKE_df = pd.read_excel(f'{self.Directory}/Raw Data/REAL_MAKE.xlsx', sheet_name=f"{year}")

            U      = USE_df.iloc[:, 1:-3].to_numpy()
            ind_Y  = np.sum(U, 0)
            B      = (U[:-3, :] @ np.diag(ind_Y**(-1))).T

            M      = MAKE_df.iloc[:, 1:].to_numpy()
            com_Y  = np.sum(M, 0)[:-2]
            A      = (M[:-2, :-2] @ np.diag(com_Y**(-1))).T

            IO = B @ A

            return IO
        
        IO_mats = {year: compute_IO(year) for year in range(BLS_year_start, Year_end+1)}
        pd.to_pickle(IO_mats, f'{self.Directory}/Clean Data/IO_Networks.pkl')
        #NAICS 2022
        
        BLS_Crosswalk_df = pd.read_excel(f'{self.Directory}/Raw Data/BLS_Crosswalk.xlsx', sheet_name="Stubs")
        BLS_Crosswalk_df['BLS_Industry'] = BLS_Crosswalk_df['Sector Number']
        
        
        # ----------------------- #
        # EPA Emissions by Sector #
        # ----------------------- #
        if API == 1:
            EPA_url = "https://pasteur.epa.gov/uploads/10.23719/1531141/GHGs_by_Detailed_Sector_US_2012-2022.xlsx"
            EPA_df = pd.read_excel(EPA_url, sheet_name="Main")
            EPA_df['year'] = EPA_df['Year']
            EPA_df.to_pickle(f'{self.Directory}/Raw Data/EPA.pkl')
        else:
            EPA_df = pd.read_pickle(f'{self.Directory}/Raw Data/EPA.pkl')
        
        EPA_df = EPA_df[EPA_df['Flowable'].isin(self.CO2e.keys())].copy()
        EPA_df['GWP'] = EPA_df['Flowable'].map(self.CO2e)
        EPA_df['FlowAmount_CO2e'] = EPA_df['FlowAmount'] * EPA_df['GWP']
        
        EPA_df['CO2e'] = EPA_df.groupby(['Sector', 'year'])['FlowAmount_CO2e'].transform("sum")
        EPA_df = EPA_df[['Sector', 'year', 'CO2e']].drop_duplicates()
        #NAICS 2017
        

        # ---------------- #
        # NAICS Crosswalks #
        # ---------------- #
        if API == 1:
            NAICS_2017_2022_url = "https://www.census.gov/naics/concordances/2017_to_2022_NAICS.xlsx"
            headers = {"User-Agent": "Mozilla/5.0"}
            r = api.get(NAICS_2017_2022_url, headers=headers)
            NAICS_2017_2022_df = pd.read_excel(io.BytesIO(r.content), skiprows=2)
            NAICS_2017_2022_df.to_pickle(f'{self.Directory}/Raw Data/NAICS.pkl')
        else:
            NAICS_2017_2022_df = pd.read_pickle(f'{self.Directory}/Raw Data/NAICS.pkl')
        
        
        # ----------------- #
        # Value Added Panel #
        # ----------------- #
        J = IO_mats[Year_end].shape[0]
        IO_df = pd.DataFrame({"BLS_Industry": np.arange(1, J+1)})
        
        va_frames = []
        for year in range(Year_start, Year_end + 1):
            USE_yr = pd.read_excel(f'{self.Directory}/Raw Data/REAL_USE.xlsx', sheet_name=f"{year}")
            U = USE_yr.iloc[:, 1:-3].to_numpy()
            ind_Y_yr = np.sum(U, 0)
            va_frames.append(pd.DataFrame({
                "BLS_Industry": np.arange(1, J+1),
                "Value_Added":  ind_Y_yr,
                "year":         year
            }))

        VA_panel = pd.concat(va_frames, ignore_index=True)
        
        
        # ----------------------------------------------------------------

        # Make unified NAICS mapping.

        # ----------------------------------------------------------------
        
        # -------------------- #
        # Clean for Comparison #
        # -------------------- #
        EPA_df['EPA_Sector'] = EPA_df['Sector'].apply(gpf.clean_naics_str)
        
        NAICS_2017_2022_df['NAICS_2017'] = NAICS_2017_2022_df['2017 NAICS Code'].apply(gpf.clean_naics_str)
        NAICS_2017_2022_df['NAICS_2022'] = NAICS_2017_2022_df['2022 NAICS Code'].apply(gpf.clean_naics_str)

        BLS_Crosswalk_df['NAICS_2022'] = BLS_Crosswalk_df['NAICS_2022'].apply(gpf.clean_naics_str)
        
        naics2017_6_universe = sorted(NAICS_2017_2022_df['NAICS_2017'].dropna().unique())
        naics2022_6_universe = sorted(NAICS_2017_2022_df['NAICS_2022'].dropna().unique())
        
        
        # ---------- #
        # Expand BLS #
        # ---------- #
        BLS_long = (BLS_Crosswalk_df
                    .assign(naics_code_list=lambda x: x['NAICS_2022'].apply(gpf.split_comma_list))
                    .explode('naics_code_list')
                    .rename(columns={'naics_code_list': 'naics_prefix'}))
        
        BLS_long['naics_prefix'] = BLS_long['naics_prefix'].apply(gpf.clean_naics_str)
        
       
        bls_expanded_rows = []
        for _, row in BLS_long.iterrows():
            bls_id = row['BLS_Industry']  # adjust column name
            children = gpf.expand_bls_row_to_6(row, naics2022_6_universe)
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
            mapped_2022_6 = gpf.map_naics2017_to_2022_6(s, naics2017_6_universe, NAICS_2017_2022_df)
            for c in mapped_2022_6:
                epa_mapping_rows.append((s, c))
        
        epa_naics2022_6 = pd.DataFrame(epa_mapping_rows,
                                       columns=['EPA_Sector', 'naics2022_6'])
        
        epa_naics2022_6 = epa_naics2022_6.drop_duplicates()
        
        
        # --------- #
        # Crosswalk #
        # --------- #
        EPA_BLS_Crosswalk = (epa_naics2022_6
                                .merge(bls_naics2022_6, on='naics2022_6', how='inner')
                                .drop_duplicates())
        EPA_BLS_Crosswalk['naics2022_6'] = pd.to_numeric(EPA_BLS_Crosswalk['naics2022_6'])
        
        BLS_Crosswalk_df.to_pickle(f'{self.Directory}/Clean Data/BLS_Crosswalk.pkl')
        
        
        # ------------------ #
        # Allocate Emissions #
        # ------------------ #
        jump_mask = (
            EPA_df
            .sort_values(['Sector', 'year'])
            .groupby('Sector')['CO2e']
            .transform(lambda x: x.apply(np.log).diff().abs())
            ) > np.log(1.5)
    
        flagged_industries = EPA_df['Sector'][jump_mask].unique()
        
        total_emissions = EPA_df['CO2e'].sum()
        flagged_emissions = EPA_df[EPA_df['Sector'].isin(flagged_industries)]['CO2e'].sum()
        
        print(f"Flagged industries account for {flagged_emissions / total_emissions:.1%} of total emissions")
        EPS_full_df = EPA_df.copy()
        EPA_df = EPA_df[~EPA_df['Sector'].isin(flagged_industries)]
        
        def Ind_em_panel(EPA_df):
            Ind_CO2_df = IO_df.merge(EPA_BLS_Crosswalk[['EPA_Sector', 'BLS_Industry']].drop_duplicates(),
                                        on='BLS_Industry',
                                        how='inner')
            
            Ind_CO2_df = Ind_CO2_df.merge(EPA_df,
                                on='EPA_Sector',
                                how='inner')
            
            Ind_CO2_df = Ind_CO2_df.merge(VA_panel,
                                on=['BLS_Industry', 'year'],
                                how='inner')
            
            Ind_CO2_df['CO2e_Industry'] = Ind_CO2_df.groupby(['BLS_Industry', 'year'])['CO2e'].transform("sum")
            Ind_CO2_df['CO2e_intensity_Industry'] = Ind_CO2_df['CO2e_Industry'] / Ind_CO2_df['Value_Added']
            
            Ind_CO2_df = Ind_CO2_df[['BLS_Industry', 'year', 'CO2e_Industry', 'CO2e_intensity_Industry']].drop_duplicates()
            
            return Ind_CO2_df

        Ind_em_panel(EPA_df).to_pickle(f'{self.Directory}/Clean Data/Ind_CO2.pkl')
        Ind_em_panel(EPS_full_df).to_pickle(f'{self.Directory}/Clean Data/Ind_CO2_full.pkl')
        
        
        # ----------------------------------------------------------------

        # Build industry patenting cross-section.

        # ----------------------------------------------------------------

        # -------------------- #
        # PatentsView Assignee #
        # -------------------- #
        if API == 1:
            PV_assignee_df = gpf.Extract_PatentsView('g_assignee_disambiguated', self.USPTO_API)
            
            PV_assignee_df.to_pickle(f'{self.Directory}/Raw Data/assignee.pkl')
        else:
            PV_assignee_df = pd.read_pickle(f'{self.Directory}/Raw Data/assignee.pkl')
            
        del PV_assignee_df

    
        # --------------------- #
        # PatentsView CPC Codes #
        # --------------------- #
        if API == 1:
            CPC_df = gpf.Extract_PatentsView('g_cpc_current', self.USPTO_API)
            
            CPC_df['patent_id'] = CPC_df['patent_id'].astype(str)
            CPC_df.to_pickle(f'{self.Directory}/Raw Data/CPC.pkl')
        else:
            CPC_df = pd.read_pickle(f'{self.Directory}/Raw Data/CPC.pkl')
                
        
        # ------------------------ #
        # PatentsView Applications #
        # ------------------------ #
        if API == 1:
            PV_applications_df = gpf.Extract_PatentsView('g_application', self.USPTO_API)
            
            PV_applications_df["year"] = pd.to_datetime(PV_applications_df["filing_date"], format="%Y-%m-%d", errors="coerce").dt.year
            PV_applications_df = PV_applications_df.dropna(subset=["year"])
            PV_applications_df = PV_applications_df[(PV_applications_df["year"] >= 1900) & (PV_applications_df["year"] <= datetime.now().year)]
            PV_applications_df['patent_id'] = PV_applications_df['patent_id'].astype(str)
            PV_applications_df.to_pickle(f'{self.Directory}/Raw Data/applications.pkl')
        else:
            PV_applications_df = pd.read_pickle(f'{self.Directory}/Raw Data/applications.pkl')
            
            
         # --------------------- #
         # PatentsView Citations #
         # --------------------- #
        if API == 1:
             citations_df = gpf.Extract_PatentsView('g_us_patent_citation', self.USPTO_API)
             
             citations_df['patent_id'] = citations_df['patent_id'].astype(str)
             citations_df['citation_patent_id'] = citations_df['citation_patent_id'].astype(str)
             citations_df.to_pickle(f'{self.Directory}/Raw Data/citations.pkl')
        else:
             citations_df = pd.read_pickle(f'{self.Directory}/Raw Data/citations.pkl')
             
             
        # --------------------- #
        # Patentsview Inventors #
        # --------------------- #
        if API == 1:
            PV_inventors_df = gpf.Extract_PatentsView('g_inventor_disambiguated', self.USPTO_API)
            PV_inventors_df['patent_id'] = PV_inventors_df['patent_id'].astype(str)
            
            PV_location_df = gpf.Extract_PatentsView('g_location_disambiguated', self.USPTO_API)
            PV_location_df = PV_location_df.dropna(subset=['state_fips'])
            
            PV_inventors_df.to_pickle(f'{self.Directory}/Raw Data/Patent_Inventors.pkl')
            PV_location_df.to_pickle(f'{self.Directory}/Raw Data/Patent_Locations.pkl')
            
        else:
            PV_inventors_df = pd.read_pickle(f'{self.Directory}/Raw Data/Patent_Inventors.pkl')
            PV_location_df = pd.read_pickle(f'{self.Directory}/Raw Data/Patent_Locations.pkl')
        
        PV_inventor_location_df = pd.merge(PV_inventors_df,
                                   PV_location_df[['location_id', 'disambig_state', 'state_fips']],
                                   on='location_id',
                                   how='inner'
                                    )
        
        PV_inventor_location_df.to_pickle(f'{self.Directory}/Clean Data/Inventor_Locations.pkl')
        del PV_inventors_df, PV_location_df
        
            
        # ---------------------- #
        # State-Level R&D Prices #
        # ---------------------- #
        state_rdp_df = pd.read_stata(f'{self.Directory}/Raw Data/RDusercost_2017.dta')
        
        state_rdp_df = state_rdp_df[['state', 'fips', 'year', 'rho_h']]
        state_rdp_df = state_rdp_df.rename(columns={"fips": "state_fips"})
        
        state_rdp_df.to_pickle(f'{self.Directory}/Clean Data/state_rd_price.pkl')
        
        
        # ------------------ #
        # Technology Classes #
        # ------------------ #
        relevant_df = CPC_df.copy()
        
        relevant_df = pd.merge(relevant_df,
                             PV_applications_df,
                             on='patent_id',
                             how='inner'
                             )
        
        relevant_df = relevant_df[(relevant_df["year"] <= Year_end)]
        
        relevant_df["cpc_group5"] = relevant_df["cpc_group"].str[:5]
        relevant_df["cpc_group6"] = relevant_df["cpc_group"].str[:6]
        
        for c in ['clean', 'dirty']:
            codes = set(self.CPC_classes[c])
            relevant_df[c] = (relevant_df["cpc_class"].isin(codes)
                                    | relevant_df["cpc_subclass"].isin(codes)
                                    | relevant_df["cpc_group"].isin(codes)
                                    | relevant_df["cpc_group5"].isin(codes)
                                    | relevant_df["cpc_group6"].isin(codes)).astype(np.int8)
        
        ice_codes = set(self.ICE_classes)
        relevant_df['ice'] = relevant_df["cpc_group"].isin(ice_codes).astype(np.int8)
        
        relevant_df['clean_full'] = relevant_df.groupby("patent_id")['clean'].transform("max")
        relevant_df['clean'] = relevant_df['clean_full'] - relevant_df.groupby("patent_id")['ice'].transform("max")
        relevant_df['dirty'] = relevant_df.groupby("patent_id")['dirty'].transform("max")
        relevant_df = relevant_df[['patent_id', 'year', 'clean', 'clean_full', 'dirty']].drop_duplicates()
        
        
        # ------------------------- #
        # Patent Citation Weighting #
        # ------------------------- #
        citations_df['cites'] = citations_df.groupby('citation_patent_id')['citation_patent_id'].transform('count')
        citations_df = citations_df[['citation_patent_id', 'cites']].drop_duplicates()
        citations_df.rename(columns={'citation_patent_id': 'patent_id'}, inplace=True)
        
        citations_df = citations_df.merge(CPC_df[['patent_id', 'cpc_class']],
                                            on='patent_id',
                                            how='right')
        citations_df = citations_df.merge(PV_applications_df[['patent_id', 'year']],
                                            on='patent_id',
                                            how='inner')
        
        citations_df['cites'] = citations_df['cites'].fillna(0)
        citations_df['cites'] = citations_df['cites'] + 1
        
        citations_df['cpc_cites'] = citations_df.groupby(['cpc_class', 'year'])['cites'].transform('mean')
        citations_df['norm_cites'] = citations_df['cites'] / citations_df.groupby('patent_id')['cpc_cites'].transform('mean')
        
        pat_df = pd.merge(
            citations_df[['patent_id', 'norm_cites']].drop_duplicates(),
            relevant_df,
            on='patent_id',
            how='inner'
        )
        
        del CPC_df, PV_applications_df, relevant_df, citations_df
        
        
        # ------------------------ #
        # Patent to Firm Crosswalk #
        # ------------------------ #
        discern_df = pd.read_csv(f'{self.Directory}/Raw Data/discern_pat_grant_1980_2021.csv', low_memory=False)
        KPSS_df = pd.read_csv(f'{self.Directory}/Raw Data/KPSS_match_patent_permno_2023.csv')
        gvkey_df = pd.read_csv(f'{self.Directory}/Raw Data/permno_gvkey.csv')
        
        KPSS_df = KPSS_df.rename(columns={"patent_num": "patent_id"})
        discern_df = discern_df.rename(columns={"permno_adj": "permno"})
        gvkey_df = gvkey_df.rename(columns={"permno_adj": "permno"})
        
        new_pats = KPSS_df[~KPSS_df['patent_id'].isin(discern_df['patent_id'])]

        pat_firm_crosswalk_df = pd.concat([discern_df, new_pats], ignore_index=True)

        pat_firm_crosswalk_df = pat_firm_crosswalk_df.merge(gvkey_df[['gvkey', 'permno']],
                                    on='permno',
                                    how='inner'
                                         )
        pat_firm_crosswalk_df = pat_firm_crosswalk_df[['patent_id', 'gvkey']]
        
        
        # --------- #
        # Compustat #
        # --------- #
        compustat_df = pd.read_csv(f'{self.Directory}/Raw Data/compustat.csv')
        
        compustat_df = compustat_df[(compustat_df['fic']=="USA") & (compustat_df['final']=="Y")]
        terry_cols = ['at', 'ppent', 'emp', 'capxv', 'sale', 'xrd']
        compustat_df = compustat_df[compustat_df[terry_cols].gt(0).all(axis=1)]
        compustat_df = compustat_df[compustat_df.groupby('gvkey')['gvkey'].transform('count') > 1]
        compustat_df.rename(columns={'fyear': 'year'}, inplace=True)
        
        compustat_df = compustat_df[(compustat_df["year"] <= Year_end)]
        compustat_df['naics2022_6'] = compustat_df['naics'] #Assume Compustat uses most up to date NAICS
        compustat_df = compustat_df[['gvkey', 'naics2022_6']].drop_duplicates()


        # --------------------------- #
        # Allocate Patents to Sectors #
        # --------------------------- #
        pat_firms_df = pat_df.merge(pat_firm_crosswalk_df,
                            on='patent_id',
                            how='inner')
        pat_firms_df = pat_firms_df.merge(compustat_df,
                            on='gvkey',
                            how='inner')
        pat_firms_df = pat_firms_df.merge(EPA_BLS_Crosswalk[['naics2022_6', 'BLS_Industry']].drop_duplicates(),
                            on='naics2022_6',
                            how='inner')
        
        pat_firms_df = pat_firms_df[['patent_id', 'year', 'gvkey', 'BLS_Industry', 'clean', 'clean_full', 'dirty', 'norm_cites']].drop_duplicates()
        pat_ind_df = pat_firms_df[['patent_id', 'year', 'BLS_Industry', 'clean', 'clean_full', 'dirty', 'norm_cites']].drop_duplicates()
        
        pat_firms_df['split_weight'] = 1 / pat_firms_df.groupby('patent_id')['gvkey'].transform('count')
        pat_firms_df.to_pickle(f'{self.Directory}/Clean Data/Pat_Firms.pkl')
        
        pat_ind_df['split_weight'] = 1 / pat_ind_df.groupby('patent_id')['BLS_Industry'].transform('count')
        
        
        # ------------------------- #
        # Clean Patenting by Sector #
        # ------------------------- #
        annual_df = pat_ind_df.copy()
        annual_df['clean_w']           = annual_df['split_weight'] * annual_df['clean']
        annual_df['clean_full_w']      = annual_df['split_weight'] * annual_df['clean_full']
        annual_df['dirty_w']           = annual_df['split_weight'] * annual_df['dirty']
        annual_df['cite_w']            = annual_df['split_weight'] * annual_df['norm_cites']
        annual_df['clean_cite_w']      = annual_df['clean_w']      * annual_df['norm_cites']
        annual_df['clean_full_cite_w'] = annual_df['clean_full_w'] * annual_df['norm_cites']
        annual_df['dirty_cite_w']      = annual_df['dirty_w']      * annual_df['norm_cites']
 
        agg_base = dict(clean_pat_count = ('clean_w',      'sum'),
                        dirty_pat_count = ('dirty_w',      'sum'),
                        pat_count       = ('split_weight', 'sum'),
                        clean_pat_cites = ('clean_cite_w', 'sum'),
                        dirty_pat_cites = ('dirty_cite_w', 'sum'),
                        pat_cites       = ('cite_w',       'sum'))
        agg_full = dict(clean_pat_count = ('clean_full_w',      'sum'),
                        dirty_pat_count = ('dirty_w',           'sum'),
                        pat_count       = ('split_weight',      'sum'),
                        clean_pat_cites = ('clean_full_cite_w', 'sum'),
                        dirty_pat_cites = ('dirty_cite_w',      'sum'),
                        pat_cites       = ('cite_w',            'sum'))

        ind_pat_df = (annual_df.groupby(['BLS_Industry', 'year'], as_index=False)
                                    .agg(**agg_base))
        ind_pat_df_full = (annual_df.groupby(['BLS_Industry', 'year'], as_index=False)
                                    .agg(**agg_full))
        panel_idx = pd.MultiIndex.from_product(
            [sorted(ind_pat_df['BLS_Industry'].unique()), 
             list(range(BLS_year_start-10, Year_end + 1))],
            names=['BLS_Industry', 'year'])
        ind_pat_df = (ind_pat_df.set_index(['BLS_Industry', 'year'])
                           .reindex(panel_idx)
                           .fillna(0.0)
                           .reset_index())
        ind_pat_df_full = (ind_pat_df_full.set_index(['BLS_Industry', 'year'])
                           .reindex(panel_idx)
                           .fillna(0.0)
                           .reset_index())
 
        ind_pat_df = ind_pat_df[ind_pat_df['BLS_Industry'] != 71]
        
        for frame in (ind_pat_df, ind_pat_df_full):
            frame['pat_count_nc'] = frame['pat_count'] - frame['clean_pat_count']
            frame['pat_cites_nc'] = frame['pat_cites'] - frame['clean_pat_cites']
 
        ind_pat_df.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat.pkl')
        ind_pat_df_full.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat_full.pkl')
            
            
            
    def Instruments(self):
        """""
        Create Series of Greenification Shocks
    
        Output: Clean Data/RD_Shocks.pkl
        """""
        
        # ----------------------------------------------------------------

        # Build instrument dataframes.

        # ----------------------------------------------------------------
        
        IV_year_start = 1980
        
        state_rdp_df = pd.read_pickle(f'{self.Directory}/Clean Data/state_rd_price.pkl')
        PV_inventor_location_df = pd.read_pickle(f'{self.Directory}/Clean Data/Inventor_Locations.pkl')
        pat_firms_df = pd.read_pickle(f'{self.Directory}/Clean Data/Pat_Firms.pkl')
        pat_firms_df = pat_firms_df[pat_firms_df['year'] >= IV_year_start]
        
        
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
 
        id_cols = ['gvkey', 'BLS_Industry', 'year']
        frames  = []
        for tname, cmap in type_map.items():
            missing = [c for c in cmap.values() if c not in firm_pat_wide_df.columns]
            if missing:
                raise KeyError(f'type "{tname}" needs columns {missing}')
            sub = firm_pat_wide_df[id_cols + list(cmap.values())].copy()
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
            'pat_count_hat__general': 'pat_count_hat',
            'pat_count_hat__clean':   'pat_count_clean_hat',
            'pat_count_hat__dirty':   'pat_count_dirty_hat',
            'pat_cites_hat__general': 'pat_cites_hat',
            'pat_cites_hat__clean':   'pat_cites_clean_hat',
            'pat_cites_hat__dirty':   'pat_cites_dirty_hat'})
        
        RD_shocks_df.to_pickle(f'{self.Directory}/Clean Data/RD_Shocks.pkl')
   
    

    def IO_Change(self, Year_start, Year_mid, Year_end, dim=3):
        """""
        Plot of Changes in IO Network from Decarbonization
    
        Output: Results/Figures/Reduced_L1_WLS.png
                Results/Figures/Reduced_L1_OLS.png
                Results/Figures/Reduced_L2_OLS.png
                Results/Figures/Reduced_L1_OLS_FE.png
                Results/Figures/Leontief_L1_WLS.png
                Results/Figures/Leontief_L1_OLS.png
                Results/Figures/Leontief_L2_OLS.png
                Results/Figures/Leontief_L1_OLS_FE.png
                Results/Figures/Reduced_full_L1_WLS.png
                Results/Figures/Reduced_full_L1_OLS.png
                Results/Figures/Reduced_full_L2_OLS.png
                Results/Figures/Reduced_full_L1_OLS_FE.png
                Results/Figures/Leontief_full_L1_WLS.png
                Results/Figures/Leontief_full_L1_OLS.png
                Results/Figures/Leontief_full_L2_OLS.png
                Results/Figures/Leontief_full_L1_OLS_FE.png
        """""
        
        # ----------------------------------------------------------------

        # Build regression dataframes.

        # ----------------------------------------------------------------
        
        BLS_Crosswalk_df = pd.read_pickle(f'{self.Directory}/Clean Data/BLS_Crosswalk.pkl')
        IO_mats = pd.read_pickle(f'{self.Directory}/Clean Data/IO_Networks.pkl')
        Ind_CO2_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_CO2.pkl')
        Ind_CO2_df_full = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_CO2_full.pkl')

        
        def IO_panel(Ind_CO2_df):
            # ------------------- #
            # Input-Output Matrix #
            # ------------------- #
            J       = IO_mats[Year_start].shape[0]
            manu    = slice(self.manu_cols[0]-1, self.manu_cols[1])  # 0-indexed rows for non-service industries
     
            I        = np.eye(J)
            LI_start = np.linalg.inv(I - IO_mats[Year_start])
            LI_mid   = np.linalg.inv(I - IO_mats[Year_mid])
            LI_end   = np.linalg.inv(I - IO_mats[Year_end])
     
            # Leontief: non-service rows, all columns
            LI_start_manu = LI_start[manu, :]
            LI_mid_manu   = LI_mid[manu, :]
            LI_end_manu   = LI_end[manu, :]
     
            # Reduced: non-service rows, fossil fuel columns dropped, renormalized
            def drop_and_normalize(IO):
                IO_manu = IO[manu, :]
                IO_r = np.delete(IO_manu, self.fossil_cols, axis=1)
                num = IO_manu.sum(axis=1, keepdims=True)
                denom = IO_r.sum(axis=1, keepdims=True)
                return IO_r * num / denom
    
            IO_start_reduced = drop_and_normalize(IO_mats[Year_start])
            IO_mid_reduced   = drop_and_normalize(IO_mats[Year_mid])
            IO_end_reduced   = drop_and_normalize(IO_mats[Year_end])
    
            def tv_metrics(A, B):
               diff  = np.abs(B - A)
               tv    = 0.5 * diff.sum(axis=1)
               tv_sq = (0.5 * (diff**2).sum(axis=1))**(1/2)
               return tv, tv_sq
    
            # Period 1: start -> mid
            tv_LI_p1,      tv_sq_LI_p1      = tv_metrics(LI_start_manu,    LI_mid_manu)
            tv_red_p1,     tv_sq_red_p1     = tv_metrics(IO_start_reduced, IO_mid_reduced)
    
            # Period 2: mid -> end
            tv_LI_p2,      tv_sq_LI_p2      = tv_metrics(LI_mid_manu,   LI_end_manu)
            tv_red_p2,     tv_sq_red_p2     = tv_metrics(IO_mid_reduced, IO_end_reduced)
    
            def make_IO_df(tv_LI, tv_sq_LI, tv_red, tv_sq_red):
                return pd.DataFrame({
                    "BLS_Industry":           np.arange(self.manu_cols[0], self.manu_cols[1]+1),
                    "TV_distance_LI":         tv_LI,
                    "TV_sq_distance_LI":      tv_sq_LI,
                    "TV_distance_reduced":    tv_red,
                    "TV_sq_distance_reduced": tv_sq_red})
    
            IO_df_p1 = make_IO_df(tv_LI_p1, tv_sq_LI_p1, tv_red_p1, tv_sq_red_p1)
            IO_df_p1['period'] = Year_mid
    
            IO_df_p2 = make_IO_df(tv_LI_p2, tv_sq_LI_p2, tv_red_p2, tv_sq_red_p2)
            IO_df_p2['period'] = Year_end
    
            IO_df = pd.concat([IO_df_p1, IO_df_p2], ignore_index=True)
    
            
            # ------------------ #
            # Allocate Emissions #
            # ------------------ #
            IO_wide_df = Ind_CO2_df.pivot(index="BLS_Industry", columns="year",
                                           values=['CO2e_intensity_Industry', 'CO2e_Industry'])
            IO_wide_df = IO_wide_df.dropna()
    
            idx1 = IO_wide_df.index.to_numpy(dtype=int)
            idx0 = idx1 - 1
    
            LI_start_sub = LI_start[np.ix_(idx0, idx0)]
            LI_mid_sub   = LI_mid[np.ix_(idx0, idx0)]
            LI_end_sub   = LI_end[np.ix_(idx0, idx0)]
    
            CO2e_LI_start = LI_start_sub @ IO_wide_df['CO2e_intensity_Industry', Year_start].to_numpy()
            CO2e_LI_mid   = LI_mid_sub   @ IO_wide_df['CO2e_intensity_Industry', Year_mid].to_numpy()
            CO2e_LI_end   = LI_end_sub   @ IO_wide_df['CO2e_intensity_Industry', Year_end].to_numpy()
    
            CO2e_lev_LI_start = LI_start_sub @ IO_wide_df['CO2e_Industry', Year_start].to_numpy()
            CO2e_lev_LI_mid   = LI_mid_sub   @ IO_wide_df['CO2e_Industry', Year_mid].to_numpy()
            
            em_p1 = pd.DataFrame({
                "BLS_Industry":       IO_wide_df.index,
                "dlog_CO2e_inten":    -(np.log(IO_wide_df['CO2e_intensity_Industry', Year_mid].to_numpy())
                                      - np.log(IO_wide_df['CO2e_intensity_Industry', Year_start].to_numpy())),
                "dlog_CO2e_inten_LI": -(np.log(CO2e_LI_mid) - np.log(CO2e_LI_start)),
                "CO2e_Industry_weight":      IO_wide_df['CO2e_Industry', Year_start].to_numpy()**(1/dim),
                "CO2e_Industry_LI_weight":   CO2e_lev_LI_start**(1/dim),
                "period": Year_mid})
    
            em_p2 = pd.DataFrame({
                "BLS_Industry":       IO_wide_df.index,
                "dlog_CO2e_inten":    -(np.log(IO_wide_df['CO2e_intensity_Industry', Year_end].to_numpy())
                                      - np.log(IO_wide_df['CO2e_intensity_Industry', Year_mid].to_numpy())),
                "dlog_CO2e_inten_LI": -(np.log(CO2e_LI_end) - np.log(CO2e_LI_mid)),
                "CO2e_Industry_weight":      IO_wide_df['CO2e_Industry', Year_mid].to_numpy()**(1/dim),
                "CO2e_Industry_LI_weight":   CO2e_lev_LI_mid**(1/dim),
                "period": Year_end})
    
            distance_cols = ['BLS_Industry', 'period',
                             'TV_distance_LI',      'TV_sq_distance_LI',
                             'TV_distance_reduced', 'TV_sq_distance_reduced']
    
            emission_cols = ['BLS_Industry', 'period',
                             'CO2e_Industry_weight', 'CO2e_Industry_LI_weight',
                             'dlog_CO2e_inten', 'dlog_CO2e_inten_LI']
    
            em_df  = pd.concat([em_p1, em_p2], ignore_index=True)
            reg_df = pd.merge(IO_df[distance_cols].drop_duplicates(),
                              em_df[emission_cols].drop_duplicates(),
                              on=['BLS_Industry', 'period'],
                              how='inner')
            
            reg_df = reg_df.merge(BLS_Crosswalk_df[["BLS_Industry", "Sector Title"]].drop_duplicates(),
                                    on="BLS_Industry",
                                    how="left"
                                )
            
            return reg_df


        # ----------------------------------------------------------------

        # Run regressions and graph.

        # ----------------------------------------------------------------
        
        # ---------------- #
        # Helper Functions #
        # ---------------- #
        
        def run_regressions(df, x_col, y_col, y_sq_col, weight_col, group_col):
            mask_pos = df[x_col] >= 0
            mask_neg = df[x_col] <  0

            x_arr    = df[x_col].to_numpy()
            y_arr    = df[y_col].to_numpy()
            y_sq_arr = df[y_sq_col].to_numpy()
            w_arr    = df[weight_col].to_numpy()
            g_arr    = df[group_col].to_numpy()

            X_kink = sm.add_constant(np.column_stack([
                x_arr * mask_neg.to_numpy(),
                x_arr * mask_pos.to_numpy(),
            ]))

            period_fe = (df['period'].to_numpy() == Year_end).astype(float)
            X_fe = np.column_stack([
                np.ones(len(x_arr)),
                period_fe,
                x_arr * mask_neg.to_numpy(),
                x_arr * mask_pos.to_numpy(),
            ])

            def resid_on_fe(v):
                means = np.where(period_fe == 0,
                                 v[period_fe == 0].mean(),
                                 v[period_fe == 1].mean())
                return v - means

            def fit(Y, X, w=None):
                cl = {'cov_type': 'cluster', 'cov_kwds': {'groups': g_arr}}
                if w is None:
                    return sm.OLS(Y, X).fit(**cl)
                return sm.WLS(Y, X, w).fit(**cl)

            return dict(
                m_l1_ols_kink = fit(y_arr,    X_kink),
                m_l1_wls_kink = fit(y_arr,    X_kink, w_arr),
                m_l2_ols_kink = fit(y_sq_arr, X_kink),
                m_l1_ols_fe   = fit(y_arr,    X_fe),
                x=x_arr, y=y_arr, y_sq=y_sq_arr, w=w_arr,
                x_resid    = resid_on_fe(x_arr),
                y_resid    = resid_on_fe(y_arr),
                y_sq_resid = resid_on_fe(y_sq_arr),
                mask_pos = mask_pos.to_numpy(),
                mask_neg = mask_neg.to_numpy(),
            )

        def plot_case(r, df, prefix, year_start, year_mid, year_end, save_dir, labels=None, top_n=1):
            x, y, y_sq           = r['x'], r['y'], r['y_sq']
            x_res, y_res = r['x_resid'], r['y_resid']
            w_raw  = r['w'] ** dim
            scale  = 1000 / w_raw.max()

            stars_idx = lambda m, k: gpf.get_stars(m.pvalues[k])

            mask_p1 = df['period'].to_numpy() == Year_mid
            mask_p2 = df['period'].to_numpy() == Year_end

            def annotate(ax, text, y_frac):
                ax.annotate(text, xy=(0.05, y_frac), xycoords='axes fraction',
                            fontsize=11, color='green',
                            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.7))

            def scatter_periods(ax, x_vals, y_arr):
                ax.scatter(x_vals[mask_p1], y_arr[mask_p1], s=w_raw[mask_p1]*scale, alpha=0.7, color='purple', label=f"Sectors: ({year_start}–{year_mid})")
                ax.scatter(x_vals[mask_p2], y_arr[mask_p2], s=w_raw[mask_p2]*scale, alpha=0.7, color='blue',   label=f"Sectors: ({year_mid}–{year_end})")

            def fix_legend(ax):
                leg = ax.legend(loc='upper right')
                for h in leg.legend_handles:
                    h._sizes = [30]

            def annotate_sectors(ax, x_vals, y_arr):
                if labels is None:
                    return
                w_avg   = pd.Series(w_raw, index=df.index).groupby(df['BLS_Industry'].values).mean()
                top_idx = w_avg.nlargest(top_n).index
                mask    = df['BLS_Industry'].isin(top_idx).to_numpy()
                for xi, yi, label in zip(x_vals[mask], y_arr[mask], labels[mask]):
                    words = [w for w in str(label).split()[:3] if w.isalpha()]
                    ax.annotate(' '.join(words), (xi, yi), fontsize=9, ha='left',
                                xytext=(11, 11), textcoords='offset points')

            def plot_single(m, y_arr, x_vals, fname, estimator_label,
                            b_idx=0, neg_idx=1, pos_idx=2):
                b   = 0 if b_idx is None else m.params[b_idx]
                s_n = m.params[neg_idx]
                s_p = m.params[pos_idx]
                xn  = np.linspace(x_vals[r['mask_neg']].min(), 0,                           100)
                xp  = np.linspace(0,                            x_vals[r['mask_pos']].max(), 100)
                fig, ax = plt.subplots(figsize=(8, 6))
                scatter_periods(ax, x_vals, y_arr)
                ax.plot(xn, b + s_n*xn, color='cyan',   linewidth=2, label=f"{estimator_label} fit (x<0)")
                ax.plot(xp, b + s_p*xp, color='orange', linewidth=2, label=f"{estimator_label} fit (x≥0)")
                annotate(ax, f"Slope (x<0)  = {s_n:.3f}{stars_idx(m, neg_idx)}", 0.95)
                annotate(ax, f"Slope (x≥0) = {s_p:.3f}{stars_idx(m, pos_idx)}", 0.88)
                annotate_sectors(ax, x_vals, y_arr)
                xlabel = "Log Emissions Intensity Reduction" 
                ylabel = "Change in Input Shares"
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
                fix_legend(ax)
                plt.savefig(f'{save_dir}/{fname}.png')
                plt.show()

            plot_single(r['m_l1_wls_kink'], y,    x,    f'{prefix}_L1_WLS', 'WLS')
            plot_single(r['m_l1_ols_kink'], y,    x,    f'{prefix}_L1_OLS', 'OLS')
            plot_single(r['m_l2_ols_kink'], y_sq, x,    f'{prefix}_L2_OLS', 'OLS')
            plot_single(r['m_l1_ols_fe'],   y_res, x_res, f'{prefix}_L1_OLS_FE', 'OLS', b_idx=None, neg_idx=2, pos_idx=3)
        
        fig_dir = f'{self.Directory}/Results/Figures'
        reg_df = IO_panel(Ind_CO2_df)


        # ------- #
        # Reduced #
        # ------- #
        r_red = run_regressions(reg_df, 'dlog_CO2e_inten', 'TV_distance_reduced', 'TV_sq_distance_reduced', 'CO2e_Industry_weight', 'BLS_Industry')
        plot_case(r_red, reg_df, 'Reduced', Year_start, Year_mid, Year_end, fig_dir,
                  labels=reg_df['Sector Title'].to_numpy())


        # ---------------- #
        # Leontief Inverse #
        # ---------------- #
        r_LI = run_regressions(reg_df, 'dlog_CO2e_inten_LI', 'TV_distance_LI', 'TV_sq_distance_LI', 'CO2e_Industry_LI_weight', 'BLS_Industry')
        plot_case(r_LI, reg_df, 'Leontief', Year_start, Year_mid, Year_end, fig_dir,
                  labels=reg_df['Sector Title'].to_numpy())
        
        
        # ---------- #
        # Winsorized #
        # ---------- #
        reg_df_full = IO_panel(Ind_CO2_df_full)
        reg_df_full = gpf.winsorize(reg_df_full,
            ['dlog_CO2e_inten'])
        
        r_red = run_regressions(reg_df_full, 'dlog_CO2e_inten', 'TV_distance_reduced', 'TV_sq_distance_reduced', 'CO2e_Industry_weight', 'BLS_Industry')
        plot_case(r_red, reg_df_full, 'Reduced_full', Year_start, Year_mid, Year_end, fig_dir,
                  labels=reg_df_full['Sector Title'].to_numpy())

        r_LI = run_regressions(reg_df_full, 'dlog_CO2e_inten_LI', 'TV_distance_LI', 'TV_sq_distance_LI', 'CO2e_Industry_LI_weight', 'BLS_Industry')
        plot_case(r_LI, reg_df_full, 'Leontief_full', Year_start, Year_mid, Year_end, fig_dir,
                  labels=reg_df_full['Sector Title'].to_numpy())
    
    
    
    def Up_Down_Green(self, BLS_year_start, Year_end, bin_len=5):
        """""
        Strategic Complementarity for Greenification
        
        Output: Results/Tables/Summary_Stats.tex
                Results/Tables/Network_Regressions_Net.tex
                Results/Tables/Network_Regressions_Net_full.tex
                Results/Tables/Network_Regressions_Net_WLS.tex
                Results/Tables/Network_Regressions_Lagged.tex
                Results/Tables/Network_Regressions_UpDown.tex
        
        """""
        
        # ----------------------------------------------------------------

        # Build regression dataframes.

        # ----------------------------------------------------------------
        
        IO_mats = pd.read_pickle(f'{self.Directory}/Clean Data/IO_Networks.pkl')
        Ind_Pat_yr_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_Pat.pkl')
        
        # Ind_Pat_df_full = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_Pat_full.pkl')
        
        RD_shocks_yr_df = pd.read_pickle(f'{self.Directory}/Clean Data/RD_Shocks.pkl')
        
        manu_idx_all = np.arange(self.manu_cols[0], self.manu_cols[1] + 1)
        
        
        # -------- #
        # Bin Data #
        # -------- # 
        bin_ends = [y for y in range(BLS_year_start, Year_end + 1, bin_len) if y in IO_mats]

        def make_bins(df, cols):
            frames = []
            rel_bins = sorted(set(df['year']).intersection(bin_ends))
            for end in rel_bins:
                w = df[(df['year'] > end - bin_len)
                       & (df['year'] <= end)]
                if w.empty:
                    continue
                frames.append(w.groupby('BLS_Industry', as_index=False)[cols]
                               .sum().assign(period=end))
            out = pd.concat(frames, ignore_index=True)
            idx = pd.MultiIndex.from_product(
                [sorted(out['BLS_Industry'].unique()), rel_bins],
                names=['BLS_Industry', 'period'])
            out = (out.set_index(['BLS_Industry', 'period'])
                      .reindex(idx).fillna(0.0).reset_index())
            return out
        
        pat_cols = ['clean_pat_count', 'dirty_pat_count', 'pat_count_nc', 'pat_count', 
                      'clean_pat_cites', 'dirty_pat_cites', 'pat_cites_nc', 'pat_cites']
        Ind_Pat_df = make_bins(Ind_Pat_yr_df, pat_cols)
        
        rd_cols   = ['pat_count_hat', 'pat_count_clean_hat', 'pat_count_dirty_hat',
                    'pat_cites_hat', 'pat_cites_clean_hat', 'pat_cites_dirty_hat']
        RD_shocks_df = make_bins(RD_shocks_yr_df, rd_cols)

       
        # ---------------- #
        # Leontief Inverse #
        # ---------------- # 
        def build_sigma_LI(IO_matrix):
            J = IO_matrix.shape[0]
            S = np.linalg.inv(np.eye(J) - IO_matrix)
            np.fill_diagonal(S, 0.0)
            return S
 
        Σ_LI = {year: build_sigma_LI(IO_mats[year]) for year in bin_ends}
        
        
        # -------------------- #
        # Greenification Rates #
        # -------------------- #
        def wide(col):
            return (Ind_Pat_df.pivot(index='period', columns='BLS_Industry', values=col)
                              .reindex(index=bin_ends, columns=manu_idx_all)
                              .sort_index())
 
        cln_p, drt_p, tot_p = wide('clean_pat_count'), wide('dirty_pat_count'), wide('pat_count')
        cln_c, drt_c, tot_c = wide('clean_pat_cites'), wide('dirty_pat_cites'), wide('pat_cites')
        dir_p, dir_c = cln_p + drt_p, cln_c + drt_c
        
        def estimate_kappa(cln_w, tot_w, min_den=100):
            c = cln_w.to_numpy(dtype=float).ravel()
            n = tot_w.to_numpy(dtype=float).ravel()
            ok = np.isfinite(c) & np.isfinite(n) & (n >= min_den)
            if ok.sum() < 20:
                print(f'  only {ok.sum()} sector-bins above {min_den}; '
                      f'falling back to kappa = 10')
                return 10.0
            p = np.nansum(c) / np.nansum(n)
            v = np.var(c[ok] / n[ok], ddof=1)
            samp = np.mean(p * (1 - p) / n[ok])
            kap = p * (1 - p) / max(v - samp, 1e-12) - 1
            return float(np.clip(kap, 1.0, 200.0))


        κ_dir_p = estimate_kappa(cln_p, dir_p)
        κ_dir_c = estimate_kappa(cln_c, dir_c)
        κ_pat   = estimate_kappa(cln_p, tot_p)
        κ_cite  = estimate_kappa(cln_c, tot_c)
        
        def shrink(num, den, kappa):
           gbar = (num.sum(axis=1) / den.sum(axis=1)).to_numpy()[:, None]
           return ((num + kappa * gbar) / (den + kappa)).where(den > 0)

        D_pat  = shrink(cln_p, dir_p, κ_dir_p)
        D_cite = shrink(cln_c, dir_c, κ_dir_c)
        G_pat  = shrink(cln_p, tot_p, κ_pat)
        G_cite = shrink(cln_c, tot_c, κ_cite)

        keep     = np.isin(manu_idx_all, Ind_Pat_df['BLS_Industry'].unique())
        keep_idx = manu_idx_all[keep]
        
        
        # ---------------------- #
        # Network Greenification #
        # ---------------------- #
        def partner_avg(S_sub, v, obs):
           v0   = np.where(obs, v, 0.0)

           up   = S_sub   @ v0 
           down = S_sub.T @ v0 
           
           return up, down

        frames = []
        for t in bin_ends:
            v_pat    = G_pat.loc[t].to_numpy(dtype=float)[keep]
            v_cite   = G_cite.loc[t].to_numpy(dtype=float)[keep]
            obs_pat  = np.isfinite(v_pat)
            obs_cite = np.isfinite(v_cite)
            
            v_pat_dir    = D_pat.loc[t].to_numpy(dtype=float)[keep]
            v_cite_dir   = D_cite.loc[t].to_numpy(dtype=float)[keep]
            obs_pat_dir  = np.isfinite(v_pat_dir)
            obs_cite_dir = np.isfinite(v_cite_dir)
 
            S          = Σ_LI[t][np.ix_(keep, keep)]
 
            up_p, dn_p = partner_avg(S, v_pat,  obs_pat)
            up_c, dn_c = partner_avg(S, v_cite, obs_cite)
            
            up_p_dir, dn_p_dir = partner_avg(S, v_pat_dir, obs_pat_dir)
            up_c_dir, dn_c_dir = partner_avg(S, v_cite_dir, obs_cite_dir)
 
            frames.append(pd.DataFrame({
                'BLS_Industry': keep_idx,
                'period':       t,
                'up_G_pat':     up_p,
                'down_G_pat':   dn_p,
                'up_G_cite':    up_c,
                'down_G_cite':  dn_c,
                'up_D_pat':     up_p_dir,
                'down_D_pat':   dn_p_dir,
                'up_D_cite':    up_c_dir,
                'down_D_cite':  dn_c_dir,
                
            }))
 
        net_df = pd.concat(frames, ignore_index=True)
        net_df['net_G_pat']  = net_df['up_G_pat']  + net_df['down_G_pat']
        net_df['net_G_cite'] = net_df['up_G_cite'] + net_df['down_G_cite']
        net_df['net_D_pat']  = net_df['up_D_pat']  + net_df['down_D_pat']
        net_df['net_D_cite'] = net_df['up_D_cite'] + net_df['down_D_cite']
 
        
        # ------------------ #
        # Own Greenification #
        # ------------------ #
        Ind_Pat_df['G_pat']  = (Ind_Pat_df['clean_pat_count']
                                / Ind_Pat_df['pat_count'].where(Ind_Pat_df['pat_count'] > 0))
        Ind_Pat_df['G_cite'] = (Ind_Pat_df['clean_pat_cites']
                                / Ind_Pat_df['pat_cites'].where(Ind_Pat_df['pat_cites'] > 0))
        
        Ind_Pat_df['clim_pat_count'] = Ind_Pat_df['clean_pat_count'] + Ind_Pat_df['dirty_pat_count'] #Move to cleaner
        Ind_Pat_df['clim_pat_cites'] = Ind_Pat_df['clean_pat_cites'] + Ind_Pat_df['dirty_pat_cites']
        
        Ind_Pat_df['D_pat']  = (Ind_Pat_df['clean_pat_count']
                                / Ind_Pat_df['clim_pat_count'].where(Ind_Pat_df['clim_pat_count'] > 0))
        Ind_Pat_df['D_cite'] = (Ind_Pat_df['clean_pat_cites']
                                / Ind_Pat_df['clim_pat_cites'].where(Ind_Pat_df['clim_pat_cites'] > 0))
 
        reg_df = net_df.merge(Ind_Pat_df, on=['BLS_Industry', 'period'], how='left')
        
        
        # ---- #
        # Lags #
        # ---- #
        lag_cols = ['up_G_pat', 'down_G_pat', 'net_G_pat', 'up_D_pat', 'down_D_pat', 'net_D_pat',
                    'up_G_cite', 'down_G_cite', 'net_G_cite', 'up_D_cite', 'down_D_cite', 'net_D_cite',
                    'G_pat', 'G_cite', 'D_pat', 'D_cite']
        lagged = reg_df[['BLS_Industry', 'period'] + lag_cols].copy()
        lagged['period'] = lagged['period'] + bin_len
        lagged = lagged.rename(columns={c: f'{c}_lag' for c in lag_cols})
        reg_df = reg_df.merge(lagged, on=['BLS_Industry', 'period'], how='left')
        
        
        # ------------------- #
        # Network Instruments #
        # ------------------- #
        RD_shocks_df['pat_count_clim_hat'] = RD_shocks_df['pat_count_clean_hat'] + RD_shocks_df['pat_count_dirty_hat'] #Move to cleaner
        RD_shocks_df['pat_cites_clim_hat'] = RD_shocks_df['pat_cites_clean_hat'] + RD_shocks_df['pat_cites_dirty_hat']
        
        S_fix = Σ_LI[BLS_year_start][np.ix_(keep, keep)]

        RD_shock_periods = sorted(set(RD_shocks_df['period']))
 
        shock_defs = {
            'rd_pat':   (RD_shocks_df,   'pat_count_clean_hat',   'pat_count_hat',
                         RD_shock_periods),
            'rd_cite':  (RD_shocks_df,   'pat_cites_clean_hat',   'pat_cites_hat',
                         RD_shock_periods),
            'rd_pat_dir':   (RD_shocks_df,   'pat_count_clean_hat',   'pat_count_clim_hat',
                         RD_shock_periods),
            'rd_cite_dir':  (RD_shocks_df,   'pat_cites_clean_hat',   'pat_cites_clim_hat',
                         RD_shock_periods),
        }

        def shock_share(src, num_col, den_col, periods, wins=0.02):
           num = (src.pivot(index='period', columns='BLS_Industry', values=num_col)
                     .reindex(index=periods, columns=manu_idx_all))
           den = (src.pivot(index='period', columns='BLS_Industry', values=den_col)
                     .reindex(index=periods, columns=manu_idx_all))
           w = num / den.where(den > 0)
           if wins:
               v = w.to_numpy(dtype=float)
               if np.isfinite(v).any():
                   lo, hi = np.nanquantile(v, [wins, 1 - wins])
                   w = w.clip(lower=lo, upper=hi)
           return w

        z_parts = []
        for tag, (src, num, den, periods) in shock_defs.items():
            Gz, rows = shock_share(src, num, den, periods), []
            for t in periods:
                v   = Gz.loc[t].to_numpy(dtype=float)[keep]
                obs = np.isfinite(v)
                if not obs.any():
                    print(f'  {tag}: no finite shares in {t}, skipped')
                    continue
                up, dn = partner_avg(S_fix, v, obs)
                rows.append(pd.DataFrame({'BLS_Industry': keep_idx, 'period': t,
                                          f'z_up_{tag}': up, f'z_dn_{tag}': dn}))
            if rows:
                z_parts.append(pd.concat(rows, ignore_index=True))
            else:
                print(f'  {tag}: NO usable periods')
 
        z_df = z_parts[0]
        for part in z_parts[1:]:
            z_df = z_df.merge(part, on=['BLS_Industry', 'period'], how='outer')
        z_cols_all = [c for c in z_df.columns if c.startswith('z_')]

        # Lag the instruments
        z_lag = z_df.copy()
        z_lag['period'] = z_lag['period'] + bin_len
        z_lag = z_lag.rename(columns={c: f'{c}_lag' for c in z_cols_all})
        reg_df = reg_df.merge(z_lag, on=['BLS_Industry', 'period'], how='left')
        
        
        # ----------------------------------------------------------------
        
        # Run regressions.
        
        # ----------------------------------------------------------------
        def fit_ppml(df, y_col, offset_col, x_cols, entity_fe=True, time_fe=True):
            "Poisson pseudo-ML with log(offset), sector and period dummies, clustered SE"
            need = [y_col, offset_col] + list(x_cols)
            d    = df.dropna(subset=need).copy()
            d    = d[d[offset_col] > 0]
 
            pos     = d.groupby('BLS_Industry')[y_col].transform('sum') > 0
            n_drop  = int(d['BLS_Industry'][~pos].nunique())
            d       = d[pos]
            if n_drop:
                print(f'  fit_ppml({y_col}): dropped {n_drop} sector(s) with no '
                      f'positive outcome in any period.')
 
            parts = [pd.Series(1.0, index=d.index, name='const'),
                     d[list(x_cols)].astype(float)]
            if entity_fe:
                parts.append(pd.get_dummies(d['BLS_Industry'], prefix='sec',
                                            drop_first=True, dtype=float))
            if time_fe:
                parts.append(pd.get_dummies(d['period'], prefix='per',
                                            drop_first=True, dtype=float))
            X = pd.concat(parts, axis=1)
            X.columns = [str(c) for c in X.columns]
 
            res = sm.GLM(d[y_col].astype(float), X,
                         family=sm.families.Poisson(),
                         offset=np.log(d[offset_col].astype(float).to_numpy())
                         ).fit(cov_type='cluster',
                               cov_kwds={'groups': d['BLS_Industry'].to_numpy()},
                               maxiter=200)
 
            fe = (['sector'] if entity_fe else []) + (['period'] if time_fe else [])
            return es.GLMWrap(res, y_col, list(x_cols), offset_col, fe,
                           n_sectors=d['BLS_Industry'].nunique())


        # ---------------------------------------------------------------- #
        # First stage: do the instruments move the endogenous regressors?   #
        # ---------------------------------------------------------------- #
        def first_stage_matrix(endogs, z_cols, label=''):
            cols = list(endogs) + list(z_cols)
            d = reg_df.dropna(subset=cols).copy()
            for c in cols:
                d[c] = d[c] - d.groupby('BLS_Industry')[c].transform('mean')
                d[c] = d[c] - d.groupby('period')[c].transform('mean')
            X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in z_cols])
            k, rows, fitted = len(z_cols), [], {}
            for e in endogs:
                Y = d[e].to_numpy(float)
                b, *_ = np.linalg.lstsq(X, Y, rcond=None)
                res = Y - X @ b
                r2  = 1 - (res**2).sum() / max(((Y - Y.mean())**2).sum(), 1e-12)
                rows.append({'endog': e, 'N': len(d),
                             'clusters': d['BLS_Industry'].nunique(),
                             'partial R2': r2,
                             'F': (r2 / max(1 - r2, 1e-12)) * (len(d) - k - 1) / k,
                             **{c: b[i + 1] for i, c in enumerate(z_cols)}})
                fitted[e] = X @ b
            print(f'\nFirst stage {label} (sector + period demeaned)')
            print(pd.DataFrame(rows).round(4).to_string(index=False))
            f = pd.DataFrame(fitted)
            if f.shape[1] == 2:
                rr = f.corr().iloc[0, 1]
                print(f'  corr(fitted {endogs[0]}, fitted {endogs[1]}) = {rr:+.3f}'
                      f'{"   <-- directions NOT separately identified" if abs(rr) > 0.9 else ""}')
            print('  F below ~10 means the instrument does not move the regressor; '
                  'IV estimates\n  and their standard errors are then unreliable '
                  'regardless of what they print.')
 
        first_stage_matrix(['up_G_pat_lag', 'down_G_pat_lag'],
                           ['z_up_rd_pat_lag', 'z_dn_rd_pat_lag'],  'RD / patents')
     
        first_stage_matrix(['up_G_cite_lag', 'down_G_cite_lag'],
                           ['z_up_rd_cite_lag', 'z_dn_rd_cite_lag'], 'RD / cites')
        
        first_stage_matrix(['up_D_pat_lag', 'down_D_pat_lag'],
                           ['z_up_rd_pat_dir_lag', 'z_dn_rd_pat_dir_lag'],  'RD / patents')
     
        first_stage_matrix(['up_D_cite_lag', 'down_D_cite_lag'],
                           ['z_up_rd_cite_dir_lag', 'z_dn_rd_cite_dir_lag'], 'RD / cites')
        

        # ---------- #
        # Estimation #
        # ---------- #
        # Green patent counts, lagged partner adoption
        m_pat_net  = fit_ppml(reg_df, 'clean_pat_count', 'dirty_pat_count',
                             ['net_D_pat_lag'])
        
        m_pat_ud  = fit_ppml(reg_df, 'clean_pat_count', 'dirty_pat_count',
                             ['up_D_pat_lag', 'down_D_pat_lag'])
        
        m_pat_net_lag  = fit_ppml(reg_df, 'clean_pat_count', 'dirty_pat_count',
                             ['net_D_pat_lag', 'D_pat_lag'])
        
        m_pat_net_gen  = fit_ppml(reg_df, 'clean_pat_count', 'pat_count_nc',
                             ['net_G_pat_lag'])
 
        # Green citations, lagged partner adoption
        m_cit_net  = fit_ppml(reg_df, 'clean_pat_cites', 'dirty_pat_cites',
                             ['net_D_cite_lag'])
        
        m_cit_ud  = fit_ppml(reg_df, 'clean_pat_cites', 'dirty_pat_cites',
                             ['up_D_cite_lag', 'down_D_cite_lag'])
        
        m_cit_net_lag  = fit_ppml(reg_df, 'clean_pat_cites', 'dirty_pat_cites',
                             ['net_D_cite_lag', 'D_cite_lag'])
        
        m_cit_net_gen  = fit_ppml(reg_df, 'clean_pat_cites', 'pat_cites_nc',
                             ['net_G_cite_lag'])
        
        # # IV
        # iv_pat_rd  = es.fit_poisson_iv(reg_df, 'clean_pat_count', 'pat_count_nc',
        #                             ['up_G_pat_lag', 'down_G_pat_lag', 'G_pat_lag'],
        #                             endog_cols=['up_G_pat_lag', 'down_G_pat_lag'],
        #                             instrument_cols=['z_up_rd_pat_lag',  'z_dn_rd_pat_lag'])
        
        # iv_pat_rd_dir  = es.fit_poisson_iv(reg_df, 'clean_pat_count', 'dirty_pat_count',
        #                             ['up_D_pat_lag', 'down_D_pat_lag', 'D_pat_lag'],
        #                             endog_cols=['up_D_pat_lag', 'down_D_pat_lag'],
        #                             instrument_cols=['z_up_rd_pat_dir_lag', 'z_dn_rd_pat_dir_lag'])
        
        
        # iv_cit_rd  = es.fit_poisson_iv(reg_df, 'clean_pat_cites', 'pat_cites_nc',
        #                             ['up_G_cite_lag', 'down_G_cite_lag', 'G_cite_lag'],
        #                             endog_cols=['up_G_cite_lag', 'down_G_cite_lag'],
        #                             instrument_cols=['z_up_rd_cite_lag', 'z_dn_rd_cite_lag'])
        
        # iv_cit_rd_dir  = es.fit_poisson_iv(reg_df, 'clean_pat_cites', 'dirty_pat_cites',
        #                             ['up_D_cite_lag', 'down_D_cite_lag', 'D_cite_lag'],
        #                             endog_cols=['up_D_cite_lag', 'down_D_cite_lag'],
        #                             instrument_cols=['z_up_rd_cite_dir_lag', 'z_dn_rd_cite_dir_lag'])
        
        Models = {'pat_net': m_pat_net, 'pat_ud': m_pat_ud, 'pat_net_lag': m_pat_net_lag, 'pat_net_gen': m_pat_net_gen,
                  'cit_net': m_cit_net, 'cit_ud': m_cit_ud, 'cit_net_lag': m_cit_net_lag, 'cit_net_gen': m_cit_net_gen}
 
        def show(models=None):
            for name, m in (models or Models).items():
                print(f'\n{"="*78}\n{name}\n{"="*78}\n{m!r}')
 
        show()
        
        
        ## Govt IV with university
        ## Other denominator for G
        ## Control for own shock on IVs
        ## Coefficiant on offset
        ## IV from network
 
        self.reg_df = reg_df
        self.Models = Models
 
        # ------------------- #
        # Summary Stats Table #
        # ------------------- #
        labels = {
            'G_pat':          'Green Patent Share',
            'G_cite':         'Green Citation Share',
            'up_G_pat_lag':   'Upstream Green Patent Share, lagged',
            'down_G_pat_lag': 'Downstream Green Patent Share, lagged',
            'net_G_pat_lag':  'Network Green Patent Share, lagged',
            'up_G_cite_lag':  'Upstream Green Citation Share, lagged',
            'down_G_cite_lag':'Downstream Green Citation Share, lagged',
            'net_G_cite_lag': 'Network Green Citation Share, lagged',
            'G_pat_lag':      'Own Green Patent Share, lagged',
            'G_cite_lag':     'Own Green Citation Share, lagged',
            'up_G_pat':       'Upstream Green Patent Share',
            'down_G_pat':     'Downstream Green Patent Share',
            'net_G_pat':      'Network Green Patent Share',
            'up_G_cite':      'Upstream Green Citation Share',
            'down_G_cite':    'Downstream Green Citation Share',
            'net_G_cite':     'Network Green Citation Share',
            's_up':           'Upstream Network Exposure',
            's_dn':           'Downstream Network Exposure',
        }
 
        stat_vars = ['G_pat', 'G_cite',
                     'up_G_pat_lag', 'down_G_pat_lag', 'net_G_pat_lag',
                     'up_G_cite_lag', 'down_G_cite_lag', 'net_G_cite_lag',
                     'G_pat_lag', 'G_cite_lag', 's_up', 's_dn']
 
        rows = []
        for col in stat_vars:
            s = reg_df[col].dropna()
            rows.append((labels[col], s.mean(), s.std(),
                         s.quantile(0.75) - s.quantile(0.25), len(s)))
 
        print(f'\n{"="*78}\nSUMMARY STATISTICS\n{"="*78}')
        print(pd.DataFrame(rows, columns=['Variable', 'Mean', 'SD', 'IQR', 'Obs'])
              .round(4).to_string(index=False))
 
        body = ''
        for label, mean, sd, iqr, n in rows:
            body += f'{label} & {mean:.3f} & {sd:.3f} & {iqr:.3f} & {n} \\\\\n'
        with open(f'{self.Directory}/Results/Tables/Summary_Stats.tex', 'w') as f:
            f.write(body)
 
 
        # ----------- #
        # Print Table #
        # ----------- #
        def build_table(models, variables, row_skip='[3pt]'):
            body = ''
            for varname in variables:
                coefs, ses = [], []
                for m in models:
                    if m is None:
                        coefs.append(''); ses.append('')
                    else:
                        c, s = gpf.fmt_coef(m, varname)
                        coefs.append(c); ses.append(s)
                body += f'{labels[varname]} & {" & ".join(coefs)} \\\\\n'
                body += f'& {" & ".join(ses)} \\\\{row_skip}\n'
            r2_vals, n_vals = [], []
            for m in models:
                if m is None:
                    r2_vals.append(''); n_vals.append('')
                else:
                    r2_vals.append('' if not np.isfinite(m.rsquared) else f'{m.rsquared:.3f}')
                    n_vals.append(str(int(m.nobs)))
            body += '\\midrule'
            body += f'Pseudo $R^2$ & {" & ".join(r2_vals)} \\\\\n'
            body += f'Obs & {" & ".join(n_vals)} \\'
            return body
 
        def print_table(title, models, variables, heads, dec=3):
            print(f'\n{"="*78}\n{title}\n{"="*78}')
            live = [i for i, m in enumerate(models) if m is not None]
            mods, cols = [models[i] for i in live], [heads[i] for i in live]
            out = []
            for var in variables:
                cs, ss = [], []
                for m in mods:
                    if var in m.params.index:
                        cs.append(f'{m.params[var]:.{dec}f}'
                                  f'{gpf.get_stars(m.pvalues[var])}')
                        ss.append(f'({m.bse[var]:.{dec}f})')
                    else:
                        cs.append(''); ss.append('')
                out.append([labels[var]] + cs)
                out.append([''] + ss)
            out.append(['Pseudo R2'] + ['' if not np.isfinite(m.rsquared)
                                        else f'{m.rsquared:.3f}' for m in mods])
            out.append(['Obs'] + [str(int(m.nobs)) for m in mods])
            print(pd.DataFrame(out, columns=[''] + cols).to_string(index=False))
 
        net_models = [m_pat_net, m_cit_net, None, m_pat_net_c]
        net_heads  = ['Patents', 'Citations', '', 'Patents (contemp.)',]
        net_vars   = ['net_G_pat_lag', 'net_G_cite_lag', 'net_G_pat',
                      'G_pat_lag', 'G_cite_lag', 's_up', 's_dn']
 
        ud_models  = [m_pat_ud, m_cit_ud, None, m_pat_ud_c]
        ud_heads   = net_heads
        ud_vars    = ['up_G_pat_lag', 'down_G_pat_lag',
                      'up_G_cite_lag', 'down_G_cite_lag',
                      'up_G_pat', 'down_G_pat',
                      'G_pat_lag', 'G_cite_lag', 's_up', 's_dn']
 
        print_table('NETWORK EFFECTS (net)',          net_models, net_vars, net_heads)
        print_table('NETWORK EFFECTS (up/down)',      ud_models,  ud_vars,  ud_heads)
 
        for tag, models, variables in [('Net',    net_models, net_vars),
                                       ('UpDown', ud_models,  ud_vars)]:
            with open(f'{self.Directory}/Results/Tables/Network_PPML_{tag}.tex', 'w') as f:
                f.write(build_table(models, variables))
    
    
    
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

    






    