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
        self.CPC_classes = ["Y02E", "Y02P", "Y02T", "B60L"]
        self.ICE_classes = ["Y02T10/10", "Y02T10/20", "Y02T10/30", "Y02T10/40"]
        self.manu_cols = [1, 93]
        self.fossil_cols = [7-1, 8-1]#, 12-1] #Exclude electricity as well ## Double check!
        
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
                Clean Data/Ind_Pat_Shares_Pre.pkl
                Clean Data/Gov_CPC.pkl
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
            
        Gov_Pats_df = PV_assignee_df[PV_assignee_df['assignee_type']==6]
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
            
        cpc4_df = CPC_df[['patent_id', 'cpc_subclass']].drop_duplicates()

                
        
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
        
        codes = set(self.CPC_classes)
        relevant_df['clean'] = (relevant_df["cpc_subclass"].isin(codes)).astype(np.int8)
        
        ice_codes = set(self.ICE_classes)
        relevant_df['ice'] = relevant_df["cpc_group"].isin(ice_codes).astype(np.int8)
        
        relevant_df['clean_full'] = relevant_df.groupby("patent_id")['clean'].transform("max")
        relevant_df['clean'] = relevant_df['clean_full'] - relevant_df.groupby("patent_id")['ice'].transform("max")
        relevant_df = relevant_df[['patent_id', 'year', 'clean', 'clean_full']].drop_duplicates()
        
        
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
        
        pat_CPC_df = pd.merge(pat_df,
                                cpc4_df,
                                on='patent_id',
                                how='inner'
                                )
        pat_CPC_df = pat_CPC_df[~((pat_CPC_df['clean'] == 0)
                                    & (pat_CPC_df['clean_full'] == 1)
                                    & (pat_CPC_df['cpc_subclass'] == 'Y02T'))]
        
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
        
        pat_firms_df = pat_firms_df[['patent_id', 'year', 'gvkey', 'BLS_Industry', 'clean', 'clean_full', 'norm_cites']].drop_duplicates()
        pat_ind_df = pat_firms_df[['patent_id', 'year', 'BLS_Industry', 'clean', 'clean_full', 'norm_cites']].drop_duplicates()
        
        pat_firms_df['split_weight'] = 1 / pat_firms_df.groupby('patent_id')['gvkey'].transform('count')
        pat_firms_df.to_pickle(f'{self.Directory}/Clean Data/Pat_Firms.pkl')
        
        pat_ind_df['split_weight'] = 1 / pat_ind_df.groupby('patent_id')['BLS_Industry'].transform('count')
        
        pat_CPC_df = pat_CPC_df.merge(pat_ind_df[['patent_id', 'BLS_Industry', 'split_weight']].drop_duplicates(),
                            on='patent_id',
                            how='left')
        
        
        # ------------------------- #
        # Clean Patenting by Sector #
        # ------------------------- #
        annual_df = pat_ind_df.copy()
        annual_df['clean_w']      = annual_df['split_weight'] * annual_df['clean']
        annual_df['clean_full_w'] = annual_df['split_weight'] * annual_df['clean_full']
        annual_df['cite_w']       = annual_df['split_weight'] * annual_df['norm_cites']
        annual_df['clean_cite_w'] = annual_df['clean_w']      * annual_df['norm_cites']
        annual_df['clean_full_cite_w'] = annual_df['clean_full_w'] * annual_df['norm_cites']
         
        agg_base = dict(clean_pat_count = ('clean_w',      'sum'),
                        pat_count       = ('split_weight', 'sum'),
                        clean_pat_cites = ('clean_cite_w', 'sum'),
                        pat_cites       = ('cite_w',       'sum'))
        agg_full = dict(clean_pat_count = ('clean_full_w',      'sum'),
                        pat_count       = ('split_weight',      'sum'),
                        clean_pat_cites = ('clean_full_cite_w', 'sum'),
                        pat_cites       = ('cite_w',            'sum'))

        bin_ends = list(range(BLS_year_start, Year_end + 1, 5))
        frames, frames_full = [], []
        for start in range(BLS_year_start - 5, Year_end, 5):
            end    = start + 5
            bin_df = annual_df[(annual_df['year'] > start) & (annual_df['year'] <= end)]
            frames.append(bin_df.groupby('BLS_Industry', as_index=False)
                                .agg(**agg_base).assign(period=end))
            frames_full.append(bin_df.groupby('BLS_Industry', as_index=False)
                                     .agg(**agg_full).assign(period=end))
 
        ind_pat_df      = pd.concat(frames,      ignore_index=True)
        ind_pat_df_full = pd.concat(frames_full, ignore_index=True)
 
        panel_idx = pd.MultiIndex.from_product(
            [sorted(ind_pat_df['BLS_Industry'].unique()), list(bin_ends)],
            names=['BLS_Industry', 'period'])
        ind_pat_df = (ind_pat_df.set_index(['BLS_Industry', 'period'])
                           .reindex(panel_idx)
                           .fillna(0.0)
                           .reset_index())
        ind_pat_df_full = (ind_pat_df_full.set_index(['BLS_Industry', 'period'])
                           .reindex(panel_idx)
                           .fillna(0.0)
                           .reset_index())
 
        ind_pat_df = ind_pat_df[ind_pat_df['BLS_Industry'] != 71]
 
        ind_pat_df.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat.pkl')
        ind_pat_df_full.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat_full.pkl')

        
        # -------------------- #
        # CPC Shares by Sector #
        # -------------------- #
        ind_pat_shares_pre_df = pat_CPC_df.dropna(subset=['BLS_Industry'])
        ind_pat_shares_pre_df = ind_pat_shares_pre_df[(ind_pat_shares_pre_df['year'] >= BLS_year_start-5-10) & (ind_pat_shares_pre_df['year'] < BLS_year_start-5)]
        
        ind_pat_shares_pre_df['pat_weight'] = ind_pat_shares_pre_df['split_weight'] / ind_pat_shares_pre_df.groupby('patent_id')['cpc_subclass'].transform('count')
        ind_pat_shares_pre_df['cite_weight'] = ind_pat_shares_pre_df['pat_weight'] * ind_pat_shares_pre_df['norm_cites']
        
        ind_pat_shares_pre_df['pat_weight_clean'] = ind_pat_shares_pre_df['clean'] * ind_pat_shares_pre_df['split_weight'] / ind_pat_shares_pre_df.groupby('patent_id')['cpc_subclass'].transform('count')
        ind_pat_shares_pre_df['cite_weight_clean'] = ind_pat_shares_pre_df['pat_weight_clean'] * ind_pat_shares_pre_df['norm_cites']
        
        
        ind_pat_shares_pre_df['cpc_pat_count'] = ind_pat_shares_pre_df.groupby(['BLS_Industry', 'cpc_subclass'])['pat_weight'].transform('sum')
        ind_pat_shares_pre_df['pat_count'] = ind_pat_shares_pre_df.groupby('BLS_Industry')['pat_weight'].transform('sum')
        ind_pat_shares_pre_df['cpc_pat_share'] = ind_pat_shares_pre_df['cpc_pat_count'] / ind_pat_shares_pre_df['pat_count']
        
        ind_pat_shares_pre_df['cpc_pat_count_clean'] = ind_pat_shares_pre_df.groupby(['BLS_Industry', 'cpc_subclass'])['pat_weight_clean'].transform('sum')
        ind_pat_shares_pre_df['pat_count_clean'] = ind_pat_shares_pre_df.groupby('BLS_Industry')['pat_weight_clean'].transform('sum')
        ind_pat_shares_pre_df['cpc_pat_share_clean'] = ind_pat_shares_pre_df['cpc_pat_count_clean'] / ind_pat_shares_pre_df['pat_count_clean']
    
        
        ind_pat_shares_pre_df['cpc_pat_cites'] = ind_pat_shares_pre_df.groupby(['BLS_Industry', 'cpc_subclass'])['cite_weight'].transform('sum')
        ind_pat_shares_pre_df['pat_cites'] = ind_pat_shares_pre_df.groupby('BLS_Industry')['cite_weight'].transform('sum')
        ind_pat_shares_pre_df['cpc_cite_share'] = ind_pat_shares_pre_df['cpc_pat_cites'] / ind_pat_shares_pre_df['pat_cites']
        
        ind_pat_shares_pre_df['cpc_pat_cites_clean'] = ind_pat_shares_pre_df.groupby(['BLS_Industry', 'cpc_subclass'])['cite_weight_clean'].transform('sum')
        ind_pat_shares_pre_df['pat_cites_clean'] = ind_pat_shares_pre_df.groupby('BLS_Industry')['cite_weight_clean'].transform('sum')
        ind_pat_shares_pre_df['cpc_cite_share_clean'] = ind_pat_shares_pre_df['cpc_pat_cites_clean'] / ind_pat_shares_pre_df['pat_cites_clean']
        
        
        ind_pat_shares_pre_df = ind_pat_shares_pre_df[['BLS_Industry', 'cpc_subclass', 'cpc_pat_share', 'cpc_pat_share_clean', 'cpc_cite_share', 'cpc_cite_share_clean']].drop_duplicates()
        panel_idx = pd.MultiIndex.from_product(
            [sorted(cpc4_df['cpc_subclass'].unique()), sorted(ind_pat_shares_pre_df['BLS_Industry'].unique())],
            names=['cpc_subclass', 'BLS_Industry'])
        ind_pat_shares_pre_df = (ind_pat_shares_pre_df.set_index(['cpc_subclass', 'BLS_Industry'])
                                .reindex(panel_idx)
                                .fillna(0.0)
                                .reset_index())
        
        ind_pat_shares_pre_df.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat_Shares_Pre.pkl')
        
        
        # ------------------------ #
        # Government Patent Series #
        # ------------------------ #
        gov_cpc_df = pd.merge(pat_CPC_df,
                             Gov_Pats_df[['patent_id']].drop_duplicates(),
                             on='patent_id',
                             how='inner')
        
        gov_cpc_df['pat_weight'] = 1.0 / gov_cpc_df.groupby('patent_id')['cpc_subclass'].transform('count')
        gov_cpc_df['cite_weight'] = gov_cpc_df['pat_weight'] * gov_cpc_df['norm_cites']
        
        gov_cpc_df['pat_weight_clean'] = gov_cpc_df['clean'] / gov_cpc_df.groupby('patent_id')['cpc_subclass'].transform('count')
        gov_cpc_df['cite_weight_clean'] = gov_cpc_df['pat_weight_clean'] * gov_cpc_df['norm_cites']
        
        gov_frames = []
        for start in range(BLS_year_start - 5, Year_end, 5):
            end    = start + 5
            bin_df = gov_cpc_df[(gov_cpc_df['year'] > start) & (gov_cpc_df['year'] <= end)]
            
            bin_df['gov_pat_count'] = bin_df.groupby(['cpc_subclass'])['pat_weight'].transform('sum')
            bin_df['gov_pat_cites'] = bin_df.groupby(['cpc_subclass'])['cite_weight'].transform('sum')
            
            bin_df['gov_pat_count_clean'] = bin_df.groupby(['cpc_subclass'])['pat_weight_clean'].transform('sum')
            bin_df['gov_pat_cites_clean'] = bin_df.groupby(['cpc_subclass'])['cite_weight_clean'].transform('sum')
            
            gov_frames.append(bin_df[['cpc_subclass', 'gov_pat_count', 'gov_pat_count_clean', 'gov_pat_cites', 'gov_pat_cites_clean']].drop_duplicates().assign(period=end))
 
        gov_cpc_df = pd.concat(gov_frames, ignore_index=True)
        
        panel_idx = pd.MultiIndex.from_product(
               [sorted(cpc4_df['cpc_subclass'].unique()), list(bin_ends)],
               names=['cpc_subclass', 'period'])
        gov_cpc_df = (gov_cpc_df.set_index(['cpc_subclass', 'period'])
                              .reindex(panel_idx)
                              .fillna(0.0)
                              .reset_index())

        gov_cpc_df.to_pickle(f'{self.Directory}/Clean Data/Gov_CPC.pkl')
            
            
            
    def Instruments(self, BLS_year_start, Year_end):
        """""
        Create Three Series of Greenification Shocks
    
        Output: Clean Data/Govt_Shocks.pkl
                Clean Data/RD_Shocks.pkl
        """""
        
        # ----------------------------------------------------------------

        # Build instrument dataframes.

        # ----------------------------------------------------------------
        
        gov_cpc_df = pd.read_pickle(f'{self.Directory}/Clean Data/Gov_CPC.pkl')
        ind_pat_shares_pre_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_Pat_Shares_Pre.pkl')
        
        state_rdp_df = pd.read_pickle(f'{self.Directory}/Clean Data/state_rd_price.pkl')
        PV_inventor_location_df = pd.read_pickle(f'{self.Directory}/Clean Data/Inventor_Locations.pkl')
        pat_firms_df = pd.read_pickle(f'{self.Directory}/Clean Data/Pat_Firms.pkl')
                
        
        # ------------------------ #
        # Government Patent Shocks #
        # ------------------------ #
        govt_shocks_df = pd.merge(gov_cpc_df,
                                    ind_pat_shares_pre_df,
                                    on='cpc_subclass',
                                    how='inner'
                                    )
        
        govt_shocks_df['weighted_pat_govt'] = govt_shocks_df['cpc_pat_share'] * govt_shocks_df['gov_pat_count']
        govt_shocks_df['pat_govt_shock'] = govt_shocks_df.groupby(['BLS_Industry', 'period'])['weighted_pat_govt'].transform('sum')
        
        govt_shocks_df['weighted_pat_govt_clean'] = govt_shocks_df['cpc_pat_share_clean'] * govt_shocks_df['gov_pat_count_clean']
        govt_shocks_df['pat_govt_shock_clean'] = govt_shocks_df.groupby(['BLS_Industry', 'period'])['weighted_pat_govt_clean'].transform('sum')
        
        govt_shocks_df['weighted_cite_govt'] = govt_shocks_df['cpc_cite_share'] * govt_shocks_df['gov_pat_cites']
        govt_shocks_df['cite_govt_shock'] = govt_shocks_df.groupby(['BLS_Industry', 'period'])['weighted_cite_govt'].transform('sum')
        
        govt_shocks_df['weighted_cite_govt_clean'] = govt_shocks_df['cpc_pat_share_clean'] * govt_shocks_df['gov_pat_cites_clean']
        govt_shocks_df['cite_govt_shock_clean'] = govt_shocks_df.groupby(['BLS_Industry', 'period'])['weighted_cite_govt_clean'].transform('sum')
        
        govt_shocks_df = govt_shocks_df[['BLS_Industry', 'period',
                                         'pat_govt_shock', 'pat_govt_shock_clean', 
                                         'cite_govt_shock', 'cite_govt_shock_clean']].drop_duplicates()
        
        govt_shocks_df.to_pickle(f'{self.Directory}/Clean Data/Govt_Shocks.pkl')
        
        
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
        firm_inv_df['pat_weight_clean'] = firm_inv_df['clean'] * firm_inv_df['split_weight'] / firm_inv_df['pat_authors']
        
        firm_inv_df['cite_weight'] = firm_inv_df['norm_cites'] * firm_inv_df['split_weight'] / firm_inv_df['pat_authors']
        firm_inv_df['cite_weight_clean'] = firm_inv_df['clean'] * firm_inv_df['norm_cites'] * firm_inv_df['split_weight'] / firm_inv_df['pat_authors']
        
        w_cols = ['pat_weight', 'pat_weight_clean', 'cite_weight', 'cite_weight_clean']
        
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
 
        fsy_win = fsy_win[(fsy_win['year'] >= BLS_year_start) & (fsy_win['year'] <= Year_end)]
 
        firm_tot = (fsy_win.groupby(['gvkey', 'year'], as_index=False)[w_cols]
                           .sum()
                           .rename(columns={c: f'{c}_tot' for c in w_cols}))
        fsy_win = fsy_win.merge(firm_tot, on=['gvkey', 'year'], how='left')
        
        share_map = {
            'firm_fips_pat_share':        ('pat_weight',        'pat_weight_tot'),
            'firm_fips_pat_share_clean':  ('pat_weight_clean',  'pat_weight_clean_tot'),
            'firm_fips_cite_share':       ('cite_weight',       'cite_weight_tot'),
            'firm_fips_cite_share_clean': ('cite_weight_clean', 'cite_weight_clean_tot'),
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
        
        firm_inv_df['weighted_rho_pats'] = firm_inv_df['firm_fips_pat_share'] * firm_inv_df['rho_h']
        firm_inv_df['weighted_rho_pats_clean'] = firm_inv_df['firm_fips_pat_share_clean'] * firm_inv_df['rho_h']
        
        firm_inv_df['weighted_rho_cites'] = firm_inv_df['firm_fips_cite_share'] * firm_inv_df['rho_h']
        firm_inv_df['weighted_rho_cites_clean'] = firm_inv_df['firm_fips_cite_share_clean'] * firm_inv_df['rho_h']
        
        firm_inv_df['E_rho_pats'] = firm_inv_df.groupby(['gvkey', 'year'])['weighted_rho_pats'].transform('sum')
        firm_inv_df['E_rho_pats_clean'] = firm_inv_df.groupby(['gvkey', 'year'])['weighted_rho_pats_clean'].transform('sum')
        
        firm_inv_df['E_rho_cites'] = firm_inv_df.groupby(['gvkey', 'year'])['weighted_rho_cites'].transform('sum')
        firm_inv_df['E_rho_cites_clean'] = firm_inv_df.groupby(['gvkey', 'year'])['weighted_rho_cites_clean'].transform('sum')
        
        firm_inv_df = firm_inv_df[['gvkey', 'year', 'E_rho_pats', 'E_rho_pats_clean', 'E_rho_cites', 'E_rho_cites_clean']].drop_duplicates()

        
        # Firm Patenting
        firm_pats_df = pat_firms_df.copy()
        
        firm_pats_df['pat_weight'] = firm_pats_df['split_weight']
        firm_pats_df['pat_weight_clean'] = firm_pats_df['clean'] * firm_pats_df['split_weight']
        
        firm_pats_df['cite_weight'] = firm_pats_df['norm_cites'] * firm_pats_df['split_weight']
        firm_pats_df['cite_weight_clean'] = firm_pats_df['clean'] * firm_pats_df['norm_cites'] * firm_pats_df['split_weight'] 
        
        firm_pats_df['pat_count'] = firm_pats_df.groupby(['gvkey', 'year'])['pat_weight'].transform('sum')
        firm_pats_df['pat_count_clean'] = firm_pats_df.groupby(['gvkey', 'year'])['pat_weight_clean'].transform('sum')
        
        firm_pats_df['pat_cites'] = firm_pats_df.groupby(['gvkey', 'year'])['cite_weight'].transform('sum')
        firm_pats_df['pat_cites_clean'] = firm_pats_df.groupby(['gvkey', 'year'])['cite_weight_clean'].transform('sum')

        
        firm_pats_df = firm_pats_df[['gvkey', 'BLS_Industry', 'year', 'pat_count', 'pat_count_clean', 'pat_cites', 'pat_cites_clean']].drop_duplicates()
        
        
        # Zero Stage Regressions
        firm_pat_panel_df = pd.merge(firm_pats_df,
                                firm_inv_df,
                                on=['gvkey', 'year'],
                                how='inner'
                                )
                
        firm_pat_panel_df['ln_pat_count'] = np.log(firm_pat_panel_df['pat_count'].where(firm_pat_panel_df['pat_count'] > 0))
        firm_pat_panel_df['ln_pat_count_clean'] = np.log(firm_pat_panel_df['pat_count_clean'].where(firm_pat_panel_df['pat_count_clean'] > 0))
        
        firm_pat_panel_df['ln_pat_cites'] = np.log(firm_pat_panel_df['pat_cites'].where(firm_pat_panel_df['pat_cites'] > 0))
        firm_pat_panel_df['ln_pat_cites_clean'] = np.log(firm_pat_panel_df['pat_cites_clean'].where(firm_pat_panel_df['pat_cites_clean'] > 0))
        
        firm_pat_panel_df['ln_E_rho_pats'] = np.log(firm_pat_panel_df['E_rho_pats'].where(firm_pat_panel_df['E_rho_pats'] > 0))
        firm_pat_panel_df['ln_E_rho_pats_clean'] = np.log(firm_pat_panel_df['E_rho_pats_clean'].where(firm_pat_panel_df['E_rho_pats_clean'] > 0))
        
        firm_pat_panel_df['ln_E_rho_cites'] = np.log(firm_pat_panel_df['E_rho_cites'].where(firm_pat_panel_df['E_rho_cites'] > 0))
        firm_pat_panel_df['ln_E_rho_cites_clean'] = np.log(firm_pat_panel_df['E_rho_cites_clean'].where(firm_pat_panel_df['E_rho_cites_clean'] > 0))
        
        firm_pat_panel_df['entity'] = firm_pat_panel_df['gvkey'].astype(str)
        firm_pat_panel_df = firm_pat_panel_df.set_index(['entity','year']).sort_index()
    
        m_pats = gpf.run_reg(firm_pat_panel_df['ln_pat_count'], firm_pat_panel_df['ln_E_rho_pats'], 'panel')
        m_pats_clean = gpf.run_reg(firm_pat_panel_df['ln_pat_count_clean'], firm_pat_panel_df['ln_E_rho_pats_clean'], 'panel')

        m_cites = gpf.run_reg(firm_pat_panel_df['ln_pat_cites'], firm_pat_panel_df['ln_E_rho_cites'], 'panel')
        m_cites_clean = gpf.run_reg(firm_pat_panel_df['ln_pat_cites_clean'], firm_pat_panel_df['ln_E_rho_cites_clean'], 'panel')
        
        firm_pat_panel_df['pat_count_hat'] = np.exp(m_pats.predict().fitted_values)
        firm_pat_panel_df['pat_count_clean_hat'] = np.exp(m_pats_clean.predict().fitted_values)
        
        firm_pat_panel_df['pat_cites_hat'] = np.exp(m_cites.predict().fitted_values)
        firm_pat_panel_df['pat_cites_clean_hat'] = np.exp(m_cites_clean.predict().fitted_values)
        firm_pat_panel_df = firm_pat_panel_df.reset_index()
        
        
        RD_frames = []
        for start in range(BLS_year_start-5, Year_end, 5):
            end = start + 5
            bin_df = firm_pat_panel_df[(firm_pat_panel_df['year'] > start) & (firm_pat_panel_df['year'] <= end)]
            RD_frames.append(bin_df.groupby('BLS_Industry', as_index=False)
                                    .agg(pat_count_hat=('pat_count_hat', 'sum'),
                                         pat_count_clean_hat=('pat_count_clean_hat', 'sum'),
                                         pat_cites_hat=('pat_cites_hat', 'sum'),
                                         pat_cites_clean_hat=('pat_cites_clean_hat', 'sum'))
                                    .assign(period=end))
            
        RD_shocks_df = pd.concat(RD_frames, ignore_index=True)
        
        panel_idx = pd.MultiIndex.from_product(
            [sorted(firm_pat_panel_df['BLS_Industry'].unique()), list(range(BLS_year_start, Year_end-5+1, 5))],
            names=['BLS_Industry', 'period'])
        RD_shocks_df = (RD_shocks_df.set_index(['BLS_Industry', 'period'])
                                .reindex(panel_idx)
                                .fillna(0.0)
                                .reset_index()
                      [['period', 'BLS_Industry', 'pat_count_hat', 'pat_count_clean_hat', 'pat_cites_hat', 'pat_cites_clean_hat']]
                      .sort_values(['BLS_Industry', 'period'])
                      .reset_index(drop=True))
        
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
    
    
    
    def Up_Down_Green(self, BLS_year_start, Year_end, dim=3):
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
        Ind_Pat_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_Pat.pkl')
        
        # Ind_CO2_df_full = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_CO2_full.pkl')
        # Ind_Pat_df_full = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_Pat_full.pkl')
        
        # govt_shocks_df = pd.read_pickle(f'{self.Directory}/Clean Data/Govt_Shocks.pkl')
        # RD_shocks_df   = pd.read_pickle(f'{self.Directory}/Clean Data/RD_Shocks.pkl')
        
        manu_idx_all = np.arange(self.manu_cols[0], self.manu_cols[1] + 1)
        M            = len(manu_idx_all)
        bin_ends     = [y for y in range(BLS_year_start, Year_end + 1, 5) if y in IO_mats]

       
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
 
        pat_wide  = wide('pat_count')
        cite_wide = wide('pat_cites')
 
        G_pat  = wide('clean_pat_count') / pat_wide.where(pat_wide  > 0)
        G_cite = wide('clean_pat_cites') / cite_wide.where(cite_wide > 0)
        
        keep     = np.isin(manu_idx_all, Ind_Pat_df['BLS_Industry'].unique())
        keep_idx = manu_idx_all[keep]
        print(f'Network universe: {int(keep.sum())} of {M} manufacturing sectors '
              f'(sectors appearing in the patent panel).')
        print(f'Bins: {bin_ends}')
        
        
        # ---------------------- #
        # Network Greenification #
        # ---------------------- #
        def partner_avg(S_sub, v, obs):
           
           v0   = np.where(obs, v, 0.0)
           o    = obs.astype(float)

           w_up = S_sub   @ o                       
           w_dn = S_sub.T @ o                      
           t_up = S_sub.sum(axis=1)               
           t_dn = S_sub.sum(axis=0)

           up   = np.where(w_up > 0, (S_sub   @ v0) / np.where(w_up > 0, w_up, 1.0), np.nan)
           down = np.where(w_dn > 0, (S_sub.T @ v0) / np.where(w_dn > 0, w_dn, 1.0), np.nan)

           cov_up = np.where(t_up > 0, w_up / np.where(t_up > 0, t_up, 1.0), np.nan)
           cov_dn = np.where(t_dn > 0, w_dn / np.where(t_dn > 0, t_dn, 1.0), np.nan)
           return up, down, cov_up, cov_dn

        frames = []
        for t in bin_ends:
            v_pat    = G_pat.loc[t].to_numpy(dtype=float)[keep]
            v_cite   = G_cite.loc[t].to_numpy(dtype=float)[keep]
            obs_pat  = np.isfinite(v_pat)
            obs_cite = np.isfinite(v_cite)
 
            S          = Σ_LI[t][np.ix_(keep, keep)]
            s_up, s_dn = S.sum(axis=1), S.sum(axis=0)
 
            up_p, dn_p, cov_up_p, cov_dn_p = partner_avg(S, v_pat,  obs_pat)
            up_c, dn_c, cov_up_c, cov_dn_c = partner_avg(S, v_cite, obs_cite)
 
            frames.append(pd.DataFrame({
                'BLS_Industry': keep_idx,
                'period':       t,
                'up_G_pat':     up_p,
                'down_G_pat':   dn_p,
                'up_G_cite':    up_c,
                'down_G_cite':  dn_c,
                's_up':         s_up,
                's_dn':         s_dn,
                'cov_up':       cov_up_p,
                'cov_dn':       cov_dn_p,
                'n_obs_up':     int(obs_pat.sum()),
            }))
 
        net_df = pd.concat(frames, ignore_index=True)
        net_df['net_G_pat']  = net_df['up_G_pat']  + net_df['down_G_pat']
        net_df['net_G_cite'] = net_df['up_G_cite'] + net_df['down_G_cite']
 
        cov_tab = (net_df.groupby('period')
                         .agg(partners_observed=('n_obs_up', 'first'),
                              mean_weight_covered=('cov_up', 'mean'),
                              min_weight_covered=('cov_up', 'min')))
        print('\nPartner coverage by bin (share of upstream network weight observed):')
        print(cov_tab.round(3).to_string())
       
        
        # ------------------ #
        # Own Greenification #
        # ------------------ #
        own_df = Ind_Pat_df[['BLS_Industry', 'period', 'clean_pat_count', 'pat_count',
                             'clean_pat_cites', 'pat_cites']].copy()
        own_df['G_pat']  = (own_df['clean_pat_count']
                            / own_df['pat_count'].where(own_df['pat_count'] > 0))
        own_df['G_cite'] = (own_df['clean_pat_cites']
                            / own_df['pat_cites'].where(own_df['pat_cites'] > 0))
 
        reg_df = net_df.merge(own_df, on=['BLS_Industry', 'period'], how='left')
        
        
        # ---- #
        # Lags #
        # ---- #
        lag_cols = ['up_G_pat', 'down_G_pat', 'net_G_pat',
                    'up_G_cite', 'down_G_cite', 'net_G_cite',
                    'G_pat', 'G_cite']
        lagged = reg_df[['BLS_Industry', 'period'] + lag_cols].copy()
        lagged['period'] = lagged['period'] + 5
        lagged = lagged.rename(columns={c: f'{c}_lag' for c in lag_cols})
        reg_df = reg_df.merge(lagged, on=['BLS_Industry', 'period'], how='left')
 
        print(f'Panel: {len(reg_df)} sector-bins; '
              f'{reg_df["net_G_pat_lag"].notna().sum()} with a lagged network term.')
        

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
            return GLMWrap(res, y_col, list(x_cols), offset_col, fe,
                           n_sectors=d['BLS_Industry'].nunique())


        # ---------- #
        # Estimation #
        # ---------- #
        reg_df['pat_count_nc'] = reg_df['pat_count'] - reg_df['clean_pat_count']
        # Green patent counts, lagged partner adoption
        m_pat_ud  = fit_ppml(reg_df, 'clean_pat_count', 'pat_count_nc',
                             ['up_G_pat_lag', 'down_G_pat_lag', 'G_pat_lag'])
        m_pat_net = fit_ppml(reg_df, 'clean_pat_count', 'pat_count',
                             ['net_G_pat_lag', 'G_pat_lag'])
 
        # Green citations, lagged partner adoption
        reg_df['pat_cites_nc'] = reg_df['pat_cites'] - reg_df['clean_pat_cites']
        m_cit_ud  = fit_ppml(reg_df, 'clean_pat_cites', 'pat_cites_nc',
                             ['up_G_cite_lag', 'down_G_cite_lag', 'G_cite_lag'])
        m_cit_net = fit_ppml(reg_df, 'clean_pat_cites', 'pat_cites',
                             ['net_G_cite_lag', 'G_cite_lag'])
 
        # Contemporaneous partner adoption (simultaneous; reported for comparison only)
        m_pat_ud_c  = fit_ppml(reg_df, 'clean_pat_count', 'pat_count',
                               ['up_G_pat', 'down_G_pat'])
        m_pat_net_c = fit_ppml(reg_df, 'clean_pat_count', 'pat_count',
                               ['net_G_pat'])
        
        m_cit_ud_c  = fit_ppml(reg_df, 'clean_pat_cites', 'pat_cites',
                               ['up_G_cite', 'down_G_cite'])
        m_cit_net_c = fit_ppml(reg_df, 'clean_pat_cites', 'pat_cites',
                               ['net_G_cite'])
 
        #### Why opposite correlations?
        ### Overlapping bins?
        
        Models = {
            'pat_ud':     m_pat_ud,     'pat_net':     m_pat_net,
            'cit_ud':     m_cit_ud,     'cit_net':     m_cit_net,
            'pat_ud_con': m_pat_ud_c,   'pat_net_con': m_pat_net_c,
            'cit_ud_con': m_cit_ud_c,   'cit_net_con': m_cit_net_c,
        }
 
        def show(models=None):
            for name, m in (models or Models).items():
                print(f'\n{"="*78}\n{name}\n{"="*78}\n{m!r}')
 
        show()
 
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

    
class GLMWrap:
    def __init__(self, res, y=None, x=None, offset=None, fe=None, n_sectors=None):
        self.glm       = res
        self.y         = y
        self.x         = list(x) if x is not None else list(res.params.index)
        self.offset    = offset
        self.fe        = fe or []
        self.n_sectors = n_sectors
        self.params    = res.params
        self.bse       = res.bse
        self.tvalues   = res.tvalues
        self.pvalues   = res.pvalues
        self.nobs      = int(res.nobs)
        try:
            self.rsquared = res.pseudo_rsquared(kind='mcf')
        except Exception:
            self.rsquared = np.nan

    def conf_int(self, alpha=0.05):
        return self.glm.conf_int(alpha=alpha)

    def frame(self):
        ci = self.conf_int()
        rows = [v for v in self.x if v in self.params.index]
        return pd.DataFrame({
            'coef':  self.params[rows].round(4),
            'se':    self.bse[rows].round(4),
            'z':     self.tvalues[rows].round(2),
            'p':     self.pvalues[rows].round(4),
            'sig':   [gpf.get_stars(p) for p in self.pvalues[rows]],
            'ci_lo': ci.iloc[:, 0][rows].round(4),
            'ci_hi': ci.iloc[:, 1][rows].round(4),
        })

    def test(self, restriction):
        try:
            w = self.glm.wald_test(restriction, use_f=False, scalar=True)
            return float(np.squeeze(w.statistic)), float(np.squeeze(w.pvalue))
        except TypeError:
            w = self.glm.wald_test(restriction, use_f=False)
            return float(np.squeeze(w.statistic)), float(np.squeeze(w.pvalue))

    def __repr__(self):
        head = f'{self.y} ~ {" + ".join(self.x)}'
        spec = (f'PPML (Poisson pseudo-ML), offset log({self.offset}), '
                f'{" + ".join(self.fe) if self.fe else "no"} FE, '
                f'SE clustered by sector')
        info = [f'N = {self.nobs}']
        if self.n_sectors is not None:
            info.append(f'sectors = {self.n_sectors}')
        if np.isfinite(self.rsquared):
            info.append(f'pseudo R2 = {self.rsquared:.4f}')
        conv = getattr(self.glm, 'converged', None)
        if conv is None:
            conv = getattr(getattr(self.glm, 'mle_retvals', {}), 'get', lambda k, d: d)('converged', None)
        if conv is not None:
            info.append(f'converged = {conv}')

        out = [head, spec, '  |  '.join(info), '', self.frame().to_string()]

        pairs = [('up_G_pat_lag', 'down_G_pat_lag'), ('up_G_cite_lag', 'down_G_cite_lag'),
                 ('up_G_pat',     'down_G_pat'),     ('up_G_cite',     'down_G_cite')]
        for a, b in pairs:
            if a in self.params.index and b in self.params.index:
                try:
                    s1, p1 = self.test(f'{a} + {b} = 0')
                    s2, p2 = self.test(f'{a} = {b}')
                    out.append(f'\nH0: up + down = 0   chi2 = {s1:.3f}  p = {p1:.4f}   '
                               f'(complementarity implies > 0)')
                    out.append(f'H0: up = down       chi2 = {s2:.3f}  p = {p2:.4f}')
                except Exception as e:
                    out.append(f'\nWald tests unavailable: {e}')
        return '\n'.join(out)

    def full(self):
        return self.glm.summary()
    
    
    
    