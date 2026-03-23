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

        
        
    def Cleaner(self, Year_start, Year_mid, Year_end, patents=1):
        """""
        Clean Data
        
        Output: Clean Data/BLS_Crosswalk.pkl
                Clean Data/Ind_CO2.pkl
                Clean Data/Ind_Pat.pkl
        """""
        
        # ----------------------------------------------------------------

        # Unpack data sets.

        # ----------------------------------------------------------------

        # ------------ #
        # BLS IO Table #
        # ------------ #
        USE_start_df = pd.read_excel(f'{self.Directory}/Raw Data/REAL_USE.xlsx', sheet_name=f"{Year_start}")
        MAKE_start_df = pd.read_excel(f'{self.Directory}/Raw Data/REAL_MAKE.xlsx', sheet_name=f"{Year_start}")
        
        USE_mid_df = pd.read_excel(f'{self.Directory}/Raw Data/REAL_USE.xlsx', sheet_name=f"{Year_mid}")
        MAKE_mid_df = pd.read_excel(f'{self.Directory}/Raw Data/REAL_MAKE.xlsx', sheet_name=f"{Year_mid}")
        
        USE_end_df = pd.read_excel(f'{self.Directory}/Raw Data/REAL_USE.xlsx', sheet_name=f"{Year_end}")
        MAKE_end_df = pd.read_excel(f'{self.Directory}/Raw Data/REAL_MAKE.xlsx', sheet_name=f"{Year_end}")
        #NAICS 2022
        
        BLS_Crosswalk_df = pd.read_excel(f'{self.Directory}/Raw Data/BLS_Crosswalk.xlsx', sheet_name="Stubs")
        BLS_Crosswalk_df['BLS_Industry'] = BLS_Crosswalk_df['Sector Number']
        
        
        # ----------------------- #
        # EPA Emissions by Sector #
        # ----------------------- #
        EPA_url = "https://pasteur.epa.gov/uploads/10.23719/1531141/GHGs_by_Detailed_Sector_US_2012-2022.xlsx"

        EPA_df = pd.read_excel(EPA_url, sheet_name="Main")    
        
        EPA_df = EPA_df[EPA_df['Flowable'].isin(self.CO2e.keys())].copy()
        EPA_df['GWP'] = EPA_df['Flowable'].map(self.CO2e)
        EPA_df['FlowAmount_CO2e'] = EPA_df['FlowAmount'] * EPA_df['GWP']
        
        EPA_df['CO2e'] = EPA_df.groupby(['Sector', 'Year'])['FlowAmount_CO2e'].transform("sum")
        EPA_df = EPA_df[['Sector', 'Year', 'CO2e']].drop_duplicates()
        
        jump_mask = (
            EPA_df
            .sort_values(['Sector', 'Year'])
            .groupby('Sector')['CO2e']
            .transform(lambda x: x.apply(np.log).diff().abs())
            ) > np.log(1.5)
    
        flagged_industries = EPA_df['Sector'][jump_mask].unique()
        
        total_emissions = EPA_df['CO2e'].sum()
        flagged_emissions = EPA_df[EPA_df['Sector'].isin(flagged_industries)]['CO2e'].sum()
        
        print(f"Flagged industries account for {flagged_emissions / total_emissions:.1%} of total emissions")
        EPA_df = EPA_df[~EPA_df['Sector'].isin(flagged_industries)]
        
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
        
        U_mid_rev = USE_mid_df.iloc[:, 1:-3].to_numpy()
        ind_Y_mid = np.sum(U_mid_rev, 0)
        B_mid = (U_mid_rev[:-3,:] @ np.diag(ind_Y_mid**(-1))).T
        
        U_end_rev = USE_end_df.iloc[:, 1:-3].to_numpy()
        ind_Y_end = np.sum(U_end_rev, 0)
        B_end = (U_end_rev[:-3,:] @ np.diag(ind_Y_end**(-1))).T

        
        # -------------------------------------- #
        # Commodity x Industry Production Shares #
        # -------------------------------------- #
        
        M_start_rev = MAKE_start_df.iloc[:, 1:].to_numpy()
        com_Y_start = np.sum(M_start_rev, 0)[:-2]
        A_start = (M_start_rev[:-2,:-2] @ np.diag(com_Y_start**(-1))).T
        
        M_mid_rev = MAKE_mid_df.iloc[:, 1:].to_numpy()
        com_Y_mid = np.sum(M_mid_rev, 0)[:-2]
        A_mid = (M_mid_rev[:-2,:-2] @ np.diag(com_Y_mid**(-1))).T
        
        M_end_rev = MAKE_end_df.iloc[:, 1:].to_numpy()
        com_Y_end = np.sum(M_end_rev, 0)[:-2]
        A_end = (M_end_rev[:-2,:-2] @ np.diag(com_Y_end**(-1))).T
        
        
        # ------------------- #
        # Input-Output Matrix #
        # ------------------- #
        self.IO_start = B_start @ A_start
        self.IO_mid = B_mid @ A_mid
        self.IO_end = B_end @ A_end
        
        J = self.IO_start.shape[0]
        IO_df = pd.DataFrame({"BLS_Industry": np.arange(1, J+1)})
        
        # ----------------- #
        # Value Added Panel #
        # ----------------- #
        va_frames = []
        for year in range(Year_start, Year_end + 1):
            USE_yr = pd.read_excel(f'{self.Directory}/Raw Data/REAL_USE.xlsx', sheet_name=f"{year}")
            U = USE_yr.iloc[:, 1:-3].to_numpy()
            ind_Y_yr = np.sum(U, 0)
            va_frames.append(pd.DataFrame({
                "BLS_Industry": np.arange(1, J+1),
                "Value_Added":  ind_Y_yr,
                "Year":         year
            }))

        VA_panel = pd.concat(va_frames, ignore_index=True)
        
        
        # ----------------------------------------------------------------

        # Make unified NAICS mapping.

        # ----------------------------------------------------------------
        
        # -------------------- #
        # Clean for Comparison #
        # -------------------- #
        EPA_df['EPA_Sector'] = EPA_df['Sector'].apply(gpf.clean_naics_str)
        
        NAICS_2012_2017_df['NAICS_2012'] = NAICS_2012_2017_df['2012 NAICS Code'].apply(gpf.clean_naics_str)
        NAICS_2012_2017_df['NAICS_2017'] = NAICS_2012_2017_df['2017 NAICS Code'].apply(gpf.clean_naics_str)
        
        NAICS_2017_2022_df['NAICS_2017'] = NAICS_2017_2022_df['2017 NAICS Code'].apply(gpf.clean_naics_str)
        NAICS_2017_2022_df['NAICS_2022'] = NAICS_2017_2022_df['2022 NAICS Code'].apply(gpf.clean_naics_str)

        BLS_Crosswalk_df['NAICS_2022'] = BLS_Crosswalk_df['NAICS_2022'].apply(gpf.clean_naics_str)
        
        naics2012_6_universe = sorted(NAICS_2012_2017_df['NAICS_2012'].dropna().unique())
        #naics2017_6_universe = sorted(NAICS_2012_2017_df['NAICS_2017'].dropna().unique())
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
            mapped_2022_6 = gpf.map_naics2012_to_2022_6(s, naics2012_6_universe, NAICS_2012_2017_df, NAICS_2017_2022_df)
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
        Ind_CO2_df = IO_df.merge(EPA_BLS_Crosswalk[['EPA_Sector', 'BLS_Industry']],
                                    on='BLS_Industry',
                                    how='inner')
        
        Ind_CO2_df = Ind_CO2_df.merge(EPA_df,
                            on='EPA_Sector',
                            how='inner')
        
        Ind_CO2_df = Ind_CO2_df.merge(VA_panel,
                            on=['BLS_Industry', 'Year'],
                            how='inner')
        
        Ind_CO2_df['CO2e_denom'] = Ind_CO2_df.groupby(['EPA_Sector', 'Year'])['Value_Added'].transform("sum")
        Ind_CO2_df['CO2e_integrand'] = Ind_CO2_df['Value_Added'] * Ind_CO2_df['CO2e'] / Ind_CO2_df['CO2e_denom']
        Ind_CO2_df['CO2e_Industry'] = Ind_CO2_df.groupby(['BLS_Industry', 'Year'])['CO2e_integrand'].transform("sum")
        Ind_CO2_df['CO2e_intensity_Industry'] = Ind_CO2_df['CO2e_Industry'] / Ind_CO2_df['Value_Added']
        
        Ind_CO2_df = Ind_CO2_df[['BLS_Industry', 'Year', 'CO2e_Industry', 'CO2e_intensity_Industry']].drop_duplicates()

        Ind_CO2_df.to_pickle(f'{self.Directory}/Clean Data/Ind_CO2.pkl')
        
        
        # ----------------------------------------------------------------

        # Build industry patenting cross-section.

        # ----------------------------------------------------------------
        if patents==1:
            # --------------------- #
            # PatentsView CPC Codes #
            # --------------------- #
            CPC_df = gpf.Extract_PatentsView('g_cpc_current')
            
            CPC_df['patent_id'] = CPC_df['patent_id'].astype(str)
                    
            
            # ------------------------ #
            # PatentsView Applications #
            # ------------------------ #
            PV_applications_df = gpf.Extract_PatentsView('g_application')
            
            PV_applications_df["year"] = pd.to_datetime(PV_applications_df["filing_date"], format="%Y-%m-%d", errors="coerce").dt.year
            PV_applications_df = PV_applications_df.dropna(subset=["year"])
            PV_applications_df = PV_applications_df[(PV_applications_df["year"] >= 1900) & (PV_applications_df["year"] <= datetime.now().year)]
            PV_applications_df['patent_id'] = PV_applications_df['patent_id'].astype(str)
            
            
            # ------------------ #
            # Technology Classes #
            # ------------------ #
            relevant_df = CPC_df.copy()
            
            relevant_df = pd.merge(relevant_df,
                                 PV_applications_df,
                                 on='patent_id',
                                 how='inner'
                                 )
            
            relevant_df = relevant_df[(relevant_df["year"] >= Year_start) & (relevant_df["year"] <= Year_end)]
            relevant_df["cpc_group5"] = relevant_df["cpc_group"].str[:5]
            relevant_df["cpc_group6"] = relevant_df["cpc_group"].str[:6]
            
            codes = set(self.CPC_classes)
            relevant_df['clean'] = (relevant_df["cpc_class"].isin(codes)
                                    | relevant_df["cpc_subclass"].isin(codes)
                                    | relevant_df["cpc_group"].isin(codes)
                                    | relevant_df["cpc_group5"].isin(codes)
                                    | relevant_df["cpc_group6"].isin(codes)).astype(np.int8)
            
            relevant_df['clean'] = relevant_df.groupby("patent_id")['clean'].transform("max")
            relevant_df = relevant_df[['patent_id', 'year', 'clean']].drop_duplicates()
                    
                    
            # --------------------- #
            # PatentsView Citations #
            # --------------------- #
            citations_df = gpf.Extract_PatentsView('g_us_patent_citation')
            
            citations_df['patent_id'] = citations_df['patent_id'].astype(str)
            citations_df['citation_patent_id'] = citations_df['citation_patent_id'].astype(str)
            
            
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
            pat_firm_crosswalk_df['split_weight'] = 1 / pat_firm_crosswalk_df.groupby('patent_id')['patent_id'].transform('count')
            
            
            # --------- #
            # Compustat #
            # --------- #
            compustat_df = pd.read_csv(f'{self.Directory}/Raw Data/compustat.csv')
            
            compustat_df = compustat_df[(compustat_df['fic']=="USA") & (compustat_df['final']=="Y")]
            terry_cols = ['at', 'ppent', 'emp', 'capxv', 'sale', 'xrd']
            compustat_df = compustat_df[compustat_df[terry_cols].gt(0).all(axis=1)]
            compustat_df = compustat_df[compustat_df.groupby('gvkey')['gvkey'].transform('size') > 1]
            compustat_df.rename(columns={'fyear': 'year'}, inplace=True)
            
            compustat_df = compustat_df[(compustat_df["year"] >= Year_start) & (compustat_df["year"] <= Year_end)]
            compustat_df['naics2022_6'] = compustat_df['naics'] #Assume Compustat uses most up to date NAICS
            compustat_df = compustat_df[['gvkey', 'naics2022_6']].drop_duplicates()
    
    
            # -------------------------- #
            # Allocate Patent to Sectors #
            # -------------------------- #
            pat_df = pat_df.merge(pat_firm_crosswalk_df,
                                on='patent_id',
                                how='inner')
            pat_df = pat_df.merge(compustat_df,
                                on='gvkey',
                                how='inner')
            pat_df = pat_df.merge(EPA_BLS_Crosswalk[['naics2022_6', 'BLS_Industry']],
                                on='naics2022_6',
                                how='inner')
            
            def compute_pat_metrics(df, suffix=''):
                df = df.copy()
                df['clean']               = df['split_weight'] * df['clean']
                df['pat_count']           = df.groupby('BLS_Industry')['split_weight'].transform('sum')
                df['clean_pat_share']     = df.groupby('BLS_Industry')['clean'].transform('sum') / df['pat_count']

                df['weighted_pat_cites']   = df['split_weight'] * df['norm_cites']
                df['weighted_clean_cites'] = df['clean'] * df['norm_cites']
                df['pat_cites']            = df.groupby('BLS_Industry')['weighted_pat_cites'].transform('sum')
                df['clean_cite_share']     = df.groupby('BLS_Industry')['weighted_clean_cites'].transform('sum') / df['pat_cites']

                cols = {'BLS_Industry':    'BLS_Industry',
                        'clean_pat_share': f'clean_pat_share{suffix}',
                        'clean_cite_share':f'clean_cite_share{suffix}'}
                return df[list(cols.keys())].drop_duplicates().rename(columns=cols)

            pat_metrics_full = compute_pat_metrics(pat_df)

            pat_metrics_late = compute_pat_metrics(
                pat_df[pat_df['year'] > Year_mid],
                suffix='_late'
            )
            
            pat_metrics_early = compute_pat_metrics(
                pat_df[pat_df['year'] <= Year_mid],
                suffix='_early'
            )

            pat_df = pat_metrics_full.merge(pat_metrics_late, on='BLS_Industry', how='left')
            pat_df = pat_df.merge(pat_metrics_early, on='BLS_Industry', how='left')
            
            pat_df.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat.pkl')



    def IO_Change(self, Year_start, Year_end):
    def IO_Change(self, Year_start, Year_mid, Year_end):
        """""
        Plot of Changes in IO Network from Decarbonization
    
        Output: Results/Tables/sector_extremes_table.tex
                Results/Figures/Baseline_L1_uw.png
                Results/Figures/Baseline_L1.png
                Results/Figures/Baseline_L2.png
                Results/Figures/Baseline_split.png
                Results/Figures/Leontief_L1_uw.png
                Results/Figures/Leontief_L1.png
                Results/Figures/Leontief_L2.png
                Results/Figures/Leontief_split.png
                Results/Figures/Reduced_L1_uw.png
                Results/Figures/Reduced_L1.png
                Results/Figures/Reduced_L2.png
                Results/Figures/Reduced_split.png
        """""
        
        # ----------------------------------------------------------------

        # Build regression dataframes.

        # ----------------------------------------------------------------
        
        BLS_Crosswalk_df = pd.read_pickle(f'{self.Directory}/Clean Data/BLS_Crosswalk.pkl')
        Ind_CO2_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_CO2.pkl')

        
        # ------------------- #
        # Input-Output Matrix #
        # ------------------- #
        J       = self.IO_start.shape[0]
        manu    = slice(7-1, 94-1)  # 0-indexed rows for non-service industries
 
        I        = np.eye(J)
        LI_start = np.linalg.inv(I - self.IO_start)
        LI_mid   = np.linalg.inv(I - self.IO_mid)
        LI_end   = np.linalg.inv(I - self.IO_end)
 
        # Baseline: non-service rows, all columns
        IO_start_manu = self.IO_start[manu, :]
        IO_mid_manu   = self.IO_mid[manu, :]
        IO_end_manu   = self.IO_end[manu, :]
 
        # Leontief: non-service rows, all columns
        LI_start_manu = LI_start[manu, :]
        LI_mid_manu   = LI_mid[manu, :]
        LI_end_manu   = LI_end[manu, :]
 
        # Reduced: non-service rows, fossil fuel columns (indices 6,7 within manu block) dropped, renormalized
        fossil_cols = [6, 7]
        def drop_and_normalize(IO):
            IO_r = np.delete(IO[manu, :], fossil_cols, axis=1)
            return IO_r / IO_r.sum(axis=1, keepdims=True)

        IO_start_reduced = drop_and_normalize(self.IO_start)
        IO_mid_reduced   = drop_and_normalize(self.IO_mid)
        IO_end_reduced   = drop_and_normalize(self.IO_end)

        def tv_metrics(A, B):
           diff  = np.abs(B - A)
           tv    = 0.5 * diff.sum(axis=1)
           tv_sq = 0.5 * ((diff**2).sum(axis=1))**(1/2)
           return tv, tv_sq

        # Period 1: start -> mid
        tv_p1,         tv_sq_p1         = tv_metrics(IO_start_manu,    IO_mid_manu)
        tv_LI_p1,      tv_sq_LI_p1      = tv_metrics(LI_start_manu,    LI_mid_manu)
        tv_red_p1,     tv_sq_red_p1     = tv_metrics(IO_start_reduced, IO_mid_reduced)

        # Period 2: mid -> end
        tv_p2,         tv_sq_p2         = tv_metrics(IO_mid_manu,    IO_end_manu)
        tv_LI_p2,      tv_sq_LI_p2      = tv_metrics(LI_mid_manu,   LI_end_manu)
        tv_red_p2,     tv_sq_red_p2     = tv_metrics(IO_mid_reduced, IO_end_reduced)

        def make_IO_df(tv, tv_sq, tv_LI, tv_sq_LI, tv_red, tv_sq_red):
            return pd.DataFrame({
                "BLS_Industry":           np.arange(7, 94),
                "TV_distance":            tv,
                "TV_sq_distance":         tv_sq,
                "TV_distance_LI":         tv_LI,
                "TV_sq_distance_LI":      tv_sq_LI,
                "TV_distance_reduced":    tv_red,
                "TV_sq_distance_reduced": tv_sq_red})

        IO_df_p1 = make_IO_df(tv_p1, tv_sq_p1, tv_LI_p1, tv_sq_LI_p1, tv_red_p1, tv_sq_red_p1)
        IO_df_p1['period'] = 1

        IO_df_p2 = make_IO_df(tv_p2, tv_sq_p2, tv_LI_p2, tv_sq_LI_p2, tv_red_p2, tv_sq_red_p2)
        IO_df_p2['period'] = 2

        IO_df = pd.concat([IO_df_p1, IO_df_p2], ignore_index=True)

        
        # ------------------ #
        # Allocate Emissions #
        # ------------------ #
        IO_wide_df = Ind_CO2_df.pivot(index="BLS_Industry", columns="Year",
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
            "CO2e_Industry":      IO_wide_df['CO2e_Industry', Year_start].to_numpy(),
            "CO2e_Industry_LI":   CO2e_lev_LI_start,
            "period": 1})

        em_p2 = pd.DataFrame({
            "BLS_Industry":       IO_wide_df.index,
            "dlog_CO2e_inten":    -(np.log(IO_wide_df['CO2e_intensity_Industry', Year_end].to_numpy())
                                  - np.log(IO_wide_df['CO2e_intensity_Industry', Year_mid].to_numpy())),
            "dlog_CO2e_inten_LI": -(np.log(CO2e_LI_end) - np.log(CO2e_LI_mid)),
            "CO2e_Industry":      IO_wide_df['CO2e_Industry', Year_mid].to_numpy(),
            "CO2e_Industry_LI":   CO2e_lev_LI_mid,
            "period": 2})

        distance_cols = ['BLS_Industry', 'period',
                         'TV_distance',         'TV_sq_distance',
                         'TV_distance_LI',      'TV_sq_distance_LI',
                         'TV_distance_reduced', 'TV_sq_distance_reduced']

        emission_cols = ['BLS_Industry', 'period',
                         'CO2e_Industry', 'CO2e_Industry_LI',
                         'dlog_CO2e_inten', 'dlog_CO2e_inten_LI']

        em_df  = pd.concat([em_p1, em_p2], ignore_index=True)
        reg_df = pd.merge(IO_df[distance_cols].drop_duplicates(),
                          em_df[emission_cols].drop_duplicates(),
                          on=['BLS_Industry', 'period'],
                          how='inner')

        
        # ------------- #
        # List Extremes #
        # ------------- #
        reg_df = reg_df.merge(BLS_Crosswalk_df[["BLS_Industry", "Sector Title"]].drop_duplicates(),
                                on="BLS_Industry",
                                how="left"
                            )
    
        largest_dlog_CO2e = (reg_df.sort_values("dlog_CO2e_inten", ascending=False)
              .head(5)[["BLS_Industry", "Sector Title", "dlog_CO2e_inten"]]
              .apply(tuple, axis=1)
              .tolist()
            )
        
        smallest_dlog_CO2e = (reg_df.sort_values("dlog_CO2e_inten", ascending=True)
              .head(5)[["BLS_Industry", "Sector Title", "dlog_CO2e_inten"]]
              .apply(tuple, axis=1)
              .tolist()
            )
        
        largest_tv = (reg_df.sort_values("TV_distance", ascending=False)
              .head(5)[["BLS_Industry", "Sector Title", "TV_distance"]]
              .apply(tuple, axis=1)
              .tolist()
            )
        
        smallest_tv = (reg_df.sort_values("TV_distance", ascending=True)
              .head(5)[["BLS_Industry", "Sector Title", "TV_distance"]]
              .apply(tuple, axis=1)
              .tolist()
            )
        
        def get_sector_names(extreme_list):
            return [x[1] for x in extreme_list]
        
        def latex_escape(s):
            return (s.replace("\\", r"\textbackslash ")
                     .replace("&", r"\&")
                     .replace("%", r"\%")
                     .replace("$", r"\$")
                     .replace("#", r"\#")
                     .replace("_", r"\_")
                     .replace("{", r"\{")
                     .replace("}", r"\}")
                     .replace("~", r"\textasciitilde ")
                     .replace("^", r"\textasciicircum "))
        
        def make_two_col_rows(left_list, right_list, n=5):
            left = [latex_escape(x) for x in left_list[:n]]
            right = [latex_escape(x) for x in right_list[:n]]
            # If one side is shorter for any reason, pad with blanks
            m = max(len(left), len(right))
            left += [""] * (m - len(left))
            right += [""] * (m - len(right))
            return "\n".join([f"{left[i]} & {right[i]} \\\\" for i in range(m)])
        
        largest_dlog_names   = get_sector_names(smallest_dlog_CO2e)
        smallest_dlog_names  = get_sector_names(largest_dlog_CO2e)
        largest_tv_names     = get_sector_names(largest_tv)
        smallest_tv_names    = get_sector_names(smallest_tv)
        
        latex_table = rf"""
        \begin{{table}}[ht]
        \centering
        \begin{{tabular}}{{p{{6cm}} p{{6cm}}}}
        \toprule
        \multicolumn{{2}}{{c}}{{\textbf{{Emission Intensity Reduction}}}} \\
        \midrule
        \textbf{{Largest}} & \textbf{{Smallest}} \\
        \midrule
        {make_two_col_rows(largest_dlog_names, smallest_dlog_names, n=5)}
        \midrule
        \multicolumn{{2}}{{c}}{{\textbf{{Input Share Change}}}} \\
        \midrule
        \textbf{{Largest}} & \textbf{{Smallest}} \\
        \midrule
        {make_two_col_rows(largest_tv_names, smallest_tv_names, n=5)}
        \bottomrule
        \end{{tabular}}
        \end{{table}}
        """
        
        output_path = f'{self.Directory}/Results/Tables/sector_extremes_table.tex'

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_table)

        # ----------------------------------------------------------------

        # Run regressions and graph.

        # ----------------------------------------------------------------
        
        # ---------------- #
        # Helper Functions #
        # ---------------- #
        
        def run_regressions(df, x_col, y_col, y_sq_col, weight_col, group_col):
            X      = sm.add_constant(df[x_col])
            Y      = df[y_col]
            Y_sq   = df[y_sq_col]
            weight = df[weight_col]
            groups = df[group_col]

            m      = sm.WLS(Y,    X, weight).fit(cov_type='cluster', cov_kwds={'groups': groups})
            m_sq   = sm.WLS(Y_sq, X, weight).fit(cov_type='cluster', cov_kwds={'groups': groups})
            m_uw   = sm.OLS(Y,    X).fit(cov_type='cluster', cov_kwds={'groups': groups})

            x_arr = df[x_col].to_numpy()
            y_arr = Y.to_numpy()
            y_sq_arr = Y_sq.to_numpy()
            w_arr = weight.to_numpy()
            g_arr = groups.to_numpy()

            # Split by sign (unweighted, clustered by group)
            mask_pos = x_arr >= 0
            mask_neg = x_arr <  0

            m_pos = sm.WLS(y_arr[mask_pos], sm.add_constant(x_arr[mask_pos]), w_arr[mask_pos]).fit(cov_type='cluster', cov_kwds={'groups': g_arr[mask_pos]})
            m_neg = sm.WLS(y_arr[mask_neg], sm.add_constant(x_arr[mask_neg]), w_arr[mask_neg]).fit(cov_type='cluster', cov_kwds={'groups': g_arr[mask_neg]})

            return dict(
                m=m, m_sq=m_sq, m_uw=m_uw, m_pos=m_pos, m_neg=m_neg,
                x=x_arr, y=y_arr, y_sq=y_sq_arr, w=w_arr,
                mask_pos=mask_pos, mask_neg=mask_neg,
                x_col=x_col
            )
        
        def plot_case(r, df, ylabel_tv, ylabel_sq, prefix, year_start, year_mid, year_end, save_dir):
            x, y, y_sq, w = r['x'], r['y'], r['y_sq'], r['w']
            x_col  = r['x_col']
            scale  = 1000 / w.max()
            stars_named = lambda m, col: gpf.get_stars(m.pvalues[col])
            stars       = lambda m, k=1: gpf.get_stars(m.pvalues[k])

            mask_p1 = df['period'].to_numpy() == 1
            mask_p2 = df['period'].to_numpy() == 2

            def annotate(ax, text, y_frac):
                ax.annotate(
                    text, xy=(0.05, y_frac), xycoords='axes fraction',
                    fontsize=11, color='green',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.7))

            def scatter_periods(ax, y_arr):
                ax.scatter(x[mask_p1], y_arr[mask_p1], s=w[mask_p1]*scale, alpha=0.7, color='purple', label=f"Sectors: ({year_start}–{year_mid})")
                ax.scatter(x[mask_p2], y_arr[mask_p2], s=w[mask_p2]*scale, alpha=0.7, color='blue',   label=f"Sectors: ({year_mid}–{year_end})")

            def fix_legend(ax):
                leg = ax.legend(loc='upper right')
                for h in leg.legend_handles:
                    h._sizes = [30]

            # --- L1 unweighted ---
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter_periods(ax, y)
            y_hat_uw = r['m_uw'].params['const'] + r['m_uw'].params[x_col] * x
            ax.plot(x, y_hat_uw, color='red', linewidth=2, label="OLS fit")
            annotate(ax, f"Slope = {r['m_uw'].params[x_col]:.3f}{stars_named(r['m_uw'], x_col)}", 0.95)
            ax.set_xlabel("-Δ ln(emissions intensity)")
            ax.set_ylabel(ylabel_tv)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
            fix_legend(ax)
            plt.savefig(f'{save_dir}/{prefix}_L1_uw.png')
            plt.show()

            # --- L1: TV distance ---
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter_periods(ax, y)
            y_hat = r['m'].params['const'] + r['m'].params[x_col] * x
            ax.plot(x, y_hat, color='red', linewidth=2, label="WLS fit")
            annotate(ax, f"Slope = {r['m'].params[x_col]:.3f}{stars_named(r['m'], x_col)}", 0.95)
            ax.set_xlabel("-Δ ln(emissions intensity)")
            ax.set_ylabel(ylabel_tv)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
            fix_legend(ax)
            plt.savefig(f'{save_dir}/{prefix}_L1.png')
            plt.show()

            # --- L2: 2-norm distance ---
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter_periods(ax, y_sq)
            y_hat_sq = r['m_sq'].params['const'] + r['m_sq'].params[x_col] * x
            ax.plot(x, y_hat_sq, color='red', linewidth=2, label="WLS fit")
            annotate(ax, f"Slope = {r['m_sq'].params[x_col]:.3f}{stars_named(r['m_sq'], x_col)}", 0.95)
            ax.set_xlabel("-Δ ln(emissions intensity)")
            ax.set_ylabel(ylabel_sq)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
            fix_legend(ax)
            plt.savefig(f'{save_dir}/{prefix}_L2.png')
            plt.show()

            # --- Split by sign ---
            x_pos = x[r['mask_pos']]
            x_neg = x[r['mask_neg']]
            y_hat_pos = r['m_pos'].params[0] + r['m_pos'].params[1] * x_pos
            y_hat_neg = r['m_neg'].params[0] + r['m_neg'].params[1] * x_neg

            fig, ax = plt.subplots(figsize=(8, 6))
            scatter_periods(ax, y)
            ax.plot(x_pos, y_hat_pos, color='orange', linewidth=2, label="WLS fit (x≥0)")
            ax.plot(x_neg, y_hat_neg, color='cyan',   linewidth=2, label="WLS fit (x<0)")
            annotate(ax, f"Slope (x≥0) = {r['m_pos'].params[1]:.3f}{stars(r['m_pos'])}", 0.95)
            annotate(ax, f"Slope (x<0) = {r['m_neg'].params[1]:.3f}{stars(r['m_neg'])}", 0.883)
            ax.set_xlabel("-Δ ln(emissions intensity)")
            ax.set_ylabel(ylabel_tv)
            ax.grid(alpha=0.3)
            fix_legend(ax)
            plt.savefig(f'{save_dir}/{prefix}_split.png')
            plt.show()

        fig_dir = f'{self.Directory}/Results/Figures'
        
        
        # -------- #
        # Baseline #
        # -------- #
        r_base = run_regressions(reg_df, 'dlog_CO2e_inten', 'TV_distance', 'TV_sq_distance', 'CO2e_Industry', 'BLS_Industry')
        plot_case(r_base, reg_df, 'TV distance (input-share change)', 'Euclidean distance (input-share change)',
                  'Baseline', Year_start, Year_mid, Year_end, fig_dir)
        
        
        # ---------------- #
        # Leontief Inverse #
        # ---------------- #
        r_LI = run_regressions(reg_df, 'dlog_CO2e_inten_LI', 'TV_distance_LI', 'TV_sq_distance_LI', 'CO2e_Industry_LI', 'BLS_Industry')
        plot_case(r_LI, reg_df, 'TV distance (input-share change)', 'Euclidean distance (input-share change)',
                  'Leontief', Year_start, Year_mid, Year_end, fig_dir)
        
        
        # ------- #
        # Reduced #
        # ------- #
        reduced_df = reg_df.dropna(subset=['TV_distance_reduced', 'TV_sq_distance_reduced'])
        r_red = run_regressions(reduced_df, 'dlog_CO2e_inten', 'TV_distance_reduced', 'TV_sq_distance_reduced', 'CO2e_Industry', 'BLS_Industry')
        plot_case(r_red, reduced_df, 'TV distance (input-share change, ex. fossil fuels)', 'Euclidean distance (input-share change, ex. fossil fuels)',
                  'Reduced', Year_start, Year_mid, Year_end, fig_dir)
        
        
        
        
    
    def Up_Down_Green(self, Year_start, Year_mid, Year_end):
        """""
        Upstream and Downstream Incentives for Greenification
        
        Output: Results/Tables/Fact_Regressions.tex
        
        """""
        
        # ----------------------------------------------------------------

        # Build regression dataframes.

        # ----------------------------------------------------------------
        
        Ind_CO2_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_CO2.pkl')
        Ind_Pat_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_Pat.pkl')


        # ------------------ #
        # Allocate Emissions #
        # ------------------ #
        np.fill_diagonal(self.IO_mid, 0)
        np.fill_diagonal(self.IO_end, 0)
        
        IO_wide_df = Ind_CO2_df.pivot(index="BLS_Industry", columns="Year", values='CO2e_intensity_Industry')
        IO_wide_df = IO_wide_df.dropna()

        idx1 = IO_wide_df.index.to_numpy(dtype=int)
        idx0 = idx1 - 1

        def compute_network_effect(IO_matrix, z, prefix):
            IO  = IO_matrix[np.ix_(idx0, idx0)]
            I   = np.eye(IO.shape[0])
            LI  = np.linalg.inv(I - IO)
            return {
                f"up_{prefix}":     (LI - I) @ z,
                f"down_{prefix}": (LI - I).T @ z,
            }

        dlog_CO2e = -(np.log(IO_wide_df[Year_end].to_numpy()) - np.log(IO_wide_df[Year_start].to_numpy()))
        dlog_CO2e_late = -(np.log(IO_wide_df[Year_end].to_numpy()) - np.log(IO_wide_df[Year_mid].to_numpy()))
        dlog_CO2e_early = -(np.log(IO_wide_df[Year_mid].to_numpy()) - np.log(IO_wide_df[Year_start].to_numpy()))

        net_em = compute_network_effect(self.IO_end, dlog_CO2e, 'dlog_CO2e_inten')
        net_em_late = compute_network_effect(self.IO_end, dlog_CO2e_late, 'dlog_CO2e_inten_late')
        net_em_early = compute_network_effect(self.IO_mid, dlog_CO2e_early, 'dlog_CO2e_inten_early')

        reg_df = pd.DataFrame({
            "BLS_Industry":    IO_wide_df.index,
            "dlog_CO2e_inten": dlog_CO2e,
            "dlog_CO2e_inten_late": dlog_CO2e_late,
            **net_em,
            **net_em_late,
            **net_em_early
        })

        reg_df = reg_df.merge(Ind_Pat_df, on='BLS_Industry', how='left')
        
        pat_count = reg_df['clean_pat_share'].fillna(0).to_numpy()
        net_pat_count = compute_network_effect(self.IO_end, pat_count, 'pat_count')
        
        pat_count_late = reg_df['clean_pat_share_late'].fillna(0).to_numpy()
        net_pat_count_late = compute_network_effect(self.IO_end, pat_count_late, 'pat_count_late')
        
        pat_count_early = reg_df['clean_pat_share_early'].fillna(0).to_numpy()
        net_pat_count_early = compute_network_effect(self.IO_mid, pat_count_early, 'pat_count_early')

        pat_cite = reg_df['clean_cite_share'].fillna(0).to_numpy()
        net_pat_cite = compute_network_effect(self.IO_end, pat_cite, 'pat_cite')
        
        pat_cite_late = reg_df['clean_cite_share_late'].fillna(0).to_numpy()
        net_pat_cite_late = compute_network_effect(self.IO_end, pat_cite_late, 'pat_cite_late')
        
        pat_cite_early = reg_df['clean_cite_share_early'].fillna(0).to_numpy()
        net_pat_cite_early = compute_network_effect(self.IO_mid, pat_cite_early, 'pat_cite_early')

        reg_df = reg_df.merge(pd.DataFrame({
            "BLS_Industry": IO_wide_df.index,
            **net_pat_count,
            **net_pat_count_late,
            **net_pat_count_early,
            **net_pat_cite,
            **net_pat_cite_late,
            **net_pat_cite_early
        }), on='BLS_Industry', how='left')
        

        # ----------------------------------------------------------------
        
        # Run regressions.
        
        # ----------------------------------------------------------------

        # -------------------- #
        # Emission Regressions #
        # -------------------- #
        X = sm.add_constant(reg_df[['up_dlog_CO2e_inten', 'down_dlog_CO2e_inten']])
        X_pat_net = sm.add_constant(reg_df[['up_pat_count', 'down_pat_count']])
        X_cite_net = sm.add_constant(reg_df[['up_pat_cite', 'down_pat_cite']])
        
        Y_em = reg_df['dlog_CO2e_inten']

        model_em = sm.OLS(Y_em, X).fit(cov_type='HC3')
        print(model_em.summary())
        
        model_em_pat_net = sm.OLS(Y_em, X_pat_net).fit(cov_type='HC3')
        print(model_em_pat_net.summary())

        model_em_cite_net = sm.OLS(Y_em, X_cite_net).fit(cov_type='HC3')
        print(model_em_cite_net.summary())
        
        # Late on Whole Sample
        Y_em_late = reg_df['dlog_CO2e_inten_late']

        model_em = sm.OLS(Y_em_late, X).fit(cov_type='HC3')
        print(model_em.summary())
        
        model_em_pat_net = sm.OLS(Y_em_late, X_pat_net).fit(cov_type='HC3')
        print(model_em_pat_net.summary())

        model_em_cite_net = sm.OLS(Y_em_late, X_cite_net).fit(cov_type='HC3')
        print(model_em_cite_net.summary())
        
        # Late on Split Sample
        X_split = sm.add_constant(reg_df[['up_dlog_CO2e_inten_late', 'down_dlog_CO2e_inten_late', 'up_dlog_CO2e_inten_early', 'down_dlog_CO2e_inten_early']])
        X_pat_net_split = sm.add_constant(reg_df[['up_pat_count_late', 'down_pat_count_late', 'up_pat_count_early', 'down_pat_count_early']])
        X_cite_net_split = sm.add_constant(reg_df[['up_pat_cite_late', 'down_pat_cite_late', 'up_pat_cite_early', 'down_pat_cite_early']])
        
        model_em = sm.OLS(Y_em_late, X_split).fit(cov_type='HC3')
        print(model_em.summary())
        
        model_em_pat_net = sm.OLS(Y_em_late, X_pat_net_split).fit(cov_type='HC3')
        print(model_em_pat_net.summary())

        model_em_cite_net = sm.OLS(Y_em_late, X_cite_net_split).fit(cov_type='HC3')
        print(model_em_cite_net.summary())


        # ------------------ #
        # Patent Regressions #
        # ------------------ #
        pat_df = reg_df.dropna(subset=['clean_pat_share', 'clean_cite_share'])

        X_pat = sm.add_constant(pat_df[['up_dlog_CO2e_inten', 'down_dlog_CO2e_inten']])
        X_pat_count_net = sm.add_constant(pat_df[['up_pat_count', 'down_pat_count']])
        X_pat_cite_net = sm.add_constant(pat_df[['up_pat_cite', 'down_pat_cite']])

        Y_pat_count = pat_df['clean_pat_share']
        
        model_pat_count = sm.OLS(Y_pat_count, X_pat).fit(cov_type='HC3')
        print(model_pat_count.summary())
        
        model_pat_count_net = sm.OLS(Y_pat_count, X_pat_count_net).fit(cov_type='HC3')
        print(model_pat_count_net.summary())

        Y_pat_cite = pat_df['clean_cite_share']
        
        model_pat_cite = sm.OLS(Y_pat_cite, X_pat).fit(cov_type='HC3')
        print(model_pat_cite.summary())
        
        model_pat_cite_net = sm.OLS(Y_pat_cite, X_pat_cite_net).fit(cov_type='HC3')
        print(model_pat_cite_net.summary())
        
        # Late on Whole Sample
        pat_df = reg_df.dropna(subset=['clean_pat_share_late', 'clean_cite_share_late'])
        X_pat = sm.add_constant(pat_df[['up_dlog_CO2e_inten', 'down_dlog_CO2e_inten']])
        X_pat_count_net = sm.add_constant(pat_df[['up_pat_count', 'down_pat_count']])
        X_pat_cite_net = sm.add_constant(pat_df[['up_pat_cite', 'down_pat_cite']])
        Y_pat_count = pat_df['clean_pat_share_late']
        
        model_pat_count = sm.OLS(Y_pat_count, X_pat).fit(cov_type='HC3')
        print(model_pat_count.summary())
        
        model_pat_count_net = sm.OLS(Y_pat_count, X_pat_count_net).fit(cov_type='HC3')
        print(model_pat_count_net.summary())

        Y_pat_cite = pat_df['clean_cite_share_late']
        
        model_pat_cite = sm.OLS(Y_pat_cite, X_pat).fit(cov_type='HC3')
        print(model_pat_cite.summary())
        
        model_pat_cite_net = sm.OLS(Y_pat_cite, X_pat_cite_net).fit(cov_type='HC3')
        print(model_pat_cite_net.summary())
        
        # Late on Split Sample
        X_pat = sm.add_constant(pat_df[['up_dlog_CO2e_inten_late', 'down_dlog_CO2e_inten_late', 'up_dlog_CO2e_inten_early', 'down_dlog_CO2e_inten_early']])
        X_pat_count_net = sm.add_constant(pat_df[['up_pat_count_late', 'down_pat_count_late', 'up_pat_count_early', 'down_pat_count_early']])
        X_pat_cite_net = sm.add_constant(pat_df[['up_pat_cite_late', 'down_pat_cite_late', 'up_pat_cite_early', 'down_pat_cite_early']])
        
        model_pat_count = sm.OLS(Y_pat_count, X_pat).fit(cov_type='HC3')
        print(model_pat_count.summary())
        
        model_pat_count_net = sm.OLS(Y_pat_count, X_pat_count_net).fit(cov_type='HC3')
        print(model_pat_count_net.summary())
        
        model_pat_cite = sm.OLS(Y_pat_cite, X_pat).fit(cov_type='HC3')
        print(model_pat_cite.summary())
        
        model_pat_cite_net = sm.OLS(Y_pat_cite, X_pat_cite_net).fit(cov_type='HC3')
        print(model_pat_cite_net.summary())
    
        
        # ----------- #
        # Print Table #
        # ----------- #
        models = [
            model_em,
            model_em_pat_net,
            model_em_cite_net,
            None,
            model_pat_count,
            model_pat_count_net,
            None,
            model_pat_cite,
            model_pat_cite_net,
        ]

        variables = [
            ('up_dlog_CO2e_inten',  'Upstream CO2e Reduction'),
            ('down_dlog_CO2e_inten','Downstream CO2e Reduction'),
            ('up_pat_count',        'Upstream Clean Patents'),
            ('down_pat_count',      'Downstream Clean Patents'),
            ('up_pat_cite',         'Upstream Clean Citations'),
            ('down_pat_cite',       'Downstream Clean Citations'),
        ]

        body = ''
        for varname, label in variables:
            coefs, ses = [], []
            for m in models:
                if m is None:
                    coefs.append('')
                    ses.append('')
                else:
                    c, s = gpf.fmt_coef(m, varname)
                    coefs.append(c)
                    ses.append(s)
            body += f'{label} & {" & ".join(coefs)} \\\\\n'
            body += f'& {" & ".join(ses)} \\\\[3pt]\n'

        r2_vals, n_vals = [], []
        for m in models:
            if m is None:
                r2_vals.append('')
                n_vals.append('')
            else:
                r2_vals.append(f'{m.rsquared:.3f}')
                n_vals.append(str(int(m.nobs)))

        body += '\\midrule\n'
        body += f'$R^2$ & {" & ".join(r2_vals)} \\\\\n'
        body += f'Obs & {" & ".join(n_vals)} \\\\\n[-7pt]'

        out_path = f'{self.Directory}/Results/Tables/Fact_Regressions.tex'
        with open(out_path, 'w') as f:
            f.write(body)
            
    
    
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



