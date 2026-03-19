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
        self.CPC_classes = ["Y02B", "Y02C", "Y02D", "Y02E", "Y02P", "Y02T", "Y02W", "B60L"]

        
        
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
        EPA_df = EPA_df[EPA_df["Year"].isin([Year_start, Year_mid, Year_end])]
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
        
        df_VA_start = pd.DataFrame({"BLS_Industry": np.arange(1, J+1),
                                    "Value_Added": ind_Y_start,
                                    "Year": Year_start})
        
        df_VA_mid = pd.DataFrame({"BLS_Industry": np.arange(1, J+1),
                                  "Value_Added": ind_Y_end,
                                  "Year": Year_mid})

        df_VA_end = pd.DataFrame({"BLS_Industry": np.arange(1, J+1),
                                  "Value_Added": ind_Y_end,
                                  "Year": Year_end})

        VA_panel = pd.concat([df_VA_start, df_VA_mid, df_VA_end], ignore_index=True)
        
        
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
            relevant_df = relevant_df[['patent_id', 'clean']].drop_duplicates()
                    
                    
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
            
            pat_df['clean'] = pat_df['split_weight'] * pat_df['clean']
            pat_df['pat_count'] = pat_df.groupby(['BLS_Industry'])['split_weight'].transform('sum')
            pat_df['clean_pat_share'] = pat_df.groupby(['BLS_Industry'])['clean'].transform('sum') / pat_df['pat_count']
            
            pat_df['weighted_pat_cites'] = pat_df['split_weight'] * pat_df['norm_cites']
            pat_df['weighted_clean_cites'] = pat_df['clean'] * pat_df['norm_cites']
            pat_df['pat_cites'] = pat_df.groupby(['BLS_Industry'])['weighted_pat_cites'].transform('sum')
            pat_df['clean_cite_share'] = pat_df.groupby(['BLS_Industry'])['weighted_clean_cites'].transform('sum') / pat_df['pat_cites']
            
            pat_df = pat_df[['BLS_Industry', 'clean_pat_share', 'clean_cite_share']].drop_duplicates()
            
            pat_df.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat.pkl')



    def IO_Change(self, Year_start, Year_mid, Year_end):
        """""
        Plot of Changes in IO Network from Decarbonization
    
        Output: Results/Tables/sector_extremes_table.tex
                Results/Figures/Baseline_L1.png
                Results/Figures/Baseline_L2.png
                Results/Figures/Leontief_L1.png
                Results/Figures/Leontief_L2.png
                Results/Figures/Reduced_L1.png
                Results/Figures/Reduced_L2.png
        """""
        
        # ----------------------------------------------------------------

        # Build regression dataframes.

        # ----------------------------------------------------------------
        
        BLS_Crosswalk_df = pd.read_pickle(f'{self.Directory}/Clean Data/BLS_Crosswalk.pkl')
        Ind_CO2_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_CO2.pkl')

        
        # ------------------- #
        # Input-Output Matrix #
        # ------------------- #
        J = self.IO_start.shape[0]
        
        I = np.eye(J)
        LI_start = np.linalg.inv(I - self.IO_start)
        LI_mid = np.linalg.inv(I - self.IO_mid)
        LI_end = np.linalg.inv(I - self.IO_end)
        
        #Cut oil and gas extraction as well as coal mining, then normalize
        def reduce_and_normalize(IO):
            IO_r = np.delete(np.delete(IO, [6, 7], axis=0), [6, 7], axis=1)
            return IO_r / IO_r.sum(axis=1, keepdims=True)

        IO_start_reduced = reduce_and_normalize(self.IO_start)
        IO_mid_reduced   = reduce_and_normalize(self.IO_mid)
        IO_end_reduced   = reduce_and_normalize(self.IO_end)

        def tv_metrics(A, B):
            diff = np.abs(B - A)
            tv   = 0.5 * diff.sum(axis=1)
            tv_sq = 0.5 * ((diff**2).sum(axis=1))**(1/2)
            return tv, tv_sq

        tv_p1,          tv_sq_p1          = tv_metrics(self.IO_start,    self.IO_mid)
        tv_LI_p1,       tv_sq_LI_p1       = tv_metrics(LI_start,         LI_mid)
        tv_reduced_p1,  tv_sq_reduced_p1  = tv_metrics(IO_start_reduced, IO_mid_reduced)

        tv_p2,          tv_sq_p2          = tv_metrics(self.IO_mid,    self.IO_end)
        tv_LI_p2,       tv_sq_LI_p2       = tv_metrics(LI_mid,         LI_end)
        tv_reduced_p2,  tv_sq_reduced_p2  = tv_metrics(IO_mid_reduced, IO_end_reduced)

        
        def make_IO_df(tv, tv_sq, tv_LI, tv_sq_LI, J):
            return pd.DataFrame({
                "BLS_Industry":      np.arange(1, J+1),
                "TV_distance":       tv,
                "TV_sq_distance":    tv_sq,
                "TV_distance_LI":    tv_LI,
                "TV_sq_distance_LI": tv_sq_LI})

        def make_reduced_df(tv_r, tv_sq_r, J):
            return pd.DataFrame({
                "BLS_Industry":           np.concatenate([np.arange(1, 7), np.arange(9, J+1)]),
                "TV_distance_reduced":    tv_r,
                "TV_sq_distance_reduced": tv_sq_r})

        IO_df_p1 = make_IO_df(tv_p1, tv_sq_p1, tv_LI_p1, tv_sq_LI_p1, J)
        IO_df_p1 = IO_df_p1.merge(make_reduced_df(tv_reduced_p1, tv_sq_reduced_p1, J), on='BLS_Industry', how='left')
        IO_df_p1['period'] = 1

        IO_df_p2 = make_IO_df(tv_p2, tv_sq_p2, tv_LI_p2, tv_sq_LI_p2, J)
        IO_df_p2 = IO_df_p2.merge(make_reduced_df(tv_reduced_p2, tv_sq_reduced_p2, J), on='BLS_Industry', how='left')
        IO_df_p2['period'] = 2

        IO_df = pd.concat([IO_df_p1, IO_df_p2], ignore_index=True)

        
        # ------------------ #
        # Allocate Emissions #
        # ------------------ #
        IO_wide_df = Ind_CO2_df.pivot(index="BLS_Industry", columns="Year", values='CO2e_intensity_Industry')
        IO_wide_df = IO_wide_df.dropna()
        
        idx1 = IO_wide_df.index.to_numpy(dtype=int)
        idx0 = idx1 - 1
        
        LI_start_sub = LI_start[np.ix_(idx0, idx0)]
        LI_mid_sub   = LI_mid[np.ix_(idx0, idx0)]
        LI_end_sub   = LI_end[np.ix_(idx0, idx0)]

        CO2e_LI_start = LI_start_sub @ IO_wide_df[Year_start].to_numpy()
        CO2e_LI_mid   = LI_mid_sub   @ IO_wide_df[Year_mid].to_numpy()
        CO2e_LI_end   = LI_end_sub   @ IO_wide_df[Year_end].to_numpy()
        
        em_p1 = pd.DataFrame({
            "BLS_Industry":       IO_wide_df.index,
            "dlog_CO2e_inten":    np.log(IO_wide_df[Year_mid].to_numpy())   - np.log(IO_wide_df[Year_start].to_numpy()),
            "dlog_CO2e_inten_LI": np.log(CO2e_LI_mid) - np.log(CO2e_LI_start),
            "period": 1})

        em_p2 = pd.DataFrame({
            "BLS_Industry":       IO_wide_df.index,
            "dlog_CO2e_inten":    np.log(IO_wide_df[Year_end].to_numpy())   - np.log(IO_wide_df[Year_mid].to_numpy()),
            "dlog_CO2e_inten_LI": np.log(CO2e_LI_end) - np.log(CO2e_LI_mid),
            "period": 2})

        em_df = pd.concat([em_p1, em_p2], ignore_index=True)

        distance_cols = ['BLS_Industry', 'period',
                         'TV_distance',       'TV_sq_distance',
                         'TV_distance_LI',    'TV_sq_distance_LI',
                         'TV_distance_reduced','TV_sq_distance_reduced']

        emission_cols = ['BLS_Industry', 'period',
                         'dlog_CO2e_inten', 'dlog_CO2e_inten_LI']

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
        
        mask_p1 = reg_df['period'] == 1
        mask_p2 = reg_df['period'] == 2
        
        # ---------- #
        # Regression #
        # ---------- #
        X = sm.add_constant(reg_df['dlog_CO2e_inten'])
        Y = reg_df['TV_distance']
        
        model = sm.OLS(Y, X).fit(cov_type='cluster', cov_kwds={'groups': reg_df['BLS_Industry']})
        #print(model.summary())
        
        beta0 = model.params['const']
        beta1 = -model.params['dlog_CO2e_inten']
        
        x = -reg_df['dlog_CO2e_inten'].to_numpy()
        y = Y.to_numpy()
        y_hat = beta0 + beta1 * x
        
        # Square Metric
        Y_sq = reg_df['TV_sq_distance']
        
        model_sq = sm.OLS(Y_sq, X).fit(cov_type='cluster', cov_kwds={'groups': reg_df['BLS_Industry']})
        #print(model_sq.summary())
        
        beta0_sq = model_sq.params['const']
        beta1_sq = -model_sq.params['dlog_CO2e_inten']
        
        y_sq = Y_sq.to_numpy()
        y_hat_sq = beta0_sq + beta1_sq * x
        
        
        # ---------- #
        # Plot Graph #
        # ---------- #
        plt.figure(figsize=(8,6))
        plt.scatter(x[mask_p1], y[mask_p1], alpha=0.7, color='purple', label=f"Sectors ({Year_start}–{Year_mid})")
        plt.scatter(x[mask_p2], y[mask_p2], alpha=0.7, color='blue',   label=f"Sectors ({Year_mid}–{Year_end})")
        plt.plot(x, y_hat, color='red', linewidth=2, label="OLS fit")
        
        p1 = model.pvalues['dlog_CO2e_inten']
        stars1 = '***' if p1 < 0.01 else '**' if p1 < 0.05 else '*' if p1 < 0.1 else ''
        plt.annotate(
            f"Slope = {beta1:.3f}{stars1}",
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=11, color='green',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.7)
        )
        plt.xlabel("-Δ ln(emissions intensity)")
        plt.ylabel("TV distance (input-share change)")
        plt.grid(alpha=0.3)
        plt.legend(loc='upper right')
        plt.savefig(f'{self.Directory}/Results/Figures/Baseline_L1.png')
        plt.show()
        
        # Square Metric
        plt.figure(figsize=(8,6))
        plt.scatter(x[mask_p1], y_sq[mask_p1], alpha=0.7, color='purple', label=f"Sectors ({Year_start}–{Year_mid})")
        plt.scatter(x[mask_p2], y_sq[mask_p2], alpha=0.7, color='blue',   label=f"Sectors ({Year_mid}–{Year_end})")
        plt.plot(x, y_hat_sq, color='red', linewidth=2, label="OLS fit")
        
        p_sq = model_sq.pvalues['dlog_CO2e_inten']
        stars_sq = '***' if p_sq < 0.01 else '**' if p_sq < 0.05 else '*' if p_sq < 0.1 else ''
        plt.annotate(
            f"Slope = {beta1_sq:.3f}{stars_sq}",
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=11, color='green',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.7)
        )
        plt.xlabel("-Δ ln(emissions intensity)")
        plt.ylabel("Euclidean distance (input-share change)")
        plt.grid(alpha=0.3)
        plt.legend(loc='upper right')
        plt.savefig(f'{self.Directory}/Results/Figures/Baseline_L2.png')
        plt.show()


        # ----------------------------------------------------------------
    
        # Leontief inverse.
    
        # ----------------------------------------------------------------
        
        # ---------- #
        # Regression #
        # ---------- #
        X = sm.add_constant(reg_df['dlog_CO2e_inten_LI'])
        Y = reg_df['TV_distance_LI']
        
        model = sm.OLS(Y, X).fit(cov_type='cluster', cov_kwds={'groups': reg_df['BLS_Industry']})
        #print(model.summary())
        
        beta0 = model.params['const']
        beta1 = -model.params['dlog_CO2e_inten_LI']
        
        x = -reg_df['dlog_CO2e_inten_LI'].to_numpy()
        y = Y.to_numpy()
        y_hat = beta0 + beta1 * x
        
        # Square Metric
        Y_sq = reg_df['TV_sq_distance_LI']
        
        model_sq = sm.OLS(Y_sq, X).fit(cov_type='cluster', cov_kwds={'groups': reg_df['BLS_Industry']})
        #print(model_sq.summary())
        
        beta0_sq = model_sq.params['const']
        beta1_sq = -model_sq.params['dlog_CO2e_inten_LI']
        
        y_sq = Y_sq.to_numpy()
        y_hat_sq = beta0_sq + beta1_sq * x
        
        
        # ---------- #
        # Plot Graph #
        # ---------- #
        plt.figure(figsize=(8,6))
        plt.scatter(x[mask_p1], y[mask_p1], alpha=0.7, color='purple', label=f"Sectors ({Year_start}–{Year_mid})")
        plt.scatter(x[mask_p2], y[mask_p2], alpha=0.7, color='blue',   label=f"Sectors ({Year_mid}–{Year_end})")
        plt.plot(x, y_hat, color='red', linewidth=2, label="OLS fit")
        
        p1 = model.pvalues['dlog_CO2e_inten_LI']
        stars1 = '***' if p1 < 0.01 else '**' if p1 < 0.05 else '*' if p1 < 0.1 else ''
        plt.annotate(
            f"Slope = {beta1:.3f}{stars1}",
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=11, color='green',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.7)
        )
        plt.xlabel("-Δ ln(emissions intensity)")
        plt.ylabel("TV distance (input-share change)")
        plt.grid(alpha=0.3)
        plt.legend(loc='upper right')
        plt.savefig(f'{self.Directory}/Results/Figures/Leontief_L1.png')
        plt.show()
        
        # Square Metric
        plt.figure(figsize=(8,6))
        plt.scatter(x[mask_p1], y_sq[mask_p1], alpha=0.7, color='purple', label=f"Sectors ({Year_start}–{Year_mid})")
        plt.scatter(x[mask_p2], y_sq[mask_p2], alpha=0.7, color='blue',   label=f"Sectors ({Year_mid}–{Year_end})")
        plt.plot(x, y_hat_sq, color='red', linewidth=2, label="OLS fit")
        
        p_sq = model_sq.pvalues['dlog_CO2e_inten_LI']
        stars_sq = '***' if p_sq < 0.01 else '**' if p_sq < 0.05 else '*' if p_sq < 0.1 else ''
        plt.annotate(
            f"Slope = {beta1_sq:.3f}{stars_sq}",
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=11, color='green',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.7)
        )
        plt.xlabel("-Δ ln(emissions intensity)")
        plt.ylabel("Euclidean distance (input-share change)")
        plt.grid(alpha=0.3)
        plt.legend(loc='upper right')
        plt.savefig(f'{self.Directory}/Results/Figures/Leontief_L2.png')
        plt.show()
        
        
        # ----------------------------------------------------------------
    
        # Reduced and normalized IO table.
    
        # ----------------------------------------------------------------
        
        reduced_df = reg_df.dropna(subset=['TV_distance_reduced', 'TV_sq_distance_reduced', 'dlog_CO2e_inten'])
        mask_p1 = reduced_df['period'] == 1
        mask_p2 = reduced_df['period'] == 2
        
        
        # ---------- #
        # Regression #
        # ---------- #
        X = sm.add_constant(reduced_df['dlog_CO2e_inten'])
        Y = reduced_df['TV_distance_reduced']
        
        model = sm.OLS(Y, X, missing='drop').fit(cov_type='cluster', cov_kwds={'groups': reduced_df['BLS_Industry']})
        #print(model.summary())
        
        beta0 = model.params['const']
        beta1 = -model.params['dlog_CO2e_inten']
        
        x = -reduced_df['dlog_CO2e_inten'].to_numpy()
        y = Y.to_numpy()
        y_hat = beta0 + beta1 * x
        
        # Square Metric
        Y_sq = reduced_df['TV_sq_distance_reduced']
        
        model_sq = sm.OLS(Y_sq, X, missing='drop').fit(cov_type='cluster', cov_kwds={'groups': reduced_df['BLS_Industry']})
        #print(model_sq.summary())
        
        beta0_sq = model_sq.params['const']
        beta1_sq = -model_sq.params['dlog_CO2e_inten']
        
        y_sq = Y_sq.to_numpy()
        y_hat_sq = beta0_sq + beta1_sq * x
        
        
        # ---------- #
        # Plot Graph #
        # ---------- #
        plt.figure(figsize=(8,6))
        plt.scatter(x[mask_p1], y[mask_p1], alpha=0.7, color='purple', label=f"Sectors ({Year_start}–{Year_mid})")
        plt.scatter(x[mask_p2], y[mask_p2], alpha=0.7, color='blue',   label=f"Sectors ({Year_mid}–{Year_end})")
        plt.plot(x, y_hat, color='red', linewidth=2, label="OLS fit")
        
        p1 = model.pvalues['dlog_CO2e_inten']
        stars1 = '***' if p1 < 0.01 else '**' if p1 < 0.05 else '*' if p1 < 0.1 else ''
        plt.annotate(
            f"Slope = {beta1:.3f}{stars1}",
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=11, color='green',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.7)
        )
        plt.xlabel("-Δ ln(emissions intensity)")
        plt.ylabel("TV distance (input-share change)")
        plt.grid(alpha=0.3)
        plt.legend(loc='upper right')
        plt.savefig(f'{self.Directory}/Results/Figures/Reduced_L1.png')
        plt.show()
        
        # Square Metric
        plt.figure(figsize=(8,6))
        plt.scatter(x[mask_p1], y_sq[mask_p1], alpha=0.7, color='purple', label=f"Sectors ({Year_start}–{Year_mid})")
        plt.scatter(x[mask_p2], y_sq[mask_p2], alpha=0.7, color='blue',   label=f"Sectors ({Year_mid}–{Year_end})")
        plt.plot(x, y_hat_sq, color='red', linewidth=2, label="OLS fit")
        
        p_sq = model_sq.pvalues['dlog_CO2e_inten']
        stars_sq = '***' if p_sq < 0.01 else '**' if p_sq < 0.05 else '*' if p_sq < 0.1 else ''
        plt.annotate(
            f"Slope = {beta1_sq:.3f}{stars_sq}",
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=11, color='green',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.7)
        )
        plt.xlabel("-Δ ln(emissions intensity)")
        plt.ylabel("Euclidean distance (input-share change)")
        plt.grid(alpha=0.3)
        plt.legend(loc='upper right')
        plt.savefig(f'{self.Directory}/Results/Figures/Reduced_L2.png')
        plt.show()
        
        
    
    def Up_Down_Green(self, Year_start, Year_end):
        """""
        Upstream and Downstream Incentives for Greenification
        
        Output: 
        """""
        
        # ----------------------------------------------------------------

        # Build regression dataframes.

        # ----------------------------------------------------------------
        
        Ind_CO2_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_CO2.pkl')
        Ind_Pat_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_Pat.pkl')


        # ------------------ #
        # Allocate Emissions #
        # ------------------ #
        np.fill_diagonal(self.IO_end, 0)

        J = self.IO_end.shape[0]
        IO_df = pd.DataFrame({"BLS_Industry": np.arange(1, J+1)})

        IO_df = IO_df.merge(Ind_CO2_df, on=['BLS_Industry'], how='inner')

        IO_wide_df = IO_df.pivot(index="BLS_Industry", columns="Year", values='CO2e_intensity_Industry')
        IO_wide_df = IO_wide_df.dropna()

        idx1 = IO_wide_df.index.to_numpy(dtype=int)
        idx0 = idx1 - 1

        def compute_network_effect(IO_matrix, dlog):
            IO  = IO_matrix[np.ix_(idx0, idx0)]
            I   = np.eye(IO.shape[0])
            LI  = np.linalg.inv(I - IO)
            return {
                "up_dlog_CO2e_inten":          - IO              @ dlog,
                "down_dlog_CO2e_inten":        - IO.T            @ dlog,
                "up_higher_dlog_CO2e_inten":   - (LI - I - IO)   @ dlog,
                "down_higher_dlog_CO2e_inten": - (LI - I - IO).T @ dlog,
            }

        dlog = np.log(IO_wide_df[Year_end].to_numpy()) - np.log(IO_wide_df[Year_start].to_numpy())
        spill = compute_network_effect(self.IO_end, dlog)

        reg_df = pd.DataFrame({
            "BLS_Industry":    IO_wide_df.index,
            "dlog_CO2e_inten": dlog,
            **spill
        })

        reg_df = reg_df.merge(Ind_Pat_df, on='BLS_Industry', how='left')

        # ----------------------------------------------------------------
        # Run regressions.
        # ----------------------------------------------------------------

        X        = sm.add_constant(reg_df[['up_dlog_CO2e_inten', 'down_dlog_CO2e_inten']])
        X_higher = sm.add_constant(reg_df[['up_dlog_CO2e_inten', 'up_higher_dlog_CO2e_inten',
                                           'down_dlog_CO2e_inten', 'down_higher_dlog_CO2e_inten']])
        cluster  = {'cov_type': 'cluster', 'cov_kwds': {'groups': reg_df['BLS_Industry']}}

        # ------------------- #
        # Emission Regression #
        # ------------------- #
        Y_em = -reg_df['dlog_CO2e_inten']

        model_em        = sm.OLS(Y_em, X).fit(**cluster)
        model_em_higher = sm.OLS(Y_em, X_higher).fit(**cluster)

        # ------------------ #
        # Patent Regressions #
        # ------------------ #
        pat_df      = reg_df.dropna(subset=['clean_pat_share', 'clean_cite_share'])
        cluster_pat = {'cov_type': 'cluster', 'cov_kwds': {'groups': pat_df['BLS_Industry']}}

        X_pat        = sm.add_constant(pat_df[['up_dlog_CO2e_inten', 'down_dlog_CO2e_inten']])
        X_pat_higher = sm.add_constant(pat_df[['up_dlog_CO2e_inten', 'up_higher_dlog_CO2e_inten',
                                               'down_dlog_CO2e_inten', 'down_higher_dlog_CO2e_inten']])

        Y_pat_count = pat_df['clean_pat_share']
        model_pat_count        = sm.OLS(Y_pat_count, X_pat).fit(**cluster_pat)
        model_pat_count_higher = sm.OLS(Y_pat_count, X_pat_higher).fit(**cluster_pat)

        Y_pat_cite = pat_df['clean_cite_share']
        model_pat_cite        = sm.OLS(Y_pat_cite, X_pat).fit(**cluster_pat)
        model_pat_cite_higher = sm.OLS(Y_pat_cite, X_pat_higher).fit(**cluster_pat)
        
        
        # ----------- #
        # Print Table #
        # ----------- #
        models = [
            model_em,
            model_em_higher,
            None,            # spacer col
            model_pat_count,
            model_pat_count_higher,
            None,            # spacer col
            model_pat_cite,
            model_pat_cite_higher,
        ]

        variables = [
            ('up_dlog_CO2e_inten',        'Upstream CO2e Reduction'),
            ('down_dlog_CO2e_inten',      'Downstream CO2e Reduction'),
            ('up_higher_dlog_CO2e_inten', 'Higher-Order Upstream CO2e Reduction'),
            ('down_higher_dlog_CO2e_inten','Higher-Order Downstream CO2e Reduction'),
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
        body += f'Obs & {" & ".join(n_vals)} \\\\\n'
        
        out_path = f'{self.Directory}/Results/Tables/Fact2_Regressions.tex'
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



