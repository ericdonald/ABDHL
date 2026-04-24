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
        self.ICE_classes = ["Y02T10/10", "Y02T10/30", "Y02T10/30", "Y02T10/40"]
        self.manu_cols = [7, 93]
        self.fossil_cols = [7-1, 8-1]#, 12-1] #Exclude electricity as well

        
        
    def Cleaner(self, BLS_year_start, Year_start, Year_mid, Year_end, patents=1):
        """""
        Clean Data
        
        Output: Clean Data/BLS_Crosswalk.pkl
                Clean Data/Ind_CO2.pkl
                Clean Data/Ind_CO2_full.pkl
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
            ind_Y  = np.sum(U, 0) #- U[-2,:] #Allows exclusion of imports from revenue
            B      = (U[:-3, :] @ np.diag(ind_Y**(-1))).T

            M      = MAKE_df.iloc[:, 1:].to_numpy()
            com_Y  = np.sum(M, 0)[:-2]
            A      = (M[:-2, :-2] @ np.diag(com_Y**(-1))).T

            IO = B @ A
            num = IO.sum(axis=1, keepdims=True)
            #np.fill_diagonal(IO, 0) #Allows exclusion of diagonal
            denom = IO.sum(axis=1, keepdims=True)

            
            return IO * num / denom

        bin_ends = list(range(BLS_year_start, Year_end+1, 5))
        self.IO = {year: compute_IO(year) for year in bin_ends}
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
        #NAICS 2017
        

        # ---------------- #
        # NAICS Crosswalks #
        # ---------------- #
        NAICS_2017_2022_url = "https://www.census.gov/naics/concordances/2017_to_2022_NAICS.xlsx"
        headers = {"User-Agent": "Mozilla/5.0"}

        r = api.get(NAICS_2017_2022_url, headers=headers)
        NAICS_2017_2022_df = pd.read_excel(io.BytesIO(r.content), skiprows=2)
        
        
        # ----------------- #
        # Value Added Panel #
        # ----------------- #
        J = self.IO[Year_end].shape[0]
        IO_df = pd.DataFrame({"BLS_Industry": np.arange(1, J+1)})
        
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
            .sort_values(['Sector', 'Year'])
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
                                on=['BLS_Industry', 'Year'],
                                how='inner')
            
            Ind_CO2_df['CO2e_Industry'] = Ind_CO2_df.groupby(['BLS_Industry', 'Year'])['CO2e'].transform("sum")
            Ind_CO2_df['CO2e_intensity_Industry'] = Ind_CO2_df['CO2e_Industry'] / Ind_CO2_df['Value_Added']
            
            Ind_CO2_df = Ind_CO2_df[['BLS_Industry', 'Year', 'CO2e_Industry', 'CO2e_intensity_Industry']].drop_duplicates()
            
            return Ind_CO2_df

        Ind_em_panel(EPA_df).to_pickle(f'{self.Directory}/Clean Data/Ind_CO2.pkl')
        Ind_em_panel(EPS_full_df).to_pickle(f'{self.Directory}/Clean Data/Ind_CO2_full.pkl')
        
        
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
            
            relevant_df = relevant_df[(relevant_df["year"] >= BLS_year_start-5) & (relevant_df["year"] <= Year_end)]
            relevant_df["cpc_group5"] = relevant_df["cpc_group"].str[:5]
            relevant_df["cpc_group6"] = relevant_df["cpc_group"].str[:6]
            
            codes = set(self.CPC_classes)
            relevant_df['clean'] = (relevant_df["cpc_class"].isin(codes)
                                    | relevant_df["cpc_subclass"].isin(codes)
                                    | relevant_df["cpc_group"].isin(codes)
                                    | relevant_df["cpc_group5"].isin(codes)
                                    | relevant_df["cpc_group6"].isin(codes)).astype(np.int8)
            
            ice_codes = set(self.ICE_classes)
            relevant_df['ice'] = relevant_df["cpc_subclass"].isin(ice_codes).astype(np.int8)
            
            relevant_df['clean_full'] = relevant_df.groupby("patent_id")['clean'].transform("max")
            relevant_df['clean'] = relevant_df['clean_full'] - relevant_df.groupby("patent_id")['ice'].transform("max")
            relevant_df = relevant_df[['patent_id', 'year', 'clean', 'clean_full']].drop_duplicates()
                    
                    
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
            
            compustat_df = compustat_df[(compustat_df["year"] >= BLS_year_start-5) & (compustat_df["year"] <= Year_end)]
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
            pat_df = pat_df.merge(EPA_BLS_Crosswalk[['naics2022_6', 'BLS_Industry']].drop_duplicates(),
                                on='naics2022_6',
                                how='inner')

            def compute_pat_metrics(df, period, _f=''):
                df = df.copy()
                df['clean']                = df['split_weight'] * df[f'clean{_f}']
                df['clean_pat_count']      = df.groupby('BLS_Industry')['clean'].transform('sum')
                df['pat_count']            = df.groupby('BLS_Industry')['split_weight'].transform('sum')
                df['clean_pat_share']      = df['clean_pat_count'] / df['pat_count']
                
                df['weighted_pat_cites']   = df['split_weight'] * df['norm_cites']
                df['weighted_clean_cites'] = df['clean'] * df['norm_cites']
                df['clean_pat_cites']      = df.groupby('BLS_Industry')['weighted_clean_cites'].transform('sum')
                df['pat_cites']            = df.groupby('BLS_Industry')['weighted_pat_cites'].transform('sum')
                df['clean_cite_share']     = df['clean_pat_cites'] / df['pat_cites']
                
                return (df[['BLS_Industry', 'clean_pat_count', 'pat_count', 'clean_pat_share', 'clean_pat_cites', 'pat_cites', 'clean_cite_share']]
                        .drop_duplicates()
                        .assign(period=period))

            bin_starts = range(BLS_year_start-5, Year_end, 5) 
            frames = []
            frames_full = []
            _f = '_full'
            for i, start in enumerate(bin_starts):
                end = start + 5 
                bin_df = pat_df[(pat_df['year'] > start) & (pat_df['year'] <= end)]
                frames.append(compute_pat_metrics(bin_df, end))
                frames_full.append(compute_pat_metrics(bin_df, end, _f))

            pat_df = pd.concat(frames, ignore_index=True)
            pat_df = pat_df[pat_df['BLS_Industry'] != 71]
            pat_df.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat.pkl')
            
            pat_df_full = pd.concat(frames_full, ignore_index=True)
            pat_df_full.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat_full.pkl')



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
        Ind_CO2_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_CO2.pkl')
        Ind_CO2_df_full = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_CO2_full.pkl')

        
        def IO_panel(Ind_CO2_df):
            # ------------------- #
            # Input-Output Matrix #
            # ------------------- #
            J       = self.IO[Year_start].shape[0]
            manu    = slice(self.manu_cols[0]-1, self.manu_cols[1])  # 0-indexed rows for non-service industries
     
            I        = np.eye(J)
            LI_start = np.linalg.inv(I - self.IO[Year_start])
            LI_mid   = np.linalg.inv(I - self.IO[Year_mid])
            LI_end   = np.linalg.inv(I - self.IO[Year_end])
     
            # Leontief: non-service rows, all columns
            LI_start_manu = LI_start[manu, :]
            LI_mid_manu   = LI_mid[manu, :]
            LI_end_manu   = LI_end[manu, :]
     
            # Reduced: non-service rows, fossil fuel columns (indices 6,7 within manu block) dropped, renormalized
            def drop_and_normalize(IO):
                IO_manu = IO[manu, :]
                IO_r = np.delete(IO_manu, self.fossil_cols, axis=1)
                num = IO_manu.sum(axis=1, keepdims=True)
                denom = IO_r.sum(axis=1, keepdims=True)
                return IO_r * num / denom
    
            IO_start_reduced = drop_and_normalize(self.IO[Year_start])
            IO_mid_reduced   = drop_and_normalize(self.IO[Year_mid])
            IO_end_reduced   = drop_and_normalize(self.IO[Year_end])
    
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
            IO_df_p1['period'] = 1
    
            IO_df_p2 = make_IO_df(tv_LI_p2, tv_sq_LI_p2, tv_red_p2, tv_sq_red_p2)
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
                "CO2e_Industry_weight":      IO_wide_df['CO2e_Industry', Year_start].to_numpy()**(1/dim),
                "CO2e_Industry_LI_weight":   CO2e_lev_LI_start**(1/dim),
                "period": 1})
    
            em_p2 = pd.DataFrame({
                "BLS_Industry":       IO_wide_df.index,
                "dlog_CO2e_inten":    -(np.log(IO_wide_df['CO2e_intensity_Industry', Year_end].to_numpy())
                                      - np.log(IO_wide_df['CO2e_intensity_Industry', Year_mid].to_numpy())),
                "dlog_CO2e_inten_LI": -(np.log(CO2e_LI_end) - np.log(CO2e_LI_mid)),
                "CO2e_Industry_weight":      IO_wide_df['CO2e_Industry', Year_mid].to_numpy()**(1/dim),
                "CO2e_Industry_LI_weight":   CO2e_lev_LI_mid**(1/dim),
                "period": 2})
    
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

            period_fe = (df['period'].to_numpy() == 2).astype(float)
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

            mask_p1 = df['period'].to_numpy() == 1
            mask_p2 = df['period'].to_numpy() == 2

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
    
    
    
    def Up_Down_Green(self, BLS_year_start, Year_start, Year_mid, Year_end, dim=3):
        """""
        Upstream and Downstream Incentives for Greenification
        
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
        
        Ind_CO2_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_CO2.pkl')
        Ind_CO2_df_full = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_CO2_full.pkl')
        Ind_Pat_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_Pat.pkl')
        Ind_Pat_df_full = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_Pat_full.pkl')

       
        # ----------- #
        # Build Panel #
        # ----------- #
        bin_ends = list(range(BLS_year_start, Year_end+1, 5))
        for year in bin_ends:
            np.fill_diagonal(self.IO[year], 0)

        J = self.IO[Year_end].shape[0]
        
        manu_mask = np.zeros(J, dtype=bool)
        manu_slice = slice(self.manu_cols[0]-1, self.manu_cols[1])
        manu_mask[manu_slice] = True
        
        IO_wide_df_full = Ind_CO2_df_full.pivot(index="BLS_Industry", columns="Year", values='CO2e_intensity_Industry')
        IO_wide_df_full = IO_wide_df_full.dropna()
        idx_full = IO_wide_df_full.index.to_numpy(dtype=int)
        
        def build_manu_IO(IO_matrix):
            A              = IO_matrix.copy()
            row_totals_excl = A.sum(axis=1)
            A_manu         = A[np.ix_(np.arange(J)[manu_slice],
                                      np.arange(J)[manu_slice])].copy()
            denom          = row_totals_excl[manu_slice, np.newaxis]
            safe_denom     = np.where(denom == 0, 1, denom)
            return A_manu / safe_denom
        
        manu_IO = {year: build_manu_IO(self.IO[year]) for year in bin_ends}
        manu_idx_all = np.arange(self.manu_cols[0], self.manu_cols[1] + 1)
        manu_mask_wide_full = np.isin(idx_full, manu_idx_all)

        def compute_network_effect(IO_manu, z_wide, prefix, idx):            
            M              = len(manu_idx_all)
            manu_mask_wide = np.isin(idx, manu_idx_all)
            manu_idx      = idx[manu_mask_wide]
            manu_embed_idx = np.searchsorted(manu_idx_all, manu_idx)
            
            I      = np.eye(M)
            LI     = np.linalg.inv(I - IO_manu)
            z_full = np.zeros(M)
            z_full[manu_embed_idx] = np.nan_to_num(z_wide)
            return {
                f"up_{prefix}":   ((LI - I)   @ z_full)[manu_embed_idx],
                f"down_{prefix}": ((LI - I).T @ z_full)[manu_embed_idx],
            }
        
        
        # --------- #
        # Emissions #
        # --------- #
        def em_panel(Ind_CO2_df):
            
            IO_wide_df = Ind_CO2_df.pivot(index="BLS_Industry", columns="Year",
                                   values=['CO2e_intensity_Industry', 'CO2e_Industry'])
            IO_wide_df = IO_wide_df.dropna()
            idx = IO_wide_df.index.to_numpy(dtype=int)
        
            dlog_p1 = -(np.log(IO_wide_df['CO2e_intensity_Industry'][Year_mid].to_numpy())
                      - np.log(IO_wide_df['CO2e_intensity_Industry'][Year_start].to_numpy()))
            dlog_p2 = -(np.log(IO_wide_df['CO2e_intensity_Industry'][Year_end].to_numpy())
                      - np.log(IO_wide_df['CO2e_intensity_Industry'][Year_mid].to_numpy()))
        
            ind_em_p1 = IO_wide_df['CO2e_Industry'][Year_start].to_numpy()
            ind_em_p2 = IO_wide_df['CO2e_Industry'][Year_mid].to_numpy()
                        
            manu_mask_wide = np.isin(idx, manu_idx_all)
            
            def make_em_period(dlog, ind_em, IO_manu, year_label):
                net = compute_network_effect(IO_manu, np.maximum(dlog[manu_mask_wide], 0), 'dlog_em', idx)
                base = pd.DataFrame({
                    "BLS_Industry":    IO_wide_df.index,
                    "period":          year_label,
                    "dlog_CO2e_inten": dlog,
                    "CO2e_Industry":   ind_em,
                })
                manu_df = pd.DataFrame({"BLS_Industry": IO_wide_df.index[manu_mask_wide], **net})
                return base.merge(manu_df, on='BLS_Industry', how='left')
    
            em_df = pd.concat([
                make_em_period(dlog_p1, ind_em_p1, manu_IO[Year_mid], Year_mid),
                make_em_period(dlog_p2, ind_em_p2, manu_IO[Year_end], Year_end),
            ], ignore_index=True)
            
            return em_df
        
        em_df = em_panel(Ind_CO2_df)
        em_df_full = em_panel(Ind_CO2_df_full)


        # ------- #
        # Patents #
        # ------- #
        def pat_panel(Ind_Pat_df):
            pat_frames = []
            for year in bin_ends:
                pat_yr = Ind_Pat_df[Ind_Pat_df['period'] == year].copy()
                if pat_yr.empty:
                    continue
                pat_yr  = pat_yr.set_index('BLS_Industry').reindex(IO_wide_df_full.index).reset_index()
                pc_raw  = pat_yr['clean_pat_share'].to_numpy()
                cc_raw  = pat_yr['clean_cite_share'].to_numpy()
                net_pc  = compute_network_effect(manu_IO[year], pc_raw[manu_mask_wide_full], 'pat_count', idx_full)
                net_cc  = compute_network_effect(manu_IO[year], cc_raw[manu_mask_wide_full], 'pat_cite', idx_full)
                manu_pat_df = pd.DataFrame({
                    "BLS_Industry":    IO_wide_df_full.index[manu_mask_wide_full],
                    **net_pc, **net_cc,
                })
                cpc_weight = pat_yr['clean_pat_count'].to_numpy()
                pc_weight = pat_yr['pat_count'].to_numpy()
                ccc_weight = pat_yr['clean_pat_cites'].to_numpy()
                cc_weight = pat_yr['pat_cites'].to_numpy()
                
                base_pat = pd.DataFrame({
                    "BLS_Industry":    IO_wide_df_full.index,
                    "period":          year,
                    "clean_pat_share": pc_raw,
                    "clean_cite_share":cc_raw,
                    "clean_pat_count":cpc_weight,
                    "pat_count":pc_weight,
                    "clean_pat_cites":ccc_weight,
                    "pat_cites":cc_weight,
                })
                pat_frames.append(base_pat.merge(manu_pat_df, on='BLS_Industry', how='left'))
    
            return pd.concat(pat_frames, ignore_index=True)
        
        pat_df = pat_panel(Ind_Pat_df)
        pat_df_full = pat_panel(Ind_Pat_df_full)
        

        # ----- #
        # Merge #
        # ----- #
        def reg_panel(em_df, pat_df):
            reg_df = pat_df.merge(em_df, on=['BLS_Industry', 'period'], how='left')
            reg_df = reg_df[reg_df['BLS_Industry'].isin(manu_idx_all)].copy()
            
            reg_df['net_dlog_em']   = reg_df['up_dlog_em']  + reg_df['down_dlog_em']
            reg_df['net_pat_count'] = reg_df['up_pat_count'] + reg_df['down_pat_count']
            reg_df['net_pat_cite']  = reg_df['up_pat_cite']  + reg_df['down_pat_cite']
            
            reg_df = reg_df.sort_values(['BLS_Industry', 'period'])
            for col in ['net_dlog_em', 'net_pat_count', 'net_pat_cite',
                        'up_dlog_em',  'down_dlog_em',
                        'up_pat_count','down_pat_count',
                        'up_pat_cite', 'down_pat_cite']:
                reg_df[f'{col}_lag'] = reg_df.groupby('BLS_Industry')[col].shift(1)
                
            return reg_df

        reg_df = reg_panel(em_df, pat_df)
        reg_df_full = reg_panel(em_df_full, pat_df_full)


        # ----------------------------------------------------------------
        
        # Run regressions.
        
        # ----------------------------------------------------------------
        def make_X(df, cols):
            fe = pd.get_dummies(df['period'], drop_first=True, dtype=float)
            return sm.add_constant(pd.concat([df[cols], fe], axis=1))
        
        def fit(df, Y_col, x_cols, w_col=None, em_sub=None):
            d  = em_sub if em_sub is not None else df
            cl = {'cov_type': 'cluster', 'cov_kwds': {'groups': d['BLS_Industry']}}
            Y  = d[Y_col]
            X  = make_X(d, x_cols)
            if w_col is None:
                return sm.OLS(Y, X).fit(**cl)
            w = d[w_col] ** (1 / dim)
            return sm.WLS(Y, X, w).fit(**cl)


        # ------------------ #
        # Estimation Samples #
        # ------------------ #
        reg_em = gpf.winsorize(
            reg_df.dropna(subset=['dlog_CO2e_inten']),
            ['dlog_CO2e_inten',
             'up_dlog_em',   'down_dlog_em',   'net_dlog_em',
             'up_pat_count', 'down_pat_count', 'net_pat_count',
             'up_pat_cite',  'down_pat_cite',  'net_pat_cite'])
        
        reg_cnt = gpf.winsorize(
            reg_df[reg_df['clean_pat_share'] > 0],
            ['clean_pat_share',
             'up_dlog_em',   'down_dlog_em',   'net_dlog_em',
             'up_pat_count', 'down_pat_count', 'net_pat_count'])
        
        reg_cit = gpf.winsorize(
            reg_df[reg_df['clean_cite_share'] > 0],
            ['clean_cite_share',
             'up_dlog_em',  'down_dlog_em',  'net_dlog_em',
             'up_pat_cite', 'down_pat_cite', 'net_pat_cite'])

        reg_cnt_em = reg_cnt.dropna(subset=['up_dlog_em', 'down_dlog_em'])
        reg_cit_em = reg_cit.dropna(subset=['up_dlog_em', 'down_dlog_em'])
        
        reg_em_full = gpf.winsorize(
            reg_df_full.dropna(subset=['dlog_CO2e_inten']),
            ['dlog_CO2e_inten',
             'up_dlog_em',   'down_dlog_em',   'net_dlog_em',
             'up_pat_count', 'down_pat_count', 'net_pat_count',
             'up_pat_cite',  'down_pat_cite',  'net_pat_cite'])
        
        reg_cnt_full = gpf.winsorize(
            reg_df_full[reg_df_full['clean_pat_share'] > 0],
            ['clean_pat_share',
             'up_dlog_em',   'down_dlog_em',   'net_dlog_em',
             'up_pat_count', 'down_pat_count', 'net_pat_count'])
        
        reg_cit_full = gpf.winsorize(
            reg_df_full[reg_df_full['clean_cite_share'] > 0],
            ['clean_cite_share',
             'up_dlog_em',  'down_dlog_em',  'net_dlog_em',
             'up_pat_cite', 'down_pat_cite', 'net_pat_cite'])

        reg_cnt_em_full = reg_cnt_full.dropna(subset=['up_dlog_em', 'down_dlog_em'])
        reg_cit_em_full = reg_cit_full.dropna(subset=['up_dlog_em', 'down_dlog_em'])
        
        lag_em_cols  = ['net_dlog_em_lag',   'net_pat_count_lag', 'net_pat_cite_lag',
                        'up_dlog_em_lag',    'down_dlog_em_lag',
                        'up_pat_count_lag',  'down_pat_count_lag',
                        'up_pat_cite_lag',   'down_pat_cite_lag']
        reg_em_lag  = gpf.winsorize(
            reg_df.dropna(subset=['dlog_CO2e_inten',
             'net_pat_count_lag', 'up_pat_count_lag', 'down_pat_count_lag',
             'net_pat_cite_lag', 'up_pat_cite_lag', 'down_pat_cite_lag']),
            ['dlog_CO2e_inten'] + lag_em_cols)
        reg_cnt_lag = gpf.winsorize(
            reg_df[reg_df['clean_pat_share'] > 0].dropna(
                subset=['net_pat_count_lag', 'up_pat_count_lag', 'down_pat_count_lag']),
             ['clean_pat_share'] + lag_em_cols)
        reg_cit_lag = gpf.winsorize(
            reg_df[reg_df['clean_cite_share'] > 0].dropna(
                subset=['net_pat_cite_lag', 'up_pat_cite_lag',  'down_pat_cite_lag']),
             ['clean_cite_share'] + lag_em_cols)

        reg_em_em_lag = gpf.winsorize(
            reg_df.dropna(subset=['dlog_CO2e_inten',
             'net_dlog_em_lag', 'up_dlog_em_lag', 'down_dlog_em_lag']),
            ['dlog_CO2e_inten'] + lag_em_cols)
        reg_cnt_em_lag = gpf.winsorize(
            reg_df[reg_df['clean_pat_share'] > 0].dropna(
                subset=['net_dlog_em_lag', 'up_dlog_em_lag', 'down_dlog_em_lag']),
             ['clean_pat_share'] + lag_em_cols)
        reg_cit_em_lag = gpf.winsorize(
            reg_df[reg_df['clean_cite_share'] > 0].dropna(
                subset=['net_dlog_em_lag', 'up_dlog_em_lag', 'down_dlog_em_lag']),
             ['clean_cite_share'] + lag_em_cols)


        # ------------------- #
        # Summary Stats Table #
        # ------------------- #
        def summary_stats(s):
            return {
                'Mean': s.mean(),
                'SD':   s.std(),
                'IQR':  s.quantile(0.75) - s.quantile(0.25),
            }

        stat_vars = [
            ('dlog_CO2e_inten', 'Emissions Intensity Reduction', reg_em),
            ('net_dlog_em',     'Network Emissions Intensity Reduction',   reg_em),
            ('clean_pat_share', 'Green Patent Share',                  reg_cnt),
            ('net_pat_count',   'Network Green Patent Share',               reg_cnt),
            ('clean_cite_share','Green Citation Share',                reg_cit),
            ('net_pat_cite',    'Network Green Citation Share',             reg_cit),
        ]

        rows = []
        for col, label, df in stat_vars:
            s    = df[col].dropna()
            stat = summary_stats(s)
            rows.append((label, stat['Mean'], stat['SD'], stat['IQR'], len(s)))

        body = ''
        for label, mean, sd, iqr, n in rows:
            body += f'{label} & {mean:.3f} & {sd:.3f} & {iqr:.3f} & {n} \\\\\n'

        out_path = f'{self.Directory}/Results/Tables/Summary_Stats.tex'
        with open(out_path, 'w') as f:
            f.write(body)
        
        # -------------------- #
        # Emission Regressions #
        # -------------------- #
       
        # Up/down split
        m_em_em  = fit(reg_em, 'dlog_CO2e_inten', ['up_dlog_em',   'down_dlog_em'])
        m_em_pat = fit(reg_em, 'dlog_CO2e_inten', ['up_pat_count', 'down_pat_count'])
        m_em_cit = fit(reg_em, 'dlog_CO2e_inten', ['up_pat_cite',  'down_pat_cite'])

        # Net current
        m_em_em_n  = fit(reg_em, 'dlog_CO2e_inten', ['net_dlog_em'])
        m_em_pat_n = fit(reg_em, 'dlog_CO2e_inten', ['net_pat_count'])
        m_em_cit_n = fit(reg_em, 'dlog_CO2e_inten', ['net_pat_cite'])

        # Net current WLS robustness
        m_em_em_n_wls  = fit(reg_em, 'dlog_CO2e_inten', ['net_dlog_em'],   'CO2e_Industry')
        m_em_pat_n_wls = fit(reg_em, 'dlog_CO2e_inten', ['net_pat_count'], 'CO2e_Industry')
        m_em_cit_n_wls = fit(reg_em, 'dlog_CO2e_inten', ['net_pat_cite'],  'CO2e_Industry')

        # Net current full sample
        m_em_em_n_full  = fit(reg_em_full, 'dlog_CO2e_inten', ['net_dlog_em'])
        m_em_pat_n_full = fit(reg_em_full, 'dlog_CO2e_inten', ['net_pat_count'])
        m_em_cit_n_full = fit(reg_em_full, 'dlog_CO2e_inten', ['net_pat_cite'])

        # Net lagged
        m_em_em_l  = fit(reg_em_em_lag, 'dlog_CO2e_inten', ['net_dlog_em_lag'])
        m_em_pat_l = fit(reg_em_lag,    'dlog_CO2e_inten', ['net_pat_count_lag'])
        m_em_cit_l = fit(reg_em_lag,    'dlog_CO2e_inten', ['net_pat_cite_lag'])

        
       
        # ------------------ #
        # Patent Regressions #
        # ------------------ #
        
        # Counts — up/down split
        m_cnt_em = fit(reg_cnt, 'clean_pat_share', ['up_dlog_em',   'down_dlog_em'],   em_sub=reg_cnt_em)
        m_cnt_pc = fit(reg_cnt, 'clean_pat_share', ['up_pat_count', 'down_pat_count'])

        # Counts — net current
        m_cnt_em_n = fit(reg_cnt, 'clean_pat_share', ['net_dlog_em'],   em_sub=reg_cnt_em)
        m_cnt_pc_n = fit(reg_cnt, 'clean_pat_share', ['net_pat_count'])

        # Counts — net current WLS robustness
        m_cnt_em_n_wls = fit(reg_cnt, 'clean_pat_share', ['net_dlog_em'],   'clean_pat_count', em_sub=reg_cnt_em)
        m_cnt_pc_n_wls = fit(reg_cnt, 'clean_pat_share', ['net_pat_count'], 'clean_pat_count')

        # Counts — net current full sample
        m_cnt_em_n_full = fit(reg_cnt_full, 'clean_pat_share', ['net_dlog_em'],   em_sub=reg_cnt_em_full)
        m_cnt_pc_n_full = fit(reg_cnt_full, 'clean_pat_share', ['net_pat_count'])

        # Counts — net lagged
        m_cnt_em_l = fit(reg_cnt_lag, 'clean_pat_share', ['net_dlog_em_lag'],   em_sub=reg_cnt_em_lag)
        m_cnt_pc_l = fit(reg_cnt_lag, 'clean_pat_share', ['net_pat_count_lag'])

        # Cites — up/down split
        m_cit_em = fit(reg_cit, 'clean_cite_share', ['up_dlog_em',  'down_dlog_em'],  em_sub=reg_cit_em)
        m_cit_cc = fit(reg_cit, 'clean_cite_share', ['up_pat_cite', 'down_pat_cite'])

        # Cites — net current
        m_cit_em_n = fit(reg_cit, 'clean_cite_share', ['net_dlog_em'],  em_sub=reg_cit_em)
        m_cit_cc_n = fit(reg_cit, 'clean_cite_share', ['net_pat_cite'])

        # Cites — net current WLS robustness
        m_cit_em_n_wls = fit(reg_cit, 'clean_cite_share', ['net_dlog_em'],  'clean_pat_cites', em_sub=reg_cit_em)
        m_cit_cc_n_wls = fit(reg_cit, 'clean_cite_share', ['net_pat_cite'], 'clean_pat_cites')

        # Cites — net current full sample
        m_cit_em_n_full = fit(reg_cit_full, 'clean_cite_share', ['net_dlog_em'],  em_sub=reg_cit_em_full)
        m_cit_cc_n_full = fit(reg_cit_full, 'clean_cite_share', ['net_pat_cite'])

        # Cites — net lagged
        m_cit_em_l = fit(reg_cit_lag, 'clean_cite_share', ['net_dlog_em_lag'],  em_sub=reg_cit_em_lag)
        m_cit_cc_l = fit(reg_cit_lag, 'clean_cite_share', ['net_pat_cite_lag'])

    
        
        # ----------- #
        # Print Table #
        # ----------- #

        def build_table(models, variables):
            body = ''
            for varname, label in variables:
                coefs, ses = [], []
                for m in models:
                    if m is None:
                        coefs.append(''); ses.append('')
                    else:
                        c, s = gpf.fmt_coef(m, varname)
                        coefs.append(c); ses.append(s)
                body += f'{label} & {" & ".join(coefs)} \\\\\n'
                body += f'& {" & ".join(ses)} \\\\[3pt]\n'
            r2_vals, n_vals = [], []
            for m in models:
                if m is None:
                    r2_vals.append(''); n_vals.append('')
                else:
                    r2_vals.append(f'{m.rsquared:.3f}')
                    n_vals.append(str(int(m.nobs)))
            body += '\midrule'
            body += f'$R^2$ & {" & ".join(r2_vals)} \\\\\n'
            body += f'Obs & {" & ".join(n_vals)} \\'
            return body
        
        def col_order(em_em, em_pat, em_cit, cnt_em, cnt_pc, cit_em, cit_cc):
            return [em_em, em_pat, em_cit, None, cnt_em, cnt_pc, None, cit_em, cit_cc]

        vars_net = [
            ('net_dlog_em',   'Network Emissions Reduction'),
            ('net_pat_count', 'Network Green Patents'),
            ('net_pat_cite',  'Network Green Citations'),
        ]
        vars_lag = [
            ('net_dlog_em_lag',   'Network Emissions Reduction (lag)'),
            ('net_pat_count_lag', 'Network Green Patents (lag)'),
            ('net_pat_cite_lag',  'Network Green Citations (lag)'),
        ]
        vars_updown = [
            ('up_dlog_em',    'Upstream Emissions Reduction'),
            ('down_dlog_em',  'Downstream Emissions Reduction'),
            ('up_pat_count',  'Upstream Green Patents'),
            ('down_pat_count','Downstream Green Patents'),
            ('up_pat_cite',   'Upstream Green Citations'),
            ('down_pat_cite', 'Downstream Green Citations'),
        ]

        t_net     = build_table(col_order(m_em_em_n,     m_em_pat_n,     m_em_cit_n,
                                          m_cnt_em_n,    m_cnt_pc_n,
                                          m_cit_em_n,    m_cit_cc_n),    vars_net)
        t_net_full = build_table(col_order(m_em_em_n_full,     m_em_pat_n_full,     m_em_cit_n_full,
                                          m_cnt_em_n_full,    m_cnt_pc_n_full,
                                          m_cit_em_n_full,    m_cit_cc_n_full),    vars_net)
        t_net_wls = build_table(col_order(m_em_em_n_wls, m_em_pat_n_wls, m_em_cit_n_wls,
                                          m_cnt_em_n_wls,m_cnt_pc_n_wls,
                                          m_cit_em_n_wls,m_cit_cc_n_wls),vars_net)
        t_lag     = build_table(col_order(m_em_em_l,     m_em_pat_l,     m_em_cit_l,
                                          m_cnt_em_l,    m_cnt_pc_l,
                                          m_cit_em_l,    m_cit_cc_l),    vars_lag)
        t_updown  = build_table(col_order(m_em_em,       m_em_pat,       m_em_cit,
                                          m_cnt_em,      m_cnt_pc,
                                          m_cit_em,      m_cit_cc),      vars_updown)


        for tag, body in [('Net',       t_net),
                          ('Net_full',  t_net_full),
                          ('Net_WLS',   t_net_wls),
                          ('Lagged',    t_lag),
                          ('UpDown',    t_updown)]:
            out_path = f'{self.Directory}/Results/Tables/Network_Regressions_{tag}.tex'
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



