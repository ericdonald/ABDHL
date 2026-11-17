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
import WRDS as wr
import Customer_Data as cud



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

        
        
    def Cleaner(self, BLS_year_start, Year_start, Year_end, API):
        """""
        Clean Data
        
        Output: Clean Data/IO_Networks.pkl
                Raw Data/EPA.pkl
                Raw Data/NAICS.pkl
                Clean Data/BLS_Crosswalk.pkl
                Clean Data/Ind_CO2.pkl
                Clean Data/Ind_CO2_full.pkl
                Raw Data/CPC.pkl
                Raw Data/applications.pkl
                Raw Data/citations.pkl
                Raw Data/Patent_Inventors.pkl
                Raw Data/Patent_Locations.pkl
                Clean Data/Inventor_Locations.pkl
                Clean Data/Tech_Pat_cg.pkl   (patents by 4-digit CPC subclass, year and type clean / general; Match_Coverage)
                Raw Data/compustat.csv, Raw Data/Compustat_Customers/*.csv   (API = 1: WRDS, see WRDS.py)
                Clean Data/Cust_{Xwalk,Patents,Class_Year,Firms,Segments,Sales}.pkl   (customer IV data: see Customer_Data.Clean)
                Clean Data/Pat_Firms.pkl
                Clean Data/Ind_Pat.pkl
                Clean Data/Ind_Pat_Y02P.pkl   (clean = Y02P only)
                Clean Data/Ind_Pat_Cite5.pkl   (citation weights from citations received within five years of the filing year; Pat_Firms.pkl gets norm_cites5)
        """""
        
        # ----------------------------------------------------------------

        # Unpack data sets.

        # ----------------------------------------------------------------
        if API == 1:
            keys_path = self.Directory / ".keys"
            keys = {}
            with open(keys_path) as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        keys[k.strip()] = v.strip()
            
            self.USPTO_API = keys.get("USPTO_API")
            self.WRDS = wr.WRDS(keys.get("WRDS_USERNAME"), keys.get("WRDS_PASSWORD"))     # connects on first use (Duo push)
            
        
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

        # --------------------- #
        # PatentsView CPC Codes #
        # --------------------- #
        if API == 1:
            CPC_df = gpf.Extract_PatentsView('g_cpc_current', self.USPTO_API)
            
            CPC_df['patent_id'] = CPC_df['patent_id'].astype(str)
            CPC_df.to_pickle(f'{self.Directory}/Raw Data/CPC.pkl')
        else:
            CPC_df = pd.read_pickle(f'{self.Directory}/Raw Data/CPC.pkl')
            
        cpc4_df = CPC_df[['patent_id', 'cpc_subclass']][CPC_df['cpc_section'] != 'Y'].drop_duplicates()
        cpc3_df = CPC_df[['patent_id', 'cpc_class']][CPC_df['cpc_section'] != 'Y'].drop_duplicates()
                
        
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
        
        relevant_df['y02p'] = (relevant_df["cpc_subclass"] == "Y02P").astype(np.int8)
        relevant_df['clean_p'] = relevant_df.groupby("patent_id")['y02p'].transform("max")
        relevant_df['clean_full'] = relevant_df.groupby("patent_id")['clean'].transform("max")
        relevant_df['clean'] = relevant_df['clean_full'] - relevant_df.groupby("patent_id")['ice'].transform("max")
        relevant_df['dirty'] = relevant_df.groupby("patent_id")['dirty'].transform("max")
        relevant_df = relevant_df[['patent_id', 'year', 'clean', 'clean_full', 'clean_p', 'dirty']].drop_duplicates()
        
        
        # ------------------------- #
        # Patent Citation Weighting #
        # ------------------------- #
        # Citations received within five years of the cited patent's filing year (citing filing year minus cited filing year
        # between 0 and 5); right-truncated for recent filings, which is not adjusted here.
        _yr = PV_applications_df.drop_duplicates('patent_id').set_index('patent_id')['year']
        _cy = citations_df['patent_id'].map(_yr)
        _ky = citations_df['citation_patent_id'].map(_yr)
        _in5 = ((_cy - _ky) >= 0) & ((_cy - _ky) <= 5)
        _cites5 = citations_df.loc[_in5, 'citation_patent_id'].value_counts()
        del _cy, _ky, _in5
        citations_df['cites'] = citations_df.groupby('citation_patent_id')['citation_patent_id'].transform('count')
        citations_df['cites5'] = citations_df['citation_patent_id'].map(_cites5)
        citations_df = citations_df[['citation_patent_id', 'cites', 'cites5']].drop_duplicates()
        citations_df.rename(columns={'citation_patent_id': 'patent_id'}, inplace=True)
        
        citations_df = citations_df.merge(cpc3_df,
                                            on='patent_id',
                                            how='right')
        citations_df = citations_df.merge(PV_applications_df[['patent_id', 'year']],
                                            on='patent_id',
                                            how='inner')
        
        citations_df['cites'] = citations_df['cites'].fillna(0)
        citations_df['cites'] = citations_df['cites'] + 1
        citations_df['cites5'] = citations_df['cites5'].fillna(0) + 1
        
        citations_df['cpc_cites'] = citations_df.groupby(['cpc_class', 'year'])['cites'].transform('mean')
        citations_df['norm_cites'] = citations_df['cites'] / citations_df.groupby('patent_id')['cpc_cites'].transform('mean')
        citations_df['cpc_cites5'] = citations_df.groupby(['cpc_class', 'year'])['cites5'].transform('mean')
        citations_df['norm_cites5'] = citations_df['cites5'] / citations_df.groupby('patent_id')['cpc_cites5'].transform('mean')
        
        pat_df = pd.merge(
            citations_df[['patent_id', 'norm_cites', 'norm_cites5']].drop_duplicates(),
            relevant_df,
            on='patent_id',
            how='inner'
        )
        
        del PV_applications_df, relevant_df, citations_df
        
        
        # ----------------------------- #
        # Patenting by Technology Class #
        # ----------------------------- #
        # Patents by 4-digit CPC subclass and year, clean / general types (a patent is split equally over its subclasses; general = not clean).
        # Used by Match_Coverage for the number of all and of clean patents filed per year.
        tech_pat_df = pat_df.merge(cpc4_df.rename(columns={'cpc_subclass': 'cpc_unit'}), on='patent_id', how='inner')
        tech_pat_df['type'] = np.where(tech_pat_df['clean'] == 1, 'c', 'g')
        tech_pat_df['tech'] = tech_pat_df['cpc_unit'] + '_' + tech_pat_df['type']
        tech_pat_df = tech_pat_df[['patent_id', 'tech', 'year', 'norm_cites']].drop_duplicates()
        tech_pat_df['split_weight'] = 1 / tech_pat_df.groupby('patent_id')['tech'].transform('count')
        tech_pat_df['cite_weight'] = tech_pat_df['split_weight'] * tech_pat_df['norm_cites']
        tech_pat_df['pat_count'] = tech_pat_df.groupby(['tech', 'year'])['split_weight'].transform('sum')
        tech_pat_df['pat_cites'] = tech_pat_df.groupby(['tech', 'year'])['cite_weight'].transform('sum')
        tech_pat_df = tech_pat_df[['tech', 'year', 'pat_count', 'pat_cites']].drop_duplicates()
        tech_pat_df.to_pickle(f'{self.Directory}/Clean Data/Tech_Pat_cg.pkl')
        
        
        # ------------------------ #
        # Patent to Firm Crosswalk #
        # ------------------------ #
        discern_df = pd.read_csv(f'{self.Directory}/Raw Data/discern_pat_grant_1980_2021.csv', low_memory=False)
        KPSS_df = pd.read_csv(f'{self.Directory}/Raw Data/KPSS_match_patent_permno_2023.csv')
        gvkey_df = pd.read_csv(f'{self.Directory}/Raw Data/permno_gvkey.csv')
        FCDI_df = pd.read_stata(f'{self.Directory}/Raw Data/cw_patent_compustat_adhps.dta')
        
        discern_df = discern_df.rename(columns={"permno_adj": "permno"})
        KPSS_df = KPSS_df.rename(columns={"patent_num": "patent_id"})
        gvkey_df = gvkey_df.rename(columns={"permno_adj": "permno"})
        FCDI_df = FCDI_df.rename(columns={"patent": "patent_id"})
        
        # patent numbers appear with and without leading zeros (and as floats) across the sources: put them on one format
        # before any comparison, so that the same patent is recognised in every source
        _norm_pid = lambda x: x.astype(str).str.strip().str.replace(r'\.0+$', '', regex=True).str.lstrip('0').replace({'': np.nan, 'nan': np.nan, 'None': np.nan})
        for _df in (discern_df, KPSS_df, FCDI_df):
            _df['patent_id'] = _norm_pid(_df['patent_id'])
        discern_df, KPSS_df, FCDI_df = (d_.dropna(subset=['patent_id']) for d_ in (discern_df, KPSS_df, FCDI_df))
        
        new_pats = KPSS_df[~KPSS_df['patent_id'].isin(discern_df['patent_id'])]

        pat_firm_crosswalk_df = pd.concat([discern_df, new_pats], ignore_index=True)

        pat_firm_crosswalk_df = pat_firm_crosswalk_df.merge(gvkey_df[['gvkey', 'permno']],
                                    on='permno',
                                    how='inner'
                                         )
        pat_firm_crosswalk_df = pat_firm_crosswalk_df[['patent_id', 'gvkey']].dropna().drop_duplicates()
        
        # Union with the third crosswalk (links already in DISCERN / KPSS are not duplicated)
        FCDI_links = FCDI_df[['patent_id', 'gvkey']].copy()
        for _c in ['gvkey']:   # gvkey is text in the .dta; put it on the dtype of the existing crosswalk
            if pd.api.types.is_numeric_dtype(pat_firm_crosswalk_df[_c]):
                FCDI_links[_c] = pd.to_numeric(FCDI_links[_c].astype(str).str.strip(), errors='coerce')
        FCDI_links = FCDI_links.dropna()
        if pd.api.types.is_numeric_dtype(pat_firm_crosswalk_df['gvkey']):
            FCDI_links['gvkey'] = FCDI_links['gvkey'].astype(pat_firm_crosswalk_df['gvkey'].dtype)
        FCDI_links = FCDI_links.drop_duplicates()
        pat_firm_crosswalk_df = pat_firm_crosswalk_df.merge(FCDI_links,
                                    on=['patent_id', 'gvkey'],
                                    how='outer')
        pat_firm_crosswalk_df = pat_firm_crosswalk_df.dropna().drop_duplicates()
        
        
        # --------- #
        # Compustat #
        # --------- #
        if API == 1:
            compustat_df = self.WRDS.Compustat()
            wr.Check_Compustat(compustat_df, self.Directory / 'Raw Data' / 'compustat.csv')   # how the new pull compares with the file it replaces
            compustat_df.to_csv(f'{self.Directory}/Raw Data/compustat.csv', index=False)
        compustat_df = pd.read_csv(f'{self.Directory}/Raw Data/compustat.csv')    # also after a download, so that API = 1 and 0 give the same data types
        
        compustat_df = compustat_df[(compustat_df['fic']=="USA") & (compustat_df['final']=="Y")]
        terry_cols = ['at', 'ppent', 'emp', 'capxv', 'sale', 'xrd']
        compustat_df = compustat_df[compustat_df[terry_cols].gt(0).all(axis=1)]
        compustat_df = compustat_df[compustat_df.groupby('gvkey')['gvkey'].transform('count') > 1]
        compustat_df.rename(columns={'fyear': 'year'}, inplace=True)
        
        compustat_df = compustat_df[(compustat_df["year"] <= Year_end)]
        compustat_df['naics2022_6'] = compustat_df['naics'] #Assume Compustat uses most up to date NAICS
        compustat_df = compustat_df[['gvkey', 'naics2022_6']].drop_duplicates()


        # ----------------- #
        # Customer IV Data  #
        # ----------------- #
        # Compustat customer segments, firm sales and names (API = 1: downloaded through WRDS and saved in Raw Data/Compustat_Customers),
        # the patent tables of the customer IV and the customer-to-firm matching: see Customer_Data.
        customer_data = cud.Customer_Data(self)
        customer_data.Collect(API, self.WRDS if API == 1 else None)
        customer_data.Clean(pat_firm_crosswalk_df, CPC_df)


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
        
        pat_firms_df = pat_firms_df[['patent_id', 'year', 'gvkey', 'BLS_Industry', 'clean', 'clean_full', 'clean_p', 'dirty', 'norm_cites', 'norm_cites5']].drop_duplicates()
        pat_ind_df = pat_firms_df[['patent_id', 'year', 'BLS_Industry', 'clean', 'clean_full', 'clean_p', 'dirty', 'norm_cites', 'norm_cites5']].drop_duplicates()
        
        pat_firms_df['split_weight'] = 1 / pat_firms_df.groupby('patent_id')['gvkey'].transform('count')
        pat_firms_df.to_pickle(f'{self.Directory}/Clean Data/Pat_Firms.pkl')
        
        pat_ind_df['split_weight'] = 1 / pat_ind_df.groupby('patent_id')['BLS_Industry'].transform('count')
        
        
        # --------------------------- #
        # Climate Patenting by Sector #
        # --------------------------- #
        annual_df = pat_ind_df.copy()
        annual_df['clean_w']           = annual_df['split_weight'] * annual_df['clean']
        annual_df['clean_p_w']         = annual_df['split_weight'] * annual_df['clean_p']
        annual_df['dirty_w']           = annual_df['split_weight'] * annual_df['dirty']
        annual_df['cite_w']            = annual_df['split_weight'] * annual_df['norm_cites']
        annual_df['clean_cite_w']      = annual_df['clean_w']      * annual_df['norm_cites']
        annual_df['clean_p_cite_w']    = annual_df['clean_p_w']    * annual_df['norm_cites']
        annual_df['dirty_cite_w']      = annual_df['dirty_w']      * annual_df['norm_cites']
        annual_df['cite5_w']           = annual_df['split_weight'] * annual_df['norm_cites5']
        annual_df['clean_cite5_w']     = annual_df['clean_w']      * annual_df['norm_cites5']
        annual_df['dirty_cite5_w']     = annual_df['dirty_w']      * annual_df['norm_cites5']
 
        agg_base = dict(clean_pat_count = ('clean_w',      'sum'),
                        dirty_pat_count = ('dirty_w',      'sum'),
                        pat_count       = ('split_weight', 'sum'),
                        clean_pat_cites = ('clean_cite_w', 'sum'),
                        dirty_pat_cites = ('dirty_cite_w', 'sum'),
                        pat_cites       = ('cite_w',       'sum'))

        agg_y02p = dict(clean_pat_count = ('clean_p_w',      'sum'),
                        dirty_pat_count = ('dirty_w',        'sum'),
                        pat_count       = ('split_weight',   'sum'),
                        clean_pat_cites = ('clean_p_cite_w', 'sum'),
                        dirty_pat_cites = ('dirty_cite_w',   'sum'),
                        pat_cites       = ('cite_w',         'sum'))

        agg_c5 = dict(clean_pat_count = ('clean_w',        'sum'),
                      dirty_pat_count = ('dirty_w',        'sum'),
                      pat_count       = ('split_weight',   'sum'),
                      clean_pat_cites = ('clean_cite5_w',  'sum'),
                      dirty_pat_cites = ('dirty_cite5_w',  'sum'),
                      pat_cites       = ('cite5_w',        'sum'))

        ind_pat_df = (annual_df.groupby(['BLS_Industry', 'year'], as_index=False)
                                    .agg(**agg_base))
        ind_pat_df_c5 = (annual_df.groupby(['BLS_Industry', 'year'], as_index=False)
                                    .agg(**agg_c5))
        ind_pat_df_y02p = (annual_df.groupby(['BLS_Industry', 'year'], as_index=False)
                                    .agg(**agg_y02p))
        panel_idx = pd.MultiIndex.from_product(
            [sorted(ind_pat_df['BLS_Industry'].unique()), 
             list(range(BLS_year_start-10, Year_end + 1))],
            names=['BLS_Industry', 'year'])
        ind_pat_df = (ind_pat_df.set_index(['BLS_Industry', 'year'])
                           .reindex(panel_idx)
                           .fillna(0.0)
                           .reset_index())
        ind_pat_df_y02p = (ind_pat_df_y02p.set_index(['BLS_Industry', 'year'])
                           .reindex(panel_idx)
                           .fillna(0.0)
                           .reset_index())
        ind_pat_df_c5 = (ind_pat_df_c5.set_index(['BLS_Industry', 'year'])
                           .reindex(panel_idx)
                           .fillna(0.0)
                           .reset_index())
 
        ind_pat_df = ind_pat_df[ind_pat_df['BLS_Industry'] != 71]
        ind_pat_df_y02p = ind_pat_df_y02p[ind_pat_df_y02p['BLS_Industry'] != 71]
        ind_pat_df_c5 = ind_pat_df_c5[ind_pat_df_c5['BLS_Industry'] != 71]
        
        for frame in (ind_pat_df, ind_pat_df_y02p, ind_pat_df_c5):
            frame['pat_count_nc'] = frame['pat_count'] - frame['clean_pat_count']
            frame['pat_cites_nc'] = frame['pat_cites'] - frame['clean_pat_cites']
            frame['clim_pat_count'] = frame['clean_pat_count'] + frame['dirty_pat_count']
            frame['clim_pat_cites'] = frame['clean_pat_cites'] + frame['dirty_pat_cites']
 
        ind_pat_df.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat.pkl')
        ind_pat_df_y02p.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat_Y02P.pkl')
        ind_pat_df_c5.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat_Cite5.pkl')
        
        
    def IO_Change(self, Year_start, Year_mid, Year_end):
        """""
        Changes in the input-output network from decarbonization: sectors whose emissions intensity fell (or rose) more
        change their input requirements more. Six two-panel figures (reduced form | Leontief inverse), kinked at zero:
        wls_sqrt (main text), wls_emis (weights = emissions), wls_cube (emissions^(1/3)), ols, wls_sqrt_fe (period effect),
        wls_sqrt_l2 (Euclidean instead of total-variation distance). Standard errors clustered by sector.

        Output: Results/Figures/IO_Change_<variant>.pdf and .png
                Specifications/Batches/io_change/io_change_slopes.csv
        """""

        # ----------------------------------------------------------------

        # Build regression dataframes.

        # ----------------------------------------------------------------
        
        BLS_Crosswalk_df = pd.read_pickle(f'{self.Directory}/Clean Data/BLS_Crosswalk.pkl')
        IO_mats = pd.read_pickle(f'{self.Directory}/Clean Data/IO_Networks.pkl')
        Ind_CO2_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_CO2.pkl')

        
        def IO_panel(Ind_CO2_df, dim):
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

        # Regressions and figures: one two-panel figure (reduced form | Leontief inverse) per variant.
        # x = log emissions-intensity reduction, y = TV distance of the input-requirement rows, kink at zero
        # (separate slopes for x < 0 and x >= 0). Weights: emissions^(1/dim) (reduced form: own emissions;
        # Leontief: emissions embodied through the Leontief inverse); marker area is proportional to emissions.

        # ----------------------------------------------------------------

        reg_dfs = {d: IO_panel(Ind_CO2_df, d) for d in (1, 2, 3)}
        VARIANTS = [   # name, weight root (None = unweighted), time effects, distance
            ('wls_sqrt',    2,    False, 'TV'),      # main text
            ('wls_emis',    1,    False, 'TV'),
            ('wls_cube',    3,    False, 'TV'),
            ('ols',         None, False, 'TV'),
            ('wls_sqrt_fe', 2,    True,  'TV'),
            ('wls_sqrt_l2', 2,    False, 'TV_sq'),
        ]
        PANELS = [('Reduced form', 'dlog_CO2e_inten', 'reduced', 'CO2e_Industry_weight'),
                  ('Leontief inverse', 'dlog_CO2e_inten_LI', 'LI', 'CO2e_Industry_LI_weight')]
        fig_dir = Path(f'{self.Directory}/Results/Figures')
        out_dir = self.Directory / 'Specifications' / 'Batches' / 'io_change'
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []

        def fit(df, x, y, w, fe):
            xn, xp = np.minimum(df[x], 0.0), np.maximum(df[x], 0.0)
            cols = [np.ones(len(df))] + ([(df['period'] == Year_end).astype(float).to_numpy()] if fe else []) + [xn.to_numpy(), xp.to_numpy()]
            X = np.column_stack(cols)
            cl = {'cov_type': 'cluster', 'cov_kwds': {'groups': df['BLS_Industry'].to_numpy()}}
            return (sm.OLS(df[y].to_numpy(), X) if w is None else sm.WLS(df[y].to_numpy(), X, w.to_numpy())).fit(**cl)

        for name, root, fe, dist in VARIANTS:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
            for ax, (title, xcol, tag, wcol) in zip(axes, PANELS):
                df = reg_dfs[root or 2]
                ycol = f'{dist}_distance_{tag}'
                w = df[wcol] if root else None
                m = fit(df, xcol, ycol, w, fe)
                x, y = df[xcol].to_numpy(), df[ycol].to_numpy()
                b, g = m.params[0], (m.params[1] if fe else 0.0)
                s_n, s_p = m.params[-2], m.params[-1]
                y_adj = y - g * (df['period'] == Year_end).to_numpy()
                emis = (df[wcol] ** (root or 2)).to_numpy()
                size = 400 * emis / emis.max() + 4
                for per, col, lab in [(Year_mid, 'purple', f'{Year_start}-{Year_mid}'), (Year_end, 'blue', f'{Year_mid}-{Year_end}')]:
                    k = (df['period'] == per).to_numpy()
                    ax.scatter(x[k], y_adj[k], s=size[k], alpha=0.6, color=col, label=lab)
                ax.plot(np.linspace(x.min(), 0, 50), b + s_n * np.linspace(x.min(), 0, 50), color='cyan', lw=2, label='fit, x < 0')
                ax.plot(np.linspace(0, x.max(), 50), b + s_p * np.linspace(0, x.max(), 50), color='orange', lw=2, label='fit, x >= 0')
                ax.text(0.03, 0.96, f'slope (x<0) = {s_n:.3f}{gpf.get_stars(m.pvalues[-2])}\nslope (x>=0) = {s_p:.3f}{gpf.get_stars(m.pvalues[-1])}',
                        transform=ax.transAxes, va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='green', alpha=0.8))
                ax.set_title(title); ax.set_xlabel('Log emissions-intensity reduction')
                ax.set_ylabel('Distance in input requirements' + (' (adjusted for period)' if fe else '')); ax.grid(alpha=0.3)
                ax.legend(loc='lower right', fontsize=8)
                rows.append(dict(variant=name, panel=title, slope_neg=s_n, se_neg=m.bse[-2], p_neg=m.pvalues[-2],
                                 slope_pos=s_p, se_pos=m.bse[-1], p_pos=m.pvalues[-1], N=int(m.nobs), sectors=df['BLS_Industry'].nunique()))
            fig.tight_layout()
            fig.savefig(fig_dir / f'IO_Change_{name}.pdf')
            fig.savefig(fig_dir / f'IO_Change_{name}.png', dpi=150)
            plt.close(fig)
        pd.DataFrame(rows).to_csv(out_dir / 'io_change_slopes.csv', index=False)
        labs = {'wls_sqrt': 'WLS, square-root weights (main text)', 'wls_emis': 'WLS, emissions weights', 'wls_cube': 'WLS, cube-root weights',
                'ols': 'OLS', 'wls_sqrt_fe': 'WLS, square-root weights, period effect', 'wls_sqrt_l2': 'WLS, square-root weights, Euclidean distance'}
        L = [r'\begin{tabular}{lcccc}', r'\toprule', r' & \multicolumn{2}{c}{Reduced form} & \multicolumn{2}{c}{Leontief inverse} \\', r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
             r' & Intensity rose & Intensity fell & Intensity rose & Intensity fell \\', r'\midrule']
        for name in labs:
            rr = {r_['panel']: r_ for r_ in rows if r_['variant'] == name}
            cells = [(rr[pn][f'slope_{sd}'], rr[pn][f'se_{sd}'], rr[pn][f'p_{sd}']) for pn in ('Reduced form', 'Leontief inverse') for sd in ('neg', 'pos')]
            L.append(labs[name] + ' & ' + ' & '.join(f'{b:.3f}{gpf.get_stars(p_)}' for b, _, p_ in cells) + r' \\')
            L.append(' & ' + ' & '.join(f'({se:.3f})' for _, se, _ in cells) + r' \\')
        L += [r'\midrule', f"Sector-windows (sectors) & \\multicolumn{{4}}{{c}}{{{rows[0]['N']} ({rows[0]['sectors']})}} \\\\", r'\bottomrule', r'\end{tabular}']
        (self.Directory / 'Results' / 'Tables' / 'IO_Change_Slopes.tex').write_text('\n'.join(L) + '\n')
        cam = {'wls_sqrt': 'WlsSqrt', 'wls_emis': 'WlsEmis', 'wls_cube': 'WlsCube', 'ols': 'Ols', 'wls_sqrt_fe': 'WlsSqrtFe', 'wls_sqrt_l2': 'WlsSqrtLtwo'}
        mac = ['% Generated by Processor.IO_Change from io_change_slopes.csv. Do not edit by hand.']
        for r in rows:
            base = 'io' + cam[r['variant']] + ('Red' if r['panel'] == 'Reduced form' else 'Leo')
            for side in ('neg', 'pos'):
                T = side.capitalize()
                mac += [f'\\newcommand{{\\{base}{T}Slope}}{{{r["slope_" + side]:.3f}}}', f'\\newcommand{{\\{base}{T}Se}}{{{r["se_" + side]:.3f}}}',
                        f'\\newcommand{{\\{base}{T}P}}{{{r["p_" + side]:.3f}}}']
        mac += [f'\\newcommand{{\\ioN}}{{{rows[0]["N"]}}}', f'\\newcommand{{\\ioSectors}}{{{rows[0]["sectors"]}}}']
        (self.Directory / 'Results' / 'Tables' / 'IO_Change_macros.tex').write_text('\n'.join(mac) + '\n')
        print(pd.DataFrame(rows).round(3).to_string())


    def Summary_Stats(self, BLS_year_start, Year_end, bin_len=5, top_n=8):
        """""
        Summary statistics of the baseline regression panel (sector x five-year bin), and the sectors with the highest
        clean patenting shares of their upstream and downstream partners (lagged network shares G, averaged over bins).

        Output: Results/Tables/Summary_Stats.tex, Results/Tables/Top_Network_Sectors.tex
                Specifications/Batches/summary_stats/summary_stats.csv, top_network_sectors.csv
        """""
        tab = self.Directory / 'Results' / 'Tables'
        out = self.Directory / 'Specifications' / 'Batches' / 'summary_stats'
        out.mkdir(parents=True, exist_ok=True)
        df, _ = self.UDG_Panel(BLS_year_start, Year_end, bin_len)
        d = df.dropna(subset=['net_G_pat_lag']).copy()
        rows = [('Patents: clean', 'clean_pat_count'), ('Patents: all', 'pat_count'),
                ('Clean share $G$, patents', 'own_G_pat'), ('Clean share $G$, citations', 'own_G_cite'),
                ('Upstream clean share, patents (lagged)', 'up_G_pat_lag'), ('Downstream clean share, patents (lagged)', 'down_G_pat_lag'),
                ('Network clean share, patents (lagged)', 'net_G_pat_lag'),
                ('Upstream clean share, citations (lagged)', 'up_G_cite_lag'), ('Downstream clean share, citations (lagged)', 'down_G_cite_lag'),
                ('Network clean share, citations (lagged)', 'net_G_cite_lag')]
        recs, lines = [], [r'\begin{tabular}{lrrrrr}', r'\toprule', r' & Mean & SD & Min & Max & N \\', r'\midrule']
        for lab, c in rows:
            x = d[c].dropna()
            recs.append(dict(variable=c, mean=x.mean(), sd=x.std(), min=x.min(), max=x.max(), N=len(x)))
            fmt = '{:.1f}' if c.endswith('count') else '{:.3f}'
            lines.append(f'{lab} & ' + ' & '.join(fmt.format(v) for v in (x.mean(), x.std(), x.min(), x.max())) + f' & {len(x)} \\\\')
        lines += [r'\midrule', f'Sectors & \\multicolumn{{5}}{{r}}{{{d.BLS_Industry.nunique()}}} \\\\',
                  f'Bins & \\multicolumn{{5}}{{r}}{{{", ".join(str(int(b)) for b in sorted(d.period.unique()))} (bin ends)}} \\\\',
                  r'\bottomrule', r'\end{tabular}']
        (tab / 'Summary_Stats.tex').write_text('\n'.join(lines) + '\n')
        pd.DataFrame(recs).to_csv(out / 'summary_stats.csv', index=False)

        names = pd.read_pickle(f'{self.Directory}/Clean Data/BLS_Crosswalk.pkl')[['BLS_Industry', 'Sector Title']].drop_duplicates('BLS_Industry')
        g = d.groupby('BLS_Industry')[['up_G_pat_lag', 'down_G_pat_lag', 'own_G_pat']].mean().reset_index().merge(names, on='BLS_Industry', how='left').rename(columns={'Sector Title': 'title'})
        g.sort_values('up_G_pat_lag', ascending=False).to_csv(out / 'top_network_sectors.csv', index=False)
        clean = lambda t: str(t).replace('&', r'\&')
        blocks = []
        for key, title in [('up_G_pat_lag', 'Highest clean share among the sector\'s suppliers (upstream partners)'),
                           ('down_G_pat_lag', 'Highest clean share among the sector\'s customers (downstream partners)')]:
            top = g.sort_values(key, ascending=False).head(top_n)
            blocks.append(r'\multicolumn{4}{l}{\textit{' + title + r'}} \\')
            blocks += [f'{clean(r.title)} & {r.up_G_pat_lag:.3f} & {r.down_G_pat_lag:.3f} & {('--' if pd.isna(r.own_G_pat) else format(r.own_G_pat, '.3f'))} \\\\' for r in top.itertuples()]
            blocks.append(r'\midrule')
        L = [r'\begin{tabular}{lccc}', r'\toprule', r' & Suppliers & Customers & Own \\', r'\midrule'] + blocks[:-1] + [r'\bottomrule', r'\end{tabular}']
        (tab / 'Top_Network_Sectors.tex').write_text('\n'.join(L) + '\n')
        print(open(tab / 'Summary_Stats.tex').read()); print(open(tab / 'Top_Network_Sectors.tex').read())


    def UpDown_Table(self, path):
        """""
        Reduced form for the main text: upstream and downstream network coefficients (clean patent counts, share G, PPML with the
        offset of non-clean patents, sector and period effects, standard errors clustered by sector) with five-year bins or annual
        outcomes, and Leontief-inverse or direct input-output weights. Reads the registry (Batch 0).

        Output: Results/Tables/Reduced_Form_UpDown.tex, Reduced_Form_macros.tex
        """""
        reg = pd.read_csv(self.Directory / 'Specifications' / 'registry.csv').set_index('spec_id')
        cols = [('b0_pat_updown', 'Leontief'), ('b0_pat_dsud', 'Direct'), ('b0_pat_annualud', 'Leontief'), ('b10_pat_annualdsud', 'Direct')]
        cell = lambda i, k, f: reg.loc[i, f'{f}_{k}']
        L = [r'\begin{tabular}{lcccc}', r'\toprule', r' & \multicolumn{2}{c}{Five-year bins} & \multicolumn{2}{c}{Annual} \\', r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
             r' & (1) & (2) & (3) & (4) \\', 'Input--output weights & ' + ' & '.join(c[1] for c in cols) + r' \\', r'\midrule']
        for k, lab in [('up', 'Upstream (suppliers)'), ('down', 'Downstream (customers)')]:
            L.append(lab + ' & ' + ' & '.join(f"{cell(i, k, 'coef'):.2f}{gpf.get_stars(cell(i, k, 'p'))}" for i, _ in cols) + r' \\')
            L.append(' & ' + ' & '.join(f"({cell(i, k, 'se'):.2f})" for i, _ in cols) + r' \\')
        L += [r'\midrule', r'$p$-value: upstream = downstream & ' + ' & '.join(f"{reg.loc[i, 'p_up_eq_down']:.3f}" for i, _ in cols) + r' \\',
              'Observations & ' + ' & '.join(f"{int(reg.loc[i, 'N']):,}" for i, _ in cols) + r' \\',
              'Sectors & ' + ' & '.join(f"{int(reg.loc[i, 'n_sectors'])}" for i, _ in cols) + r' \\', r'\bottomrule', r'\end{tabular}']
        Path(path).write_text('\n'.join(L) + '\n')
        mac = {}
        for (i, _), nm in zip(cols, ['BinsL', 'BinsD', 'AnnL', 'AnnD']):
            for k, kk in [('up', 'Up'), ('down', 'Down')]:
                mac[f'rf{nm}{kk}'] = f"{cell(i, k, 'coef'):.2f}"; mac[f'rf{nm}{kk}P'] = f"{cell(i, k, 'p'):.3f}"
        Path(path).with_name('Reduced_Form_macros.tex').write_text('% Generated by Processor.UpDown_Table. Do not edit by hand.\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in mac.items()))


    def UpDown_Robust(self, path):
        """""
        Appendix table: upstream and downstream coefficients (clean patent counts, share G, PPML) under alternative constructions.
        Reads the registry (Batch 0).

        Output: Results/Tables/Reduced_Form_Robust.tex
        """""
        reg = pd.read_csv(self.Directory / 'Specifications' / 'registry.csv').set_index('spec_id')
        rows = [('Baseline (Table 1, column 1)', 'b0_pat_updown'), ('Control: sum of weights', 'b0_pat_wsumud'),
                ('Normalized weights', 'b0_pat_normud'), ('No shrinkage of shares', 'b0_pat_noshrud'),
                ('Offset: all patents', 'b0_pat_offallud'), ('Network-mass $\\times$ period controls', 'b0_pat_mtrendud'),
                ('Fixed first-bin weights', 'b0_pat_fixwud'), ('Partners without patents at the aggregate share', 'b0_pat_m1ud'), ('Outcome bins up to 2017', 'b0_pat_m2ud'),
                ('Direct input--output coefficients', 'b0_pat_dsud'), ('Placebo: dirty share of partners', 'b0_pat_plcud')]
        L = [r'\begin{tabular}{lccc}', r'\toprule', r' & Upstream & Downstream & Observations \\', r'\midrule']
        for lab, i in rows:
            r = reg.loc[i]
            L.append(lab + f" & {r.coef_up:.2f}{gpf.get_stars(r.p_up)} & {r.coef_down:.2f}{gpf.get_stars(r.p_down)} & {int(r.N):,} \\\\")
            L.append(f" & ({r.se_up:.2f}) & ({r.se_down:.2f}) & \\\\")
        L += [r'\bottomrule', r'\end{tabular}']
        Path(path).write_text('\n'.join(L) + '\n')


    UDG_BASE = dict(share='G',          # G = clean/all patents, D = clean/climate patents
                    measure='pat',      # pat = counts, cite = citation-weighted
                    bin_len=5,
                    normalise=False,    # partner_avg normalisation of network weights
                    shrink=True,        # kappa shrinkage toward the period mean
                    direction='net',    # net = up + down, updown = separate
                    own_lag=False,      # control for own lagged share
                    rhs=None,           # network variable: None = same as share; P = placebo (dirty/all); E = emissions reduction
                    rhs_meas=None,      # measure (pat/cite) of the network variable; None = same as measure
                    controls=(),        # extra network controls, e.g. 'T_pat' (patent share), 'Q_cite' (relative cites/patent)
                    annual=False,       # True = single-year outcome/offset, regressor from the rolling window ending bin_len years earlier
                    mass_trend=False,   # control for (sector's fixed network weight sum) x (period effects)
                    fill_missing=False, # partners with undefined share enter at the aggregate share of the bin
                    direct_s=False,     # direct input-output coefficients instead of the Leontief inverse
                    max_period=None,    # last outcome bin used in the estimation sample
                    drop_sectors=(),    # industries left out of the estimation sample (they remain partners)
                    free_offset=False,  # log(offset) as a regressor with an estimated coefficient (PPML only)
                    fixed_w=False,      # network sums use the first-bin Leontief inverse (also for the regressor and controls)
                    clean='all',        # clean-patent definition: 'all' (Y02E/P/T, B60L) or 'y02p' (Y02P only)
                    iv_wins=None,       # winsorise the instrument share at [w, 1-w] across industries within period
                    iv_lags=(1,),       # instrument lags in bins (1 = same lag as the regressor, 2 = one bin earlier)
                    iv=None,            # None = PPML; 'gov'/'ks'/'rd' = FE-Poisson IV with that instrument; 'none' = same estimator, no instrument
                    entity_fe=True,
                    time_fe=True,
                    offset=None,        # None -> non-clean patents (G) / dirty patents (D)
                    table='main',       # which table the spec is shown in
                    panel='A')          # panel of the main table (baseline is repeated in each panel)


    UDG_WINS = 0.05   # winsorising of the emissions RHS (cross-section within period)


    @staticmethod
    def partner_avg(S_sub, v, obs, normalise=False):
        "Upstream and downstream network sums of v (optionally weight-normalised)."
        v0 = np.where(obs, v, 0.0)
        up, down = S_sub @ v0, S_sub.T @ v0
        if not normalise:
            return up, down
        o = obs.astype(float)
        w_up, w_dn = S_sub @ o, S_sub.T @ o
        return (np.where(w_up > 0, up / np.where(w_up > 0, w_up, 1.0), np.nan),
                np.where(w_dn > 0, down / np.where(w_dn > 0, w_dn, 1.0), np.nan))


    def UDG_Panel(self, BLS_year_start, Year_end, bin_len, normalise=False, shrink=True, ivs=(), annual=False, iv_wins=None, clean='all', fixed_w=False, fill_missing=False, direct_s=False):
        """""
        Regression panel for strategic complementarity: for each (share, measure) in
        {G, D} x {pat, cite} builds the own share, up, down, net network shares, and their lags.
        For each source in ivs ('gov', 'ks', 'rd') it also builds the instrument share Z<src> (clean/gen
        shock ratio, see UDG_Instrument) and its network sums, using the Leontief inverse fixed at the first bin.
        iv_wins: if set, instrument shares are winsorised at [iv_wins, 1-iv_wins] across industries within each period.
        clean: 'all' = baseline clean definition (Y02E, Y02P, Y02T, B60L); 'y02p' = Y02P only (Ind_Pat_Y02P.pkl);
               'entry' = firms' first clean patents over firms at risk (Ind_Pat_Entry.pkl, see Firm_Entry).
               'cite5' = citation weights from citations received within five years after the filing year (Ind_Pat_Cite5.pkl).
        fill_missing: a partner whose share is undefined (no patents in the bin) enters at the aggregate share of the bin
        (the shrinkage formula at zero patents) instead of contributing zero.
        direct_s: network weights are the direct input-output coefficients (diagonal zeroed) instead of the Leontief inverse.
        fixed_w: all network sums (regressors and controls, not only the instruments) use the Leontief inverse of the first
        bin instead of the inverse of each bin, so that changes in the network cannot drive the regressors.
        annual: regressors are built from rolling bin_len-year windows ending in every year (so the regressor for
        year y is the window ending y - bin_len), and the outcome and offset are single-year counts.

        Returns: reg_df, kappas
        """""
        IO_mats = pd.read_pickle(f'{self.Directory}/Clean Data/IO_Networks.pkl')
        Ind_Pat_yr_df = pd.read_pickle(f"{self.Directory}/Clean Data/Ind_Pat{ {'y02p': '_Y02P', 'entry': '_Entry', 'cap': '_Cap', 'cite5': '_Cite5'}.get(clean, '') }.pkl")
        manu_idx_all = np.arange(self.manu_cols[0], self.manu_cols[1] + 1)
        if annual:
            bin_ends = sorted(y for y in range(BLS_year_start, Year_end + 1) if y in IO_mats)
        else:
            bin_ends = sorted(y for y in range(Year_end, BLS_year_start - 1, -bin_len) if y in IO_mats)

        # -------- #
        # Bin Data #
        # -------- #
        pat_cols = ['clean_pat_count', 'dirty_pat_count', 'clim_pat_count', 'pat_count_nc', 'pat_count',
                    'clean_pat_cites', 'dirty_pat_cites', 'clim_pat_cites', 'pat_cites_nc', 'pat_cites']
        rel_bins = sorted(set(Ind_Pat_yr_df['year']).intersection(bin_ends))
        frames = []
        for end in rel_bins:
            w = Ind_Pat_yr_df[(Ind_Pat_yr_df['year'] > end - bin_len) & (Ind_Pat_yr_df['year'] <= end)]
            if not w.empty:
                frames.append(w.groupby('BLS_Industry', as_index=False)[pat_cols].sum().assign(period=end))
        Ind_Pat_df = pd.concat(frames, ignore_index=True)
        idx = pd.MultiIndex.from_product([sorted(Ind_Pat_df['BLS_Industry'].unique()), rel_bins],
                                         names=['BLS_Industry', 'period'])
        Ind_Pat_df = (Ind_Pat_df.set_index(['BLS_Industry', 'period'])
                                .reindex(idx).fillna(0.0).reset_index())

        # ---------------- #
        # Leontief Inverse #
        # ---------------- #
        def build_sigma_LI(IO_matrix):
            S = IO_matrix.copy() if direct_s else np.linalg.inv(np.eye(IO_matrix.shape[0]) - IO_matrix)
            np.fill_diagonal(S, 0.0)
            return S
        Sigma_LI = {year: build_sigma_LI(IO_mats[year]) for year in bin_ends}

        # ------ #
        # Shares #
        # ------ #
        def wide(col):
            return (Ind_Pat_df.pivot(index='period', columns='BLS_Industry', values=col)
                              .reindex(index=bin_ends, columns=manu_idx_all).sort_index())

        def estimate_kappa(cln_w, tot_w, min_den=100):
            c = cln_w.to_numpy(dtype=float).ravel()
            n = tot_w.to_numpy(dtype=float).ravel()
            ok = np.isfinite(c) & np.isfinite(n) & (n >= min_den)
            if ok.sum() < 20:
                print(f'  only {ok.sum()} sector-bins above {min_den}; falling back to kappa = 10')
                return 10.0
            p = np.nansum(c) / np.nansum(n)
            v = np.var(c[ok] / n[ok], ddof=1)
            samp = np.mean(p * (1 - p) / n[ok])
            return float(np.clip(p * (1 - p) / max(v - samp, 1e-12) - 1, 1.0, 200.0))

        def shrink_to_mean(num, den, kappa):
            gbar = (num.sum(axis=1) / den.sum(axis=1)).to_numpy()[:, None]
            out = (num + kappa * gbar) / (den + kappa)
            return out if fill_missing else out.where(den > 0)

        def raw_share(num, den):
            r = num / den.where(den > 0)
            if fill_missing:
                r = r.where(den > 0, (num.sum(axis=1) / den.sum(axis=1)).to_numpy()[:, None])
            return r

        shares, kappas = {}, {}
        for meas, suf in [('pat', 'count'), ('cite', 'cites')]:
            cln = wide(f'clean_pat_{suf}')
            for sh, den in [('G', wide(f'pat_{suf}')), ('D', wide(f'clim_pat_{suf}'))]:
                if shrink:
                    kappas[(sh, meas)] = estimate_kappa(cln, den)
                    shares[(sh, meas)] = shrink_to_mean(cln, den, kappas[(sh, meas)])
                else:
                    kappas[(sh, meas)] = np.nan
                    shares[(sh, meas)] = raw_share(cln, den)
            # Placebo: dirty / all patents
            dty, tot = wide(f'dirty_pat_{suf}'), wide(f'pat_{suf}')
            if shrink:
                kappas[('P', meas)] = estimate_kappa(dty, tot)
                shares[('P', meas)] = shrink_to_mean(dty, tot, kappas[('P', meas)])
            else:
                kappas[('P', meas)] = np.nan
                shares[('P', meas)] = raw_share(dty, tot)
            # Placebo for D: dirty / (clean + dirty)
            clm = wide(f'clim_pat_{suf}')
            if shrink:
                kappas[('PD', meas)] = estimate_kappa(dty, clm)
                shares[('PD', meas)] = shrink_to_mean(dty, clm, kappas[('PD', meas)])
            else:
                kappas[('PD', meas)] = np.nan
                shares[('PD', meas)] = raw_share(dty, clm)

        # Controls: sum of Leontief weights (v = 1 for every sector, so up = row sum, down = column sum)
        shares[('W', 'w')] = pd.DataFrame(1.0, index=bin_ends, columns=manu_idx_all)
        kappas[('W', 'w')] = np.nan

        # Controls: total-patent share and relative citations per patent
        tot_p, tot_c = wide('pat_count'), wide('pat_cites')
        shares[('T', 'pat')] = tot_p.div(tot_p.sum(axis=1), axis=0)
        cpp = tot_c / tot_p.where(tot_p > 0)
        shares[('Q', 'cite')] = cpp.div(tot_c.sum(axis=1) / tot_p.sum(axis=1), axis=0)
        kappas[('T', 'pat')] = kappas[('Q', 'cite')] = np.nan

        # Level specification: inverse hyperbolic sine of clean patents, winsorised across industries within period
        for meas, suf in [('pat', 'count'), ('cite', 'cites')]:
            a = np.arcsinh(wide(f'clean_pat_{suf}').astype(float))
            for t in a.index:
                v = a.loc[t].to_numpy(dtype=float)
                if np.isfinite(v).sum() > 2:
                    lo, hi = np.nanquantile(v, [self.UDG_WINS, 1 - self.UDG_WINS])
                    a.loc[t] = np.clip(v, lo, hi)
            shares[('A', meas)] = a
            kappas[('A', meas)] = np.nan

        # Instruments: clean/gen shock ratio by source and measure
        # Periods before the first bin, needed only for the two-bin-lagged instruments
        step = 1 if annual else bin_len
        extra_ends = list(range(min(bin_ends) - bin_len, min(bin_ends), step))
        z_extra = {}
        for src in ivs:
            for meas in ('pat', 'cite'):
                z = self.UDG_Instrument(src, meas, extra_ends + bin_ends, bin_len, manu_idx_all)
                if iv_wins:
                    for t in z.index:
                        v = z.loc[t].to_numpy(dtype=float)
                        if np.isfinite(v).sum() > 2:
                            lo, hi = np.nanquantile(v, [iv_wins, 1 - iv_wins])
                            z.loc[t] = np.clip(v, lo, hi)
                shares[(f'Z{src}', meas)] = z.loc[bin_ends]
                z_extra[(f'Z{src}', meas)] = z.loc[extra_ends]
                kappas[(f'Z{src}', meas)] = np.nan

        # Emissions: ln reduction in emissions intensity (winsorised by period)
        co2 = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_CO2.pkl')
        co2 = co2[co2['year'].isin(bin_ends) & co2['BLS_Industry'].isin(manu_idx_all)].copy()
        co2['ln_int'] = np.log(co2['CO2e_intensity_Industry'].where(co2['CO2e_intensity_Industry'] > 0))
        ln_w = (co2.pivot(index='year', columns='BLS_Industry', values='ln_int')
                   .reindex(index=bin_ends, columns=manu_idx_all))
        dln_w = -(ln_w - ln_w.shift(bin_len if annual else 1))
        for t in dln_w.index:
            v = dln_w.loc[t].to_numpy(dtype=float)
            if np.isfinite(v).any():
                lo, hi = np.nanquantile(v, [self.UDG_WINS, 1 - self.UDG_WINS])
                dln_w.loc[t] = np.clip(v, lo, hi)
        shares[('E', 'em')] = dln_w
        kappas[('E', 'em')] = np.nan

        # ------- #
        # Network #
        # ------- #
        keep = np.isin(manu_idx_all, Ind_Pat_df['BLS_Industry'].unique())
        keep_idx = manu_idx_all[keep]
        net_frames = []
        for t in bin_ends:
            S = Sigma_LI[min(bin_ends) if fixed_w else t][np.ix_(keep, keep)]
            S_fix = Sigma_LI[min(bin_ends)][np.ix_(keep, keep)]
            row = {'BLS_Industry': keep_idx, 'period': t}
            for (sh, meas), sv in shares.items():
                v = sv.loc[t].to_numpy(dtype=float)[keep]
                if np.isfinite(v).any():
                    up, dn = self.partner_avg(S_fix if sh.startswith('Z') else S, v, np.isfinite(v), normalise)
                else:
                    up = dn = np.full(len(v), np.nan)
                row[f'own_{sh}_{meas}'] = v
                row[f'up_{sh}_{meas}'] = up
                row[f'down_{sh}_{meas}'] = dn
                row[f'net_{sh}_{meas}'] = up + dn
            net_frames.append(pd.DataFrame(row))
        net_df = pd.concat(net_frames, ignore_index=True)
        # Instrument network sums for the periods before the first bin (fixed first-bin weights)
        extra_frames = []
        if z_extra:
            S_fix = Sigma_LI[min(bin_ends)][np.ix_(keep, keep)]
            for t in extra_ends:
                row = {'BLS_Industry': keep_idx, 'period': t}
                for (sh, meas), sv in z_extra.items():
                    v = sv.loc[t].to_numpy(dtype=float)[keep]
                    if np.isfinite(v).any():
                        up, dn = self.partner_avg(S_fix, v, np.isfinite(v), normalise)
                    else:
                        up = dn = np.full(len(v), np.nan)
                    row[f'own_{sh}_{meas}'] = v
                    row[f'up_{sh}_{meas}'] = up
                    row[f'down_{sh}_{meas}'] = dn
                    row[f'net_{sh}_{meas}'] = up + dn
                extra_frames.append(pd.DataFrame(row))
        if annual:
            outc = Ind_Pat_yr_df[Ind_Pat_yr_df['year'].isin(bin_ends)
                                 & Ind_Pat_yr_df['BLS_Industry'].isin(Ind_Pat_df['BLS_Industry'].unique())]
            outc = outc[['BLS_Industry', 'year'] + pat_cols].rename(columns={'year': 'period'})
        else:
            outc = Ind_Pat_df
        reg_df = net_df.merge(outc, on=['BLS_Industry', 'period'], how='left')

        # ---- #
        # Lags #
        # ---- #
        lag_cols = [c for c in net_df.columns if c not in ('BLS_Industry', 'period')]
        lagged = net_df[['BLS_Industry', 'period'] + lag_cols].copy()
        lagged['period'] = lagged['period'] + bin_len
        lagged = lagged.rename(columns={c: f'{c}_lag' for c in lag_cols})
        reg_df = reg_df.merge(lagged, on=['BLS_Industry', 'period'], how='left')
        z_cols = [c for c in lag_cols if '_Z' in c]
        if z_cols:
            lagged2 = pd.concat([net_df[['BLS_Industry', 'period'] + z_cols]] + [f[['BLS_Industry', 'period'] + z_cols] for f in extra_frames],
                                ignore_index=True)
            lagged2['period'] = lagged2['period'] + 2 * bin_len
            lagged2 = lagged2.rename(columns={c: f'{c}_lag2' for c in z_cols})
            reg_df = reg_df.merge(lagged2, on=['BLS_Industry', 'period'], how='left')
        return reg_df, kappas


    def UDG_Instrument(self, src, meas, bin_ends, bin_len, manu_idx_all):
        """""
        Instrument share by industry and bin: (clean-type shock) / (gen-type shock), with numerator and
        denominator summed over the bin before taking the ratio. Missing where the denominator is zero.
        src: 'govr'/'ksr' (all-patent class distribution times the class-level ratio government clean / government all,
        or spillovers received by clean / received by all; 'govrn'/'ksrn' renormalise over classes with a defined ratio),
        'gov' (government/university patents by CPC), 'ks' (knowledge spillovers, all sources), 'kswt' (spillovers
        to clean from clean over spillovers to general from general), 'rd' (R&D price exposure).
        Returns: DataFrame, index bin_ends, columns manu_idx_all.
        """""
        D = f'{self.Directory}/Clean Data/'
        lnsfx = '_asinh' if '_asinh' in src else ''   # patent levels entering the shocks as inverse hyperbolic sines
        src = src.replace('_asinh', '')
        dg = '_Diag' if src.endswith('_diag') else ''   # spillover network that keeps the own-technology term
        tg = '_Type' if src.endswith('_type') else ''   # government series of the recipient's own type
        cu = '_Cur' if src.endswith('_cur') else ''     # contemporaneous class distribution of the industry
        src = src[:-5] if dg else src[:-5] if tg else src[:-4] if cu else src
        sfx = '_c3' if src.endswith('_c3') else '_mg' if src.endswith('_mg') else ''   # 3-digit class / main-group versions
        src = src[:-3] if sfx else src
        if src in ('govcl', 'ksagg'):
            # Aggregate shock x sector exposure: the sector's pre-sample clean share times a single national series,
            #   govcl: government clean patenting;  ksagg: spillovers received by all clean patents.
            pc = pd.read_pickle(D + 'Ind_Pre_Clean.pkl').set_index('BLS_Industry')
            share = (pc['pre_clean_pat'] / pc['pre_all_pat'] if meas == 'pat' else pc['pre_clean_cites'] / pc['pre_all_cites'])
            share = share.reindex(manu_idx_all)
            if src == 'govcl':
                g = pd.read_pickle(D + f'Gov_CPC{sfx}.pkl')
                ser = g.groupby('year')['gov_clean_count' if meas == 'pat' else 'gov_clean_cites'].sum()
            else:
                a = pd.read_pickle(D + f'KS_CleanAgg_cg{sfx}{lnsfx}.pkl').set_index('year')
                ser = a['recv_cnt' if meas == 'pat' else 'recv_cit']
            sums = {end: ser[(ser.index > end - bin_len) & (ser.index <= end)].sum() for end in bin_ends}
            if lnsfx and src == 'govcl':   # government clean patenting as asinh (the spillover series are transformed already)
                out = {end: share * np.arcsinh(sums[end]) for end in bin_ends}
            else:
                scale = np.nanmean([v for v in sums.values() if v > 0]) or 1.0   # the series is scaled to mean one across bins
                out = {end: share * (sums[end] / scale) for end in bin_ends}
            return pd.DataFrame(out).T.reindex(index=bin_ends)
        if src in ('govr', 'govrn', 'ksr', 'ksrn'):
            # Industry all-patent class distribution times a class-level ratio:
            #   gov: government clean / government all;  ks: spillovers received by clean / received by all types.
            # Ratio formed on class sums over the window; classes with an undefined ratio contribute zero ('govr', 'ksr')
            # or are dropped and the weights renormalised ('govrn', 'ksrn').
            renorm = src.endswith('n')
            base = src[:-1] if renorm else src
            c, n = ('cnt', 'pat') if meas == 'pat' else ('cit', 'cite')
            sh = pd.read_pickle(D + f'Ind_Pat_Shares_Pre{sfx}.pkl')
            sh = sh[sh['type'] == 'all']
            W = (sh.pivot_table(index='BLS_Industry', columns='cpc_subclass',
                                values='cpc_pat_share' if meas == 'pat' else 'cpc_cite_share', aggfunc='sum', fill_value=0.0)
                   .reindex(manu_idx_all, fill_value=0.0))
            if base == 'govr':
                g = pd.read_pickle(D + f'Gov_CPC{sfx}.pkl')
                t = pd.DataFrame({'cpc_subclass': g['cpc_subclass'], 'year': g['year'],
                                  'num': g[f'gov_clean_{"count" if meas == "pat" else "cites"}'],
                                  'den': g[f'gov_pat_{"count" if meas == "pat" else "cites"}']})
            else:
                kr = pd.read_pickle(D + f'KS_Recv_cg{sfx}{dg}{lnsfx}.pkl')   # clean / general types only
                t = pd.DataFrame({'cpc_subclass': kr['cpc_subclass'], 'year': kr['year'],
                                  'num': kr[f'recv_{c}_clean'], 'den': kr[f'recv_{c}_gen']})
            Wm = W.to_numpy(float)
            has_w = Wm.sum(axis=1) > 0
            out = {}
            for end in bin_ends:
                w = t[(t['year'] > end - bin_len) & (t['year'] <= end)].groupby('cpc_subclass')[['num', 'den']].sum()
                r = (w['num'] / w['den'].where(w['den'] > 0)).reindex(W.columns).to_numpy(float)
                ok = np.isfinite(r)
                if not ok.any():
                    out[end] = pd.Series(np.nan, index=manu_idx_all)
                    continue
                z = Wm[:, ok] @ r[ok]
                if renorm:
                    cov = Wm[:, ok].sum(axis=1)
                    z = np.where(cov > 0, z / np.where(cov > 0, cov, 1.0), np.nan)
                z = np.where(has_w, z, np.nan)
                out[end] = pd.Series(z, index=manu_idx_all)
            return pd.DataFrame(out).T.reindex(index=bin_ends)
        c, n = ('cnt', 'pat') if meas == 'pat' else ('cit', 'cite')
        if src == 'gov':
            df = pd.read_pickle(D + f'Govt_Shocks{sfx}{tg}{cu}{lnsfx}.pkl')
            num, den = df[f'{n}_govt_shock_clean'], df[f'{n}_govt_shock_gen']
        elif src == 'ks':
            df = pd.read_pickle(D + f'KS_Shocks_cg{sfx}{dg}{cu}{lnsfx}.pkl')   # clean / general types only
            num = df[[x for x in df.columns if x.startswith(f'spill_shock_{c}_clean_')]].fillna(0.0).sum(axis=1)
            den = df[[x for x in df.columns if x.startswith(f'spill_shock_{c}_gen_')]].fillna(0.0).sum(axis=1)
        elif src == 'kswt':
            df = pd.read_pickle(D + f'KS_Shocks_cg{sfx}{dg}{cu}{lnsfx}.pkl')
            num, den = df[f'spill_shock_{c}_clean_c'].fillna(0.0), df[f'spill_shock_{c}_gen_g'].fillna(0.0)
        elif src in ('rd', 'rdcf'):
            df = pd.read_pickle(D + ('RD_Shocks_cf.pkl' if src == 'rdcf' else 'RD_Shocks.pkl'))
            k = 'count' if meas == 'pat' else 'cites'
            num, den = df[f'pat_{k}_clean_hat'], df[f'pat_{k}_hat_gen']
        else:
            raise ValueError(f'unknown instrument source {src}')
        d = pd.DataFrame({'BLS_Industry': df['BLS_Industry'], 'year': df['year'], 'num': num, 'den': den})
        out = {}
        for end in bin_ends:
            w = d[(d['year'] > end - bin_len) & (d['year'] <= end)]
            g = w.groupby('BLS_Industry')[['num', 'den']].sum()
            out[end] = (g['num'] / g['den'].where(g['den'] > 0)).reindex(manu_idx_all)
        return pd.DataFrame(out).T.reindex(index=bin_ends)


    def UDG_FirstStage(self, df, endog, instruments, exog, y_col, offset_col, entity_fe=True):
        """""
        OLS first stages on the IV estimation sample (same filters as es.fit_poisson_iv), with sector and
        period effects, SEs clustered by sector. For each endogenous regressor reports the coefficient on its
        own-slot instrument, the cluster-robust F of all excluded instruments, and the partial (within) R2.

        Returns: dict slot -> dict(coef, se, F, pr2), and the sample size.
        """""
        need = [y_col, offset_col] + list(endog) + list(exog) + list(instruments)
        d = df.dropna(subset=need).copy()
        d = d[d[offset_col] > 0]
        d = d[d.groupby('BLS_Industry')[y_col].transform('sum') > 0]
        fe = [pd.get_dummies(d['BLS_Industry'], prefix='sec', drop_first=True, dtype=float)] if entity_fe else []
        if d['period'].nunique() > 1:
            fe.append(pd.get_dummies(d['period'], prefix='per', drop_first=True, dtype=float))
        base = pd.concat([pd.Series(1.0, index=d.index, name='const'), d[list(exog)].astype(float)] + fe, axis=1)
        X = pd.concat([d[list(instruments)].astype(float), base], axis=1)
        X.columns = [str(c) for c in X.columns]
        out = {}
        for e in endog:
            y = d[e].astype(float)
            res = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': d['BLS_Industry'].to_numpy()})
            ins = [str(c) for c in instruments]
            R = np.zeros((len(ins), X.shape[1]))
            for i, c in enumerate(ins):
                R[i, list(X.columns).index(c)] = 1.0
            ft = res.f_test(R)
            F, pF = float(np.squeeze(ft.fvalue)), float(np.squeeze(ft.pvalue))
            # partial R2: residualise y and the instruments on the non-instrument regressors
            B = base.to_numpy(float)
            proj = lambda M: M - B @ np.linalg.lstsq(B, M, rcond=None)[0]
            ry, rz = proj(y.to_numpy(float)[:, None])[:, 0], proj(d[list(instruments)].to_numpy(float))
            fit = rz @ np.linalg.lstsq(rz, ry, rcond=None)[0]
            pr2 = 1 - np.sum((ry - fit) ** 2) / np.sum(ry ** 2)
            slot = e.split('_')[0]
            own = next((c for c in ins if c.split('_')[0] == slot), ins[0])
            out[slot] = dict(coef=float(res.params[own]), se=float(res.bse[own]), p=float(res.pvalues[own]), F=F, pF=pF, pr2=float(pr2))
        return out, len(d)


    def UDG_Fit(self, df, y_col, offset_col, x_cols, entity_fe=True, time_fe=True, free_offset=False):
        """Poisson pseudo-ML with log(offset), sector and period dummies, SEs clustered by sector.
        free_offset: log(offset) enters as a regressor with a free coefficient instead of an offset with coefficient one."""
        d = df.dropna(subset=[y_col, offset_col] + list(x_cols)).copy()
        d = d[d[offset_col] > 0]
        pos = d.groupby('BLS_Industry')[y_col].transform('sum') > 0
        n_drop = int(d['BLS_Industry'][~pos].nunique())
        d = d[pos]
        if n_drop:
            print(f'  UDG_Fit({y_col}): dropped {n_drop} sector(s) with no positive outcome.')
        ln_off = np.log(d[offset_col].astype(float))
        parts = [pd.Series(1.0, index=d.index, name='const'), d[list(x_cols)].astype(float)]
        if free_offset:
            parts.append(ln_off.rename('ln_offset'))
        if entity_fe:
            parts.append(pd.get_dummies(d['BLS_Industry'], prefix='sec', drop_first=True, dtype=float))
        time_fe = time_fe and d['period'].nunique() > 1
        if time_fe:
            parts.append(pd.get_dummies(d['period'], prefix='per', drop_first=True, dtype=float))
        X = pd.concat(parts, axis=1)
        X.columns = [str(c) for c in X.columns]
        res = sm.GLM(d[y_col].astype(float), X, family=sm.families.Poisson(),
                     offset=None if free_offset else ln_off.to_numpy()
                     ).fit(cov_type='cluster', cov_kwds={'groups': d['BLS_Industry'].to_numpy()},
                           maxiter=200)
        fe = (['sector'] if entity_fe else []) + (['period'] if time_fe else [])
        xs = list(x_cols) + (['ln_offset'] if free_offset else [])
        return es.GLMWrap(res, y_col, xs, None if free_offset else offset_col, fe,
                          n_sectors=d['BLS_Industry'].nunique()), d['period'].nunique()


    def UDG_OLS(self, df, y_col, x_cols, entity_fe=True, time_fe=True):
        "Linear regression with sector and period dummies, SEs clustered by sector."
        d = df.dropna(subset=[y_col] + list(x_cols)).copy()
        parts = [pd.Series(1.0, index=d.index, name='const'), d[list(x_cols)].astype(float)]
        if entity_fe:
            parts.append(pd.get_dummies(d['BLS_Industry'], prefix='sec', drop_first=True, dtype=float))
        time_fe = time_fe and d['period'].nunique() > 1
        if time_fe:
            parts.append(pd.get_dummies(d['period'], prefix='per', drop_first=True, dtype=float))
        X = pd.concat(parts, axis=1)
        X.columns = [str(c) for c in X.columns]
        res = sm.OLS(d[y_col].astype(float), X).fit(cov_type='cluster', cov_kwds={'groups': d['BLS_Industry'].to_numpy()})
        fe = (['sector'] if entity_fe else []) + (['period'] if time_fe else [])
        m = es.GLMWrap(res, y_col, list(x_cols), None, fe, n_sectors=d['BLS_Industry'].nunique())
        m.rsquared = float(res.rsquared)
        return m, d['period'].nunique()


    def UDG_Run(self, batch, specs, BLS_year_start, Year_end, description=''):
        """""
        Run a batch of specifications. Each spec is a dict with a unique 'id' and any overrides
        of Processor.UDG_BASE. Re-running a batch replaces its rows in the registry.

        Output: Specifications/registry.csv (one row per spec)
                Specifications/Batches/<batch>/table_<measure>_<panel>.tex, models.txt, specs.csv
        """""
        outdir = self.Directory / 'Specifications' / 'Batches' / batch
        outdir.mkdir(parents=True, exist_ok=True)
        panels = {}
        rows, models, fulls = [], {}, []
        g_sectors = {}
        ivs = tuple(sorted({sp['iv'] for sp in specs if str(sp.get('iv')).replace('_c3', '').replace('_mg', '').replace('_asinh', '').replace('_diag', '').replace('_type', '').replace('_cur', '') in ('gov', 'ks', 'kswt', 'rd', 'rdcf', 'govr', 'govrn', 'ksr', 'ksrn', 'govcl', 'ksagg')}))

        for sp in specs:
            spec = {**self.UDG_BASE, **sp}
            key = (spec['bin_len'], spec['normalise'], spec['shrink'], spec['annual'], spec['iv_wins'], spec['clean'], spec['fixed_w'], spec['fill_missing'], spec['direct_s'])
            if key not in panels:
                panels[key] = self.UDG_Panel(BLS_year_start, Year_end, key[0], key[1], key[2], ivs=ivs,
                                             annual=key[3], iv_wins=key[4], clean=key[5], fixed_w=key[6], fill_missing=key[7], direct_s=key[8])
            reg_df, kappas = panels[key]
            if spec['max_period']:
                reg_df = reg_df[reg_df['period'] <= spec['max_period']]
            if spec['drop_sectors']:   # not estimated, but still partners of the other industries
                reg_df = reg_df[~reg_df['BLS_Industry'].isin(spec['drop_sectors'])]

            sh, meas = spec['share'], spec['measure']
            suf = {'pat': 'count', 'cite': 'cites'}[meas]
            y_col = f'clean_pat_{suf}'
            if sh == 'A':
                y_col = f'asinh_clean_pat_{suf}'
                # restrict to the sectors that enter the G baseline (positive non-clean offset, some clean outcome)
                if meas not in g_sectors:
                    if (5, False, True, False, None, 'all', False, False, False) not in panels:
                        panels[(5, False, True, False, None, 'all', False, False, False)] = self.UDG_Panel(BLS_year_start, Year_end, 5, False, True, ivs=ivs)
                    gb = panels[(5, False, True, False, None, 'all', False, False, False)][0]
                    gb = gb.dropna(subset=[f'clean_pat_{suf}', f'pat_{suf}_nc', f'net_G_{meas}_lag'])
                    gb = gb[gb[f'pat_{suf}_nc'] > 0]
                    tot = gb.groupby('BLS_Industry')[f'clean_pat_{suf}'].sum()
                    g_sectors[meas] = set(tot[tot > 0].index)
                reg_df = reg_df[reg_df['BLS_Industry'].isin(g_sectors[meas])]
                reg_df = reg_df.assign(**{y_col: np.arcsinh(reg_df[f'clean_pat_{suf}'])})
            offset = spec['offset'] or (f'pat_{suf}_nc' if sh == 'G' else f'dirty_pat_{suf}' if sh == 'D' else 'none')
            rhs = spec['rhs'] or sh
            rm = spec['rhs_meas'] or meas
            rk = f'{rhs}_em' if rhs == 'E' else f'{rhs}_{rm}'
            dirs = ['net'] if spec['direction'] == 'net' else ['up', 'down']
            x_cols = [f'{d}_{rk}_lag' for d in dirs]
            for c in spec['controls']:
                x_cols += [f'{d}_{c}_lag' for d in dirs]
            if spec['own_lag']:
                x_cols.append(f'own_{sh}_{meas}_lag')

            if spec['mass_trend']:
                # Unnormalised network sums scale with the weight mass of the sector; with period effects, sectors with a
                # larger mass load more on common time trends. Control for the fixed (first-bin) mass interacted with period.
                if 'mass_i' not in locals():
                    IOm = pd.read_pickle(f'{self.Directory}/Clean Data/IO_Networks.pkl')[BLS_year_start]
                    Sm = np.linalg.inv(np.eye(IOm.shape[0]) - IOm)
                    np.fill_diagonal(Sm, 0.0)
                    mid = np.arange(self.manu_cols[0], self.manu_cols[1] + 1)
                    mass_i = pd.Series((Sm.sum(axis=1) + Sm.sum(axis=0))[mid - 1], index=mid)
                # drop the first period of the *estimation* sample (all periods would be collinear with sector effects)
                need = list(x_cols)
                if spec['iv'] not in (None, 'none'):
                    need += [f"{d}_Z{spec['iv']}_{rm}{'_lag' if l == 1 else f'_lag{l}'}" for l in spec['iv_lags'] for d in dirs]
                pers = sorted(reg_df.dropna(subset=need)['period'].unique())[1:]
                mt = pd.DataFrame({f'mass_x_per{p}': reg_df['BLS_Industry'].map(mass_i) * (reg_df['period'] == p) for p in pers})
                reg_df = pd.concat([reg_df.drop(columns=[c for c in mt.columns if c in reg_df.columns]), mt], axis=1)
                x_cols = x_cols + list(mt.columns)
            iv, fs = spec['iv'], {}
            if sh == 'A':
                m, n_per = self.UDG_OLS(reg_df, y_col, x_cols, spec['entity_fe'], spec['time_fe'])
            elif iv:
                if not spec['entity_fe']:
                    raise ValueError('IV estimator always uses sector effects')
                if iv != 'none':
                    assert rhs == 'G', 'instruments are defined for the clean share G only'
                    endog = x_cols[:len(dirs)]
                    instr = [f"{d}_Z{iv}_{rm}{'_lag' if l == 1 else f'_lag{l}'}" for l in spec['iv_lags'] for d in dirs]
                    exog = [c for c in x_cols if c not in endog]
                else:
                    endog, instr = [], []
                m = es.fit_poisson_iv(reg_df, y_col, offset, x_cols, endog, instr, time_fe=spec['time_fe'])
                dd = reg_df.dropna(subset=[y_col, offset] + x_cols + instr)
                dd = dd[dd[offset] > 0]
                n_per = dd[dd.groupby('BLS_Industry')[y_col].transform('sum') > 0]['period'].nunique()
                if iv != 'none':
                    fs, _ = self.UDG_FirstStage(reg_df, endog, instr, exog, y_col, offset)
                m.fe, m.n_sectors = ['sector'] + (['period'] if spec['time_fe'] else []), m.n_clusters
            else:
                m, n_per = self.UDG_Fit(reg_df, y_col, offset, x_cols, spec['entity_fe'], spec['time_fe'], spec['free_offset'])
            models[spec['id']] = m

            row = {'spec_id': spec['id'], 'batch': batch, 'label': spec.get('label', ''),
                   'outcome': y_col, 'share': sh, 'rhs': rhs, 'rhs_meas': rm, 'controls': '+'.join(spec['controls']),
                   'table': spec['table'], 'panel': spec['panel'], 'short': spec.get('short', str(spec['id']).split('_', 1)[-1]), 'measure': meas, 'bin_len': spec['bin_len'],
                   'weights': ('normalised' if spec['normalise'] else 'non-normalised') + (', fixed first-bin' if spec['fixed_w'] else ''),
                   'shrinkage': 'winsorised' if sh == 'A' else 'kappa' if spec['shrink'] else 'none',
                   'kappa': kappas[(rhs, 'em' if rhs == 'E' else rm)], 'direction': spec['direction'],
                   'own_lag': spec['own_lag'], 'offset': offset,
                   'estimator': 'OLS' if sh == 'A' else 'PPML' if not iv else ('FE-Poisson (GMM)' if iv == 'none' else 'FE-Poisson IV (GMM)'),
                   'instrument': iv or '', 'outcome_freq': 'annual' if spec['annual'] else 'bin', 'clean_def': spec['clean'],
                   'mass_trend': bool(spec['mass_trend']), 'free_offset': bool(spec['free_offset']),
                   'fill_missing': bool(spec['fill_missing']), 'direct_s': bool(spec['direct_s']),
                   'max_period': spec['max_period'] if spec['max_period'] else np.nan,
                   'iv_wins': spec['iv_wins'] if spec['iv_wins'] else np.nan,
                   'iv_lags': '+'.join(str(l) for l in spec['iv_lags']) if iv and iv != 'none' else '',
                   'hansen_p': getattr(m, 'J_p', np.nan) if iv else np.nan,
                   'fe': '+'.join(m.fe) if m.fe else 'none', 'cluster': 'sector',
                   'N': m.nobs, 'n_sectors': m.n_sectors, 'n_periods': n_per,
                   'pseudo_r2': m.rsquared,
                   'converged': m.converged if iv and sh != 'A' else getattr(m.glm, 'converged', np.nan),
                   'run_date': datetime.now().strftime('%Y-%m-%d')}
            for slot, var in [('net', f'net_{rk}_lag'), ('up', f'up_{rk}_lag'),
                              ('down', f'down_{rk}_lag'), ('lag', f'own_{sh}_{meas}_lag')]:
                if var in m.params.index:
                    row[f'coef_{slot}'], row[f'se_{slot}'], row[f'p_{slot}'] = m.params[var], m.bse[var], m.pvalues[var]
                else:
                    row[f'coef_{slot}'] = row[f'se_{slot}'] = row[f'p_{slot}'] = np.nan
            for slot, r in fs.items():
                for k, v in r.items():
                    row[f'fs_{k}_{slot}'] = v
            row['coef_off'], row['se_off'], row['p_off_eq1'] = (1.0, np.nan, np.nan)
            if spec['free_offset']:
                row['coef_off'], row['se_off'] = float(m.params['ln_offset']), float(m.bse['ln_offset'])
                row['p_off_eq1'] = m.test('ln_offset = 1')[1]
            if spec['direction'] != 'net':
                if iv:
                    row['p_up_plus_down'] = m.test(x_cols[0], x_cols[1], 'sum')[3]
                    row['p_up_eq_down'] = m.test(x_cols[0], x_cols[1], 'diff')[3]
                else:
                    _, row['p_up_plus_down'] = m.test(f'{x_cols[0]} + {x_cols[1]} = 0')
                    _, row['p_up_eq_down'] = m.test(f'{x_cols[0]} = {x_cols[1]}')
            row['notes'] = spec.get('notes', '')
            rows.append(row)
            print(f'\n{"="*78}\n{spec["id"]}  {spec.get("label", "")}\n{"="*78}\n{m!r}')

        # -------- #
        # Registry #
        # -------- #
        new = pd.DataFrame(rows)
        reg_path = self.Directory / 'Specifications' / 'registry.csv'
        if reg_path.exists():
            old = pd.read_csv(reg_path)
            old = old[old['batch'] != batch]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(reg_path, index=False)
        pd.DataFrame(rows).to_csv(outdir / 'specs.csv', index=False)
        bad = [r['spec_id'] for r in rows if r['converged'] is False or r['converged'] == False]
        with open(outdir / 'nonconverged.tex', 'w') as f:
            f.write('All specifications in this batch converged.' if not bad else
                    'The following specifications did \\emph{not} converge, and their estimates should not be interpreted: '
                    + ', '.join(r'\texttt{' + i.replace('_', r'\_') + '}' for i in bad) + '.')
        with open(outdir / 'models.txt', 'w') as f:
            for k, m in models.items():
                f.write(f'\n{"="*78}\n{k}\n{"="*78}\n{m!r}\n')

        # ----- #
        # Table #
        # ----- #
        out = pd.DataFrame(rows)
        main = out[out['table'] == 'main']
        for meas in main['measure'].unique():
            mm = main[main['measure'] == meas]
            base = mm[mm['spec_id'].str.endswith('_base')]
            for p in sorted(mm['panel'].unique()):
                sub = mm[mm['panel'] == p]
                sub = pd.concat([base[~base['spec_id'].isin(sub['spec_id'])], sub])
                self.UDG_Table(sub, outdir / f'table_{meas}_{p}.tex')
        for name in [t for t in out['table'].unique() if t != 'main']:
            ids = out.loc[out['table'] == name, 'spec_id']
            self.UDG_ModelTable({i: models[i] for i in ids}, out.set_index('spec_id'), outdir / f'table_{name}.tex')
        return models


    def Firm_Entry(self, first_year=1987):
        """""
        Extensive-margin adoption from firm patenting: a firm "adopts" clean technology when it files its first clean patent.
        For each industry and year: entry = number of firms whose first clean patent is filed that year (split over the
        industries of those patents); at risk = number of firms active in the industry (a patent there up to that year,
        split equally over their industries) that have not filed a clean patent before the year. Firms whose first clean
        patent predates the sample are treated as already adopted.

        Output: Clean Data/Ind_Pat_Entry.pkl in the layout of Ind_Pat.pkl, with clean_pat_count = entry and
        pat_count = firm-years at risk, so that G = entry / at risk is an adoption hazard (UDG_Panel clean='entry').
        """""
        pf = pd.read_pickle(f'{self.Directory}/Clean Data/Pat_Firms.pkl')[['patent_id', 'year', 'gvkey', 'BLS_Industry', 'clean']]
        pf = pf.drop_duplicates(['patent_id', 'gvkey', 'BLS_Industry'])
        first_clean = pf.loc[pf['clean'] == 1].groupby('gvkey')['year'].min()
        # entry: share of the firm's first-year clean patents in each industry
        c = pf[(pf['clean'] == 1) & (pf['year'] == pf['gvkey'].map(first_clean))]
        w = c.groupby(['gvkey', 'BLS_Industry']).size().rename('n').reset_index()
        w['entry'] = w['n'] / w.groupby('gvkey')['n'].transform('sum')
        w['year'] = w['gvkey'].map(first_clean)
        entry = w.groupby(['BLS_Industry', 'year'])['entry'].sum()
        # at risk: firm-industry pairs with a first patent in the industry up to the year, firm not adopted before the year
        fs = pf.groupby(['gvkey', 'BLS_Industry'])['year'].min().rename('fs_year').reset_index()
        fs['fc'] = fs['gvkey'].map(first_clean).fillna(9999)
        years = range(first_year, int(pf['year'].max()) + 1)
        rows = []
        for y in years:
            a = fs[(fs['fs_year'] <= y) & (fs['fc'] >= y)]
            n_ind = fs[fs['fs_year'] <= y].groupby('gvkey')['BLS_Industry'].transform('count')   # industries of the firm so far
            wt = 1.0 / n_ind.reindex(a.index)
            r = (a.assign(w=wt).groupby('BLS_Industry')['w'].sum().rename('at_risk').reset_index())
            r['year'] = y
            rows.append(r)
        risk = pd.concat(rows, ignore_index=True)
        ind = sorted(risk['BLS_Industry'].unique())
        idx = pd.MultiIndex.from_product([ind, list(years)], names=['BLS_Industry', 'year'])
        out = (risk.set_index(['BLS_Industry', 'year'])[['at_risk']].reindex(idx).fillna(0.0)
                   .join(entry.rename('entry'), how='left').fillna({'entry': 0.0}).reset_index())
        out['entry'] = out['entry'].clip(upper=out['at_risk'].clip(lower=0))   # adopters not active before are counted as at risk
        res = pd.DataFrame({'BLS_Industry': out['BLS_Industry'], 'year': out['year'],
                            'clean_pat_count': out['entry'], 'dirty_pat_count': 0.0, 'pat_count': out['at_risk'],
                            'clean_pat_cites': out['entry'], 'dirty_pat_cites': 0.0, 'pat_cites': out['at_risk']})
        res['pat_count_nc'] = res['pat_count'] - res['clean_pat_count']
        res['pat_cites_nc'] = res['pat_count_nc']
        res['clim_pat_count'] = res['clean_pat_count']
        res['clim_pat_cites'] = res['clean_pat_cites']
        res = res[res['BLS_Industry'] != 71]
        res.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat_Entry.pkl')
        return res


    def IO_Validation(self, batch, BLS_year_start, Year_end, bin_len=5, dim=3):
        """""
        Does clean patenting predict changes in the supply chain? Following the figures of IO_Change: the outcome is the
        total-variation distance between an industry's input-requirement row at the start and end of a bin, in the
        reduced form (fossil-fuel columns dropped and rows renormalised) and from the Leontief inverse. Regressors are
        the industry's clean patenting in the same bin or the bin before (asinh of clean patents; clean share; first-clean
        entry). Columns: OLS and WLS (weights = emissions^(1/dim) in 2012) with period effects, and WLS with sector and
        period effects; standard errors clustered by industry.

        Output: Specifications/Batches/<batch>/io_validation.csv, table_io_validation.tex
        """""
        outdir = self.Directory / 'Specifications' / 'Batches' / batch
        outdir.mkdir(parents=True, exist_ok=True)
        D = f'{self.Directory}/Clean Data/'
        IO = pd.read_pickle(D + 'IO_Networks.pkl')
        pat = pd.read_pickle(D + 'Ind_Pat.pkl')
        co2 = pd.read_pickle(D + 'Ind_CO2.pkl')
        ent_path = Path(D + 'Ind_Pat_Entry.pkl')
        ent = pd.read_pickle(ent_path) if ent_path.exists() else None
        manu = np.arange(self.manu_cols[0] - 1, self.manu_cols[1])
        J = IO[BLS_year_start].shape[0]

        def reduced(M):
            Mm = M[manu, :]
            r = np.delete(Mm, self.fossil_cols, axis=1)
            return r * Mm.sum(axis=1, keepdims=True) / r.sum(axis=1, keepdims=True)

        def leontief(M):
            return np.linalg.inv(np.eye(J) - M)[manu, :]

        def tv(A, B):
            return 0.5 * np.abs(B - A).sum(axis=1)

        ends = [e for e in range(BLS_year_start + bin_len, Year_end + 1, bin_len)]
        def window(df, col, e):
            w = df[(df['year'] > e - bin_len) & (df['year'] <= e)]
            return w.groupby('BLS_Industry')[col].sum()

        w12 = co2[co2['year'] == 2012].set_index('BLS_Industry')['CO2e_Industry'] ** (1 / dim)
        rows = []
        for e in ends:
            A, B = IO[e - bin_len], IO[e]
            d = pd.DataFrame({'BLS_Industry': manu + 1, 'period': e,
                              'tv_red': tv(reduced(A), reduced(B)), 'tv_li': tv(leontief(A), leontief(B))})
            for lab, ee in [('', e), ('_lag', e - bin_len)]:
                cl, tot = window(pat, 'clean_pat_count', ee), window(pat, 'pat_count', ee)
                d['ln_clean' + lab] = np.arcsinh(d['BLS_Industry'].map(cl).fillna(0.0))
                d['G' + lab] = (d['BLS_Industry'].map(cl) / d['BLS_Industry'].map(tot).where(d['BLS_Industry'].map(tot) > 0))
                if ent is not None:
                    en = window(ent, 'clean_pat_count', ee)
                    d['ln_entry' + lab] = np.arcsinh(d['BLS_Industry'].map(en).fillna(0.0))
            d['w'] = d['BLS_Industry'].map(w12)
            rows.append(d)
        df = pd.concat(rows, ignore_index=True)
        df = df[df['w'].notna() & df['BLS_Industry'].isin(pat['BLS_Industry'].unique())]

        def fit(d, x, y, weighted, sector_fe):
            d = d.dropna(subset=[x, y])
            parts = [pd.Series(1.0, index=d.index, name='const'), d[[x]].astype(float),
                     pd.get_dummies(d['period'], prefix='p', drop_first=True, dtype=float)]
            if sector_fe:
                parts.append(pd.get_dummies(d['BLS_Industry'], prefix='s', drop_first=True, dtype=float))
            X = pd.concat(parts, axis=1)
            X.columns = [str(c) for c in X.columns]
            kw = dict(cov_type='cluster', cov_kwds={'groups': d['BLS_Industry'].to_numpy()})
            m = (sm.WLS(d[y], X, weights=d['w']) if weighted else sm.OLS(d[y], X)).fit(**kw)
            return m.params[x], m.bse[x], m.pvalues[x], len(d), d['BLS_Industry'].nunique(), m.rsquared

        regs = [('ln_clean', 'asinh(clean patents)'), ('G', 'Clean share $G$')]
        if ent is not None:
            regs.append(('ln_entry', 'asinh(first clean patents)'))
        specs = [('OLS, period FE', False, False), ('WLS, period FE', True, False), ('WLS, sector + period FE', True, True)]
        out = []
        for y, ylab in [('tv_red', 'Reduced form (fossil columns dropped)'), ('tv_li', 'Leontief inverse')]:
            for x, xlab in regs:
                for timing in ['', '_lag']:
                    for slab, wt, sfe in specs:
                        b, se, p, n, ns, r2 = fit(df, x + timing, y, wt, sfe)
                        out.append(dict(outcome=y, regressor=x, timing='same bin' if timing == '' else 'previous bin',
                                        spec=slab, coef=b, se=se, p=p, N=n, n_sectors=ns, r2=r2))
        out = pd.DataFrame(out)
        out.to_csv(outdir / 'io_validation.csv', index=False)
        cols = [(t, s_) for t in ['same bin', 'previous bin'] for s_, _, _ in specs]
        lines = [r'\begin{tabular}{l' + 'c' * len(cols) + '}', r'\toprule',
                 r' & \multicolumn{3}{c}{Same bin} & \multicolumn{3}{c}{Previous bin} \\',
                 r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
                 ' & ' + ' & '.join(['OLS', 'WLS', 'WLS, FE'] * 2) + r' \\', r'\midrule']
        for y, ylab in [('tv_red', 'Reduced form (fossil columns dropped)'), ('tv_li', 'Leontief inverse')]:
            lines.append(r'\multicolumn{7}{l}{\textit{Outcome: TV distance, ' + ylab + r'}} \\')
            for x, xlab in regs:
                sub = out[(out['outcome'] == y) & (out['regressor'] == x)]
                cs, ss = [], []
                for t, sl in cols:
                    r = sub[(sub['timing'] == t) & (sub['spec'] == [sp[0] for sp in specs][[sp[0] for sp in specs].index(sl)])].iloc[0]
                    cs.append(f"{r.coef:.3f}{gpf.get_stars(r.p)}"); ss.append(f"({r.se:.3f})")
                lines += [xlab + ' & ' + ' & '.join(cs) + r' \\', ' & ' + ' & '.join(ss) + r' \\']
            lines.append(r'\midrule')
        n0 = out.iloc[0]
        lines += ['Observations (first column) & ' + str(int(n0.N)) + ' & & & & & ' + r' \\', r'\bottomrule', r'\end{tabular}']
        with open(outdir / 'table_io_validation.tex', 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(out.round(3).to_string())
        return out


    def Old_Benchmark(self, batch, BLS_year_start, Year_start, Year_mid, Year_end, bin_len=5, dim=3, lower=0.1):
        """""
        Benchmark: the network regressions of the previously submitted version of the paper (branch Old, Up_Down_Green:
        Network_Regressions_Net / _Net_WLS / _Lagged / _UpDown), run on the current Clean Data. Outcomes: reduction in
        log emissions intensity, clean patent share, clean citation share, on the network of the partners' emission
        reduction, clean patent share or clean citation share. OLS (WLS: weights^(1/dim)) with bin effects, standard errors
        clustered by industry, estimation samples winsorised at the 10th and 90th percentiles, shares > 0. The network is
        the old one: the manufacturing block of the input-output matrix (diagonal zeroed, rows divided by the full-economy
        row totals), its Leontief inverse minus the identity for upstream, the transpose for downstream; the shares of
        industries without data are set to zero; the emission regressor uses the positive part of the reduction.

        Output: Specifications/Batches/<batch>/old_benchmark.csv, table_old_{net,wls,lag,updown}.tex
        """""
        outdir = self.Directory / 'Specifications' / 'Batches' / batch
        outdir.mkdir(parents=True, exist_ok=True)
        D = f'{self.Directory}/Clean Data/'
        IO = pd.read_pickle(D + 'IO_Networks.pkl')
        co2 = pd.read_pickle(D + 'Ind_CO2.pkl')
        co2_full = pd.read_pickle(D + 'Ind_CO2_full.pkl')
        ann = pd.read_pickle(D + 'Ind_Pat.pkl')
        bin_ends = list(range(BLS_year_start, Year_end + 1, bin_len))
        manu_slice = slice(self.manu_cols[0] - 1, self.manu_cols[1])
        manu_idx_all = np.arange(self.manu_cols[0], self.manu_cols[1] + 1)
        M = len(manu_idx_all)

        manu_IO = {}
        for y in bin_ends:
            A = IO[y].copy()
            np.fill_diagonal(A, 0.0)
            tot = A.sum(axis=1)
            denom = tot[manu_slice, np.newaxis]
            manu_IO[y] = A[manu_slice, manu_slice] / np.where(denom == 0, 1, denom)

        def network(IO_manu, z_wide, idx):
            in_manu = np.isin(idx, manu_idx_all)
            emb = np.searchsorted(manu_idx_all, idx[in_manu])
            LI = np.linalg.inv(np.eye(M) - IO_manu)
            z = np.zeros(M)
            z[emb] = np.nan_to_num(z_wide)
            return pd.DataFrame({'BLS_Industry': idx[in_manu], 'up': ((LI - np.eye(M)) @ z)[emb],
                                 'down': ((LI - np.eye(M)).T @ z)[emb]})

        # emissions
        def em_panel(df):
            w = df.pivot(index='BLS_Industry', columns='year', values=['CO2e_intensity_Industry', 'CO2e_Industry']).dropna()
            idx = w.index.to_numpy(dtype=int)
            in_manu = np.isin(idx, manu_idx_all)
            parts = []
            for (a, b), ye in [((Year_start, Year_mid), Year_mid), ((Year_mid, Year_end), Year_end)]:
                dlog = -(np.log(w['CO2e_intensity_Industry'][b].to_numpy()) - np.log(w['CO2e_intensity_Industry'][a].to_numpy()))
                base = pd.DataFrame({'BLS_Industry': idx, 'period': ye, 'dlog_CO2e_inten': dlog,
                                     'CO2e_Industry': w['CO2e_Industry'][a].to_numpy()})
                net = network(manu_IO[ye], np.maximum(dlog[in_manu], 0), idx).rename(columns={'up': 'up_dlog_em', 'down': 'down_dlog_em'})
                parts.append(base.merge(net, on='BLS_Industry', how='left'))
            return pd.concat(parts, ignore_index=True)

        em_df = em_panel(co2)
        wide_full = co2_full.pivot(index='BLS_Industry', columns='year', values='CO2e_intensity_Industry').dropna()
        idx_full = wide_full.index.to_numpy(dtype=int)

        # patents
        frames = []
        for y in bin_ends:
            w = ann[(ann['year'] > y - bin_len) & (ann['year'] <= y)]
            g = w.groupby('BLS_Industry')[['clean_pat_count', 'pat_count', 'clean_pat_cites', 'pat_cites']].sum().reindex(wide_full.index)
            g['clean_pat_share'] = g['clean_pat_count'] / g['pat_count'].where(g['pat_count'] > 0)
            g['clean_cite_share'] = g['clean_pat_cites'] / g['pat_cites'].where(g['pat_cites'] > 0)
            out = g.reset_index()
            out['period'] = y
            in_manu = np.isin(idx_full, manu_idx_all)
            for col, nm in [('clean_pat_share', 'pat_count'), ('clean_cite_share', 'pat_cite')]:
                net = network(manu_IO[y], g[col].to_numpy()[in_manu], idx_full).rename(columns={'up': f'up_{nm}', 'down': f'down_{nm}'})
                out = out.merge(net, on='BLS_Industry', how='left')
            frames.append(out)
        pat = pd.concat(frames, ignore_index=True)
        reg = pat.merge(em_df, on=['BLS_Industry', 'period'], how='left')
        reg = reg[reg['BLS_Industry'].isin(manu_idx_all)].sort_values(['BLS_Industry', 'period']).copy()
        for k in ['dlog_em', 'pat_count', 'pat_cite']:
            reg[f'net_{k}'] = reg[f'up_{k}'] + reg[f'down_{k}']
        for c in ['net_dlog_em', 'net_pat_count', 'net_pat_cite', 'up_dlog_em', 'down_dlog_em', 'up_pat_count',
                  'down_pat_count', 'up_pat_cite', 'down_pat_cite']:
            reg[c + '_lag'] = reg.groupby('BLS_Industry')[c].shift(1)

        def wins(d, cols):
            d = d.copy()
            for c in cols:
                lo, hi = d[c].quantile(lower), d[c].quantile(1 - lower)
                d[c] = d[c].clip(lower=lo, upper=hi)
            return d

        def fit(df, y, xs, wcol=None, sub=None):
            d = sub if sub is not None else df
            fe = pd.get_dummies(d['period'], drop_first=True, dtype=float)
            X = sm.add_constant(pd.concat([d[xs], fe], axis=1))
            kw = dict(cov_type='cluster', cov_kwds={'groups': d['BLS_Industry']})
            return (sm.OLS(d[y], X) if wcol is None else sm.WLS(d[y], X, d[wcol] ** (1 / dim))).fit(**kw)

        allc = ['up_dlog_em', 'down_dlog_em', 'net_dlog_em', 'up_pat_count', 'down_pat_count', 'net_pat_count',
                'up_pat_cite', 'down_pat_cite', 'net_pat_cite']
        reg_em = wins(reg.dropna(subset=['dlog_CO2e_inten']), ['dlog_CO2e_inten'] + allc)
        reg_cnt = wins(reg[reg['clean_pat_share'] > 0], ['clean_pat_share', 'up_dlog_em', 'down_dlog_em', 'net_dlog_em',
                                                         'up_pat_count', 'down_pat_count', 'net_pat_count'])
        reg_cit = wins(reg[reg['clean_cite_share'] > 0], ['clean_cite_share', 'up_dlog_em', 'down_dlog_em', 'net_dlog_em',
                                                          'up_pat_cite', 'down_pat_cite', 'net_pat_cite'])
        reg_cnt_em = reg_cnt.dropna(subset=['up_dlog_em', 'down_dlog_em'])
        reg_cit_em = reg_cit.dropna(subset=['up_dlog_em', 'down_dlog_em'])
        lagc = ['net_dlog_em_lag', 'net_pat_count_lag', 'net_pat_cite_lag', 'up_dlog_em_lag', 'down_dlog_em_lag',
                'up_pat_count_lag', 'down_pat_count_lag', 'up_pat_cite_lag', 'down_pat_cite_lag']
        reg_em_lag = wins(reg.dropna(subset=['dlog_CO2e_inten', 'net_pat_count_lag', 'up_pat_count_lag', 'down_pat_count_lag',
                                             'net_pat_cite_lag', 'up_pat_cite_lag', 'down_pat_cite_lag']), ['dlog_CO2e_inten'] + lagc)
        reg_cnt_lag = wins(reg[reg['clean_pat_share'] > 0].dropna(subset=['net_pat_count_lag', 'up_pat_count_lag', 'down_pat_count_lag']),
                           ['clean_pat_share'] + lagc)
        reg_cit_lag = wins(reg[reg['clean_cite_share'] > 0].dropna(subset=['net_pat_cite_lag', 'up_pat_cite_lag', 'down_pat_cite_lag']),
                           ['clean_cite_share'] + lagc)
        reg_em_em_lag = wins(reg.dropna(subset=['dlog_CO2e_inten', 'net_dlog_em_lag', 'up_dlog_em_lag', 'down_dlog_em_lag']),
                             ['dlog_CO2e_inten'] + lagc)
        reg_cnt_em_lag = wins(reg[reg['clean_pat_share'] > 0].dropna(subset=['net_dlog_em_lag', 'up_dlog_em_lag', 'down_dlog_em_lag']),
                              ['clean_pat_share'] + lagc)
        reg_cit_em_lag = wins(reg[reg['clean_cite_share'] > 0].dropna(subset=['net_dlog_em_lag', 'up_dlog_em_lag', 'down_dlog_em_lag']),
                              ['clean_cite_share'] + lagc)

        # seven columns: emissions reduction on (em, patents, citations); patent share on (em, patents); citation share on (em, citations)
        def cols_for(kind):
            if kind == 'net':
                return [(reg_em, 'dlog_CO2e_inten', ['net_dlog_em'], None, None), (reg_em, 'dlog_CO2e_inten', ['net_pat_count'], None, None),
                        (reg_em, 'dlog_CO2e_inten', ['net_pat_cite'], None, None),
                        (reg_cnt, 'clean_pat_share', ['net_dlog_em'], None, reg_cnt_em), (reg_cnt, 'clean_pat_share', ['net_pat_count'], None, None),
                        (reg_cit, 'clean_cite_share', ['net_dlog_em'], None, reg_cit_em), (reg_cit, 'clean_cite_share', ['net_pat_cite'], None, None)]
            if kind == 'wls':
                return [(reg_em, 'dlog_CO2e_inten', ['net_dlog_em'], 'CO2e_Industry', None), (reg_em, 'dlog_CO2e_inten', ['net_pat_count'], 'CO2e_Industry', None),
                        (reg_em, 'dlog_CO2e_inten', ['net_pat_cite'], 'CO2e_Industry', None),
                        (reg_cnt, 'clean_pat_share', ['net_dlog_em'], 'clean_pat_count', reg_cnt_em), (reg_cnt, 'clean_pat_share', ['net_pat_count'], 'clean_pat_count', None),
                        (reg_cit, 'clean_cite_share', ['net_dlog_em'], 'clean_pat_cites', reg_cit_em), (reg_cit, 'clean_cite_share', ['net_pat_cite'], 'clean_pat_cites', None)]
            if kind == 'lag':
                return [(reg_em_em_lag, 'dlog_CO2e_inten', ['net_dlog_em_lag'], None, None), (reg_em_lag, 'dlog_CO2e_inten', ['net_pat_count_lag'], None, None),
                        (reg_em_lag, 'dlog_CO2e_inten', ['net_pat_cite_lag'], None, None),
                        (reg_cnt_lag, 'clean_pat_share', ['net_dlog_em_lag'], None, reg_cnt_em_lag), (reg_cnt_lag, 'clean_pat_share', ['net_pat_count_lag'], None, None),
                        (reg_cit_lag, 'clean_cite_share', ['net_dlog_em_lag'], None, reg_cit_em_lag), (reg_cit_lag, 'clean_cite_share', ['net_pat_cite_lag'], None, None)]
            return [(reg_em, 'dlog_CO2e_inten', ['up_dlog_em', 'down_dlog_em'], None, None), (reg_em, 'dlog_CO2e_inten', ['up_pat_count', 'down_pat_count'], None, None),
                    (reg_em, 'dlog_CO2e_inten', ['up_pat_cite', 'down_pat_cite'], None, None),
                    (reg_cnt, 'clean_pat_share', ['up_dlog_em', 'down_dlog_em'], None, reg_cnt_em), (reg_cnt, 'clean_pat_share', ['up_pat_count', 'down_pat_count'], None, None),
                    (reg_cit, 'clean_cite_share', ['up_dlog_em', 'down_dlog_em'], None, reg_cit_em), (reg_cit, 'clean_cite_share', ['up_pat_cite', 'down_pat_cite'], None, None)]

        row_defs = {
            'net': [('net_dlog_em', 'Network log emissions-intensity reduction'), ('net_pat_count', 'Network clean patent share'),
                    ('net_pat_cite', 'Network clean citation share')],
            'lag': [('net_dlog_em_lag', 'Network log emissions-intensity reduction (lag)'), ('net_pat_count_lag', 'Network clean patent share (lag)'),
                    ('net_pat_cite_lag', 'Network clean citation share (lag)')],
            'updown': [('up_dlog_em', 'Upstream log emissions-intensity reduction'), ('down_dlog_em', 'Downstream log emissions-intensity reduction'),
                       ('up_pat_count', 'Upstream clean patent share'), ('down_pat_count', 'Downstream clean patent share'),
                       ('up_pat_cite', 'Upstream clean citation share'), ('down_pat_cite', 'Downstream clean citation share')]}
        row_defs['wls'] = row_defs['net']
        all_rows = []
        for kind in ['net', 'wls', 'lag', 'updown']:
            models = []
            for d, y, xs, w, sub in cols_for(kind):
                m = fit(d, y, xs, w, sub)
                models.append(m)
                for x in xs:
                    all_rows.append(dict(table=kind, outcome=y, term=x, coef=m.params[x], se=m.bse[x], p=m.pvalues[x],
                                         N=int(m.nobs), r2=m.rsquared))
            lines = [r'\begin{tabular}{lccccccc}', r'\toprule',
                     r' & \multicolumn{3}{c}{Emissions-intensity reduction} & \multicolumn{2}{c}{Clean patent share} & \multicolumn{2}{c}{Clean citation share} \\',
                     r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
                     ' & ' + ' & '.join(f'({i})' for i in range(1, 8)) + r' \\', r'\midrule']
            for var, lab in row_defs[kind]:
                cs, ss = [], []
                for m in models:
                    if var in m.params.index:
                        cs.append(f"{m.params[var]:.3f}{gpf.get_stars(m.pvalues[var])}"); ss.append(f"({m.bse[var]:.3f})")
                    else:
                        cs.append(''); ss.append('')
                lines += [lab + ' & ' + ' & '.join(cs) + r' \\', ' & ' + ' & '.join(ss) + r' \\']
            lines += [r'\midrule', '$R^2$ & ' + ' & '.join(f'{m.rsquared:.3f}' for m in models) + r' \\',
                      'Observations & ' + ' & '.join(str(int(m.nobs)) for m in models) + r' \\', r'\bottomrule', r'\end{tabular}']
            with open(outdir / f'table_old_{kind}.tex', 'w') as f:
                f.write('\n'.join(lines) + '\n')
        out = pd.DataFrame(all_rows)
        out.to_csv(outdir / 'old_benchmark.csv', index=False)
        print(out.round(3).to_string())
        return out


    def Cap_Citations(self, q=0.99):
        """""
        Industry-year patent table with citation weights capped at their q-quantile (unweighted, over all patent-industry
        records), rebuilt from Pat_Firms.pkl with the construction of the Cleaner (patents split equally over their
        industries). Counts are unchanged. A coauthor suggestion (see the note in the repository discussion).

        Output: Clean Data/Ind_Pat_Cap.pkl in the layout of Ind_Pat.pkl (used by UDG_Panel with clean='cap')
        """""
        D = f'{self.Directory}/Clean Data/'
        pf = pd.read_pickle(D + 'Pat_Firms.pkl')[['patent_id', 'year', 'BLS_Industry', 'clean', 'dirty', 'norm_cites']].drop_duplicates()
        pf['w'] = 1.0 / pf.groupby('patent_id')['BLS_Industry'].transform('count')
        cap = pf['norm_cites'].quantile(q)
        pf['nc'] = pf['norm_cites'].clip(upper=cap)
        pf['clean_w'], pf['dirty_w'] = pf['w'] * pf['clean'], pf['w'] * pf['dirty']
        pf['cite_w'], pf['clean_cite_w'], pf['dirty_cite_w'] = pf['w'] * pf['nc'], pf['clean_w'] * pf['nc'], pf['dirty_w'] * pf['nc']
        g = pf.groupby(['BLS_Industry', 'year'], as_index=False).agg(
            clean_pat_count=('clean_w', 'sum'), dirty_pat_count=('dirty_w', 'sum'), pat_count=('w', 'sum'),
            clean_pat_cites=('clean_cite_w', 'sum'), dirty_pat_cites=('dirty_cite_w', 'sum'), pat_cites=('cite_w', 'sum'))
        ref = pd.read_pickle(D + 'Ind_Pat.pkl')
        idx = pd.MultiIndex.from_frame(ref[['BLS_Industry', 'year']])
        g = g.set_index(['BLS_Industry', 'year']).reindex(idx).fillna(0.0).reset_index()
        g['pat_count_nc'] = g['pat_count'] - g['clean_pat_count']
        g['pat_cites_nc'] = g['pat_cites'] - g['clean_pat_cites']
        g['clim_pat_count'] = g['clean_pat_count'] + g['dirty_pat_count']
        g['clim_pat_cites'] = g['clean_pat_cites'] + g['dirty_pat_cites']
        g.to_pickle(D + 'Ind_Pat_Cap.pkl')
        chk = ref.merge(g, on=['BLS_Industry', 'year'], suffixes=('', '_c'))
        print(f'cap = {cap:.2f}; max |count difference| vs Ind_Pat: {(chk.clean_pat_count - chk.clean_pat_count_c).abs().max():.2e}')
        return g


    def Match_Coverage(self, batch, BLS_year_start, Year_end, bin_len=5):
        """""
        Share of US patents (any assignee) that are matched to a Compustat firm and an input-output industry, by filing-year
        window, for all patents and for clean patents. All patents = the sum of the saved patent counts by technology and
        year (each patent has total weight one), clean patents = the clean technologies of the clean / general types.

        Output: Specifications/Batches/<batch>/match_coverage.csv, table_match_coverage.tex; Results/Tables/Match_Coverage.tex (appendix exhibit)
        """""
        outdir = self.Directory / 'Specifications' / 'Batches' / batch
        outdir.mkdir(parents=True, exist_ok=True)
        D = f'{self.Directory}/Clean Data/'
        tp = pd.read_pickle(D + 'Tech_Pat_cg.pkl')
        tot = tp.groupby('year')['pat_count'].sum()
        clean_tot = tp[tp['tech'].str.endswith('_c')].groupby('year')['pat_count'].sum()
        pf = pd.read_pickle(D + 'Pat_Firms.pkl')[['patent_id', 'year', 'clean']].drop_duplicates('patent_id')
        m, mc = pf.groupby('year').size(), pf[pf['clean'] == 1].groupby('year').size()
        df = pd.DataFrame({'all': tot, 'matched': m, 'clean_all': clean_tot, 'clean_matched': mc}).fillna(0.0)
        wins = [(e - bin_len + 1, e) for e in range(BLS_year_start, Year_end + 1, bin_len)]
        wins += [(BLS_year_start - bin_len + 1, Year_end)]
        rows = []
        for lo, hi in wins:
            d = df.loc[lo:hi].sum()
            rows.append(dict(window=f'{lo}--{hi}', all=d['all'], matched=d['matched'], share=d['matched'] / d['all'],
                             clean_all=d['clean_all'], clean_matched=d['clean_matched'], clean_share=d['clean_matched'] / d['clean_all']))
        out = pd.DataFrame(rows)
        out.to_csv(outdir / 'match_coverage.csv', index=False)
        lines = [r'\begin{tabular}{lrrcrrc}', r'\toprule',
                 r' & \multicolumn{3}{c}{All patents} & \multicolumn{3}{c}{Clean patents} \\',
                 r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
                 r'Filing years & Filed & Matched & Share & Filed & Matched & Share \\', r'\midrule']
        for k, r in out.iterrows():
            if k == len(out) - 1:
                lines.append(r'\midrule')
            lines.append(f"{r['window']} & {r['all']:,.0f} & {r['matched']:,.0f} & {r['share']:.3f} & {r['clean_all']:,.0f} & "
                         f"{r['clean_matched']:,.0f} & {r['clean_share']:.3f} \\\\")
        lines += [r'\bottomrule', r'\end{tabular}']
        for path in (outdir / 'table_match_coverage.tex', self.Directory / 'Results' / 'Tables' / 'Match_Coverage.tex'):
            with open(path, 'w') as f:
                f.write('\n'.join(lines) + '\n')
        print(out.round(3).to_string())
        return out


    def Theory_Check(self, alpha=0.10, max_codes=5):
        """""
        Which specifications are consistent with the theory: strategic complementarity in clean adoption, upstream and downstream
        (network coefficients positive and significant), stronger downstream (downstream coefficient larger than upstream).
        Criteria, on converged specifications whose regressor is the network clean share (G, D or the asinh level):
          net specifications: coefficient > 0 and p < alpha;
          up/down specifications: both coefficients > 0 and p < alpha, and downstream > upstream ('confirms' = all three).
        For instrumented specifications the count of those with first-stage F >= 10 is also given.
        Output: Specifications/Batches/<batch>/check_<measure>_<panel>.tex (one boxed line for each table), Specifications/Batches/diag_old_benchmark/
                check_old_*.tex, Specifications/Summary/table_scorecard.tex (one row for each batch and measure).
        """""
        R = self.Directory / 'Specifications'
        reg = pd.read_csv(R / 'registry.csv')
        reg = reg[(reg['table'] == 'main') & reg['rhs'].isin(['G', 'D', 'A']) & (reg['converged'].fillna(1) == 1)].copy()

        def stats(d):
            net = d[d['direction'] == 'net']
            ud = d[d['direction'] == 'updown']
            n_ok = net[(net['coef_net'] > 0) & (net['p_net'] < alpha)]
            both = ud[(ud['coef_up'] > 0) & (ud['p_up'] < alpha) & (ud['coef_down'] > 0) & (ud['p_down'] < alpha)]
            larger = ud[ud['coef_down'] > ud['coef_up']]
            larger_sig = larger[larger['p_up_eq_down'] < alpha]
            conf = both[both['coef_down'] > both['coef_up']]
            strong_net = n_ok[n_ok['fs_F_net'] >= 10] if 'fs_F_net' in n_ok else n_ok.iloc[0:0]
            return dict(n_net=len(net), net_ok=n_ok, n_ud=len(ud), both=both, larger=larger, larger_sig=larger_sig, conf=conf,
                        strong_net=strong_net, iv=net['instrument'].notna().any() or ud['instrument'].notna().any())

        def codes(df):
            c = [f'\\texttt{{{s}}}'.replace('_', '\\_') for s in df['short']]
            if not c:
                return 'none'
            return ', '.join(c[:max_codes]) + (f' and {len(c) - max_codes} more' if len(c) > max_codes else '')

        def line(s):
            parts = []
            if s['n_net']:
                parts.append(f"net: {len(s['net_ok'])}/{s['n_net']} positive and significant" + (f" ({len(s['strong_net'])} with first-stage $F\\geq10$)" if s['iv'] else '')
                             + (f" ({codes(s['net_ok'])})" if 0 < len(s['net_ok']) <= max_codes else ''))
            if s['n_ud']:
                parts.append(f"up/down: both positive and significant {len(s['both'])}/{s['n_ud']}; downstream larger {len(s['larger'])}/{s['n_ud']}"
                             f" (significantly {len(s['larger_sig'])})")
                parts.append(f"all three: {codes(s['conf'])}")
            return '; '.join(parts)

        def box(text):
            return ('\\fcolorbox{black!35}{black!4}{\\parbox{\\dimexpr0.97\\linewidth-2\\fboxsep-2\\fboxrule\\relax}{\\footnotesize\\textbf{Theory check} '
                    '(positive and significant at 10\\%, downstream larger). ' + text + '.}}\n')

        rows = []
        for (batch, meas, panel), d in reg.groupby(['batch', 'measure', 'panel']):
            s = stats(d)
            if s['n_net'] + s['n_ud'] == 0:
                continue
            with open(R / 'Batches' / batch / f'check_{meas}_{panel}.tex', 'w') as f:
                f.write(box(line(s)))
        for (batch, meas), d in reg.groupby(['batch', 'measure']):
            s = stats(d)
            rows.append((batch, meas, s))
        # previous specifications (OLS, emissions / share outcomes)
        old = pd.read_csv(R / 'Batches' / 'diag_old_benchmark' / 'old_benchmark.csv')
        old_rows = []
        for t in ['net', 'wls', 'lag']:
            d = old[old['table'] == t]
            ok = d[(d['coef'] > 0) & (d['p'] < alpha)]
            with open(R / 'Batches' / 'diag_old_benchmark' / f'check_old_{t}.tex', 'w') as f:
                f.write(box(f'net: {len(ok)}/{len(d)} positive and significant'))
            old_rows.append((t, len(ok), len(d)))
        ud = old[old['table'] == 'updown'].copy()
        ud['base'] = ud['term'].str.replace('^(up|down)_', '', regex=True)
        ud['side'] = ud['term'].str.extract('^(up|down)_')[0]
        pairs = ud.pivot_table(index=['outcome', 'base'], columns='side', values=['coef', 'p'])
        both = pairs[(pairs[('coef', 'up')] > 0) & (pairs[('p', 'up')] < alpha) & (pairs[('coef', 'down')] > 0) & (pairs[('p', 'down')] < alpha)]
        larger = pairs[pairs[('coef', 'down')] > pairs[('coef', 'up')]]
        conf = both[both[('coef', 'down')] > both[('coef', 'up')]]
        with open(R / 'Batches' / 'diag_old_benchmark' / 'check_old_updown.tex', 'w') as f:
            f.write(box(f'up/down: both positive and significant {len(both)}/{len(pairs)}; downstream larger {len(larger)}/{len(pairs)}; all three {len(conf)}/{len(pairs)}'))
        # scorecard
        names = {'batch0_base': 'Batch 0: $G$, reduced form', 'batch1_D': 'Batch 1: $D$, reduced form', 'batch2_gov': 'Batch 2: government patents (IV)',
                 'batch3_rd': 'Batch 3: R\\&D prices (IV)', 'batch3_rdcf': 'Batch 3: R\\&D prices, clean firms (IV)', 'batch4_ks': 'Batch 4: spillovers (IV)',
                 'batch5_asinh': 'Batch 5: asinh levels (OLS)', 'batch6_govr': 'Batch 6: government ratio (IV)', 'batch6_ksr': 'Batch 6: spillover ratio (IV)',
                 'batch7_govcl': 'Batch 7: aggregate government (IV)', 'batch7_ksagg': 'Batch 7: aggregate spillovers (IV)', 'batch8_entry': 'Batch 8: first clean patents'}
        lines = [r'\begin{tabular}{lcccc}', r'\toprule',
                 r' & Net: positive & Up/down: both positive & Downstream & All three \\',
                 r' & and significant & and significant & larger & \\', r'\midrule']
        def cellfrac(k, n):
            return f'{k}/{n}' if n else '--'
        for batch, meas, s in sorted(rows, key=lambda r: (list(names).index(r[0]) if r[0] in names else 99, r[1] != 'pat')):
            lab = names.get(batch, batch) + (', counts' if meas == 'pat' else ', citations')
            net_c = cellfrac(len(s['net_ok']), s['n_net']) + (f" ({len(s['strong_net'])})" if s['iv'] and s['n_net'] else '')
            lines.append(f"{lab} & {net_c} & {cellfrac(len(s['both']), s['n_ud'])} & {cellfrac(len(s['larger']), s['n_ud'])} & {cellfrac(len(s['conf']), s['n_ud'])} \\\\")
        lines.append(r'\midrule')
        lines.append(f"Previous specifications (OLS, Table~\\ref{{tab:oldnet}}) & {sum(k for _, k, _ in old_rows)}/{sum(n for _, _, n in old_rows)} & {len(both)}/{len(pairs)} & {len(larger)}/{len(pairs)} & {len(conf)}/{len(pairs)} \\\\")
        lines += [r'\bottomrule', r'\end{tabular}']
        with open(R / 'Summary' / 'table_scorecard.tex', 'w') as f:
            f.write('\n'.join(lines) + '\n')
        return rows


    def Linkage_Comparison(self, path):
        """""
        Before / after table for the extended patent-to-firm crosswalk (union of three sources): coverage of the match and the
        baseline reduced-form estimates. The 'before' registry and coverage were saved in Specifications/Batches/diag_linkage/ before
        the data were rebuilt with the third crosswalk.
        """""
        R = self.Directory / 'Specifications'
        L = R / 'Batches' / 'diag_linkage'
        reg0, reg1 = (pd.read_csv(L / 'registry_before.csv').set_index('spec_id'), pd.read_csv(R / 'registry.csv').set_index('spec_id'))
        cov0 = pd.read_csv(L / 'match_coverage_before.csv')
        cov1 = pd.read_csv(R / 'Batches' / 'diag_match_coverage' / 'match_coverage.csv')
        def cov(df, col):
            r = df.iloc[-1]; return r[col]
        def est(reg, i):
            r = reg.loc[i]; return f"{r.coef_net:.2f}{gpf.get_stars(r.p_net)}", f"({r.se_net:.2f})"
        lines = [r'\begin{tabular}{lcc}', r'\toprule', r' & Two crosswalks & Three crosswalks \\', r'\midrule',
                 'Patents matched to a firm and industry, 1993--2022 & ' + ' & '.join(f"{cov(c, 'matched'):,.0f}" for c in (cov0, cov1)) + r' \\',
                 'Share of patents filed & ' + ' & '.join(f"{cov(c, 'share'):.3f}" for c in (cov0, cov1)) + r' \\',
                 'Share of clean patents filed & ' + ' & '.join(f"{cov(c, 'clean_share'):.3f}" for c in (cov0, cov1)) + r' \\',
                 'Last bin (2018--2022), share of patents filed & ' + ' & '.join(f"{c.iloc[-2]['share']:.3f}" for c in (cov0, cov1)) + r' \\',
                 r'\midrule', r'\multicolumn{3}{l}{\textit{Network coefficient, PPML}} \\']
        for lab, i in [('$G$, counts', 'b0_pat_base'), ('$G$, citations', 'b0_cite_base'), ('$D$, counts', 'b1_pat_base'),
                       ('$D$, citations', 'b1_cite_base'), ('$G$, counts, direct weights (M3)', 'b0_pat_ds')]:
            a, b = est(reg0, i), est(reg1, i)
            lines += [lab + f' & {a[0]} & {b[0]} \\\\', f' & {a[1]} & {b[1]} \\\\']
        lines += [r'\midrule', 'Industries, $G$ ($D$) baseline & ' + ' & '.join(f"{int(r.loc['b0_pat_base','n_sectors'])} ({int(r.loc['b1_pat_base','n_sectors'])})" for r in (reg0, reg1)) + r' \\',
                  'Observations, $G$ ($D$) baseline & ' + ' & '.join(f"{int(r.loc['b0_pat_base','N'])} ({int(r.loc['b1_pat_base','N'])})" for r in (reg0, reg1)) + r' \\',
                  r'\bottomrule', r'\end{tabular}']
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')


    def Cite5_Summary(self, path):
        """""
        Citation-weighted clean share: citations counted over the whole life of the patent (baseline) against citations received
        within five years of filing (variant c5), for G (batch 0) and D (batch 1); net network coefficient and, for the baseline
        and c5, the upstream / downstream split.
        """""
        reg = pd.read_csv(self.Directory / 'Specifications' / 'registry.csv').set_index('spec_id')
        def cell(i, col):
            if i not in reg.index or pd.isna(reg.loc[i, f'coef_{col}']):
                return '', ''
            r = reg.loc[i]
            return f"{r[f'coef_{col}']:.2f}{gpf.get_stars(r[f'p_{col}'])}", f"({r[f'se_{col}']:.2f})"
        lines = [r'\begin{tabular}{lcc}', r'\toprule', r' & $G$, citations & $D$, citations \\', r'\midrule']
        for lab, sp, col in [('Baseline (net)', 'base', 'net'), ('Five-year window (net)', 'c5', 'net'),
                             ('Baseline, upstream', 'updown', 'up'), ('Baseline, downstream', 'updown', 'down'),
                             ('Five-year window, upstream', 'c5ud', 'up'), ('Five-year window, downstream', 'c5ud', 'down'),
                             ('M1 + M2', 'm12', 'net'), ('M1 + M2, five-year window', 'm12c5', 'net'),
                             ('M1 + M2 + M3', 'm12ds', 'net'), ('M1 + M2 + M3, five-year window', 'm12dsc5', 'net')]:
            c = [cell(f'b0_cite_{sp}', col), cell(f'b1_cite_{sp}', col)]
            lines += [lab + ' & ' + ' & '.join(x[0] for x in c) + r' \\', ' & ' + ' & '.join(x[1] for x in c) + r' \\']
        lines += [r'\bottomrule', r'\end{tabular}']
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')


    def UDG_Summary(self, path, rows, cols):
        """""
        Compact summary table from the registry: one row per (specification, term), one column per outcome.
        rows: list of (label, short, slot) with slot in {net, up, down, lag}; cols: list of (label, id prefix, measure).
        The spec id is <prefix>_<measure>_<short>. Non-convergent specifications are shown as 'not convergent'.
        """""
        reg = pd.read_csv(self.Directory / 'Specifications' / 'registry.csv').set_index('spec_id')
        tex = lambda t: str(t).replace('_', r'\_')
        lines = [r'\begin{tabular}{l' + 'c' * len(cols) + '}', r'\toprule',
                 ' & ' + ' & '.join(c[0] for c in cols) + r' \\', r'\midrule']
        for label, short, slot in rows:
            cs, ss = [], []
            for _, tag, meas in cols:
                sid = f'{tag}_{meas}_{short}'
                if slot == 'eq':
                    cs.append(f"{reg.loc[sid, 'p_up_eq_down']:.3f}" if sid in reg.index and np.isfinite(reg.loc[sid, 'p_up_eq_down']) else '')
                    ss.append('')
                elif sid not in reg.index or not np.isfinite(reg.loc[sid, f'coef_{slot}']):
                    cs.append(''); ss.append('')
                elif reg.loc[sid, 'converged'] == 0 and reg.loc[sid, 'estimator'] != 'PPML':
                    cs.append('not convergent'); ss.append('')
                else:
                    r = reg.loc[sid]
                    cs.append(f"{r[f'coef_{slot}']:.2f}{gpf.get_stars(r[f'p_{slot}'])}")
                    ss.append(f"({r[f'se_{slot}']:.2f})")
            lines += [tex(label) + ' & ' + ' & '.join(cs) + r' \\', ' & ' + ' & '.join(ss) + r' \\']
        base = rows[0][1]
        ns = [str(int(reg.loc[f'{tag}_{meas}_{base}', 'N'])) if f'{tag}_{meas}_{base}' in reg.index else '' for _, tag, meas in cols]
        lines += [r'\midrule', 'Observations (first row) & ' + ' & '.join(ns) + r' \\', r'\bottomrule', r'\end{tabular}']
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')


    def UDG_Table(self, reg, path):
        "Write a batch table (one column per spec) from registry rows. All numbers come from the rows."
        tex = lambda s: str(s).replace('_', r'\_')
        labels = [('net', 'Net network'), ('up', 'Upstream'), ('down', 'Downstream'), ('lag', 'Own lagged share')]
        cols = len(reg)
        nc = [bool(v == False) for v in reg['converged']] if 'converged' in reg else [False] * cols   # explicit False only
        lines = [r'\begin{tabular}{l' + 'c' * cols + '}', r'\toprule',
                 ' & ' + ' & '.join(f'({i + 1})' for i in range(cols)) + r' \\',
                 ' & ' + ' & '.join(tex(s) for s in reg['short']) + r' \\', r'\midrule']
        for slot, name in labels:
            if reg[f'coef_{slot}'].isna().all():
                continue
            cs, ss = [], []
            for (_, r), bad in zip(reg.iterrows(), nc):
                if bad:
                    cs.append('not convergent' if slot == ([sl for sl, _n in labels if reg[f'coef_{sl}'].notna().any()] or ['net'])[0] else '')
                    ss.append('')
                elif np.isfinite(r[f'coef_{slot}']):
                    cs.append(f"{r[f'coef_{slot}']:.3f}{gpf.get_stars(r[f'p_{slot}'])}")
                    ss.append(f"({r[f'se_{slot}']:.3f})")
                else:
                    cs.append(''); ss.append('')
            lines += [f'{name} & ' + ' & '.join(cs) + r' \\', ' & ' + ' & '.join(ss) + r' \\']
        if 'free_offset' in reg and reg['free_offset'].any():
            co = [f"{c:.3f}" if fo else '1 (imposed)' for c, fo in zip(reg['coef_off'], reg['free_offset'])]
            so = [f"({s:.3f})" if fo else '' for s, fo in zip(reg['se_off'], reg['free_offset'])]
            po = [f"{p:.3f}" if fo else '' for p, fo in zip(reg['p_off_eq1'], reg['free_offset'])]
            lines += [r'\midrule', 'Coefficient on log offset & ' + ' & '.join(co) + r' \\', ' & ' + ' & '.join(so) + r' \\',
                      r'$p$-value, coefficient $=1$ & ' + ' & '.join(po) + r' \\']
        if 'instrument' in reg and reg['instrument'].fillna('').ne('').any():
            lines += [r'\midrule', r'\multicolumn{%d}{l}{\textit{First stage (OLS, sector and period effects)}} \\' % (cols + 1)]
            for slot, name in labels[:3]:
                if f'fs_F_{slot}' not in reg or reg[f'fs_F_{slot}'].isna().all():
                    continue
                f = lambda col, fmt: [fmt.format(v) if v == v else '' for v in reg[col]]
                st = lambda col, pcol, fmt: [fmt.format(v) + gpf.get_stars(p) if v == v else '' for v, p in zip(reg[col], reg[pcol])]
                lines += [f'{name}: coefficient on instrument & ' + ' & '.join(st(f'fs_coef_{slot}', f'fs_p_{slot}', '{:.3f}')) + r' \\',
                          ' & ' + ' & '.join('(' + x + ')' if x else '' for x in f(f'fs_se_{slot}', '{:.3f}')) + r' \\',
                          f'{name}: $F$ (excluded instruments) & ' + ' & '.join(st(f'fs_F_{slot}', f'fs_pF_{slot}', '{:.1f}')) + r' \\',
                          f'{name}: partial $R^2$ & ' + ' & '.join(f(f'fs_pr2_{slot}', '{:.3f}')) + r' \\']
        lines += [r'\midrule',
                  'Observations & ' + ' & '.join(f'{int(n):,}' for n in reg['N']) + r' \\',
                  'Sectors & ' + ' & '.join(f'{int(n)}' for n in reg['n_sectors']) + r' \\',
                  'Network variable & ' + ' & '.join(tex(s) for s in reg['rhs']) + r' \\',
                  *(['Outcome frequency & ' + ' & '.join(tex(s) for s in reg['outcome_freq']) + r' \\']
                    if 'outcome_freq' in reg and reg['outcome_freq'].nunique() > 1 else []),
                  *(['Network mass $\\times$ period controls & ' + ' & '.join('yes' if v else 'no' for v in reg['mass_trend']) + r' \\\\']
                    if 'mass_trend' in reg and reg['mass_trend'].any() else []),
                  *(['Clean definition & ' + ' & '.join(tex(s) for s in reg['clean_def']) + r' \\']
                    if 'clean_def' in reg and reg['clean_def'].nunique() > 1 else []),
                  *(['Instrument measured (bins before outcome) & ' + ' & '.join(tex(s) if s == s and s else '--' for s in reg['iv_lags']) + r' \\']
                    if 'iv_lags' in reg and reg['iv_lags'].fillna('').nunique() > 1 else []),
                  *(['Instrument winsorised & ' + ' & '.join(f'{w * 100:.0f}\\%' if w == w else 'no' for w in reg['iv_wins']) + r' \\']
                    if 'iv_wins' in reg and reg['iv_wins'].notna().any() else []),
                  *(['Hansen $J$ ($p$-value) & ' + ' & '.join(f'{w:.3f}' if (w == w and not bad) else '--' for w, bad in zip(reg['hansen_p'], nc)) + r' \\']
                    if 'hansen_p' in reg and reg['hansen_p'].notna().any() else []),
                  *([ 'Instrument & ' + ' & '.join(tex(s) if s == s and s else '--' for s in reg['instrument']) + r' \\']
                    if 'instrument' in reg and reg['instrument'].fillna('').ne('').any() else []),
                  'Weights & ' + ' & '.join(tex(s) for s in reg['weights']) + r' \\',
                  'Bin length (years) & ' + ' & '.join(f'{int(n)}' for n in reg['bin_len']) + r' \\',
                  'Offset & ' + ' & '.join(tex(s) for s in reg['offset']) + r' \\',
                  'Controls & ' + ' & '.join(tex(s).replace('W\\_w', 'weight sum') or 'none' if s else 'none' for s in reg['controls']) + r' \\',
                  r'Shrinkage $\kappa$ & ' + ' & '.join(f'{k:.1f}' if np.isfinite(k) else '--' for k in reg['kappa']) + r' \\',
                  'Fixed effects & ' + ' & '.join(tex(s) for s in reg['fe']) + r' \\',
                  r'\bottomrule', r'\end{tabular}']
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')


    @staticmethod
    def udg_label(v):
        "Readable row label for a regressor column name like up_G_cite_lag."
        pre, _, stem = v.replace('_lag', '').partition('_')
        stems = {'G_pat': 'clean share (patents)', 'G_cite': 'clean share (citations)',
                 'A_pat': 'asinh clean patents (counts)', 'A_cite': 'asinh clean patents (citations)',
                 'D_pat': 'clean/(clean+dirty) (patents)', 'D_cite': 'clean/(clean+dirty) (citations)',
                 'P_pat': 'dirty share (patents)', 'P_cite': 'dirty share (citations)',
                 'PD_pat': 'dirty/(clean+dirty) (patents)', 'PD_cite': 'dirty/(clean+dirty) (citations)',
                 'W_w': 'sum of Leontief weights',
                 'E_em': 'emissions-intensity reduction', 'T_pat': 'total patents (share)',
                 'Q_cite': 'citations per patent (relative)'}
        pres = {'up': 'Upstream', 'down': 'Downstream', 'net': 'Net', 'own': 'Own'}
        return f'{pres.get(pre, pre)} {stems.get(stem, stem)}'.replace('_', r'\_')


    def UDG_ModelTable(self, models, info, path):
        "Generic table: one column per model, one row per non-FE regressor. info is indexed by spec_id."
        tex = lambda s: str(s).replace('_', r'\_')
        ids = list(models)
        rows_v = []
        for i in ids:
            for v in models[i].x:
                if v not in rows_v:
                    rows_v.append(v)
        lines = [r'\begin{tabular}{l' + 'c' * len(ids) + '}', r'\toprule',
                 ' & ' + ' & '.join(f'({k + 1})' for k in range(len(ids))) + r' \\',
                 ' & ' + ' & '.join(tex(info.loc[i, 'short']) for i in ids) + r' \\', r'\midrule']
        for v in rows_v:
            cs, ss = [], []
            for i in ids:
                m = models[i]
                if v in m.params.index:
                    cs.append(f'{m.params[v]:.3f}{gpf.get_stars(m.pvalues[v])}')
                    ss.append(f'({m.bse[v]:.3f})')
                else:
                    cs.append(''); ss.append('')
            lines += [self.udg_label(v) + ' & ' + ' & '.join(cs) + r' \\', ' & ' + ' & '.join(ss) + r' \\']
        lines += [r'\midrule',
                  'Outcome & ' + ' & '.join(tex(info.loc[i, 'outcome']) for i in ids) + r' \\',
                  'Offset & ' + ' & '.join(tex(info.loc[i, 'offset']) for i in ids) + r' \\',
                  'Observations & ' + ' & '.join(f'{models[i].nobs:,}' for i in ids) + r' \\',
                  'Sectors & ' + ' & '.join(f'{models[i].n_sectors}' for i in ids) + r' \\',
                  'Fixed effects & ' + ' & '.join(tex('+'.join(models[i].fe) or 'none') for i in ids) + r' \\',
                  r'\bottomrule', r'\end{tabular}']
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')


    def UDG_Corr(self, batch, BLS_year_start, Year_end, bin_len=5):
        """""
        Correlations among the lagged up/down clean shares (patents and citations).
        Panel A: raw. Panel B: after removing sector and period means.

        Output: Specifications/Batches/<batch>/table_corr.tex
        """""
        outdir = self.Directory / 'Specifications' / 'Batches' / batch
        outdir.mkdir(parents=True, exist_ok=True)
        reg_df, _ = self.UDG_Panel(BLS_year_start, Year_end, bin_len)
        vs = ['up_G_pat_lag', 'down_G_pat_lag', 'up_G_cite_lag', 'down_G_cite_lag']
        names = ['Up (patents)', 'Down (patents)', 'Up (citations)', 'Down (citations)']
        d = reg_df.dropna(subset=vs)[['BLS_Industry', 'period'] + vs].copy()
        within = d.copy()
        for _ in range(100):
            for g in ['BLS_Industry', 'period']:
                within[vs] = within[vs] - within.groupby(g)[vs].transform('mean')

        def panel(c, title):
            c = c.to_numpy()
            out = [r'\multicolumn{5}{l}{\textit{' + title + r'}} \\']
            for i, nm in enumerate(names):
                out.append(nm + ' & ' + ' & '.join(f'{c[i, j]:.3f}' if j <= i else '' for j in range(4)) + r' \\')
            return out
        lines = [r'\begin{tabular}{lcccc}', r'\toprule', ' & ' + ' & '.join(names) + r' \\', r'\midrule']
        lines += panel(d[vs].corr(), f'Panel A: raw (N = {len(d)})') + [r'\midrule']
        lines += panel(within[vs].corr(), 'Panel B: sector and period means removed') + [r'\bottomrule', r'\end{tabular}']
        with open(outdir / 'table_corr.tex', 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(d[vs].corr().round(3)); print(within[vs].corr().round(3))


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

    






    