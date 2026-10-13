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

        
        
    def Cleaner(self, BLS_year_start, Year_start, Year_end, bin_len, API, k_min=3, k_max=10):
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
                Clean Data/Spill_Net.pkl
                Clean Data/Tech_Pat.pkl
                Clean Data/Pat_Firms.pkl
                Clean Data/Ind_Pat.pkl
                Clean Data/Ind_Pat_Shares_Pre.pkl
                Clean Data/Gov_CPC.pkl
                Clean Data/Ind_Pat_Y02P.pkl   (clean = Y02P only)
                Clean Data/Spill_Net_cg{,_c3}.pkl, Tech_Pat_cg{,_c3}.pkl, Ind_Pat_Shares_Pre_cg{,_c3}.pkl
                    (clean / general types only: dirty patents are general)
                Clean Data/Spill_Clean_Net_cg{,_c3}.pkl   (citations of all clean patents to technologies, by lag)
                Clean Data/Ind_Pre_Clean.pkl   (pre-sample clean share of each sector)
                Clean Data/Spill_Net_c3.pkl, Tech_Pat_c3.pkl, Ind_Pat_Shares_Pre_c3.pkl, Gov_CPC_c3.pkl
                    (same objects at 3-digit CPC class; the column named cpc_subclass holds the class)
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
        
        Gov_Pats_df = PV_assignee_df[['patent_id']][PV_assignee_df['assignee_type']==6].drop_duplicates()
        del PV_assignee_df
        
        univ_pats_df = pd.read_csv(f'{self.Directory}/Raw Data/patentsview_govt_univ.csv')
        univ_pats_df['patent_id'] = univ_pats_df['patent_id'].astype(str)
        
        Gov_Pats_df = Gov_Pats_df.merge(univ_pats_df[['patent_id']][univ_pats_df['univ_or_gov_nonprofit'] == 1],
                             on='patent_id',
                             how='outer')

    
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
        
        relevant_df['y02p'] = (relevant_df["cpc_subclass"] == "Y02P").astype(np.int8)
        relevant_df['clean_p'] = relevant_df.groupby("patent_id")['y02p'].transform("max")
        relevant_df['clean_full'] = relevant_df.groupby("patent_id")['clean'].transform("max")
        relevant_df['clean'] = relevant_df['clean_full'] - relevant_df.groupby("patent_id")['ice'].transform("max")
        relevant_df['dirty'] = relevant_df.groupby("patent_id")['dirty'].transform("max")
        relevant_df = relevant_df[['patent_id', 'year', 'clean', 'clean_full', 'clean_p', 'dirty']].drop_duplicates()
        
        
        # Technology classes at two granularities: 4-digit subclass (baseline files) and 3-digit class (files with suffix _c3).
        # In the _c3 files the column called cpc_subclass holds the 3-character CPC class.
        cpc_levels = [('cpc_subclass', ''), ('cpc_class', '_c3')]
        def cpc_unit_df(unit):
            d = cpc4_df if unit == 'cpc_subclass' else cpc3_df
            return d.rename(columns={unit: 'cpc_unit'})
        # Patent types for the spillover files: 'cdg' = clean / dirty / general (baseline files);
        # 'cg' = clean / general only, where general means not clean (files with suffix _cg; dirty patents are general).
        type_schemes = [('cdg', ''), ('cg', '_cg')]
        def patent_types(clean, dirty, scheme):
            if scheme == 'cg':
                return [['c'] if c == 1 else ['g'] for c in clean]
            return [(['c'] if c == 1 else []) + (['d'] if d == 1 else []) or ['g'] for c, d in zip(clean, dirty)]
        
        # ----------------- #
        # Spillover Network #
        # ----------------- #
        for _ts, _tsfx in type_schemes:
            rel = relevant_df[['patent_id', 'year', 'clean', 'dirty']].copy()
            rel['types'] = patent_types(rel['clean'], rel['dirty'], _ts)
            rel = rel[['patent_id', 'year', 'types']]
    
            for _unit, _sfx in cpc_levels:
                cpc = cpc_unit_df(_unit)
                cpc_counts = cpc.groupby('patent_id').size().rename('n_cpc').reset_index()
    
                cit = citations_df[['patent_id', 'citation_patent_id']].drop_duplicates()
                cit = cit.merge(rel.rename(columns={'year': 'citing_year', 'types': 'citing_types'}), on='patent_id', how='inner')
                cit = cit.merge(rel.rename(columns={'patent_id': 'citation_patent_id', 'year': 'cited_year', 'types': 'cited_types'}), on='citation_patent_id', how='inner')
                cit = cit.merge(cpc_counts.rename(columns={'patent_id': 'citation_patent_id', 'n_cpc': 'n_cpc_q'}), on='citation_patent_id', how='inner')

                cit = cit[(cit['citing_year'] > BLS_year_start-bin_len-10) 
                                                      & (cit['citing_year'] <= BLS_year_start-bin_len)]
                cit['k'] = cit['citing_year'] - cit['cited_year']
                cit = cit[(cit['k'] >= k_min) & (cit['k'] <= k_max)].copy()
        
                cit['n_types_q'] = cit['cited_types'].apply(len)
                cit['cited_split'] = cit['n_cpc_q'] * cit['n_types_q']
                citing_cpc = cpc.rename(columns={'cpc_unit': 'cpc_p'})
                left = cit.merge(citing_cpc, on='patent_id', how='inner')
                left = left.explode('citing_types').rename(columns={'citing_types': 'type_p'})
                left['i'] = left['cpc_p'] + '_' + left['type_p']
        
                cited_cpc = cpc.rename(columns={'patent_id': 'citation_patent_id', 'cpc_unit': 'cpc_q'})
                if _ts == 'cg':
                    # Spillovers from all clean citing patents (one aggregate recipient): share of their citations, by lag,
                    # that go to each technology j (class x clean/general), pre-sample window as above.
                    cc = cit[cit['citing_types'].apply(lambda t: 'c' in t)][['patent_id', 'citation_patent_id', 'k', 'cited_types', 'cited_split']]
                    cc = cc.merge(cited_cpc, on='citation_patent_id', how='inner').explode('cited_types')
                    cc['j'] = cc['cpc_q'] + '_' + cc['cited_types']
                    cc['weight'] = 1.0 / cc['cited_split']
                    ccn = cc.groupby(['j', 'k'])['weight'].sum().reset_index(name='N_jk')
                    ccn['S_jk'] = ccn['N_jk'] / ccn.groupby('k')['N_jk'].transform('sum')
                    ccn[['j', 'k', 'S_jk']].to_pickle(f'{self.Directory}/Clean Data/Spill_Clean_Net{_tsfx}{_sfx}.pkl')
                    del cc, ccn
                long = left.merge(cited_cpc, on='citation_patent_id', how='inner')
                long = long.explode('cited_types').rename(columns={'cited_types': 'type_q'})
                long['j'] = long['cpc_q'] + '_' + long['type_q']
                long['weight'] = 1.0 / long['cited_split']
        
                N = long.groupby(['i', 'j', 'k'])['weight'].sum().reset_index(name='N_ijk')
                TotalN = N.groupby(['i', 'k'])['N_ijk'].sum().reset_index(name='TotalN_jk')
        
                diag = N[N['i'] == N['j']][['i', 'k', 'N_ijk']].rename(columns={'N_ijk': 'N_iik'})

                out = N.merge(TotalN, on=['i', 'k'], how='left')
                out = out.merge(diag, on=['i', 'k'], how='left')
                out['N_iik'] = out['N_iik'].fillna(0.0)
        
                out = out[out['i'] != out['j']].copy()
                out['denom'] = out['TotalN_jk'] - out['N_iik']
                out['S_ijk'] = np.where(out['denom'] > 0, out['N_ijk'] / out['denom'], np.nan)
        
                spill_net_df = out[['i', 'j', 'k', 'S_ijk']]
                spill_net_df.to_pickle(f'{self.Directory}/Clean Data/Spill_Net{_tsfx}{_sfx}.pkl')
                del cit, left, long, N, TotalN, diag, out, spill_net_df, cpc, cpc_counts, citing_cpc, cited_cpc
        
        
        
        # ------------------------- #
        # Patent Citation Weighting #
        # ------------------------- #
        citations_df['cites'] = citations_df.groupby('citation_patent_id')['citation_patent_id'].transform('count')
        citations_df = citations_df[['citation_patent_id', 'cites']].drop_duplicates()
        citations_df.rename(columns={'citation_patent_id': 'patent_id'}, inplace=True)
        
        citations_df = citations_df.merge(cpc3_df,
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
        
        del PV_applications_df, relevant_df, citations_df
        
        
        # ----------------------------- #
        # Patenting by Technology Class #
        # ----------------------------- #
        for _ts, _tsfx in type_schemes:
            for _unit, _sfx in cpc_levels:
                tech_pat_df = pat_df.merge(cpc_unit_df(_unit),
                                            on='patent_id',
                                            how='inner')
        
                tech_pat_df['type'] = patent_types(tech_pat_df['clean'], tech_pat_df['dirty'], _ts)
                tech_pat_df = tech_pat_df.explode('type')
                tech_pat_df['tech'] = tech_pat_df['cpc_unit'] + '_' + tech_pat_df['type']
        
                tech_pat_df = tech_pat_df[['patent_id', 'tech', 'year', 'norm_cites']].drop_duplicates()
                tech_pat_df['split_weight'] = 1 / tech_pat_df.groupby('patent_id')['tech'].transform('count')
                tech_pat_df['cite_weight'] = tech_pat_df['split_weight'] * tech_pat_df['norm_cites']
        
                tech_pat_df['pat_count'] = tech_pat_df.groupby(['tech', 'year'])['split_weight'].transform('sum')
                tech_pat_df['pat_cites'] = tech_pat_df.groupby(['tech', 'year'])['cite_weight'].transform('sum')
        
                tech_pat_df = tech_pat_df[['tech', 'year', 'pat_count', 'pat_cites']].drop_duplicates()
                tech_pat_df.to_pickle(f'{self.Directory}/Clean Data/Tech_Pat{_tsfx}{_sfx}.pkl')
        
        
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
        
        pat_firms_df = pat_firms_df[['patent_id', 'year', 'gvkey', 'BLS_Industry', 'clean', 'clean_full', 'clean_p', 'dirty', 'norm_cites']].drop_duplicates()
        pat_ind_df = pat_firms_df[['patent_id', 'year', 'BLS_Industry', 'clean', 'clean_full', 'clean_p', 'dirty', 'norm_cites']].drop_duplicates()
        
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

        ind_pat_df = (annual_df.groupby(['BLS_Industry', 'year'], as_index=False)
                                    .agg(**agg_base))
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
 
        ind_pat_df = ind_pat_df[ind_pat_df['BLS_Industry'] != 71]
        ind_pat_df_y02p = ind_pat_df_y02p[ind_pat_df_y02p['BLS_Industry'] != 71]
        
        for frame in (ind_pat_df, ind_pat_df_y02p):
            frame['pat_count_nc'] = frame['pat_count'] - frame['clean_pat_count']
            frame['pat_cites_nc'] = frame['pat_cites'] - frame['clean_pat_cites']
            frame['clim_pat_count'] = frame['clean_pat_count'] + frame['dirty_pat_count']
            frame['clim_pat_cites'] = frame['clean_pat_cites'] + frame['dirty_pat_cites']
 
        ind_pat_df.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat.pkl')
        ind_pat_df_y02p.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat_Y02P.pkl')
        
        
        # -------------------- #
        # CPC Shares by Sector #
        # -------------------- #
        # Pre-sample clean share of each sector's patents (used by the aggregate-shock instruments)
        _pre = pat_ind_df[(pat_ind_df['year'] > BLS_year_start-bin_len-10) & (pat_ind_df['year'] <= BLS_year_start-bin_len)].copy()
        _pre['w'] = _pre['split_weight']
        _pre['wc'] = _pre['w'] * _pre['norm_cites']
        _pre['w_clean'] = _pre['w'] * _pre['clean']
        _pre['wc_clean'] = _pre['wc'] * _pre['clean']
        (_pre.groupby('BLS_Industry')[['w', 'wc', 'w_clean', 'wc_clean']].sum()
             .rename(columns={'w': 'pre_all_pat', 'wc': 'pre_all_cites', 'w_clean': 'pre_clean_pat', 'wc_clean': 'pre_clean_cites'})
             .reset_index().to_pickle(f'{self.Directory}/Clean Data/Ind_Pre_Clean.pkl'))
        del _pre
        
        for _ts, _tsfx in type_schemes:
            for _unit, _sfx in cpc_levels:
                ind_pat_cpc_pre_df = pat_ind_df.merge(cpc_unit_df(_unit),
                                                    on='patent_id',
                                                    how='inner')
        
        
                ind_pat_cpc_pre_df = ind_pat_cpc_pre_df[(ind_pat_cpc_pre_df['year'] > BLS_year_start-bin_len-10) 
                                                              & (ind_pat_cpc_pre_df['year'] <= BLS_year_start-bin_len)]
                panel_idx = pd.MultiIndex.from_product(
                    [sorted(CPC_df[_unit].unique()), sorted(ind_pat_cpc_pre_df['BLS_Industry'].unique())],
                    names=['cpc_unit', 'BLS_Industry'])
        
                frames = []
                for ty in (['clean', 'dirty', 'gen', 'all'] if _ts == 'cdg' else ['clean', 'gen']):
                    clim_share = ind_pat_cpc_pre_df.copy()
                    if ty == 'gen' and _ts == 'cg':
                        clim_share = clim_share[clim_share['clean'] == 0]
                    elif ty == 'gen':
                        clim_share = clim_share[(clim_share['clean'] == 0) & (clim_share['dirty'] == 0)]
                    elif ty == 'all':
                        pass
                    else:
                        clim_share = clim_share[clim_share[ty] == 1]
            
                    clim_share['pat_weight'] = clim_share['split_weight'] / clim_share.groupby('patent_id')['cpc_unit'].transform('nunique')
                    clim_share['cite_weight'] = clim_share['pat_weight'] * clim_share['norm_cites']
        
                    clim_share['cpc_pat_count'] = clim_share.groupby(['BLS_Industry', 'cpc_unit'])['pat_weight'].transform('sum')
                    clim_share['pat_count'] = clim_share.groupby('BLS_Industry')['pat_weight'].transform('sum')
                    clim_share['cpc_pat_share'] = clim_share['cpc_pat_count'] / clim_share['pat_count']
        
                    clim_share['cpc_pat_cites'] = clim_share.groupby(['BLS_Industry', 'cpc_unit'])['cite_weight'].transform('sum')
                    clim_share['pat_cites'] = clim_share.groupby('BLS_Industry')['cite_weight'].transform('sum')
                    clim_share['cpc_cite_share'] = clim_share['cpc_pat_cites'] / clim_share['pat_cites']
        
                    clim_share = clim_share[['BLS_Industry', 'cpc_unit', 'cpc_pat_share', 'cpc_cite_share']].drop_duplicates()
                    clim_share = (clim_share.set_index(['cpc_unit', 'BLS_Industry'])
                                            .reindex(panel_idx)
                                            .fillna(0.0)
                                            .reset_index())
                    clim_share['type'] = ty
                    frames.append(clim_share)
            
                ind_pat_shares_pre_df = pd.concat(frames, ignore_index=True).rename(columns={'cpc_unit': 'cpc_subclass'})
                ind_pat_shares_pre_df.to_pickle(f'{self.Directory}/Clean Data/Ind_Pat_Shares_Pre{_tsfx}{_sfx}.pkl')
        
        
        # ------------------------ #
        # Government Patent Series #
        # ------------------------ #
        Gov_Pats_df = Gov_Pats_df.merge(pat_df,
                             on='patent_id',
                             how='inner')
        
        for _unit, _sfx in cpc_levels:
            gov_cpc_df = Gov_Pats_df.merge(cpc_unit_df(_unit),
                                 on='patent_id',
                                 how='inner')
        
            gov_cpc_df['pat_weight'] = 1.0 / gov_cpc_df.groupby('patent_id')['cpc_unit'].transform('nunique')
            gov_cpc_df['cite_weight'] = gov_cpc_df['pat_weight'] * gov_cpc_df['norm_cites']
        
            gov_cpc_df['clean_weight'] = gov_cpc_df['pat_weight'] * gov_cpc_df['clean']
            gov_cpc_df['clean_cite_weight'] = gov_cpc_df['cite_weight'] * gov_cpc_df['clean']
        
            gov_cpc_df['gov_pat_count'] = gov_cpc_df.groupby(['cpc_unit', 'year'])['pat_weight'].transform('sum')
            gov_cpc_df['gov_pat_cites'] = gov_cpc_df.groupby(['cpc_unit', 'year'])['cite_weight'].transform('sum')
            gov_cpc_df['gov_clean_count'] = gov_cpc_df.groupby(['cpc_unit', 'year'])['clean_weight'].transform('sum')
            gov_cpc_df['gov_clean_cites'] = gov_cpc_df.groupby(['cpc_unit', 'year'])['clean_cite_weight'].transform('sum')
        
            gov_cpc_df = gov_cpc_df[['cpc_unit', 'year', 'gov_pat_count', 'gov_pat_cites',
                                     'gov_clean_count', 'gov_clean_cites']].drop_duplicates()
        
            panel_idx = pd.MultiIndex.from_product(
                   [sorted(CPC_df[_unit].unique()), list(range(BLS_year_start-2*bin_len+1, Year_end+1))],
                   names=['cpc_unit', 'year'])
            gov_cpc_df = (gov_cpc_df.set_index(['cpc_unit', 'year'])
                                  .reindex(panel_idx)
                                  .fillna(0.0)
                                  .reset_index())

            gov_cpc_df.rename(columns={'cpc_unit': 'cpc_subclass'}).to_pickle(f'{self.Directory}/Clean Data/Gov_CPC{_sfx}.pkl')
        
        


    def Instruments(self, sfx='', asinh_pat=False):
        """""
        Create Series of Greenification Shocks

        sfx: '' = 4-digit CPC subclasses (baseline); '_c3' = 3-digit CPC classes. For '_c3' only the government and
        spillover shocks are built (the R&D-price shock does not depend on the technology class).

        Output: Clean Data/Govt_Shocks.pkl
                Clean Data/KS_Shocks.pkl
                Clean Data/RD_Firm_Panel.pkl  (zero-stage estimation sample)
                Clean Data/RD_Shocks.pkl
        """""
        
        # ----------------------------------------------------------------

        # Build instrument dataframes.

        # ----------------------------------------------------------------
        
        gov_cpc_df = pd.read_pickle(f'{self.Directory}/Clean Data/Gov_CPC{sfx}.pkl')
        ind_pat_shares_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_Pat_Shares_Pre{sfx}.pkl')

        
        
        
        # ------------------------ #
        # Government Patent Shocks #
        # ------------------------ #
        if asinh_pat:   # government patenting as asinh(patents) by class-year; only the government shock is rebuilt
            gov_cpc_df = gov_cpc_df.copy()
            gov_cpc_df[['gov_pat_count', 'gov_pat_cites']] = np.arcsinh(gov_cpc_df[['gov_pat_count', 'gov_pat_cites']])
        govt_shocks_df = pd.merge(gov_cpc_df,
                                    ind_pat_shares_df,
                                    on='cpc_subclass',
                                    how='inner'
                                    )
        
        frames = []
        for ty in ['clean', 'dirty', 'gen']:
            shock = govt_shocks_df.copy()
            shock = shock[shock['type'] == ty]
            
            shock['weighted_pat_govt'] = shock['cpc_pat_share'] * shock['gov_pat_count']
            shock['pat_govt_shock'] = shock.groupby(['BLS_Industry', 'year'])['weighted_pat_govt'].transform('sum')
            
            shock['weighted_cite_govt'] = shock['cpc_cite_share'] * shock['gov_pat_cites']
            shock['cite_govt_shock'] = shock.groupby(['BLS_Industry', 'year'])['weighted_cite_govt'].transform('sum')
            
            shock = shock[['BLS_Industry', 'year', 'type',
                           'pat_govt_shock', 'cite_govt_shock']].drop_duplicates()

            frames.append(shock)
                
        govt_shocks_df = pd.concat(frames, ignore_index=True)
        
        govt_shocks_wide = govt_shocks_df.pivot(index=['BLS_Industry', 'year'], columns=['type'], 
                                              values=['pat_govt_shock', 'cite_govt_shock'])

        govt_shocks_wide.columns = [f'{v}_{t}' for v, t in govt_shocks_wide.columns]
        govt_shocks_df = govt_shocks_wide.reset_index()
        
        govt_shocks_df.to_pickle(f"{self.Directory}/Clean Data/Govt_Shocks{sfx}{'_asinh' if asinh_pat else ''}.pkl")
        if asinh_pat:
            return
        
        
        self.Spillover_Shocks(sfx)
        if sfx:
            return

        IV_year_start = 1980
        
        state_rdp_df = pd.read_pickle(f'{self.Directory}/Clean Data/state_rd_price.pkl')
        PV_inventor_location_df = pd.read_pickle(f'{self.Directory}/Clean Data/Inventor_Locations.pkl')
        pat_firms_df = pd.read_pickle(f'{self.Directory}/Clean Data/Pat_Firms.pkl')
        pat_firms_df = pat_firms_df[pat_firms_df['year'] >= IV_year_start]
        # Keep only the inventor rows that can merge (memory); the inner merge below is unchanged by this
        PV_inventor_location_df = PV_inventor_location_df.loc[
            PV_inventor_location_df['patent_id'].isin(pat_firms_df['patent_id']),
            ['patent_id', 'inventor_id', 'state_fips']]
        
        
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
 
        # Two versions: all firms, and only firms that ever have clean patents (suffix _cf)
        _clean_firms = set(firm_pat_wide_df.loc[firm_pat_wide_df['pat_count_clean'] > 0, 'gvkey'])
        for _v, _vsfx in [('all', ''), ('cf', '_cf')]:
            _wide = firm_pat_wide_df if _v == 'all' else firm_pat_wide_df[firm_pat_wide_df['gvkey'].isin(_clean_firms)]
            id_cols = ['gvkey', 'BLS_Industry', 'year']
            frames  = []
            for tname, cmap in type_map.items():
                missing = [c for c in cmap.values() if c not in _wide.columns]
                if missing:
                    raise KeyError(f'type "{tname}" needs columns {missing}')
                sub = _wide[id_cols + list(cmap.values())].copy()
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
            firm_pat_panel_df.reset_index().to_pickle(f'{self.Directory}/Clean Data/RD_Firm_Panel{_vsfx}.pkl')
    
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
                'pat_count_hat__general': 'pat_count_hat_gen',
                'pat_count_hat__clean':   'pat_count_clean_hat',
                'pat_count_hat__dirty':   'pat_count_dirty_hat',
                'pat_cites_hat__general': 'pat_cites_hat_gen',
                'pat_cites_hat__clean':   'pat_cites_clean_hat',
                'pat_cites_hat__dirty':   'pat_cites_dirty_hat'})
        
            RD_shocks_df.to_pickle(f'{self.Directory}/Clean Data/RD_Shocks{_vsfx}.pkl')
   
    

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


    def UDG_Panel(self, BLS_year_start, Year_end, bin_len, normalise=False, shrink=True, ivs=(), annual=False, iv_wins=None, clean='all', fixed_w=False):
        """""
        Regression panel for strategic complementarity: for each (share, measure) in
        {G, D} x {pat, cite} builds the own share, up, down, net network shares, and their lags.
        For each source in ivs ('gov', 'ks', 'rd') it also builds the instrument share Z<src> (clean/gen
        shock ratio, see UDG_Instrument) and its network sums, using the Leontief inverse fixed at the first bin.
        iv_wins: if set, instrument shares are winsorised at [iv_wins, 1-iv_wins] across industries within each period.
        clean: 'all' = baseline clean definition (Y02E, Y02P, Y02T, B60L); 'y02p' = Y02P only (Ind_Pat_Y02P.pkl).
        fixed_w: all network sums (regressors and controls, not only the instruments) use the Leontief inverse of the first
        bin instead of the inverse of each bin, so that changes in the network cannot drive the regressors.
        annual: regressors are built from rolling bin_len-year windows ending in every year (so the regressor for
        year y is the window ending y - bin_len), and the outcome and offset are single-year counts.

        Returns: reg_df, kappas
        """""
        IO_mats = pd.read_pickle(f'{self.Directory}/Clean Data/IO_Networks.pkl')
        Ind_Pat_yr_df = pd.read_pickle(f"{self.Directory}/Clean Data/Ind_Pat{'_Y02P' if clean == 'y02p' else ''}.pkl")
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
            S = np.linalg.inv(np.eye(IO_matrix.shape[0]) - IO_matrix)
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
            return ((num + kappa * gbar) / (den + kappa)).where(den > 0)

        shares, kappas = {}, {}
        for meas, suf in [('pat', 'count'), ('cite', 'cites')]:
            cln = wide(f'clean_pat_{suf}')
            for sh, den in [('G', wide(f'pat_{suf}')), ('D', wide(f'clim_pat_{suf}'))]:
                if shrink:
                    kappas[(sh, meas)] = estimate_kappa(cln, den)
                    shares[(sh, meas)] = shrink_to_mean(cln, den, kappas[(sh, meas)])
                else:
                    kappas[(sh, meas)] = np.nan
                    shares[(sh, meas)] = (cln / den.where(den > 0))
            # Placebo: dirty / all patents
            dty, tot = wide(f'dirty_pat_{suf}'), wide(f'pat_{suf}')
            if shrink:
                kappas[('P', meas)] = estimate_kappa(dty, tot)
                shares[('P', meas)] = shrink_to_mean(dty, tot, kappas[('P', meas)])
            else:
                kappas[('P', meas)] = np.nan
                shares[('P', meas)] = dty / tot.where(tot > 0)
            # Placebo for D: dirty / (clean + dirty)
            clm = wide(f'clim_pat_{suf}')
            if shrink:
                kappas[('PD', meas)] = estimate_kappa(dty, clm)
                shares[('PD', meas)] = shrink_to_mean(dty, clm, kappas[('PD', meas)])
            else:
                kappas[('PD', meas)] = np.nan
                shares[('PD', meas)] = dty / clm.where(clm > 0)

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


    def RD_ZeroStage(self, batch):
        """""
        Zero-stage regressions behind the R&D-price shock: log firm patenting on log state R&D-price exposure,
        with entity (firm x type) and year effects, SEs clustered by entity. Whole sample (general, clean and dirty
        firm-type panels stacked, as used to build the instrument) and each type on its own. Not used to build
        instruments except the whole-sample fit.

        Output: Results/Batches/<batch>/zero_stage.csv, table_zero_stage.tex
        """""
        outdir = self.Directory / 'Results' / 'Batches' / batch
        outdir.mkdir(parents=True, exist_ok=True)
        panels_rd = {'': pd.read_pickle(f'{self.Directory}/Clean Data/RD_Firm_Panel.pkl').set_index(['entity', 'year']).sort_index()}
        cf_path = Path(f'{self.Directory}/Clean Data/RD_Firm_Panel_cf.pkl')
        if cf_path.exists():
            panels_rd['cf'] = pd.read_pickle(cf_path).set_index(['entity', 'year']).sort_index()
        samples = [('all', 'Whole sample', None, ''), ('clean', 'Clean only', 'clean', ''),
                   ('dirty', 'Dirty only', 'dirty', ''), ('general', 'General only', 'general', '')]
        if 'cf' in panels_rd:   # firms that ever have clean patents (all three types)
            samples += [('cf_all', 'Clean firms: whole sample', None, 'cf'), ('cf_clean', 'Clean firms: clean only', 'clean', 'cf'),
                        ('cf_general', 'Clean firms: general only', 'general', 'cf')]
        rows = []
        for meas, y, x in [('pat', 'ln_pat_count', 'ln_E_rho_pats'), ('cite', 'ln_pat_cites', 'ln_E_rho_cites')]:
            for key, lab, ty, pk in samples:
                panel = panels_rd[pk]
                d = panel if ty is None else panel[panel['type'] == ty]
                m = gpf.run_reg(d[y], d[[x]], 'panel')
                rows.append(dict(measure=meas, sample=key, label=lab, coef=float(m.params[x]), se=float(m.std_errors[x]),
                                 p=float(m.pvalues[x]), t=float(m.tstats[x]), N=int(m.nobs), n_entities=int(m.entity_info['total']),
                                 n_firms=int(d['gvkey'].nunique()), r2_within=float(m.rsquared)))
        out = pd.DataFrame(rows)
        out.to_csv(outdir / 'zero_stage.csv', index=False)
        def write_table(smp, fname):
            ns = len(smp)
            lines = [r'\begin{tabular}{l' + 'c' * ns + '}', r'\toprule',
                     ' & ' + ' & '.join(f'({i + 1})' for i in range(ns)) + r' \\',
                     ' & ' + ' & '.join(l for _, l, _, _ in smp) + r' \\', r'\midrule']
            for meas, title in [('pat', 'Dependent variable: log patent count'), ('cite', 'Dependent variable: log normalised citations')]:
                sub = out[out['measure'] == meas].set_index('sample').loc[[k for k, _, _, _ in smp]]
                lines += [r'\multicolumn{%d}{l}{\textit{' % (ns + 1) + title + r'}} \\',
                          'Log state R\\&D-price exposure & ' + ' & '.join(f"{r.coef:.3f}{gpf.get_stars(r.p)}" for r in sub.itertuples()) + r' \\',
                          ' & ' + ' & '.join(f'({r.se:.3f})' for r in sub.itertuples()) + r' \\',
                          '$t$-statistic & ' + ' & '.join(f'{r.t:.2f}' for r in sub.itertuples()) + r' \\',
                          'Observations & ' + ' & '.join(f'{r.N:,}' for r in sub.itertuples()) + r' \\',
                          'Firm-type entities & ' + ' & '.join(f'{r.n_entities:,}' for r in sub.itertuples()) + r' \\',
                          'Firms & ' + ' & '.join(f'{r.n_firms:,}' for r in sub.itertuples()) + r' \\',
                          '$R^2$ (effects removed) & ' + ' & '.join(f'{r.r2_within:.4f}' for r in sub.itertuples()) + r' \\', r'\midrule']
            lines += ['Fixed effects & ' + ' & '.join(['entity+year'] * ns) + r' \\', r'\bottomrule', r'\end{tabular}']
            with open(outdir / fname, 'w') as f:
                f.write('\n'.join(lines) + '\n')

        write_table([x for x in samples if x[3] == ''], 'table_zero_stage.tex')
        cf = [x for x in samples if x[3] == 'cf']
        if cf:   # clean-only (unchanged by the restriction) next to the same regressions on firms with clean patents
            ref = [('clean', 'Reference: all firms, clean only', 'clean', '')]
            write_table(ref + [x for x in cf if x[0] in ('cf_all', 'cf_general')], 'table_zero_stage_cf.tex')
        print(out.round(4).to_string())
        return out


    def KS_Shares(self, batch, BLS_year_start, Year_end, bin_len=5, sfx='_cg'):
        """""
        Composition of the spillovers each recipient type receives: for recipient type r in {clean, dirty, gen} the share
        of its received spillovers that originates in source type {clean, dirty, general} (the nine KS_Shocks columns).
        Pooled shares sum the shocks over manufacturing industries and years; sector-average shares are the mean of
        the industry-level shares (industry-years with no spillovers received are skipped).

        Output: Results/Batches/<batch>/ks_shares.csv, table_ks_shares.tex, table_ks_shares_bins.tex
        """""
        outdir = self.Directory / 'Results' / 'Batches' / batch
        outdir.mkdir(parents=True, exist_ok=True)
        ks = pd.read_pickle(f'{self.Directory}/Clean Data/KS_Shocks{sfx}.pkl')
        manu = np.arange(self.manu_cols[0], self.manu_cols[1] + 1)
        ks = ks[ks['BLS_Industry'].isin(manu) & (ks['year'] > BLS_year_start - bin_len) & (ks['year'] <= Year_end)].copy()
        ks['period'] = Year_end - ((Year_end - ks['year']) // bin_len) * bin_len
        if sfx.startswith('_cg'):
            recips, srcs = ['clean', 'gen'], ['c', 'g']
        else:
            recips, srcs = ['clean', 'dirty', 'gen'], ['c', 'd', 'g']
        names = {'clean': 'Clean', 'dirty': 'Dirty', 'gen': 'General'}
        src_names = {k: v for k, v in {'c': 'clean', 'd': 'dirty', 'g': 'general'}.items() if k in srcs}
        ns = len(srcs)
        rows = []
        for meas, c in [('pat', 'cnt'), ('cite', 'cit')]:
            for r in recips:
                cols = [f'spill_shock_{c}_{r}_{x}' for x in srcs]
                v = ks[['BLS_Industry', 'period'] + cols].copy()
                v[cols] = v[cols].fillna(0.0)
                v.columns = ['BLS_Industry', 'period'] + srcs
                for lab, d in [('all', v)] + [(int(p), g) for p, g in v.groupby('period')]:
                    tot = d[srcs].sum()
                    pooled = tot / tot.sum()
                    ind = d.groupby('BLS_Industry')[srcs].sum()
                    ind = ind[ind.sum(axis=1) > 0]
                    avg = ind.div(ind.sum(axis=1), axis=0).mean()
                    for x in srcs:
                        rows.append(dict(measure=meas, recipient=r, period=lab, source=src_names[x], pooled=pooled[x],
                                         sector_avg=avg[x], received=tot.sum()))
        out = pd.DataFrame(rows)
        out.to_csv(outdir / 'ks_shares.csv', index=False)
        hdr = [r'\begin{tabular}{l' + 'c' * (2 * ns) + '}', r'\toprule',
               r' & \multicolumn{%d}{c}{Pooled} & \multicolumn{%d}{c}{Sector average} \\' % (ns, ns),
               r'\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}' % (ns + 1, ns + 2, 2 * ns + 1),
               'Recipient & ' + ' & '.join(['From ' + v for v in src_names.values()] * 2) + r' \\', r'\midrule']
        lines = list(hdr)
        for meas, title in [('pat', 'Patent counts'), ('cite', 'Citation-weighted')]:
            lines.append(r'\multicolumn{%d}{l}{\textit{' % (2 * ns + 1) + title + r'}} \\')
            for r in recips:
                d = out[(out['measure'] == meas) & (out['recipient'] == r) & (out['period'] == 'all')].set_index('source')
                lines.append(names[r] + ' & ' + ' & '.join(f"{d.loc[x, 'pooled']:.3f}" for x in src_names.values()) + ' & '
                             + ' & '.join(f"{d.loc[x, 'sector_avg']:.3f}" for x in src_names.values()) + r' \\')
            lines.append(r'\midrule')
        lines[-1] = r'\bottomrule'; lines.append(r'\end{tabular}')
        with open(outdir / 'table_ks_shares.tex', 'w') as f:
            f.write('\n'.join(lines) + '\n')
        periods = sorted(p for p in out['period'].unique() if p != 'all')
        lines = [r'\begin{tabular}{ll' + 'c' * len(periods) + '}', r'\toprule',
                 'Recipient & Source & ' + ' & '.join(str(p) for p in periods) + r' \\', r'\midrule']
        for meas, title in [('pat', 'Patent counts'), ('cite', 'Citation-weighted')]:
            lines.append(r'\multicolumn{%d}{l}{\textit{%s (pooled shares)}} \\' % (len(periods) + 2, title))
            for r in recips:
                for x in src_names.values():
                    d = out[(out['measure'] == meas) & (out['recipient'] == r) & (out['source'] == x)].set_index('period')
                    lines.append((names[r] if x == 'clean' else '') + f' & {x} & ' + ' & '.join(f"{d.loc[p, 'pooled']:.3f}" for p in periods) + r' \\')
            lines.append(r'\midrule')
        lines[-1] = r'\bottomrule'; lines.append(r'\end{tabular}')
        with open(outdir / 'table_ks_shares_bins.tex', 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(out[out['period'] == 'all'].round(3).to_string())
        return out


    def Spillover_Shocks(self, sfx='', asinh_pat=False):
        """""
        Knowledge-spillover shocks received by each industry and recipient type, by source type and year.
        sfx selects the files: '' (clean/dirty/general, 4-digit), '_c3' (3-digit), '_cg' (clean/general only, 4-digit),
        '_cg_c3' (clean/general only, 3-digit).

        Output: Clean Data/KS_Shocks{sfx}.pkl
        """""
        spill_net_df = pd.read_pickle(f'{self.Directory}/Clean Data/Spill_Net{sfx}.pkl')
        tech_pat_df = pd.read_pickle(f'{self.Directory}/Clean Data/Tech_Pat{sfx}.pkl')
        if asinh_pat:
            tech_pat_df[['pat_count', 'pat_cites']] = np.arcsinh(tech_pat_df[['pat_count', 'pat_cites']])
        ind_pat_shares_df = pd.read_pickle(f'{self.Directory}/Clean Data/Ind_Pat_Shares_Pre{sfx}.pkl')
        ind_pat_shares_df = ind_pat_shares_df[ind_pat_shares_df['type'] != 'all']
        
        ind_pat_shares_df['tech'] = ind_pat_shares_df['cpc_subclass'] + '_' + ind_pat_shares_df['type'].str[:1]
        
        KS_df = pd.merge(spill_net_df,
                        ind_pat_shares_df.rename(columns={'tech': 'i'}), 
                        on='i', how='inner')
        
        KS_df['BLS_tech_loading_cnt'] = KS_df['cpc_pat_share'] * KS_df['S_ijk']
        KS_df['BLS_tech_loading_cit'] = KS_df['cpc_cite_share'] * KS_df['S_ijk']
                
        KS_df['total_BLS_tech_loading_cnt'] = KS_df.groupby(['BLS_Industry', 'type', 'j', 'k'])['BLS_tech_loading_cnt'].transform('sum')
        KS_df['total_BLS_tech_loading_cit'] = KS_df.groupby(['BLS_Industry', 'type', 'j', 'k'])['BLS_tech_loading_cit'].transform('sum')
        #Diagonal already removed
        KS_df = KS_df[['BLS_Industry', 'type', 'j', 'k', 'total_BLS_tech_loading_cnt', 'total_BLS_tech_loading_cit']].drop_duplicates()
        
        KS_df = KS_df.merge(tech_pat_df.rename(columns={'tech': 'j', 'year': 'spill_send_year'}), 
                            on='j', how='inner')
        
        KS_df['year'] = KS_df['spill_send_year'] + KS_df['k']
        KS_df['src_type'] = KS_df['j'].str[-1]
        
        KS_df['BLS_tech_receipt_cnt'] = KS_df['total_BLS_tech_loading_cnt'] * KS_df['pat_count']
        KS_df['BLS_tech_receipt_cit'] = KS_df['total_BLS_tech_loading_cit'] * KS_df['pat_cites']
        
        KS_df['spill_shock_cnt'] = KS_df.groupby(['BLS_Industry', 'type', 'src_type', 'year'])['BLS_tech_receipt_cnt'].transform('sum')
        KS_df['spill_shock_cit'] = KS_df.groupby(['BLS_Industry', 'type', 'src_type', 'year'])['BLS_tech_receipt_cit'].transform('sum')
        
        KS_df = KS_df[['BLS_Industry', 'type', 'src_type', 'year', 'spill_shock_cnt', 'spill_shock_cit']].drop_duplicates()
        
        KS_wide = KS_df.pivot(index=['BLS_Industry', 'year'], columns=['type', 'src_type'], 
                       values=['spill_shock_cnt', 'spill_shock_cit'])

        KS_wide.columns = [f'{v}_{t}_{s}' for v, t, s in KS_wide.columns]
        KS_df = KS_wide.reset_index()
        
        KS_df.to_pickle(f"{self.Directory}/Clean Data/KS_Shocks{sfx}{'_asinh' if asinh_pat else ''}.pkl")
        return KS_df


    def KS_CleanAgg(self, sfx='_cg', asinh_pat=False):
        """""
        Spillovers received by all clean patents (one aggregate recipient), by year:
        R_y = sum_k sum_j S_clean,j,k * Pat_j,y-k, with S from the clean-citing-patent network of the pre-sample window.

        Output: Clean Data/KS_CleanAgg{sfx}.pkl  (year, recv_cnt, recv_cit)
        """""
        D = f'{self.Directory}/Clean Data/'
        sn = pd.read_pickle(D + f'Spill_Clean_Net{sfx}.pkl')
        tp = pd.read_pickle(D + f'Tech_Pat{sfx}.pkl').rename(columns={'tech': 'j', 'year': 'send_year'})
        if asinh_pat:
            tp[['pat_count', 'pat_cites']] = np.arcsinh(tp[['pat_count', 'pat_cites']])
        m = sn.merge(tp, on='j', how='inner')
        m['year'] = m['send_year'] + m['k'].astype(int)
        m['recv_cnt'] = m['S_jk'] * m['pat_count']
        m['recv_cit'] = m['S_jk'] * m['pat_cites']
        out = m.groupby('year', as_index=False)[['recv_cnt', 'recv_cit']].sum()
        out.to_pickle(D + f"KS_CleanAgg{sfx}{'_asinh' if asinh_pat else ''}.pkl")
        return out


    def KS_Recv(self, sfx='', asinh_pat=False):
        """""
        Spillovers received by recipient technology class, type and year: for each recipient (class, type) the
        citation-share-weighted patenting of its source technologies, recv(i, y) = sum_j S_ij,k * Pat_j,y-k.
        The class-level ratio recv(clean) / recv(clean + dirty + gen) is formed in UDG_Instrument.

        Output: Clean Data/KS_Recv.pkl  (columns cpc_subclass, year, recv_{cnt,cit}_{clean,dirty,gen})
        """""
        D = f'{self.Directory}/Clean Data/'
        sn = pd.read_pickle(D + f'Spill_Net{sfx}.pkl')
        tp = pd.read_pickle(D + f'Tech_Pat{sfx}.pkl').rename(columns={'tech': 'j', 'year': 'send_year'})
        if asinh_pat:   # source patenting as asinh(patents), so that large technologies do not dominate
            tp[['pat_count', 'pat_cites']] = np.arcsinh(tp[['pat_count', 'pat_cites']])
        parts = []
        for k, sub in sn.groupby('k'):
            m = sub[['i', 'j', 'S_ijk']].merge(tp, on='j', how='inner')
            m['year'] = m['send_year'] + int(k)
            m['cnt'] = m['S_ijk'] * m['pat_count']
            m['cit'] = m['S_ijk'] * m['pat_cites']
            parts.append(m.groupby(['i', 'year'], as_index=False)[['cnt', 'cit']].sum())
            del m
        r = pd.concat(parts, ignore_index=True).groupby(['i', 'year'], as_index=False)[['cnt', 'cit']].sum()
        r['cpc_subclass'] = r['i'].str[:-2]
        r['type'] = r['i'].str[-1].map({'c': 'clean', 'd': 'dirty', 'g': 'gen'})
        wide = r.pivot(index=['cpc_subclass', 'year'], columns='type', values=['cnt', 'cit']).fillna(0.0)
        wide.columns = [f'recv_{v}_{t}' for v, t in wide.columns]
        wide = wide.reset_index()
        wide.to_pickle(D + f"KS_Recv{sfx}{'_asinh' if asinh_pat else ''}.pkl")
        return wide


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
        sfx = '_c3' if src.endswith('_c3') else ''   # 3-digit CPC class version of the government / spillover shocks
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
                kr = pd.read_pickle(D + f'KS_Recv_cg{sfx}{lnsfx}.pkl')   # clean / general types only
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
            df = pd.read_pickle(D + f'Govt_Shocks{sfx}{lnsfx}.pkl')
            num, den = df[f'{n}_govt_shock_clean'], df[f'{n}_govt_shock_gen']
        elif src == 'ks':
            df = pd.read_pickle(D + f'KS_Shocks_cg{sfx}{lnsfx}.pkl')   # clean / general types only
            num = df[[x for x in df.columns if x.startswith(f'spill_shock_{c}_clean_')]].fillna(0.0).sum(axis=1)
            den = df[[x for x in df.columns if x.startswith(f'spill_shock_{c}_gen_')]].fillna(0.0).sum(axis=1)
        elif src == 'kswt':
            df = pd.read_pickle(D + f'KS_Shocks_cg{sfx}{lnsfx}.pkl')
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


    def UDG_OwnFirstStage(self, batch, BLS_year_start, Year_end, srcs=('gov', 'ks', 'kswt', 'rd')):
        """""
        Diagnostic first stages, before and after the network aggregation. For each instrument source and measure:
        OLS of the lagged own-sector clean share on the lagged own-sector instrument, and of the lagged network (net)
        clean share on the lagged network instrument, with sector and period effects and sector-clustered SEs.
        The sample is the G-baseline estimation sample (positive non-clean offset, sectors with a clean outcome), so the
        network rows are the first stages shown in the instrumented batches.

        Output: Results/Batches/<batch>/own_first_stage.csv, table_own_first_stage.tex
        """""
        outdir = self.Directory / 'Results' / 'Batches' / batch
        outdir.mkdir(parents=True, exist_ok=True)
        reg_df, _ = self.UDG_Panel(BLS_year_start, Year_end, 5, ivs=tuple(srcs))
        rows = []
        for meas, suf in [('pat', 'count'), ('cite', 'cites')]:
            for src in srcs:
                for lvl in ['own', 'net']:
                    y, z = f'{lvl}_G_{meas}_lag', f'{lvl}_Z{src}_{meas}_lag'
                    fs, n = self.UDG_FirstStage(reg_df, [y], [z], [], f'clean_pat_{suf}', f'pat_{suf}_nc')
                    r = fs[lvl]
                    rows.append(dict(measure=meas, source=src, level=lvl, coef=r['coef'], se=r['se'], p=r['p'],
                                     F=r['F'], pF=r['pF'], pr2=r['pr2'], N=n))
        out = pd.DataFrame(rows)
        out.to_csv(outdir / 'own_first_stage.csv', index=False)
        names = {'gov': 'Government patents', 'ks': 'Spillovers, all sources', 'kswt': 'Spillovers, within type',
                 'rd': 'State R\\&D prices', 'rdcf': 'R\\&D prices, clean firms', 'govcl': 'Gov. clean patenting (aggregate)', 'ksagg': 'Clean spillovers (aggregate)', 'govr': 'Gov. clean / gov. all ratio', 'govrn': 'Gov. ratio, renormalised', 'ksr': 'Spillover clean / all ratio', 'ksrn': 'Spillover ratio, renormalised', 'gov_c3': 'Government patents (3-digit)', 'ks_c3': 'Spillovers (3-digit)'}
        lines = [r'\begin{tabular}{llcccccc}', r'\toprule',
                 r' & & \multicolumn{3}{c}{Patent counts} & \multicolumn{3}{c}{Citations} \\',
                 r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
                 'Instrument & Level & Coefficient (SE) & $F$ & Partial $R^2$' + ' & Coefficient (SE) & $F$ & Partial $R^2$' + r' \\', r'\midrule']
        for src in srcs:
            for lvl, lname in [('own', 'Own sector'), ('net', 'Network (net)')]:
                cells = []
                for meas in ['pat', 'cite']:
                    r = out[(out['measure'] == meas) & (out['source'] == src) & (out['level'] == lvl)].iloc[0]
                    cells += [f"{r.coef:.3f}{gpf.get_stars(r.p)} ({r.se:.3f})", f"{r.F:.1f}{gpf.get_stars(r.pF)}", f"{r.pr2:.3f}"]
                lines.append((names.get(src, src) if lvl == 'own' else '') + f' & {lname} & ' + ' & '.join(cells) + r' \\')
            lines.append(r'\midrule')
        lines[-1] = r'\bottomrule'; lines.append(r'\end{tabular}')
        with open(outdir / 'table_own_first_stage.tex', 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(out.round(3).to_string())
        return out


    def UDG_Run(self, batch, specs, BLS_year_start, Year_end, description=''):
        """""
        Run a batch of specifications. Each spec is a dict with a unique 'id' and any overrides
        of Processor.UDG_BASE. Re-running a batch replaces its rows in the registry.

        Output: Results/registry.csv (one row per spec)
                Results/Batches/<batch>/table_<measure>_<panel>.tex, models.txt, specs.csv
        """""
        outdir = self.Directory / 'Results' / 'Batches' / batch
        outdir.mkdir(parents=True, exist_ok=True)
        panels = {}
        rows, models, fulls = [], {}, []
        g_sectors = {}
        ivs = tuple(sorted({sp['iv'] for sp in specs if str(sp.get('iv')).replace('_c3', '').replace('_asinh', '') in ('gov', 'ks', 'kswt', 'rd', 'rdcf', 'govr', 'govrn', 'ksr', 'ksrn', 'govcl', 'ksagg')}))

        for sp in specs:
            spec = {**self.UDG_BASE, **sp}
            key = (spec['bin_len'], spec['normalise'], spec['shrink'], spec['annual'], spec['iv_wins'], spec['clean'], spec['fixed_w'])
            if key not in panels:
                panels[key] = self.UDG_Panel(BLS_year_start, Year_end, key[0], key[1], key[2], ivs=ivs,
                                             annual=key[3], iv_wins=key[4], clean=key[5], fixed_w=key[6])
            reg_df, kappas = panels[key]

            sh, meas = spec['share'], spec['measure']
            suf = {'pat': 'count', 'cite': 'cites'}[meas]
            y_col = f'clean_pat_{suf}'
            if sh == 'A':
                y_col = f'asinh_clean_pat_{suf}'
                # restrict to the sectors that enter the G baseline (positive non-clean offset, some clean outcome)
                if meas not in g_sectors:
                    if (5, False, True, False, None, 'all', False) not in panels:
                        panels[(5, False, True, False, None, 'all', False)] = self.UDG_Panel(BLS_year_start, Year_end, 5, False, True, ivs=ivs)
                    gb = panels[(5, False, True, False, None, 'all', False)][0]
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
        reg_path = self.Directory / 'Results' / 'registry.csv'
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


    def UDG_Summary(self, path, rows, cols):
        """""
        Compact summary table from the registry: one row per (specification, term), one column per outcome.
        rows: list of (label, short, slot) with slot in {net, up, down, lag}; cols: list of (label, id prefix, measure).
        The spec id is <prefix>_<measure>_<short>. Non-convergent specifications are shown as 'not convergent'.
        """""
        reg = pd.read_csv(self.Directory / 'Results' / 'registry.csv').set_index('spec_id')
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


    def UDG_InstrumentSummary(self, path, instruments):
        """""
        One row per instrument: network first-stage F and second-stage estimate for the baseline net specification,
        with the network-mass x period controls and with fixed first-bin network weights. Counts and citations panels.
        instruments: list of (label, id prefix).
        """""
        reg = pd.read_csv(self.Directory / 'Results' / 'registry.csv').set_index('spec_id')
        tex = lambda t: str(t).replace('_', r'\_')
        variants = [('', 'Baseline'), ('mtrend', 'Network-mass $\\times$ period'), ('fixw', 'Fixed weights')]
        lines = [r'\begin{tabular}{l' + 'c' * 6 + '}', r'\toprule',
                 r' & \multicolumn{3}{c}{First-stage $F$} & \multicolumn{3}{c}{Second-stage estimate (SE)} \\',
                 r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
                 'Instrument & ' + ' & '.join(v[1] for v in variants) * 1 + ' & ' + ' & '.join(v[1] for v in variants) + r' \\', r'\midrule']
        for meas, title in [('pat', 'Patent counts'), ('cite', 'Citation-weighted patents')]:
            lines.append(r'\multicolumn{7}{l}{\textit{' + title + r'}} \\')
            first = instruments[0][1]
            refs = []
            for key in ('noiv', 'noivmt', 'noivfw'):
                sid = f'{first}_{meas}_{key}'
                r = reg.loc[sid]
                refs.append(f"{r['coef_net']:.2f}{gpf.get_stars(r['p_net'])} ({r['se_net']:.2f})")
            lines.append('No instrument (PPML-type estimate) & & & & ' + ' & '.join(refs) + r' \\')
            for label, tag in instruments:
                fs, ss = [], []
                for key, _ in variants:
                    sid = f"{tag}_{meas}_{key or 'base'}"
                    if sid not in reg.index:
                        fs.append(''); ss.append(''); continue
                    r = reg.loc[sid]
                    fs.append(f"{r['fs_F_net']:.1f}{gpf.get_stars(r['fs_pF_net'])}" if np.isfinite(r['fs_F_net']) else '')
                    if r['converged'] == 0:
                        ss.append('not conv.')
                    else:
                        ss.append(f"{r['coef_net']:.1f}{gpf.get_stars(r['p_net'])} ({r['se_net']:.1f})")
                lines.append(tex(label) + ' & ' + ' & '.join(fs) + ' & ' + ' & '.join(ss) + r' \\')
            lines.append(r'\midrule')
        lines[-1] = r'\bottomrule'
        lines.append(r'\end{tabular}')
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

        Output: Results/Batches/<batch>/table_corr.tex
        """""
        outdir = self.Directory / 'Results' / 'Batches' / batch
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

    






    