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
        
        
        BLS_Crosswalk_df = pd.read_excel(f'{self.Directory}/Raw Data/BLS_Crosswalk.xlsx', sheet_name="Stubs")
        
        
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
        Y_start = np.sum(U_start_rev, 0)
        B_start = (U_start_rev[:-3,:] @ np.diag(Y_start**(-1))).T
        
        U_end_rev = USE_end_df.iloc[:, 1:-3].to_numpy()
        Y_end = np.sum(U_end_rev, 0)
        B_end = (U_end_rev[:-3,:] @ np.diag(Y_end**(-1))).T
        
        
        # -------------------------------------- #
        # Commodity x Industry Production Shares #
        # -------------------------------------- #
        
        
        # ------------------- #
        # Input-Output Matrix #
        # ------------------- #
        # ----------------------------------------------------------------

        # Make unified NAICS mapping.

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



