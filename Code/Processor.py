"""""""""""
Processor Module

Notes: This file defines a class for processing the workflow of "Transition to Green Technology along the Supply Chain".

"""""""""""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
    
        Output: Results/Tables/sector_extremes_table.tex
                Results/Figures/Baseline_L1.png
                Results/Figures/Baseline_L2.png
                Results/Figures/Leontief_L1.png
                Results/Figures/Leontief_L2.png
                Results/Figures/Reduced_L1.png
                Results/Figures/Reduced_L2.png
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
        
        I = np.eye(J)
        LI_start = np.linalg.inv(I - IO_start)
        LI_end = np.linalg.inv(I - IO_end)
        
        #Cut oil and gas extraction as well as coal mining, then normalize
        IO_start_reduced = np.delete(np.delete(IO_start, [6, 7], axis=0), [6, 7], axis=1)
        IO_start_reduced = IO_start_reduced / IO_start_reduced.sum(axis=1, keepdims=True)
        
        IO_end_reduced = np.delete(np.delete(IO_end, [6, 7], axis=0), [6, 7], axis=1)
        IO_end_reduced = IO_end_reduced / IO_end_reduced.sum(axis=1, keepdims=True)

        
        diff = np.abs(IO_end - IO_start)
        tv_by_industry = 0.5 * diff.sum(axis=1)
        tv_sq_by_industry = 0.5 * ((diff**(2)).sum(axis=1))**(1/2)
        
        diff_LI = np.abs(LI_end - LI_start)
        tv_by_industry_LI = 0.5 * diff_LI.sum(axis=1)
        tv_sq_by_industry_LI = 0.5 * ((diff_LI**(2)).sum(axis=1))**(1/2)
        
        diff_reduced = np.abs(IO_end_reduced - IO_start_reduced)
        tv_by_industry_reduced = 0.5 * diff_reduced.sum(axis=1)
        tv_sq_by_industry_reduced = 0.5 * ((diff_reduced**(2)).sum(axis=1))**(1/2)
        
        IO_df = pd.DataFrame({
            "BLS_Industry": np.arange(1, J+1),
            "TV_distance": tv_by_industry,
            "TV_sq_distance": tv_sq_by_industry,
            "TV_distance_LI": tv_by_industry_LI,
            "TV_sq_distance_LI": tv_sq_by_industry_LI})
        
        IO_reduced_df = pd.DataFrame({
            "BLS_Industry": np.concatenate([np.arange(1, 7), np.arange(9, J+1)]),
            "TV_distance_reduced": tv_by_industry_reduced,
            "TV_sq_distance_reduced": tv_sq_by_industry_reduced})
        
        IO_df = IO_df.merge(IO_reduced_df,
                                    on='BLS_Industry',
                                    how='left')
        
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
        #naics2017_6_universe = sorted(NAICS_2012_2017_df['NAICS_2017'].dropna().unique())
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
        
        
        # --------- #
        # Crosswalk #
        # --------- #
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
        
        
        # ------------------ #
        # Allocate Emissions #
        # ------------------ #
        
        IO_df['CO2e_denom'] = IO_df.groupby(['EPA_Sector', 'Year'])['Value_Added'].transform("sum")
        IO_df['CO2e_integrand'] = IO_df['Value_Added'] * IO_df['CO2e'] / IO_df['CO2e_denom']
        IO_df['CO2e_Industry'] = IO_df.groupby(['BLS_Industry', 'Year'])['CO2e_integrand'].transform("sum")
        IO_df['CO2e_intensity_Industry'] = IO_df['CO2e_Industry'] / IO_df['Value_Added']
        
        IO_df = IO_df[['BLS_Industry', 'Year', 'TV_distance', 'TV_sq_distance', 'TV_distance_LI', 'TV_sq_distance_LI', 'TV_distance_reduced', 'TV_sq_distance_reduced', 'CO2e_intensity_Industry']].drop_duplicates()
        
        IO_wide_df = IO_df.pivot(index="BLS_Industry", columns="Year", values='CO2e_intensity_Industry')
        IO_wide_df = IO_wide_df.dropna()
        
        idx1 = IO_wide_df.index.to_numpy(dtype=int)
        idx0 = idx1 - 1
        
        LI_start_sub = LI_start[np.ix_(idx0, idx0)]
        LI_end_sub = LI_end[np.ix_(idx0, idx0)]
        
        CO2e_intensity_Industry_LI_start = LI_start_sub @ IO_wide_df[Year_start].to_numpy()
        CO2e_intensity_Industry_LI_end = LI_end_sub @ IO_wide_df[Year_end].to_numpy()
        
        IO_wide_df["dlog_CO2e_inten"] = np.log(IO_wide_df[Year_end]) - np.log(IO_wide_df[Year_start])
        IO_wide_df["dlog_CO2e_inten_LI"] = np.log(CO2e_intensity_Industry_LI_end) - np.log(CO2e_intensity_Industry_LI_start)
        IO_wide_df = IO_wide_df.reset_index()
        
        reg_df = pd.merge(IO_df[['BLS_Industry', 'TV_distance', 'TV_sq_distance', 'TV_distance_LI', 'TV_sq_distance_LI', 'TV_distance_reduced', 'TV_sq_distance_reduced']].drop_duplicates(),
                          IO_wide_df[['BLS_Industry', 'dlog_CO2e_inten', 'dlog_CO2e_inten_LI']].drop_duplicates(),
                          on='BLS_Industry',
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
        
        # ---------- #
        # Regression #
        # ---------- #
        X = sm.add_constant(reg_df['dlog_CO2e_inten'])
        y = reg_df['TV_distance']
        
        model = sm.OLS(y, X).fit()
        print(model.summary())
        
        beta0 = model.params['const']
        beta1 = -model.params['dlog_CO2e_inten']
        
        x = -reg_df['dlog_CO2e_inten'].to_numpy()
        y = y.to_numpy()
        y_hat = beta0 + beta1 * x
        
        # Two Metric
        y_sq = reg_df['TV_sq_distance']
        
        model_sq = sm.OLS(y_sq, X).fit()
        print(model_sq.summary())
        
        beta0_sq = model_sq.params['const']
        beta1_sq = -model_sq.params['dlog_CO2e_inten']
        
        y_sq = y_sq.to_numpy()
        y_hat_sq = beta0_sq + beta1_sq * x
        
        
        # ---------- #
        # Plot Graph #
        # ---------- #
        plt.figure(figsize=(8,6))
        plt.scatter(x, y, alpha=0.7, label="Industries")
        plt.plot(x, y_hat, color='red', linewidth=2, label="OLS fit")
        
        plt.xlabel("-Δ log(emissions intensity)")
        plt.ylabel("TV distance (input-share change)")
        plt.grid(alpha=0.3)
        plt.title("IO Table - 1 Norm")
        plt.legend()
        plt.savefig(f'{self.Directory}/Results/Figures/Baseline_L1.png')
        plt.show()
        
        # Square Root Metric
        plt.figure(figsize=(8,6))
        plt.scatter(x, y_sq, alpha=0.7, label="Industries")
        plt.plot(x, y_hat_sq, color='red', linewidth=2, label="OLS fit")
        
        plt.xlabel("-Δ log(emissions intensity)")
        plt.ylabel("2 Norm distance (input-share change)")
        plt.grid(alpha=0.3)
        plt.title("IO Table - 2 Norm")
        plt.legend()
        plt.savefig(f'{self.Directory}/Results/Figures/Baseline_L2.png')
        plt.show()


        # ----------------------------------------------------------------
    
        # Leontief inverse.
    
        # ----------------------------------------------------------------
        
        # ---------- #
        # Regression #
        # ---------- #
        X = sm.add_constant(reg_df['dlog_CO2e_inten_LI'])
        y = reg_df['TV_distance_LI']
        
        model = sm.OLS(y, X).fit()
        print(model.summary())
        
        beta0 = model.params['const']
        beta1 = -model.params['dlog_CO2e_inten_LI']
        
        x = -reg_df['dlog_CO2e_inten_LI'].to_numpy()
        y = y.to_numpy()
        y_hat = beta0 + beta1 * x
        
        # Two Metric
        y_sq = reg_df['TV_sq_distance_LI']
        
        model_sq = sm.OLS(y_sq, X).fit()
        print(model_sq.summary())
        
        beta0_sq = model_sq.params['const']
        beta1_sq = -model_sq.params['dlog_CO2e_inten_LI']
        
        y_sq = y_sq.to_numpy()
        y_hat_sq = beta0_sq + beta1_sq * x
        
        
        # ---------- #
        # Plot Graph #
        # ---------- #
        plt.figure(figsize=(8,6))
        plt.scatter(x, y, alpha=0.7, label="Industries")
        plt.plot(x, y_hat, color='red', linewidth=2, label="OLS fit")
        
        plt.xlabel("-Δ log(emissions intensity)")
        plt.ylabel("TV distance (input-share change)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.title("Leontief Inverse - 1 Norm")
        plt.savefig(f'{self.Directory}/Results/Figures/Leontief_L1.png')
        plt.show()
        
        # Square Root Metric
        plt.figure(figsize=(8,6))
        plt.scatter(x, y_sq, alpha=0.7, label="Industries")
        plt.plot(x, y_hat_sq, color='red', linewidth=2, label="OLS fit")
        
        plt.xlabel("-Δ log(emissions intensity)")
        plt.ylabel("2 Norm distance (input-share change)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.title("Leontief Inverse - 2 Norm")
        plt.savefig(f'{self.Directory}/Results/Figures/Leontief_L2.png')
        plt.show()
        
        
        # ----------------------------------------------------------------
    
        # Reduced and normalized IO table.
    
        # ----------------------------------------------------------------
        
        # ---------- #
        # Regression #
        # ---------- #
        X = sm.add_constant(reg_df['dlog_CO2e_inten'])
        y = reg_df['TV_distance_reduced']
        
        model = sm.OLS(y, X, missing='drop').fit()
        print(model.summary())
        
        beta0 = model.params['const']
        beta1 = -model.params['dlog_CO2e_inten']
        
        x = -reg_df['dlog_CO2e_inten'].to_numpy()
        y = y.to_numpy()
        y_hat = beta0 + beta1 * x
        
        # Two Metric
        y_sq = reg_df['TV_sq_distance_reduced']
        
        model_sq = sm.OLS(y_sq, X, missing='drop').fit()
        print(model_sq.summary())
        
        beta0_sq = model_sq.params['const']
        beta1_sq = -model_sq.params['dlog_CO2e_inten']
        
        y_sq = y_sq.to_numpy()
        y_hat_sq = beta0_sq + beta1_sq * x
        
        
        # ---------- #
        # Plot Graph #
        # ---------- #
        plt.figure(figsize=(8,6))
        plt.scatter(x, y, alpha=0.7, label="Industries")
        plt.plot(x, y_hat, color='red', linewidth=2, label="OLS fit")
        
        plt.xlabel("-Δ log(emissions intensity)")
        plt.ylabel("TV distance (input-share change)")
        plt.grid(alpha=0.3)
        plt.title("Reduced IO Table - 1 Norm")
        plt.legend()
        plt.savefig(f'{self.Directory}/Results/Figures/Reduced_L1.png')
        plt.show()
        
        # Square Root Metric
        plt.figure(figsize=(8,6))
        plt.scatter(x, y_sq, alpha=0.7, label="Industries")
        plt.plot(x, y_hat_sq, color='red', linewidth=2, label="OLS fit")
        
        plt.xlabel("-Δ log(emissions intensity)")
        plt.ylabel("2 Norm distance (input-share change)")
        plt.grid(alpha=0.3)
        plt.title("Reduced IO Table - 2 Norm")
        plt.legend()
        plt.savefig(f'{self.Directory}/Results/Figures/Reduced_L2.png')
        plt.show()
 
    
    
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



