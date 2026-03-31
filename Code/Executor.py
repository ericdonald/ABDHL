"""""""""""
Executor

Notes: This file executes the code for "Transition to Green Technology along the Supply Chain".
    
"""""""""""

import Processor as p


# ----------------------------------------------------------------

# Define project objects.

# ----------------------------------------------------------------

P = p.Processor()
BLS_year_start, Year_start, Year_mid, Year_end = (1997, 2012, 2017, 2022)

# ----------------------------------------------------------------

# Run project methods.

# ----------------------------------------------------------------

# ---------- #
# Clean Data #
# ---------- #
P.Cleaner(BLS_year_start, Year_start, Year_mid, Year_end, 0)


# ---------------- #
# IO Change Graphs #
# ---------------- #
#P.IO_Change(Year_start, Year_mid, Year_end)


# --------------------- #
# Directional Incentive #
# --------------------- #
P.Up_Down_Green(BLS_year_start, Year_start, Year_mid, Year_end)


# ----------------------- #
# Record Package Versions #
# ----------------------- #
packages = ["matplotlib", "numpy", "openpyxl", "pandas", "statsmodels"]
P.write_package_versions(packages)




