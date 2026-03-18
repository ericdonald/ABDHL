"""""""""""
Executor

Notes: This file executes the code for "Transition to Green Technology along the Supply Chain".
    
"""""""""""

import Processor as p


# ----------------------------------------------------------------

# Define project objects.

# ----------------------------------------------------------------

P = p.Processor()
Year_start, Year_end = (2012, 2022)

# ----------------------------------------------------------------

# Run project methods.

# ----------------------------------------------------------------

# ---------- #
# Clean Data #
# ---------- #
P.Cleaner(Year_start, Year_end, 0)


# ---------------- #
# IO Change Graphs #
# ---------------- #
P.IO_Change(Year_start, Year_end)


# --------------------- #
# Directional Incentive #
# --------------------- #
P.Up_Down_Green(Year_start, Year_end)


# ----------------------- #
# Record Package Versions #
# ----------------------- #
packages = ["matplotlib", "numpy", "openpyxl", "pandas", "statsmodels"]
P.write_package_versions(packages)




