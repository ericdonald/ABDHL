"""""""""""
Executor

Notes: This file executes the code for "Transition to Green Technology along the Supply Chain".
    
"""""""""""

import Processor as p


# ----------------------------------------------------------------

# Define project objects.

# ----------------------------------------------------------------

P = p.Processor()
Year_start, Year_end = (2012, 2020)

# ----------------------------------------------------------------

# Run project methods.

# ----------------------------------------------------------------

# ---------- #
# Clean Data #
# ---------- #
P.Cleaner(Year_start, Year_end)


# ---------------- #
# IO Change Graphs #
# ---------------- #
P.IO_Change(Year_start, Year_end)


# ----------------------- #
# Record Package Versions #
# ----------------------- #
packages = ["openpyxl", "pandas"]
P.write_package_versions(packages)




