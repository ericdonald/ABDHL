"""""""""""
Executor

Notes: This file executes the code for "Transition to Green Technology along the Supply Chain".
    
"""""""""""

import Processor as p


# ----------------------------------------------------------------

# Define project objects.

# ----------------------------------------------------------------

P = p.Processor()


# ----------------------------------------------------------------

# Run project methods.

# ----------------------------------------------------------------

# --------------- #
# IO Change Graph #
# --------------- #
P.IO_Change(2012, 2020)


# ----------------------- #
# Record Package Versions #
# ----------------------- #
packages = ["openpyxl", "pandas"]
P.write_package_versions(packages)




