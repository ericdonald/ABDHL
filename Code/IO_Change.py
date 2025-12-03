"""""""""""
New Supply Chains

Notes: This file plots Figure 1 of "Transition to Green Technology along the Supply Chain"

"""""""""""

import pandas as pd



# ----------------------------------------------------------------

# Unpack data sets.

# ----------------------------------------------------------------

# ------------ #
# BLS IO Table #
# ------------ #
# ----------------------- #
# EPA Emissions by Sector #
# ----------------------- #
# --------------- #
# GWP-100 Factors #
# --------------- #
CO2e = {'Methane': 28,
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


# ---------------- #
# NAICS Crosswalks #
# ---------------- #


# ----------------------------------------------------------------

# Build industry IO matrix.

# ----------------------------------------------------------------


# ----------------------------------------------------------------

# Make unified NAICS mapping.

# ----------------------------------------------------------------