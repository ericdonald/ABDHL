# Replication Package <br> [Transition to Clean Technology along the Supply Chain](https://www.ericdonald.com/research/transition-to-green-technology-along-the-supply-chain)

## Data Sources:

Below is the list of all data sources required for replication. All necessary data files (except those requiring a license) are included in the Raw Data folder, so the replication code can be run immediately with `API=0`. Setting `API=1` will re-download the data from the original sources, which may produce small numerical differences if the underlying databases have been updated since the archived data was collected.

The first group are those that can be programmatically retrieved via APIs or direct download, the second group are those contained exclusively in the Raw Data folder and do not make use of an API, and the third group are those that require the user to have a license. The links below are for reference only; a user does not need to visit these sites to extract the data.

To make use of the API commands, the user will need to make a `.keys` file with the following lines:

```
USPTO_API = XX
WRDS_USERNAME = XX
WRDS_PASSWORD = XX
```

where `XX` is the user's API key for the relevant data source.

### API/Web Accessible:

- EPA [Emissions by Sector](https://catalog.data.gov/dataset/2012-2022-national-level-greenhouse-gas-emission-totals-by-industry)
- Census [NAICS Concordances](https://www.census.gov/naics/?68967)
- PatentsView
  - [CPC Codes](https://patentsview.org/download/data-download-tables)
  - [Applications](https://patentsview.org/download/data-download-tables)
  - [Citations](https://patentsview.org/download/data-download-tables)
  - [Inventors](https://data.uspto.gov/bulkdata/datasets/pvgpatdis)
  - [Location Crosswalk](https://data.uspto.gov/bulkdata/datasets/pvgpatdis)

### Contained in Raw Data:

- BLS [Input-Output Matrix](https://www.bls.gov/emp/data/input-output-matrix.htm)
- Crosswalks of Patents to Firms from [Kogan et al. (2017)](https://github.com/KPSS2017/Technological-Innovation-Resource-Allocation-and-Growth-Extended-Data), [Autor et al. (2020)](https://www.ddorn.net/data.htm), and [Arora et al. (2021)](https://zenodo.org/records/13619821)

### Requires License:
- WRDS
  - [Compustat](https://wrds-www.wharton.upenn.edu/pages/get-data/compustat-capital-iq-standard-poors/)
  - [CRSP](https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/)

## Software Requirements:

### Python

All of the replication codes run on Python `3.12.11`. Prior to running the codes, install the following packages:

| Package | Version |
|---------|---------|
| matplotlib | 3.10.8 |
| numpy | 2.4.2 |
| openpyxl | 3.1.5 |
| pandas | 3.0.0 |
| statsmodels | 0.14.6 |
| scipy | 1.18.1 |
| linearmodels | 7.0 |
| rapidfuzz | 3.14.6 (customer-name matching) |
| psycopg2 | 2.9.12 (WRDS connection; do not install the `wrds` package, it downgrades pandas) |

## Description of Code:

### Setup Instructions

Before running the code, create the following folders in the repository root:

```
Raw Data/
Clean Data/
Results/
Results/Tables/
Results/Figures/
```

Download the [raw data](https://github.com/ericdonald/ABDHL/releases/download/v1.0.0/Raw.Data.zip), unzip, and place the file(s) directly in `Raw Data/`.

## List of Tables and Figures:

Run `Code/Executor.py`; it writes every exhibit below to `Results/Tables` and `Results/Figures`. The draft `Results/empirical_draft.tex` inputs them.

| Exhibit | File | Function |
|---------|------|----------|
| Figure 1: input-output change vs. emissions-intensity reduction | `Figures/IO_Change_wls_sqrt` | `Processor.IO_Change` |
| Table 1: reduced form, upstream and downstream | `Tables/Reduced_Form_UpDown` | `Processor.UpDown_Table` |
| Table 2: customer IV | `Tables/Customer_IV` | `Customer_Tables.Tables` |
| Figure 2: timing of both directions | `Figures/timing_both` | `Customer_Tables.Tables` |
| Appendix: IO-change slopes | `Tables/IO_Change_Slopes` | `Processor.IO_Change` |
| Appendix: reduced-form robustness | `Tables/Reduced_Form_Robust` | `Processor.UpDown_Robust` |
| Appendix: clean and dirty CPC codes | `Tables/CPC_Codes` | `Processor.CPC_Codes_Table` |
| Appendix: summary statistics, top network sectors | `Tables/Summary_Stats`, `Tables/Top_Network_Sectors` | `Processor.Summary_Stats` |
| Appendix: match coverage | `Tables/Match_Coverage` | `Processor.Match_Coverage` |
| Appendix: shrinkage and other customer-IV choices | `Tables/Robust_Lambda`, `Tables/Robust_Other` | `Customer_Tables.Tables` |
