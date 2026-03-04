# Replication Package <br> [Transition to Clean Technology along the Supply Chain](https://www.ericdonald.com/research/transition-to-green-technology-along-the-supply-chain)

## Data Sources:

Below is the list of all data sources required for replication. The first group are those programmatically retrieved via APIs or direct download, the second group are those contained in the Raw Data folder, and the third group are those that require the user to have a license. The links below are for reference only; a user does not need to visit these sites to extract the data.

### API/Web Acessible:

- EPA [Emissions by Sector](https://catalog.data.gov/dataset/2012-2022-national-level-greenhouse-gas-emission-totals-by-industry)
- Census [NAICS Concordances](https://www.census.gov/naics/?68967)
- PatentsView
  - [CPC Codes](https://patentsview.org/download/data-download-tables)
  - [Applications](https://patentsview.org/download/data-download-tables)
  - [Citations](https://patentsview.org/download/data-download-tables)

### Contained in Raw Data:

- BLS [Input-Output Matrix](https://www.bls.gov/emp/data/input-output-matrix.htm)
- Crosswalks of Patents to Firms from [Kogan et al. (2017)](https://github.com/KPSS2017/Technological-Innovation-Resource-Allocation-and-Growth-Extended-Data) and [Arora et al. (2021)](https://zenodo.org/records/13619821)

### Requires License:
- [Compustat](https://wrds-www.wharton.upenn.edu/)

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

## Description of Code:

## List of Tables and Figures:
