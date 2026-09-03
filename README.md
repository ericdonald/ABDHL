# Replication Package <br> [Transition to Clean Technology along the Supply Chain](https://www.ericdonald.com/research/transition-to-green-technology-along-the-supply-chain)

## Data Sources:

Below is the list of all data sources required for replication. All necessary data files (except those requiring a license) are included in the Raw Data folder, so the replication code can be run immediately with `API=0`. Setting `API=1` will re-download the data from the original sources, which may produce small numerical differences if the underlying databases have been updated since the archived data was collected.

The first group are those that can be programmatically retrieved via APIs or direct download, the second group are those contained exclusively in the Raw Data folder and do not make use of API, and the third group are those that require the user to have a license. The links below are for reference only; a user does not need to visit these sites to extract the data.

To make use of the API commands, the user will need to make a `.keys` file with the following lines:

```
USPTO_API = XX
```

where `XX` is the user's API key for the relevant data source.

### API/Web Accessible:

- EPA [Emissions by Sector](https://catalog.data.gov/dataset/2012-2022-national-level-greenhouse-gas-emission-totals-by-industry)
- Census [NAICS Concordances](https://www.census.gov/naics/?68967)
- PatentsView
  - [Assignee](https://patentsview.org/download/data-download-tables)
  - [CPC Codes](https://patentsview.org/download/data-download-tables)
  - [Applications](https://patentsview.org/download/data-download-tables)
  - [Citations](https://patentsview.org/download/data-download-tables)
  - [Inventors](https://data.uspto.gov/bulkdata/datasets/pvgpatdis)
  - [Location Crosswalk](https://data.uspto.gov/bulkdata/datasets/pvgpatdis)

### Contained in [Raw Data](https://github.com/ericdonald/ABDHL/releases/download/v1.0.0/Raw.Data.zip):

- BLS [Input-Output Matrix](https://www.bls.gov/emp/data/input-output-matrix.htm)
- Crosswalks of Patents to Firms from [Kogan et al. (2017)](https://github.com/KPSS2017/Technological-Innovation-Resource-Allocation-and-Growth-Extended-Data) and [Arora et al. (2021)](https://zenodo.org/records/13619821)
- State-Level R&D Prices from [Lucking et al. (2019)](https://www.dropbox.com/s/d1nrtacxk6qke0a/spillovers_rep.zip?dl=0)

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

### Setup Instructions

Before running the code, create the following folders in the repository root:

```
Raw Data/
Clean Data/
Results/
Results/Tables/
Results/Figures/
```

Download the [raw data](), unzip, and place the file(s) directly in `Raw Data/`.

## List of Tables and Figures:
