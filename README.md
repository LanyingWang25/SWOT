## SWOT
Inference of cell-type composition and single-cell spatial maps from spatial transcriptomics data with SWOT


## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)



## Overview
This project is a python implementation of SWOT, which is introduced in the paper "Inference of cell-type composition and single-cell spatial maps from spatial transcriptomics data with SWOT"

SWOT is a spatially weighted optimal transport method that integrates single-cell RNA sequencing data with spatial transcriptomics data for the inference of cell-type composition and single-cell spatial maps. It contains two principal components: an optimal transport module for learning a cell-to-spot mapping and a cell mapping module for estimating cell-type proportions, cell numbers, and cell coordinates per spot.

All the processed data required to produce figures presented in the manuscript can be found at Zenodo: https://doi.org/10.5281/zenodo.16737576.

## Installation
We suggest using a separate conda environment for installing SWOT. SWOT can be run in Windows Powershell or Linux Bash shell.

1. Create conda environment and install `SWOT` package.

```bash
conda create -y -n SWOT_env python=3.9

conda activate SWOT_env
```

2. Clone this repository and cd into it as below.

```bash
git clone https://github.com//GaoLabXDU/SWOT.git
```

3. Install requirements directly.

```bash
cd SWOT

pip install -r requirements.txt
```



## Usage

The SWOT algorithm is used in a spot-resolution spatial transcriptomics data to infer the cell-type composition and single-cell spatial map of a given tissue. 
SWOT inputs a gene expression matrix with cell type information of scRNA-seq data and a gene expression matrix with spatial coordinates of spatial transcriptomics data. 


### Run Example
An example for pancreatic ductal adenocarcinoma (PDAC) dataset can be found under the directory "Example/", and the example data are under the directory "Data_PDAC/".

```bash
cd Example

unzip Data_PDAC.zip

python PDAC.py
```
The results of SWOT are saved in "SWOT_files/".
* "CellMapping/": inferred results
  * "Celltype_composition.csv", represents the estimated cell-type composition matrix with rows being spots and columns being cell types. 
  * "Cell_maps_xy.csv", represents the estimated spatial coordinates with rows being estimated cells and columns being meta information.
  * "Cell_maps_exp.csv", represents the gene expression profiles of single-cell spatial maps with rows being genes and columns being cells.
* "OptimalTransport/": intermediate results
  * "D_cell.csv", represents the gene expression distance among cells in scRNA-seq data. 
  * "D_pos.csv", represents the spatial coordinates distance among spots in spatial transcriptomics data.
  * "D_spot.csv", represents the gene expression distance among spots in spatial transcriptomics data.
  * "D_cell_spot.csv", represents the gene expression distance between cells and spots.
  * "TransportPlan.csv", represents the learned transport plan with rows being cells and columns being spots.
  * "spa_cost.csv", represents the spatially weighted distance matrix among spots in spatial transcriptomics data.

### Run your own data
When using your own data to run SWOT, you should provide as:
* The gene expression matrix of scRNA-seq and spatial transcriptomics data, rows represent genes and columns represent cells or spots, and saved as .csv format.
* The cell type labels matrix of scRNA-seq data, rows represent cells and columns represent cell type information having 'celltype' for labels, and saved as .csv format.
* The spatial coordinates matrix of spatial transcriptomics data, rows represent spots and columns represent coordinates information having 'X' and 'Y', and saved as .csv format.


