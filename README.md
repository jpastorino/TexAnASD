# TexAnASD - Text Analytics for ASD Risk Gene Predictions
- IEEE BIBM 2019 - Workshop on Machine Learning and Artificial Intelligence in Bioinformatics and Medical Informatics
- San Diego, CA, USA, November 18 - 21, 2019
- Access the paper [here](https://ieeexplore.ieee.org/abstract/document/8983107).

## Table of Contents
1. [Code Files Description](#Code-Files-Description)
1. [Data Directory Description](#DAta-Directory-Description)
1. [Authors](#Authors)
1. [Citation](#Citation)

## Code Files Description
* **File**: `./auxiliary_func.py`: Auxiliary functions.
* **File**: `./config.json`: Configuration file.
* **File**: `./omim_main.ipynb`: Main Program
* **File**: `./omim_pipeline.py`: Functions to run the pipeline.
* **File**: `./omim.py`: Functions to retrieve and process OMIM data.
* **File**: `./param_tunning.csv`: Configuration file of the meta-parameter tuning experiments.


## Data Directory Description
### ForecASD Data
* **Directory**: `./data/forecASD`: This data was extracted or regenerated by using the code available at [ForecASD GitHub](https://github.com/LeoBman/forecASD)


### SFARI Data
**File**: `./data/sfari_gene/SFARI-Gene_genes_06-20-2019release_07-10-2019export.csv`: SFARI Gene Database. Can be downloaded [here](https://gene.sfari.org/database/human-gene/) *(download this dataset link)*.
 
 

### StringDB Data
**File**: `./data/stringdb/9606.protein.info.v11.0.txt`: STRING DB Protein Information. Can be downloaded [here](https://string-db.org/cgi/download.pl) 


### OMIM Data
Use the config file to configure the data loading.
* **Directory**: `./data/omim/genes`: Contains the data cache from reading the OMIM dataset.
* **Directory**: `./data/omim/processed`: Contains the processed data from the cache. 
* **File**: `./data/omim/mim2gene.txt`: Mapping between OMIM gene id and other systems. 

### TexAnASD Data
* **Directory**: `./data/texanasd`: Mainly output data.
* **Directory**: `./data/texanasd/computed_data`: Storage of temprary computations if required.
* **Directory**: `./data/texanasd/crossvalidation`: Cross-validation output data and logs. 
* **File**: `./data/texanasd/crossvalidation/crossvalidation_summary.csv`: stores the summary of AUC and runtimes for each cross-validation iteration and fold.
* **Directory**: `./data/texanasd/output/data`: Pipeline output. Stores the predicted scores, classes and ground truth labels for the test data.
* **File**: `./data/texanasd/output/main_summary.csv`: stores the results of the pipeline in terms of auc,accuracy,recall,precision,F1 and runtime.
* **File**: `./data/texanasd/output/parameter_tuning_summary.csv`: stores the results meta-parameter tuning experiments.
* **File**: `./data/texanasd/output/req_omim_ids.csv`: the list of ids to retrieve from OMIM Web Service and the mapping with other naming systems.
* **File**: `./data/texanasd/output/test_samples_labels.csv`: labels of the test samples
* **File**: `./data/texanasd/output/train_samples_labels.csv`: labels of the train samples


## Authors
* [Javier Pastorino](http://cse.ucdenver.edu/~pastorij)
* [Ashis Kumer Biswas](http://cse.ucdenver.edu/~biswasa)


- [Machine Learning Laboratory](http://ml.cse.ucdenver.edu)
- Computer Science and Engineering - University of Colorado, Denver

## Citation
Please if you want to reference our work in yours, please cite this paper as follows.

J. Pastorino and A. K. Biswas, "TexAnASD: Text Analytics for ASD Risk Gene Predictions," 2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), San Diego, CA, USA, 2019, pp. 1350-1357.
