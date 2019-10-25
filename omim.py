###########################################################
# Author: J. Pastorino, Ashis K. Biswas
#
# Description: Methods to implement functionality to 
#              retrieve and process OMIM data.


import numpy as np
import pandas as pd
import json                  # For reading config. / and managing objects
import requests              # Fore reading the  API

import logging
import os
import ast

from time import time
import datetime
import pickle
import concurrent.futures

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

##########


def generateOMIM_ids(configOmimToGeneFile,configProteinInfoFile,configSFari, saveToFilename):

    logging.info("Generating OMIM id file.")
    t0=time()

    df_omim_to_gene = pd.read_csv(configOmimToGeneFile, sep="\t",comment='#', header=None)
    df_omim_to_gene.columns=["OMIM_ID","TYPE","ENTREZ_ID","HGNC","ENSEMBL"]
    df_omim_to_gene = df_omim_to_gene[df_omim_to_gene["TYPE"]=='gene']
    df_omim_to_gene.dropna(subset=["HGNC"],inplace=True)
    
    df_protein_info = pd.read_csv(configProteinInfoFile, sep="\t",comment='#')
    df_protein_info.dropna(subset=["preferred_name"],inplace=True)

    req_omim_ids = []

    for protein_idx,protein_row in df_protein_info.iterrows():
        t_hgnc    = protein_row["preferred_name"]
        t_protein = protein_row["protein_external_id"].split('.')[1]
        for omim_idx, omim_row in df_omim_to_gene[df_omim_to_gene["HGNC"]==t_hgnc].iterrows():
            t_entrez  = omim_row.ENTREZ_ID
            t_omim    = omim_row.OMIM_ID
            t_ensembl = omim_row.ENSEMBL
            req_omim_ids.append({"HGNC":t_hgnc,
                                 "ENTREZ_ID":int(t_entrez),
                                 "OMIM_ID":t_omim,
                                 "ENSEMBL_ID":t_ensembl,
                                 "ENSP":t_protein
                                })

    t1=time()
    
    logging.info("Done in {:3.2f}sec".format(t1-t0))
    df_req_omim_ids = pd.DataFrame(req_omim_ids, columns=["HGNC","ENTREZ_ID","OMIM_ID","ENSEMBL_ID","ENSP"])
    logging.info("Done. There are {} OMIM ids to retrieve.".format(df_req_omim_ids.shape[0]))
    
    
    t0=time()
    logging.info("Reading ASD Class Labels...")
    df_sfari_gene = pd.read_csv(configSFari)
    df_req_omim_ids["ASD"] = [ int(gene in df_sfari_gene['gene-symbol'].values) for gene in df_req_omim_ids["HGNC"].values]
    
    t1=time()
    logging.info("Done in {:3.2f}sec".format(t1-t0))
    logging.info("Saving to file...")
    df_req_omim_ids.to_csv(saveToFilename,index=False)
    logging.info("Done!")
    
    return df_req_omim_ids
    
    
# https://api.omim.org/api/entry/search?search=approved_gene_symbol:ARF5&apiKey=<apiKey>&format=json
# https://api.omim.org/api/entry?mimNumber=<omimid>&apiKey=<apiKey>&format=json&include=all

def getAPIRequest(omim_id, config):
    return  config["omim"]["url"]+ \
            "?mimNumber="+str(omim_id)+ \
            "&apiKey="+config["omim"]["apiKey"]+ \
            "&format="+config["omim"]["format"]+\
            "&include="+config["omim"]["include"]



def readOmimDataFromWeb(df_omim_ids, OMIM_GENE_PATH, config):
    logging.info("Reading omim data from the web.")
    for i,gene in df_omim_ids.iterrows():
        url = getAPIRequest(gene["OMIM_ID"], config)
        filename = OMIM_GENE_PATH+"/"+gene["HGNC"]+".json"

        resp = requests.get(url)
        if resp.status_code == 200:
            file = open(filename,"w")
            file.write( str(resp.json()))
            file.close()
        else:
            logging.warning("ERROR:"+str(resp.status_code))
    
    logging.info("Done!")
    
    
def processOmimDataFiles(df_omim_ids, OMIM_GENE_PATH, saveToFilename ):
    
    logging.info("Processing OMIM data files to generate a summary file.")
    
    omim_info = []
    cols = []
    
    for i,gene in df_omim_ids.iterrows():
        filename = OMIM_GENE_PATH+"/"+gene["HGNC"]+".json"
        file = open(filename,"r")
        d = ast.literal_eval(file.read())
        file.close()

        new_val={}
        
        new_val["gene"]=gene["HGNC"]
        new_val["text"]=""
        
        for i in range(len(d["omim"]["entryList"][0]["entry"]["textSectionList"])):
            new_val["text"] +=d["omim"]["entryList"][0]["entry"]["textSectionList"][i]["textSection"]["textSectionContent"]+"\n"

        ### Gene MAP
        try:
            new_val["geneMapExists"] = d["omim"]["entryList"][0]["entry"]["geneMapExists"]
            try:    new_val["chromosome"] = int(d["omim"]["entryList"][0]["entry"]["geneMap"]["chromosome"])
            except: new_val["chromosome"] = np.nan

            try:    new_val["locationStart"] = int(d["omim"]["entryList"][0]["entry"]["geneMap"]["chromosomeLocationStart"])
            except: new_val["locationStart"] = np.nan

            try:    new_val["locationEnd"] = int(d["omim"]["entryList"][0]["entry"]["geneMap"]["chromosomeLocationEnd"])
            except: new_val["locationEnd"] = np.nan
        except:
            new_val["geneMapExists"] = False
            new_val["chromosome"] = np.nan
            new_val["locationStart"] = np.nan
            new_val["locationEnd"] = np.nan

    
        ### Clinical Synopsys
        try:
            new_val["clinicalSynopsisExists"] = False
            new_val["clinicalSynopsisExists"] = d["omim"]["entryList"][0]["entry"]["clinicalSynopsisExists"]
            for key in d["omim"]["entryList"][0]["entry"]["clinicalSynopsis"]:
                cols.append(key)
                new_val[key] = d["omim"]["entryList"][0]["entry"]["clinicalSynopsis"][key]
        except: 
            pass
        
        omim_info.append(new_val)

    logging.info("Saving data...")
    
    dict_columns = ["gene","geneMapExists","chromosome","locationStart","locationEnd","clinicalSynopsisExists"]
    dict_columns.extend(set(cols))
    dict_columns.append("text")
    df_omim_info = pd.DataFrame(omim_info, columns= dict_columns)
    
    df_omim_info.to_csv(saveToFilename,index=False)
    logging.info("Done.")
    
    return df_omim_info



