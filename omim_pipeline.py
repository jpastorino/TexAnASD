###########################################################
# Author: J. Pastorino, Ashis K. Biswas
#
# Description: Methods to implement the main pipeline. 
#              To be called from the Jupyter Notebook: omim_main.ipynb


import numpy as np
import pandas as pd
import logging
from time import time
import os
import datetime
import pickle
import concurrent.futures

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition           import TruncatedSVD
from sklearn.pipeline                import make_pipeline
from sklearn.preprocessing           import Normalizer

from sklearn import svm

from sklearn         import metrics
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

import nltk

nltk.download('all-corpora')

########################################################################################################################################
def getCorpus(df_text_data):
    corpus=[]
    for i,row in df_text_data.iterrows():
        corpus.append(row["text"])

    return corpus



########################################################################################################################################
def tfidf_dimReduced(corpus, log_id="<<id>>", number_features=10000):
    """
        Vectorizes the corpus using -IDF metric and then reduces the dimension of the features to number_features.
        Vectorizer results are normaliTFzed. Since LSA/SVD results are not normalized, we have to redo the normalization.
    
    Parameters
    ----------------
    corpus (array): array of text documents.
    login_id [optional] (string): id to display in the logs. Usually represents the fold. 
    number_features [optional](int): numbers of features to keep after SVD.
    
    Returns
    ----------------
    lsa (object): the learned model.
    X (object): the matrix of learned features representing the corpus.
    """

    wnl = nltk.stem.PorterStemmer()

    logging.info(log_id+"Running Stemming...")
    t0 = time()
    
    corporea = []
    for doc in corpus:
        new_doc=[]
        # removing autism and asd words.
        doc = doc.lower().replace("autism","").replace("asd","")
        
        tokens = nltk.word_tokenize(doc)
        for token in tokens:
            new_doc.append(wnl.stem(token))
        corporea.append(' '.join(new_doc))

    logging.info(log_id+"Stemming done in %fs" % (time() - t0))
    
    
    
    logging.info(log_id+"Running tfidf...")
    t0 = time()
    
    vectorizer = TfidfVectorizer(max_df=0.5, #max_features=20000,
                                 min_df=1, stop_words='english', use_idf=True)
    
    logging.info(log_id+"tf-idf done in %fs" % (time() - t0))

    
    logging.info(log_id+"Running SVD Dim Reduction...")
    t0 = time()
    svd = TruncatedSVD(number_features)
    
    normalizer = Normalizer(copy=False)
    
    lsa = make_pipeline(vectorizer, svd, normalizer)

    X = lsa.fit_transform(corporea)

    explained_variance = svd.explained_variance_ratio_.sum()

    logging.info(log_id+"SVD explained variance: {}%".format(int(explained_variance * 100)))
    logging.info(log_id+"SVD done in %fs" % (time() - t0))

    return lsa, X

########################################################################################################################################
def shuffleData(data,labels):
    """
    Shuffels the data randomly.
    
    Parameters
    ----------------
    data (array): array of data samples.
    labels (array): array of labels for the samples
    
    Returns
    ----------------
    new_data (array): shuffled data samples.
    new_labels (array): shuffled lables samples (sync).
    """
    new_data   =[]
    new_labels =[]
    
    arr = np.arange(len(data))
    np.random.shuffle(arr)

    for i in arr:
        new_data.append(data[i])
        new_labels.append(labels[i])
    
    return new_data, new_labels


########################################################################################################################################
def cross_validation_split(data, labels, k_folds=5):
    """
    Splits the dataset into k-folds partitions to run cross validation
    
    Parameters: 
    data (array): the data samples.
    labels (array): the class labels.
    k_folds (array): optional the number of folds.
  
    Returns: 
    array: {data, labels} - a set of dictionary objects with data and labels for each partition.
    """
    if k_folds <2:        raise Exception("k-folds must be at least 2")
    
    samples_per_fold = len(data) // k_folds
    last_fold_size   = len(data) - (samples_per_fold * (k_folds-1))
    
    folds = []
   
    for i in range(k_folds-1):
        s = i*samples_per_fold
        e = (i+1)*samples_per_fold
        folds.append({"data":data[s:e],"labels":labels[s:e]})

    #Last fold
    s = (k_folds-1) * samples_per_fold
    e = k_folds * last_fold_size

    folds.append({"data":data[s:e],"labels":labels[s:e]})
    
    return folds



########################################################################################################################################
def plotROC_Fold(fold, doPlot=True):
    y_score = np.load(PROJECT_DATA+"/crossvalidation/fold_"+str(fold)+"_yscores.npy")
    y_test  = np.load(PROJECT_DATA+"/crossvalidation/fold_"+str(fold)+"_ytest.npy")

    plotROC(y_score, y_test, doPlot, savePlot=True,filename=PROJECT_DATA+"/crossvalidation/roc_plot_fold_{:1d}.svg".format(fold))

########################################################################################################################################
def plotROC_Model(y_scores_filename, y_test_filename, figure_output_filename):
    y_score = np.load(y_scores_filename)
    y_test  = np.load(y_test_filename)

    plotROC(y_score, y_test, doPlot=True, savePlot=True,filename=figure_output_filename)


    
    
########################################################################################################################################
def plotROC(y_score, y_test, doPlot=True, savePlot=False, filename=""):
    """
    PLOT/COMPUTE ROC
    Compute ROC curve and ROC area for each class
    """
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:], y_score[:])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    if doPlot:
        plt.figure()
        lw = 2
        plt.plot(fpr[0], tpr[0], color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0] )
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        if savePlot:
            plt.savefig(filename, format='svg')
        plt.show()
    
    return roc_auc[0]    





########################################################################################################################################
def runPipeline(train_samples, train_labels, test_samples, test_labels,parameters=None, pipelineID="id", 
                log_id="<<id>>", path_results=None, plot_roc=False, summary_filename="runPipeLine_summary.csv"):
    """
    Runs the pipeline for the given samples. 
    
    Parameters:
    ----------------
    train_samples, train_labels (array,array): the samples and labels for training.
    test_samples, test_labels (array,array): the samples and labels for testing.
    run_parallel [optional](bool): True for parallel processing, False otherwise. 
    pipelineID [optional] (string): an id to save into the filename the results.
    path_results   [optional] (string): path to save the data and the summary file. 
    plot_roc   [optional] (bool): true to plot the ROC.
        
    Returns:
    ----------------
    dictionary : obtained scores for the pipeline. 
    
    """
    logging.info(log_id+"Starting running pipeline for pipeline id:{}".format(pipelineID))
    
    if parameters is None:
        logging.debug(log_id+"Parameters: Default") 
    else:
        logging.debug(log_id+"Parameters: "+''.join(''.join((k,"->", str(v),";")) for k,v in parameters.items())) 
    
    scores = {"auc":0, "accuracy":0, "recall":0, "precision":0, "F1":0}
    
    t0 = time()
    
    logging.info(log_id+"Training TF-IDF.")

    lsa_model, trained_features = tfidf_dimReduced(train_samples, log_id=log_id)

    logging.info(log_id+"Transforming Testing Samples.")
    test_features = lsa_model.transform(test_samples)
    
    if parameters is None:
        svc = svm.SVC(gamma='scale',probability=True)
    else:
        svc = svm.SVC(**parameters)

    logging.info(log_id+"Training SVC Model")
    fitted_model = svc.fit(trained_features, train_labels)
    
    logging.info(log_id+"Predicting Test Class using SVC trained model.")
    y_predict    = svc.predict(test_features)
    
    logging.info(log_id+"Computing Metrics.")
    scores_auc   = np.array(fitted_model.decision_function(test_features))
    
    scores["accuracy"]  = metrics.accuracy_score(test_labels, y_predict)
    scores["recall"]    = metrics.recall_score(test_labels, y_predict)
    scores["precision"] = metrics.precision_score(test_labels, y_predict)
    scores["F1"]        = metrics.f1_score(test_labels, y_predict)
    scores["auc"]       = metrics.roc_auc_score(test_labels, scores_auc)

    logging.info(log_id+"Saving to disc Y-Scores, Y-TrueLabels and Y-Predictions")
    y_scores_filename  = path_results+"/data/"+str(pipelineID)+"_yscores"
    y_test_filename    = path_results+"/data/"+str(pipelineID)+"_ytest"
    y_predict_filename = path_results+"/data/"+str(pipelineID)+"_ypredicted"
    
    np.save(y_scores_filename,  scores_auc  ) # Saving svm scores to compute AUC-ROC
    np.save(y_test_filename,    test_labels ) # Saving test ground truth to compute AUC-ROC
    np.save(y_predict_filename, y_predict )   # Saving predicted y labels.
    
    logging.info(log_id+"y-predicted labels:"+y_predict_filename+".npy")
    logging.info(log_id+"y-scores:"+y_scores_filename+".npy")
    logging.info(log_id+"y-ground truth labels:"+y_test_filename+".npy")

    t1=time()
    
    summary = "{},{},{:10.8f},{:10.8f},{:10.8f},{:10.8f},{:10.8f},{:10.2f}\n".format(str(datetime.datetime.now()), pipelineID, scores["auc"], scores["accuracy"], 
                                                                                     scores["recall"],scores["precision"],scores["F1"], t1-t0)
 
    logging.info(log_id+"Saving Summary information (scores) to file.")
    
    #Saving Results.
    if not os.path.exists(path_results+"/"+str(summary_filename)): 
        summary_file = open(path_results+"/"+str(summary_filename),"w")
        summary_file.write("date_time,pipelineID,auc,accuracy,recall,precision,F1,runtime_sec\n")
    else: 
        summary_file = open(path_results+"/"+str(summary_filename),"a")
        
    summary_file.write(summary)
    summary_file.close() 

    
    logging.info(log_id+"End running pipeline for pipeline id:{}. Runtime:{:3.2f}sec".format(pipelineID,t1-t0))

    if plot_roc: 
        plotROC_Model(y_scores_filename+".npy",y_test_filename+".npy",path_results+"/data/"+str(pipelineID)+"_roc_plot.svg")

    
    return scores



########################################################################################################################################
def runParallelCrossValidation(fold, data_folds, cross_validation_path, iteration=0, parameters=None):
    """
    Runs a single thread computing a fold with the given data. This method prepares the data (train/test) and launchs the pipeline.
    
    Parameters:
    ----------------
    fold (number): number of the fold to run
    data_folds (array): array of dictionary (data/labels) objects
    cross_validation_path (string): path to save the output data
    iteration [optional] (int): number of iteration. By default is 0.
    
    Returns:
    ----------------
    A dictionary with the scores (auc,accuracy,recall,precision,F1)
    """
    
    t0=time()
    
    log_id = "<<{:2d}-fold>> ".format(fold) 

    logging.info(log_id+"Train-Validation. Saving information to {}".format(cross_validation_path) )

    # Merging Train folds intro one single object.    
    train_folds = []
    for j in range(len(data_folds)):
        if not j==fold:
            train_folds.append(data_folds[j])

    #Test fold is the one excluded.
    test_samples = data_folds[fold]["data"]
    test_labels = data_folds[fold]["labels"]
    
    #Building Train data and labels arrays. 
    train_data   = []
    train_labels = []
    for i in range(len(train_folds)):
        train_data.extend(train_folds[i]["data"])
        train_labels.extend(train_folds[i]["labels"])
        logging.info(log_id+"Train set {} => Train has {} Samples and {} Labels.".format(i,len(train_folds[i]["data"]),len(train_folds[i]["labels"])) )
            
    logging.info(log_id+"Full Train Dataset has {} Samples and {} Labels.".format(len(train_data),len(train_labels)))
    logging.info(log_id+"Test Dataset has {} Samples and {} Labels.".format(len(test_samples),len(test_labels)) )

    
    #Running the Pipeline. 
    score = runPipeline (train_data, train_labels, test_samples, test_labels, 
                         pipelineID = "iter_"+str(iteration)+"_fold_"+str(fold), 
                         log_id=log_id,
                         path_results = cross_validation_path,
                         parameters=parameters,
                         summary_filename= "crossvalidation_summary.csv" )

    t1=time()
    
    logging.info(log_id+"runParallelCorssValidation ran in {:4.2f} sec.".format(t1-t0))
    logging.info(log_id+"Scores:"+ ''.join(''.join((k,"->", str(v),";")) for k,v in score.items())) 
    
    return score
    
########################################################################################################################################
def runModelTrainTest(train_samples,train_labels,test_samples,test_labels, output_path, parameters=None):
    """
    Runs a single thread computing a fold with the given data. This method prepares the data (train/test) and launchs the pipeline.
    
    Parameters:
    ----------------
    fold (number): number of the fold to run
    data_folds (array): array of dictionary (data/labels) objects
    cross_validation_path (string): path to save the output data
    iteration [optional] (int): number of iteration. By default is 0.
    
    Returns:
    ----------------
    A dictionary with the scores (auc,accuracy,recall,precision,F1)
    """
    
    t0=time()
    
    logging.info("[[main]] Model Train-Test. Saving information to {}".format(output_path) )
        
    logging.info("[[main]] Full Train Dataset has {} Samples and {} Labels.".format(len(train_samples),len(train_labels)))
    logging.info("[[main]] Test Dataset has {} Samples and {} Labels.".format(len(test_samples),len(test_labels)) )

    #Running the Pipeline. 
    score = runPipeline (train_samples, train_labels, test_samples, test_labels, 
                         pipelineID = "main", 
                         log_id="[[main]] ",
                         path_results = output_path,
                         parameters=parameters,
                         plot_roc=True,
                         summary_filename= "main_summary.csv" )

    t1=time()
    
    logging.info("[[main]] runModelTrainTest ran in {:4.2f} sec.".format(t1-t0))
    logging.info("[[main]] Scores:"+ ''.join(''.join((k,"->", str(v),";")) for k,v in score.items())) 
    
    return score
    