#!/usr/bin/env python

#  Author: Shammur Absar Chowdhury; Feb/2020
# *************************************** #

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
from math import sqrt
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import datetime
import sys
from os.path import basename
import warnings
import sklearn.metrics as metrics
from sklearn import preprocessing
from datetime import datetime
import optparse
import os, errno
from nltk.corpus import stopwords
from time import time
from sklearn.model_selection import KFold
import pickle
import random
import aidrtokenize as aidrtokenize
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
import performance as performance
import logging
import data_preparation as data_preparation


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
print(__doc__)


def remove_stop_words(text, stop_words):
    text_arr = text.split()
    new_text_arr = []
    for word in text_arr:
        if (word in stop_words):
            continue
        else:
            new_text_arr.append(word)
    text = " ".join(new_text_arr)
    return text


"""
Reads the data and do some preprocessing
"""


def read_train_data(data_file, delim, lang="en"):
    """
    Prepare the data
    """
    data = []
    labels = []

    with open(data_file, 'rb') as input_file:
        next(input_file)
        for line in input_file:
            line = line.decode(encoding='utf-8', errors='strict')
            line = line.strip()
            if (line == ""):
                continue
            row = line.split(delim)
            txt = row[1].strip().lower()
            txt = aidrtokenize.tokenize(txt)
            #txt = remove_stop_words(txt, stop_words)
            label = row[2]
            if (len(txt) < 1):
                print (txt)
                continue
            # if(isinstance(txt, str)):
            data.append(txt)
            labels.append(label)
            # else:
            #     print(txt)
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(labels)

    return np.array(data), y, label_encoder


"""
Reads the data and do some preprocessing
"""


def read_test_data(data_file, delim, label_encoder, lang="en"):
    """
    Prepare the data
    """

    data = []
    labels = []
    with open(data_file, 'rU') as f:
        next(f)
        for line in f:
            line = line.strip()
            if (line == ""):
                continue
            row = line.split(delim)
            txt = row[1].strip().lower()
            txt = aidrtokenize.tokenize(txt)
            #txt = remove_stop_words(txt, stop_words)
            label = row[2]
            if (len(txt) < 1):
                print (txt)
                continue
            data.append(txt)
            labels.append(label)
    y = label_encoder.fit_transform(labels)

    return np.array(data), y, label_encoder



def save_model(data_x, data_y, model_dir, model_file_name, tfidf_vectorizer, label_encoder):
    train_x_feat = tfidf_vectorizer.fit_transform(data_x)
    classifier = RandomForestClassifier(n_estimators=100, n_jobs=10).fit(train_x_feat, data_y)

    base_name = os.path.basename(model_file_name)
    base_name = os.path.splitext(base_name)[0]
    model_file = model_dir + "/" + base_name + "_svm.hdf5"
    tokenizer_file = model_dir + "/" + base_name + "_svm.tokenizer"
    label_encoder_file = model_dir + "/" + base_name + "_svm.label_encoder"

    configfile = model_dir + "/" + base_name + "_svm.config"
    configFile = open(configfile, "w")
    configFile.write("model_file=" + model_file + "\n")
    configFile.write("tokenizer_file=" + tokenizer_file + "\n")
    configFile.write("label_encoder_file=" + label_encoder_file + "\n")
    configFile.close()

    # files = []
    # files.append(configfile)

    # serialize weights to HDF5

    with open(model_file, 'wb') as file:
        pickle.dump(classifier, file)

    # saving tokenizer
    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tfidf_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # saving label_encoder
    with open(label_encoder_file, 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

def training(train_file, dev_file, tst_file):
    dirname = os.path.dirname(train_file)
    base = os.path.basename(train_file)
    file_name = os.path.splitext(base)[0]

    model_dir = "models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_model_path = model_dir + file_name + "_svm.model"

    n_features = 20000
    lang = "english"

    delim = ","
    train_x, train_y, train_le = data_preparation.read_train_data(train_file, delim)
    val_x, val_y, _ = data_preparation.read_test_data(dev_file, delim,train_le)
    test_x, test_y, _ = data_preparation.read_test_data(tst_file, delim,train_le)
    print(set(train_y))
    print(set(val_y))
    print(set(test_y))
    labels = list(train_le.classes_)
    print(labels)

    # sys.exit(0)
    # Feature Extraction
    tfidf_vectorizer = TfidfVectorizer(encoding='utf-8', lowercase=True, ngram_range=(1, 4), norm='l2', use_idf=True,
                                       max_df=0.95, min_df=3, max_features=n_features)
    train_x_feat = tfidf_vectorizer.fit_transform(train_x)

    val_x_feat = tfidf_vectorizer.transform(val_x)
    test_x_feat = tfidf_vectorizer.transform(test_x)
    random_seed = 2814
    random.seed(random_seed)
    classifier = LinearSVC(random_state=random_seed, C=1.5, class_weight='balanced', tol=0.0001).fit(train_x_feat, train_y)
    #class_weight='balanced',


    # dirname = os.path.dirname(tst_file)
    base = os.path.basename(tst_file)
    file_name = os.path.splitext(base)[0]
    results_dir = "results/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    output_file = results_dir + file_name + "_results_svm.txt"

    output_text_file = open(output_file, "w")
    output_text_file.write("acc\tprecision\trecall\tF1\tAUC\n");

    train_pred_y = classifier.predict(train_x_feat)
    train_pred_y = train_le.inverse_transform(train_pred_y)
    train_true_y = train_le.inverse_transform(train_y)

    AUC, acc, precision, recall, F1, report = performance.performance_measure(train_true_y, train_pred_y, train_le)
    result = str("{0:.2f}".format(acc)) + "\t" + str("{0:.2f}".format(precision)) + "\t" + str(
        "{0:.2f}".format(recall)) + "\t" + str("{0:.2f}".format(F1)) + "\t" + str("{0:.2f}".format(AUC)) + "\n"

    output_text_file.write("train-set" + "\n" + result + "\n")
    output_text_file.write(report + "\n")

    print ("train set:\t" + result)
    print(report)


    dev_pred_y = classifier.predict(val_x_feat)
    dev_pred_y = train_le.inverse_transform(dev_pred_y)
    dev_true_y = train_le.inverse_transform(val_y)

    AUC, acc, precision, recall, F1, report = performance.performance_measure(dev_true_y, dev_pred_y, train_le)
    result = str("{0:.2f}".format(acc)) + "\t" + str("{0:.2f}".format(precision)) + "\t" + str(
        "{0:.2f}".format(recall)) + "\t" + str("{0:.2f}".format(F1)) + "\t" + str("{0:.2f}".format(AUC)) + "\n"

    output_text_file.write("dev-set" + "\n" + result + "\n")
    output_text_file.write(report + "\n")

    print ("dev set:\t" + result)
    print(report)

    test_pred_y = classifier.predict(test_x_feat)
    test_pred_y = train_le.inverse_transform(test_pred_y)


    test_true_y = train_le.inverse_transform(test_y)


    AUC, acc, precision, recall, F1, report = performance.performance_measure(test_true_y, test_pred_y, train_le)
    result = str("{0:.2f}".format(acc)) + "\t" + str("{0:.2f}".format(precision)) + "\t" + str(
        "{0:.2f}".format(recall)) + "\t" + str("{0:.2f}".format(F1)) + "\t" + str("{0:.2f}".format(AUC)) + "\n"

    output_text_file.write("test-set" + "\n" + result + "\n")
    output_text_file.write(report + "\n\n")

    conf_mat_str = performance.format_conf_mat(test_true_y, test_pred_y, train_le)

    output_text_file.write(conf_mat_str + "\n")

    print ("Test set:\t" + result)
    print(report)

    output_text_file.close()

    save_model(train_x, train_y, model_dir, best_model_path, tfidf_vectorizer, train_le)

    return result

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="train_data", default=None, type="string")
    parser.add_option('-v', action="store", dest="val_data", default=None, type="string")
    parser.add_option('-t', action="store", dest="test_data", default=None, type="string")
    # parser.add_option('-o', action="store", dest="output_file", default=None, type="string")
    # parser.add_option('-m', action="store", dest="model_file", default=None, type="string")

    options, args = parser.parse_args()
    a = datetime.now().replace(microsecond=0)

    train_file = options.train_data
    dev_file = options.val_data
    tst_file = options.test_data
    # results_file = options.output_file
    # best_model_path = options.model_file


    dirname = os.path.dirname(train_file)
    base = os.path.basename(train_file)
    file_name = os.path.splitext(base)[0]

    training(train_file, dev_file, tst_file)

    b = datetime.now().replace(microsecond=0)
    print ("time taken:")
    print(b - a)
