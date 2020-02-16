#!/usr/bin/env python

#  Author: Shammur Absar Chowdhury; Feb/2020
# *************************************** #

import os
import logging
from nltk.corpus import stopwords
import datetime
import sys
from os.path import basename
import warnings
from sklearn import preprocessing
from sklearn.externals import joblib
import optparse
import pickle
import data_preparation as data_preparation
import performance as performance

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
print(__doc__)

logger = logging.getLogger(__name__)


"""
Reads the data and do some preprocessing
"""
def predict(loaded_model, tokenizer, label_encoder, data_in_file,data_out_file,delim=","):
    """
    Prepare the data
    """
    data = []
    text_out_file=open(data_out_file,"w")
    text_out_file.write("text\tclass_label\tconfidence\n");

    data_x = data_preparation.read_data_for_classification(data_in_file, delim)

    count=1
    for txt in data_x:
        if (len(txt) < 1):
            print (txt)
            continue
        data.append(txt)
        if(count%10000==0):
            dev_x_feat = tokenizer.transform(data)
            result = loaded_model.predict(dev_x_feat)
            result = label_encoder.inverse_transform(result)

            for lab, txt in zip(result,data):
                # prob_per_class_dictionary = dict(zip(label_encoder.classes_, class_prob))
                # prob = prob_per_class_dictionary.get(lab)
                output_data=txt+"\t"+map_labels(lab)
                text_out_file.write(output_data+"\n");
            data=[]
        count=count+1

    dev_x_feat = tokenizer.transform(data)
    result = loaded_model.predict(dev_x_feat)
    result = label_encoder.inverse_transform(result)


    for lab, txt in zip(result, data):


        output_data = txt + "\t" + map_labels(lab)
        text_out_file.write(output_data + "\n");

    text_out_file.close

"""
Reads the data and do some preprocessing
"""
def eval(loaded_model, tokenizer, label_encoder, data_in_file,data_out_file,delim=","):
    """
    Prepare the data
    """
    data = []
    text_out_file=open(data_out_file,"w")
    text_out_file.write("text\tclass_label\tconfidence\n");

    data_x, data_y, _ = data_preparation.read_test_data(data_in_file, delim, label_encoder)
    data_x_feat = tokenizer.transform(data_x)
    pred_y = loaded_model.predict(data_x_feat)
    data_y = label_encoder.inverse_transform(data_y)
    pred_y = label_encoder.inverse_transform(pred_y)
    AUC, acc, precision, recall, F1, report = performance.performance_measure(data_y, pred_y, label_encoder)
    result = str("{0:.2f}".format(acc)) + "\t" + str("{0:.2f}".format(precision)) + "\t" + str(
        "{0:.2f}".format(recall)) + "\t" + str("{0:.2f}".format(F1)) + "\t" + str("{0:.2f}".format(AUC)) + "\n"
    print(result)
    print(report)

"""
It is better to use the following function to process a single item, which also takes care of empty vector and returns None
"""
def predict_single_item(loaded_model, tokenizer, label_encoder, text):
    data=[]
    data.append(text)
    x_feat = tokenizer.transform(data)
    dense = x_feat.toarray()
    if(dense[0].sum()==0.0):
        lab = "None"
        prob = 1.0
        return lab, prob
    result = loaded_model.predict(x_feat)
    result = label_encoder.inverse_transform(result)

    class_probabilities = loaded_model.predict_proba(x_feat)[0]

    prob_per_class_dictionary = dict(zip(label_encoder.classes_, class_probabilities))
    lab=result[0]
    prob = prob_per_class_dictionary.get(lab)

    return lab,prob

def read_config(configfile):
    configdict = {}
    with open(configfile, 'r') as f:
        for line in f:
            line = line.strip()
            if (line == ""):
                continue
            row = line.split("=")
            configdict[row[0]] = row[1]
    return configdict

def load_models(config_dictionary):

    # Load from file
    with open(config_dictionary["model_file"], 'rb') as file:
        loaded_model = pickle.load(file)

    tokenizer_file = config_dictionary["tokenizer_file"]
    label_encoder_file = config_dictionary["label_encoder_file"]

    # loading tokenizer
    with open(tokenizer_file, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # loading label_encoder
    with open(label_encoder_file, 'rb') as handle:
        label_encoder = pickle.load(handle)

    return loaded_model, tokenizer, label_encoder


"""
Mapping Labels to readable format:
"""
Label_mapping={
"art-and-entertainment":"Culture, Art and Entertainment" ,
"business-and-economy" : "Business and Economy",
"crime_and_security" : "Crime and Security",
"education" : "Education",
"environment" : "Environment",
"health" : "Health",
"human-rights-press-freedom" : "Human Rights and Freedom of Speech",
"others" : "Other Categories",
"politics" : "Politics",
"science-and-technology" : "Science and Technology",
"spiritual" : "Religion",
"sports" : "Sports",
"war-conflict" : "War and Conflicts"

}

def map_labels(lab):
    return Label_mapping[lab]



if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option('-c', action="store", dest="config_file")
    parser.add_option('-d', action="store", dest="data_file")
    parser.add_option('-o', action="store", dest="output_file")
    parser.add_option('-l', action="store", dest="language", default="english", type="string")
    parser.add_option('--eval', action="store", dest="evaluation", default="no", type="string")

    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)

    config_file_name = options.config_file
    data_file_name = options.data_file
    output_file_name = options.output_file
    lang = options.language
    eval=options.evaluation

    delim = "\t"

    config_dictionary = read_config(config_file_name)

    loaded_model, tokenizer, label_encoder = load_models(config_dictionary)
    text=""

    #lab,prob=predict_single_item(loaded_model, tokenizer, label_encoder, text)
    #print("class label: %s, confidence: %s" % (lab,prob))
    if eval == 'no':
        predict(loaded_model, tokenizer, label_encoder, data_file_name, output_file_name, delim=delim)
    else:
        eval(loaded_model, tokenizer, label_encoder, data_file_name, output_file_name, delim=delim)


    b = datetime.datetime.now().replace(microsecond=0)
    print ("time taken:")
    print(b-a)


