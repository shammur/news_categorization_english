#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun June  2 10:33:22 2019

#  Author: Shammur Absar Chowdhury; June/2019, Feb/2020
# *************************************** #

"""

import os
import re

import numpy as np
import pandas as pd
from nltk import tokenize
import keras
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import string
from sklearn.utils import shuffle

import wordsegment as ws
# from wordsegment import segment
ws.load()
from emoji import UNICODE_EMOJI
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from nltk.stem.porter import *
from nltk.stem import PorterStemmer
from sklearn import preprocessing
from nltk.corpus import stopwords
english_stopwords = set(stopwords.words('english'))
porter = PorterStemmer()


def read_data_for_classification(filename,delim):
    data = []
    input_data = pd.read_csv(filename, sep=delim)
    for idx in range(input_data.Content.shape[0]):
        text=input_data.Content[idx]
        text=clean_content(text)
        if  isinstance(text,str):
            data.append(text)
    return data

def read_train_data(filename,delim):
    # articles = []
    labels = []
    data = []
    # nlp = spacy.load('en')
    train_slb = LabelEncoder()
    input_data = pd.read_csv(filename, sep=delim)
    input_data = shuffle(input_data)

    for idx in range(input_data.Content.shape[0]):
        text=input_data.Content[idx]
        text=clean_content(text)
        if  isinstance(text,str):
            data.append(text)
            class_label = input_data.Class[idx].lstrip().rstrip()
            if(class_label=="fashion-and-lifestyle"):
                class_label = "others"
            #else:
            #    class_label = input_data.Class[idx]
            labels.append(class_label)

    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(labels)
    return data,y, label_encoder


def read_test_data(filename,delim,label_encoder):
    # articles = []
    labels = []
    data = []
    # nlp = spacy.load('en')
    train_slb = LabelEncoder()
    input_data = pd.read_csv(filename, sep=delim)
    input_data = shuffle(input_data)

    for idx in range(input_data.Content.shape[0]):
        text=input_data.Content[idx]
        text=clean_content(text)
        if  isinstance(text,str):
            data.append(text)
            class_label = input_data.Class[idx].lstrip().rstrip()
            if(class_label=="fashion-and-lifestyle"):
                class_label = "others"
            #else:
            #    class_label = input_data.Class[idx]
            labels.append(class_label)
    y = label_encoder.transform(labels)
    return data,y, label_encoder


def read_test_data_cnn(filename, maxlen, max_words, tokenizer, train_slb, delim):
    # articles = []
    labels = []
    texts = []
    # nlp = spacy.load('en')

    input_data = pd.read_csv(filename, sep=delim)
    # input_data = input_data[0:100]
    for idx in range(input_data.Content.shape[0]):
        text=input_data.Content[idx]
        text = clean_content(text)
        if  isinstance(text,str):
            # text = dps.clean_article(text)
            # sentences = []
            # sentences.append(text)  # tokenize.sent_tokenize(text)
            # sentences = tokenize.sent_tokenize(text)
            texts.append(text)
            # articles.append(sentences)
            labels.append((input_data.Class[idx]).lstrip().rstrip())



    data = np.zeros((len(texts), maxlen), dtype='int32')


    for i, sent in enumerate(texts):
        # sent = sentences[0]
        # for j, sent in enumerate(sentences):
        #     if j < max_sentences:
        wordTokens = text_to_word_sequence(sent)
        k = 0
        for _, word in enumerate(wordTokens):
            if k< maxlen and word in tokenizer.word_index and tokenizer.word_index[word]< (max_words+1):
                data[i, k] = tokenizer.word_index[word]
                k = k + 1
            elif k < maxlen and word not in tokenizer.word_index:
                data[i, k] = tokenizer.word_index[tokenizer.oov_token]
                k = k + 1

    print('Shape of data tensor:', data.shape)
    yL = train_slb.fit_transform(labels)
    data_class_index = list(train_slb.classes_)
    shuf_lab = yL.tolist()
    yC = len(set(shuf_lab))
    yR = len(shuf_lab)
    data_y = np.zeros((yR, yC))
    data_y[np.arange(yR), yL] = 1
    data_y = np.array(data_y, dtype=np.int32)
    return data, data_y, data_class_index

def read_test_data_with_info(filename, maxlen, max_words, tokenizer, train_slb, delim):
    # articles = []
    labels = []
    texts = []
    ids = []
    # nlp = spacy.load('en')

    input_data = pd.read_csv(filename, sep=delim)
    # input_data = input_data[0:100]
    for idx in range(input_data.Content.shape[0]):
        text=input_data.Content[idx]
        text = clean_content(text)
        if  isinstance(text,str):
            # text = dps.clean_article(text)
            # sentences = []
            # sentences.append(text)  # tokenize.sent_tokenize(text)
            # sentences = tokenize.sent_tokenize(text)
            texts.append(text)
            # articles.append(sentences)
            labels.append((input_data.Class[idx]).lstrip().rstrip())
            ids.append(input_data.ID[idx])


    data = np.zeros((len(texts), maxlen), dtype='int32')


    for i, sent in enumerate(texts):
        # sent = sentences[0]
        # for j, sent in enumerate(sentences):
        #     if j < max_sentences:
        wordTokens = text_to_word_sequence(sent)
        k = 0
        for _, word in enumerate(wordTokens):
            if k < maxlen and word in tokenizer.word_index and tokenizer.word_index[word] < (max_words + 1):
                data[i, k] = tokenizer.word_index[word]
                k = k + 1
            elif k < maxlen and word not in tokenizer.word_index:
                data[i, k] = tokenizer.word_index[tokenizer.oov_token]
                k = k + 1
    # word_index = tokenizer.word_index
    # print('Total %s unique tokens.' % len(word_index))
    print('Shape of data tensor:', data.shape)
    yL = train_slb.fit_transform(labels)
    data_class_index = list(train_slb.classes_)
    shuf_lab = yL.tolist()
    yC = len(set(shuf_lab))
    yR = len(shuf_lab)
    data_y = np.zeros((yR, yC))
    data_y[np.arange(yR), yL] = 1
    data_y = np.array(data_y, dtype=np.int32)
    return data, data_y, data_class_index, ids



def read_test_data_with_info_pred(filename, maxlen, max_words,tokenizer,train_slb,delim):
    # articles = []
    # labels = []
    texts = []
    ids = []
    # nlp = spacy.load('en')

    input_data = pd.read_csv(filename, sep=delim)
    # input_data = input_data[0:100]
    for idx in range(input_data.Content.shape[0]):
        text=input_data.Content[idx]
        text = clean_content(text)
        if  isinstance(text,str):
            text = (text)
            # sentences = []
            # sentences.append(text)  # tokenize.sent_tokenize(text)
            # sentences = tokenize.sent_tokenize(text)
            texts.append(text)
            # articles.append(sentences)
            # labels.append((input_data.Classes[idx]).split())
            ids.append(input_data.ID[idx])


    #tokenizer = Tokenizer(num_words=max_words)
    #tokenizer.fit_on_texts(texts)
    data = np.zeros((len(texts), maxlen), dtype='int32')
    # data = np.zeros((len(texts), max_sentences, maxlen), dtype='int32')

    for i, sent in enumerate(texts):
        # sent=sentences[0]
        # for j, sent in enumerate(sentences):
            # if j < max_sentences:
        wordTokens = text_to_word_sequence(sent)
        k = 0
        for _, word in enumerate(wordTokens):
            if k< maxlen and word in tokenizer.word_index and tokenizer.word_index[word]< (max_words+1):
                data[i, k] = tokenizer.word_index[word]
                k = k + 1
            elif k < maxlen and word not in tokenizer.word_index:
                data[i, k] = tokenizer.word_index[tokenizer.oov_token]
                k = k + 1
    # word_index = tokenizer.word_index
    # print('Total %s unique tokens.' % len(word_index))
    print('Shape of data tensor:', data.shape)
    # data_y = train_mlb.transform(labels)
    data_class_index = list(train_slb.classes_)
    return data, data_class_index, ids

def inverse_transform_to_labels(binary_array, class_indexes):
    # print(binary_array[4])
    # print(binary_array[5])
    cls=[ class_indexes [idx] for idx, val in enumerate(binary_array) if val== 1]
    if not cls : cls.append('None')
    return ','.join(cls)


def load_embedding(file_name):
    print('Indexing word vectors.')
    embeddings_index = {}

    #f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    f = open(file_name)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))
    return embeddings_index;


def prepare_embedding(word_index, embeddings_index, embedding_dim):
    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        try:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        except KeyError:
            continue
    return embedding_matrix;

#Data processing pipeline:


def clean_content(line):
    if (isinstance(line, float)):
        return None
    line.replace('\n', ' ')
    line = remove_emails(line)
    line = remove_urls(line)

    # Check if # or @ is there with word
    linelst = []
    # if '#' in line or '@' in line:
    for w in line.split():
        if ("#" in w or "@" in w):
            # print(w)
            linelst.append(' '.join(ws.segment(w)))
        elif 'http' not in w:
            linelst.append(w)
    line = ' '.join(linelst)

    # add spaces between punc,
    line = line.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))

    # then remove punc,
    # line = line.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    translator = str.maketrans('', '', string.punctuation)
    line = line.translate(translator)

    ## convert to lower case
    line = line.lower()

    # stemming:
    token_words = word_tokenize(line)
    # token_words
    stem_sentence = [porter.stem(word) if not hasDigits(word) else '<NUM>' for word in token_words]

    stem_sentence = removeConsecutiveSameNum(stem_sentence)
    # for word in token_words:
    #
    #     stem_sentence.append(porter.stem(word))
    # print(type (stem_sentence))
    line = " ".join(stem_sentence)

    removed_stops = [w for w in line.split() if not w in english_stopwords and len(w) != 1]
    line = ' '.join(removed_stops)

    return line

def remove_urls (text):
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
    return text

def remove_emails(text):
    text = re.sub(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", "",  text, flags=re.MULTILINE)
    return text

def is_emoji(s):
    return s in UNICODE_EMOJI

# add space near your emoji
def add_space_with_emojis(text):
    return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()


def removeConsecutiveSameNum(v):
    st = []
    lines=[]

    # Start traversing the sequence
    for i in range(len(v)):

        # Push the current string if the stack
        # is empty
        if (len(st) == 0):
            st.append(v[i])
            lines.append(v[i])
        else:
            Str = st[-1]

            # compare the current string with stack top
            # if equal, pop the top
            if (Str == v[i] and Str == '<NUM>'):
                st.pop()

                # Otherwise push the current string
            else:
                lines.append(v[i])
                st.pop()
                # st.append(v[i])

                # Return stack size
    return lines

def hasDigits(s):
    return any( 48 <= ord(char) <= 57 for char in s)
