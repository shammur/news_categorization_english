

# Social Media News Post Categorization model.
This release includes model for categorizing topics (13 categories) for social media posts from news agency. The model is trained on dataset collected from different online platforms (e.g., Twitter, Facebook, Youtube) of a well-know news organisation. It includes posts from 2016 to Jan, 2020.

The categories are:
* Culture, Art and Entertainment
* Business and Economy
* Crime and Security
* War and Conflict
* Education
* Environment
* Health
* Human Rights and Freedom of Speech
* Politics
* Science and Technology
* Religion
* Sports
* Others Categories - representing categories that are not mentioned above like travel blogs, news related to fashion among others.

## Data Annotation
To train the model, we annotated ~10K amount of data.
The contents are collected from the following sources:
* Twitter
* Youtube
* Facebook
* Instagram

The annotation of the collected dataset is obtained using Amazon Mechanical Turk (AMT). To ensure the quality of the annotation and language proficiency, we utilized two different evaluation criteria of the annotator. For more details, check the below paper:

**Comming Soon**
Cite [the Arxiv paper](https://arxiv.org/):
Containing details of data collection method, annotation guideline, with link to dataset and model performance.
<!-- ```
@inproceedings{shammur2020offensive,
  title={A Multi-Platform Arabic News Comment Dataset for Offensive Language Detection},
  author={Chowdhury, Shammur Absar  and Mubarak, Hamdy and Abdelali, Ahmed and Jung, Soon-gyo and Jansen, Bernard J and Salminen, Joni},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC'20)},
  year={2020}
}
``` -->

In addition to the social media post dataset, we also used News_Category_Dataset -- containing around 200k news headlines from the year 2012 to 2018 obtained from HuffPost. [Available through https://rishabhmisra.github.io/publications/#datasets]

The overall data distribution is presented in below Table:

Table 1: Train, dev and test distribution

Train | Dev | Test
:------: | :------:| :------:
26094 | 34 | 64
7339 | 41 | 73
3651 | 65 | 132
1017 | 8 | 17
4208 | 62 | 122
17429 | 28 | 57
14436 | 94 | 187
51615 | 27 | 53
32651 | 250 | 483
4123 | 19 | 39
2236 | 12 | 24
4616 | 22 | 41
1435 | 126 | 251

Total |Train | Dev | Test
:------: |:------: | :------:| :------:
Total | 170850 | 788 | 1543


## Model training and evaluation
To train the model we conducted several experiments consisting of different machine learning algorithms and different feature representations. The following two models are provided.
* SVM
* BERT (please download it from the local server or email shammurchowdhury@gmail.com).

The SVM model is designed using word ngrams. The motivation for using Support Vector is that it is better for a limited data set and also good for imbalanced class distribution present in the dataset (see Table 1, for more details). 

The model is evaluated using:
* Dev and test set (see Table 1, for more details) for evaluating the in-domain data performance

### SVM
For the training the classifier with SVM, we used TF-IDF representations for word ngrams. The reason to choose SVM with TF-IDF is their simplicity.



## Predicting using the models

## Data Format
### Input data format
The input file should have the following fields, including
`<Input ID>\t<Content>\t<Class>` *(Field names must be maintained as Content and Class (optional for evaluation only) - else the code will break)*
however when the model is not used to evaluate the performance, `<CLass>` is optional field.
*!!! The text/input should have each datapoint in a single line, if the intend post contain new lines (\n), this should be preprocessed separately before using the model !!!*

### Output data format
The output of the file will include the following fields

* While running the model just for prediction:
`<id>\t<text>\t<class_label>`

The output are mapped to make label for readable (see Table 2 for more details).


### SVM
To run the classification model please use python version 3.7, install dependencies

To install the requirements:
```
pip install -r requirements.txt
```

The model can be used in two ways, either using batch of data or single data points.
<!-- Even though for single datapoint the batch processing script can be used, we suggest to use the example provided in `run_airline_post_cat_models_for_single_text.ipynb` -->

For batch classification of data:

```
python bin/prediction_model.py -c models/sm_news_en_trn_svm.config -d sample_data/sample_test.tsv -o results/sample_tst_predicted.tsv
```
For evaluation of batch with reference label, just add
the following flag to `prediction_model.py`

```
  --eval yes
```

The results of the model on the given dataset will be printed in the i/o
Example:
```
python bin/prediction_model.py -c models/sm_news_en_trn_svm.config -d sample_data/sample_test.tsv -o results/sample_tst_predicted.tsv --eval yes
```


### BERT
The model was fine-tuned with 20 epoch with BERT network followed by a softmax layer. The model is trained and tested on using GPUs.
To run the BERT-based classification model, please follow the steps below:

#### Create a virtual environment
```
python3 -m venv news_cat_bert_env
```
#### Activate your virtual environment
```
source $PATH_TO_ENV/news_cat_bert_env/bin/activate
```

#### Install dependencies
```
pip install -r requirements_py3.7_bert.txt
```

#### Run the classification script

```
bash bin/bert_multiclass_classification.sh
```


## Classification Results

As mentioned earlier, the performance of the model is tested using official dev and test sets.

Table 2: Overall Performance of the model on cross-validation


Overall| Macro	F1| Weighted F1
--------| :------: | :------:
SVM-Test | 0.56 | 0.64
BERT-Test | 0.57 | 0.64
<!-- Dev | 0.60 | 0.65 -->


Table 3: Class wise Performance of the SVM and BERT models on test set

Output | Class | SVM | BERT
------| :------: | :------:| :------:
Culture, Art and Entertainment  | art-and-entertainment | 0.55 | 0.59
Business and Economy | business-and-economy | 0.57 | 0.54
Crime and Security | crime_and_security | 0.59 | 0.57
Education | education | 0.39 | 0.46
Environment | environment | 0.76 | 0.72
Health | health | 0.49 | 0.68
Human Rights and Freedom of Speech | human-rights-press-freedom | 0.45 | 0.45
Politics | politics | 0.77 | 0.78
Science and Technology | science-and-technology | 0.46 | 0.35
Religion | spiritual | 0.61 | 0.43
Sports | sports | 0.72 | 0.81
War and Conflicts | war-conflict | 0.70 | 0.68
Others Categories | others | 0.22 | 0.38
