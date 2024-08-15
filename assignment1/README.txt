Team Members:
1. Goda Nagakalyani (214050010)


Requirements:
Pytorch
Python3
Keras
Transformers
Datasets
numpy
scipy
sklearn
tensorflow
sklearn_crfsuite

Hardware Requirements:
>15 GB Ram
For BERT based model: GPU

Setup before running:
preprocesses data can be donwloaded from: https://drive.google.com/drive/folders/13bpmwRF5TV9ssU8eQgwXlvaLvp0nvqvW?usp=sharing
from folder additional. those are (train/test/dev)_data_preprocessed.json
keep them in the same folder as code.


Execution and Reproducing Results:

CRF:
Folder: /code/CRF_NER
File: crf_sklearn.py

To train using test data and predict for test data 
Run: python crf_sklearn.py
Make sure the files used in this code, train_data_preprocessed.json etc. exist in the same folder.
You can also try running CRF.py which is an extention Prof. Soumen's code. But will take more that 6Hrs
for 1 epoch in cpu. Was failing to load in GPU because of the size.

LSTM:
Folder: /code/LSTM_NER
File: lstm_ner.py

To train call the train() function.
The test() function is also included to test for one example.
All the predefined parameters are included. (epochs etc.)
Make sure the files used in this code, train_data_preprocessed.json etc. exist in the same folder.

BERT:

Folder: /code/BERT_NER
File: bert_ner.py

To train call the train() function.
The test() function is also included to test for one example.

Note: 

All the predefined parameters are included. (epochs etc.)
Make sure the files used in this code, train_data_preprocessed.json etc. exist in the same folder.
Also make sure you have GPU. Also, for seqeval of distilbert-base-uncased, Nvidia libraries are required. Depends on GPU configuration, so not mentioning them here. The model checkpoint used Hugging face API to access the base model, so make sure connection is accessible.
