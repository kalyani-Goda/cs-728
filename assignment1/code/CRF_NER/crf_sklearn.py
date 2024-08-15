import json
#import torch
from itertools import chain

#import nltk
import sklearn
from sklearn.model_selection import cross_val_predict, cross_val_score
import sklearn_crfsuite
from sklearn_crfsuite import scorers,CRF
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn_crfsuite import metrics
from sklearn.metrics import classification_report
from tqdm import tqdm


data_folder = "./"
train_file = "train_data_preprocessed.json"
dev_file = "dev_data_preprocessed.json"
test_file = "test_data_preprocessed.json"


#train_file = dev_file
train_data = []
test_data = []
with open(data_folder+ train_file) as f:     #change files as required
    train_data = json.load(f)

with open(data_folder+ test_file) as f:     #change files as required
    test_data = json.load(f)

# Print tag stats and prepare tag dictionary.
STATE_INIT = 0
tag_name_to_id = dict()
tag_name_to_id["init"] = STATE_INIT

# for i in range(len(data)):
#     for j in range(len(data[i]["tags"])):
#         tag = data[i]["tags"][j]
#         if tag not in tag_name_to_id:
#             tag_name_to_id[tag] = len(tag_name_to_id)

NUM_STATES = len(tag_name_to_id)
#print("Number of states", NUM_STATES)

def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent["sent"], i) for i in range(len(sent["sent"]))]

def sent2labels(sent):
    return sent["tags"]


def gen_data():
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for sentence in tqdm(train_data, mininterval=1):
        X_train.append(sent2features(sentence))
        y_train.append(sentence["tags"])
    for sentence in tqdm(test_data):
        X_test.append(sent2features(sentence))
        y_test.append(sentence["tags"])
    return X_train,y_train, X_test, y_test


X_train, y_train, X_test, y_test = gen_data()
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

#Creating the CRF model
crf = CRF(algorithm='lbfgs',
          c1=0.25,
          c2=0.3,
          max_iterations=5,
          all_possible_transitions=True,
          verbose = True)

crf.fit(X_train,y_train)

y_pred_data = crf.predict(X_test)

y_true = []
y_pred = []

for i in range(len(y_test)):
    if (len(y_pred_data[i]) != len(y_test[i])): 
        print(i,"pred: ",len(y_pred_data[i]))
        print(i,"test: ",len(y_test[i]))
        print()
    y_true.extend(y_test[i])
    y_pred.extend(y_pred_data[i])
print(len(y_true))
print(len(y_pred))

report = classification_report(y_true, y_pred)
print("Writing report into: report.txt")
with open('report.txt', 'w') as f:
    print(report, file=f)
print(report)

for i in tqdm(range(len(test_data))):
    test_data[i]["tags"] = y_pred_data[i]

print("Writing into output into: ", "pred_"+test_file)
with open("test_data_predicted_CRF.json", 'w') as fp:
    json.dump(test_data, fp)
