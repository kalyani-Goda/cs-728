import json

def f1(p, r):
    if r == 0.:
        return 0.
    return 2 * p * r / float( p + r )


def loose_macro(true, pred):
    num_entities = len(true)
    p = 0.
    r = 0.
    for true_labels, predicted_labels in zip(true, pred):
        if len(predicted_labels) > 0:
            p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
        if len(true_labels):
            r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
    precision = p / num_entities
    recall = r / num_entities
    return precision, recall, f1( precision, recall)


def loose_micro(true, pred):
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    for true_labels, predicted_labels in zip(true, pred):
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set(predicted_labels).intersection(set(true_labels))) 
    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.
    recall = num_correct_labels / num_true_labels
    return precision, recall, f1( precision, recall)

with open('test_data_predicted_LSTM_2.json') as f:           #change the file name as required (in /output folder)
    data = json.load(f)

predicted_list = []

for i in range(len(data)):
    tags_list = data[i]['tags']
    predicted_list+= tags_list


with open('test_data_preprocessed.json') as f:              #can be found in /additional folder
    data = json.load(f)

true_list = []

for i in range(len(data)):
    tags_list = data[i]['tags']
    true_list+= tags_list

print(len(predicted_list),len(true_list))

precision,recall,f1_micro = loose_micro(true_list,predicted_list)
print(precision,recall,f1_micro)
precision,recall,f1_macro = loose_macro(true_list,predicted_list)
print(precision,recall,f1_macro)

matched_count = 0
for i in range(len(true_list)):
    if true_list[i] == predicted_list[i]:
        matched_count+=1

print(matched_count)

