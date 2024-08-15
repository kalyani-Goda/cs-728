import os
import itertools
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch
import json

with open('train_data_preprocessed.json') as f:
	data = json.load(f)

train_sentences, train_tags = [], []

for i in range(len(data)):
	tags_list = data[i]['tags']
	sent_list = data[i]['sent']
	train_sentences.append(np.array(sent_list))
	train_tags.append(np.array(tags_list))

with open('test_data_preprocessed.json') as f:
	test_data = json.load(f)

test_sentences, test_tags = [], []

for i in range(len(test_data)):
	tags_list = data[i]['tags']
	sent_list = data[i]['sent']
	test_sentences.append(np.array(sent_list))
	test_tags.append(np.array(tags_list))


words, tags = set([]), set([])

 
for s in train_sentences:
	for w in s:
		words.add(w)
 
for ts in train_tags:
	for t in ts:
		tags.add(t)

label_list = list(tags)

task = "ner" 
model_checkpoint = "distilbert-base-uncased"
batch_size = 16
	
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def get_all_tokens_and_ner_tags(name):
	return pd.concat([get_tokens_and_ner_tags(name)]).reset_index().drop('index', axis=1)
	
def get_tokens_and_ner_tags(name):
	if name=='train': 
		return pd.DataFrame({'tokens':train_sentences , 'ner_tags':train_tags })
	if name=='test':
		return pd.DataFrame({'tokens':test_sentences , 'ner_tags':test_tags })
  
def get_dataset():
	train_df = get_all_tokens_and_ner_tags('train')
	test_df = get_all_tokens_and_ner_tags('test')
	train_dataset = Dataset.from_pandas(train_df)
	test_dataset = Dataset.from_pandas(test_df)

	return (train_dataset, test_dataset)

train_dataset, test_dataset = get_dataset()

print(train_dataset)

def tokenize_and_align_labels(examples):
	label_all_tokens = True
	tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

	labels = []
	for i, label in enumerate(examples[f"{task}_tags"]):
		word_ids = tokenized_inputs.word_ids(batch_index=i)
		previous_word_idx = None
		label_ids = []
		for word_idx in word_ids:
			if word_idx is None:
				label_ids.append(-100)
			elif label[word_idx] == '0':
				label_ids.append(0)
			previous_word_idx = word_idx
		labels.append(label_ids)
		
	tokenized_inputs["labels"] = labels
	return tokenized_inputs


train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)

print(train_tokenized_datasets)

def train():
	model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

	args = TrainingArguments(
		f"test-{task}",
		evaluation_strategy = "epoch",
		learning_rate=1e-4,
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=batch_size,
		num_train_epochs=50,
		weight_decay=1e-5,
	)

	data_collator = DataCollatorForTokenClassification(tokenizer)
	metric = load_metric("seqeval")


	def compute_metrics(p):
		predictions, labels = p
		predictions = np.argmax(predictions, axis=2)

		true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
		true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

		results = metric.compute(predictions=true_predictions, references=true_labels)
		return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}
		
	trainer = Trainer(
		model,
		args,
		train_dataset=train_tokenized_datasets,
		eval_dataset=test_tokenized_datasets,
		data_collator=data_collator,
		tokenizer=tokenizer,
		compute_metrics=compute_metrics
	)

	trainer.train()
	trainer.evaluate()
	trainer.save_model('bert.model')


def test():
	tokenizer = AutoTokenizer.from_pretrained('./bert.model/')

	sentence = 'Delhi is capital of India.'
	tokens = tokenizer(sentence)
	torch.tensor(tokens['input_ids']).unsqueeze(0).size()

	model = AutoModelForTokenClassification.from_pretrained('./bert.model/', num_labels=len(label_list))
	predictions = model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0), attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
	predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
	predictions = [label_list[i] for i in preds]

	words = tokenizer.batch_decode(tokens['input_ids'])
	print(words)
	print(predictions)

#train()

#test()