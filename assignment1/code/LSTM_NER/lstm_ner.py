import numpy as np
import json


with open('train_data_preprocessed.json') as f:
	data = json.load(f)

with open('test_data_preprocessed.json') as f:
	test_data = json.load(f)

train_sentences, train_tags = [], []

for i in range(len(data)):
	tags_list = data[i]['tags']
	sent_list = data[i]['sent']
	train_sentences.append(np.array(sent_list))
	train_tags.append(np.array(tags_list))

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

print(tags)


 
word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0  # The special value used for padding
word2index['-OOV-'] = 1  # The special value used for OOVs
 
tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0  # The special value used to padding

train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []
 
for s in train_sentences:
	s_int = []
	for w in s:
		try:
			s_int.append(word2index[w])
		except KeyError:
			s_int.append(word2index['-OOV-'])
 
	train_sentences_X.append(s_int)
 
for s in test_sentences:
	s_int = []
	for w in s:
		try:
			s_int.append(word2index[w])
		except KeyError:
			s_int.append(word2index['-OOV-'])
 
	test_sentences_X.append(s_int)
 
for s in train_tags:
	train_tags_y.append([tag2index[t] for t in s])
 
for s in test_tags:
	test_tags_y.append([tag2index[t] for t in s])
 
print(train_sentences_X[0])
print(test_sentences_X[0])
print(train_tags_y[0])
print(test_tags_y[0])
 

MAX_LENGTH = len(max(train_sentences_X, key=len))
print(MAX_LENGTH)


from keras.preprocessing.sequence import pad_sequences
 
train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')
 
print(train_sentences_X[0])
print(test_sentences_X[0])
print(train_tags_y[0])
print(test_tags_y[0])


from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam


def to_categorical(sequences, categories):
	cat_sequences = []
	for s in sequences:
		cats = []
		for item in s:
			cats.append(np.zeros(categories))
			cats[-1][item] = 1.0
		cat_sequences.append(cats)
	return np.array(cat_sequences)



 
def train():
	model = Sequential()
	model.add(InputLayer(input_shape=(MAX_LENGTH, )))
	model.add(Embedding(len(word2index), 128))
	model.add(Bidirectional(LSTM(256, return_sequences=True)))
	model.add(TimeDistributed(Dense(len(tag2index))))
	model.add(Activation('softmax'))
	 
	model.compile(loss='categorical_crossentropy',
				  optimizer=Adam(0.001),
				  metrics=['accuracy'])
	 
	model.summary()

	cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
	print(cat_train_tags_y[0])


	model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=200, validation_split=0.2)
	 
	scores = model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))
	print(f"{model.metrics_names[1]}: {scores[1] * 100}")   # acc: 99.09751977804825


	model.save('lstm_model') 


def logits_to_tokens(sequences, index):
	token_sequences = []
	for categorical_sequence in sequences:
		token_sequence = []
		for categorical in categorical_sequence:
			token_sequence.append(index[np.argmax(categorical)])
 
		token_sequences.append(token_sequence)
 
	return token_sequences



from tensorflow import keras


def test():

	model = keras.models.load_model('lstm_model')


	test_samples = [
		"Running is very important for IIT Bombay students.".split()
	]
	print(test_samples)



	test_samples_X = []
	for s in test_samples:
		s_int = []
		for w in s:
			try:
				s_int.append(word2index[w.lower()])
			except KeyError:
				s_int.append(word2index['-OOV-'])
		test_samples_X.append(s_int)
	 
	test_samples_X = pad_sequences(test_samples_X, maxlen=MAX_LENGTH, padding='post')
	print(test_samples_X)



	predictions = model.predict(test_samples_X)
	print(predictions, predictions.shape)
	print(logits_to_tokens(predictions, {i: t for t, i in tag2index.items()}))


#train()


#test()







 
