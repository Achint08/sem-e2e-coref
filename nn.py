import numpy as np
import preprop as pp
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from keras.utils import Progbar, plot_model
from keras.backend import max
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform

trainSentences = pp.readfile("end.v4_auto_conll")
# devSentences = pp.readfile("dev.english.v4_auto_conll")
# testSentences = pp.readfile("test.english.v4_gold_conll")

words = {}
wordEmbeddings = []
word2Idx = {}

for dataset in [trainSentences]:
	for sentence in dataset:
		for genre,token,char,speaker in sentence[0]:
			words[token.lower()] = True


fEmbeddings = open("glove.6B.50d.txt", encoding="utf-8")
turianEmbeddings = open("turian.50d.txt", encoding="utf-8")

for line in fEmbeddings:
	split = line.strip().split(" ")
	word = split[0]
	if len(word2Idx) == 0:
		word2Idx["PADDING_TOKEN"] = len(word2Idx)
		vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
		wordEmbeddings.append(vector)
		word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
		vector = np.zeros(len(split)-1)
		wordEmbeddings.append(vector)
	if split[0].lower() in words:
		vector = np.array([float(num) for num in split[1:]])
		wordEmbeddings.append(vector)
		word2Idx[split[0]] = len(word2Idx)

wordEmbeddings = np.array(wordEmbeddings)
print(wordEmbeddings.shape)
char2Idx = {"PADDING":0, "UNKNOWN":1}
for c in "!\"#$%&'*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{}~·ÌÛàòö˙ِ’→■□●【】の・一（）＊：￥":
	char2Idx[c] = len(char2Idx)

train_set = pp.padding(pp.createMatrices(trainSentences,word2Idx,char2Idx))
# dev_set = padding(createMatrices(trainSentences,word2Idx,char2Idx))
# test_set = padding(createMatrices(trainSentences,word2Idx,char2Idx))

train_batch = pp.createBatches(train_set)
# dev_batch = pp.createBatches(dev_set)
# test_batch = pp.createBatches(test_set)

# words_input = Input(shape=(None,),dtype='int32',name='words_input')
# words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1], weights=[wordEmbeddings], trainable=False)(words_input)
# dropout = Dropout(0.5)(words)
# character_input=Input(shape=(None,50,),name='char_input')
# embed_char_out=Embedding(len(char2Idx),8,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5))(character_input)
# conv1d_out_0= Conv1D(kernel_size=3, filters=50, use_bias=True, padding='same',activation='relu', strides= 1)(embed_char_out)
# conv1d_out_1= Conv1D(kernel_size=4, filters=50, use_bias=True, padding='same',activation='relu', strides= 1)(embed_char_out)
# conv1d_out_2= Conv1D(kernel_size=5, filters=50, use_bias=True, padding='same',activation='relu', strides= 1)(embed_char_out)
# maxpool_out= MaxPooling1D(50)(conv1d_out)
# char = TimeDistributed(Flatten())(maxpool_out)
# char = Dropout(0.5)(char)
# output = concatenate([words, char])
# output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.2, recurrent_initializer= "Orthogonal", recurrent_dropout=0.2))(output)
# model = Model(inputs=[words_input, character_input], outputs=[output])
# model.summary()
# plot_model(model, to_file='model.png')
