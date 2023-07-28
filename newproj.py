import seaborn as sns
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os

import emoji

import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, SimpleRNN, LSTM, Activation, Bidirectional

#import tensorflow as tf

from sklearn.metrics import confusion_matrix


# load the train data:
train_data = pd.read_csv('train_emoji_n.csv', header=None)
train_data.head()

# Load the test data

test_data = pd.read_csv('test_emoji_n1.csv', header=None)
test_data.head()


# drop columns 2 and 3 from our train data:

train_data.drop(labels=[2, 3], axis=1, inplace=True)
train_data.head()

# Here

emoji_dict = {
    '0': ':grinning_face:',
    '1': ':pensive_face:',
    '2': ':angry_face:',
    '3': ':anguished_face:'
}

# for e in emoji_dict.values():
#     print(emoji.emojize(e), end=' ')

# pre processing:

X_train = train_data[0].values
Y_train = train_data[1].values

X_train[:10]

Y_train[:10]

# X_train.shape, Y_train.shape

# We are embedding the text as we are going to create RNN model:

f = open('glove.6B.50d.txt', encoding='utf8', mode='r')

embedding_matrix = {}

for line in f:
    values = line.split()
    word = values[0]
    emb = np.array(values[1:], dtype='float')

    embedding_matrix[word] = emb

# embedding_matrix

# We will create a function which will give embedding of our text data:


def get_embedding_matrix_for_data(data):
    max_len = 10
    embedding_data = np.zeros((len(data), 12, 50))

    for x in range(data.shape[0]):
        word_in_sen = data[x].split()

        for y in range(len(word_in_sen)):
            if embedding_matrix.get(word_in_sen[y].lower()) is not None:
                embedding_data[x][y] = embedding_matrix[word_in_sen[y].lower()]

    return embedding_data


X_train = get_embedding_matrix_for_data(X_train)

# X_train.shape

# covert the output to categorical:


Y_train = to_categorical(Y_train)

# Y_train

# Create Model:

# build our model:

model = Sequential()
model.add(Bidirectional(LSTM(64, input_shape=(12, 50), return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128, return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))
model.add(Input(shape=(1,)))

model.build(input_shape=(1, 12, 50))

model.summary()

model.compile(optimizer='adam',
              loss=keras.losses.categorical_crossentropy, metrics=['acc'])

history = model.fit(X_train, Y_train, validation_split=0.2,
                    batch_size=32, epochs=50)

# plot acuuracy and loss graph:

plt.figure(figsize=(8, 6))
plt.title(' Accuracy scores')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['accuracy', 'val_accuracy'])
# plt.show()

plt.figure(figsize=(8, 6))
plt.title('Loss value')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
# plt.show()

print(model.evaluate(X_train, Y_train)[1])

X_test = test_data[0].values
Y_test = test_data[1].values

# print(X_test)

# print(X_test.shape)

X_test = get_embedding_matrix_for_data(X_test)
Y_test = to_categorical(Y_test)

# print(Y_test)

print(model.evaluate(X_test, Y_test)[1])

# Predicting through constructed model

Y_pred = model.predict(X_test)
classes_x = np.argmax(Y_pred, axis=1)

# for t in range(len(test_data)):
#     print(test_data[0].iloc[t])
#     print("predicted:", emoji.emojize(
#         emoji_dict[str(np.argmax(Y_pred[t], axis=0))]))
#     print('Actual: ', emoji.emojize(emoji_dict[str(test_data[1].iloc[t])]))
#     print("")

# Confusion matrix for multi-class classification
y_predict = model.predict(X_test)
y_predict = np.argmax(y_predict, axis=1)
y_testing = np.argmax(Y_test, axis=1)
confusion_matrix(y_testing, y_predict)
cm = confusion_matrix(y_testing, y_predict, labels=[0, 1, 2, 3])

# print(cm)

# Plotting the confusion matrix
# sn = sns.heatmap(confusion_matrix(y_testing, y_predict), label=[
#                  'Actual Values', 'Predicted Values'], annot=True, cmap='Purples')
# sn.set(xlabel='Predicted', ylabel='Actual Values')
# sn.show()

epochs = 50

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), history.history["loss"], label="Train Loss")
plt.plot(np.arange(0, epochs), history.history["acc"], label="Train Acc")

plt.title("Loss and Accuracy plot")
plt.xlabel("Epoch")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.jpg")
# plt.show()

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
