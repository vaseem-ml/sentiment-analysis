import os
import tensorflow as tf

import pandas as pd
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import h5py



#import dataset
data=pd.read_csv('drugsCom_raw/drugsComTest_raw.tsv', sep="\t")
data['rating'] = [1 if int(x)>5 else 0 for x in data['rating']]
X = data.iloc[:,[3]]
Y = data.iloc[:,4]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("TRAIN size:", len(X_train))
print("TEST size:", len(X_test))



# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 3
W2V_MIN_COUNT = 10
 
documents = [_text.split() for _text in X_train.review] 
w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 
                                            window=W2V_WINDOW, 
                                            min_count=W2V_MIN_COUNT, 
                                            workers=8)
w2v_model.build_vocab(documents)

words = w2v_model.wv.vocab.keys()

vocab_size = len(words)

w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)



# Tokenizing

 
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 300
# This is fixed.
EMBEDDING_DIM = 300
 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train.review)
word_index = tokenizer.word_index
vocab_size = len(word_index)
print('Found %s unique tokens.' % len(word_index))
 
# Convert the data to padded sequences
X_train_padded = tokenizer.texts_to_sequences(X_train.review)
X_train_padded = pad_sequences(X_train_padded, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X_train_padded.shape)


# Embedding matrix for the embedding layer
embedding_matrix = np.zeros((vocab_size+1, W2V_SIZE))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
print(embedding_matrix.shape)





def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Embedding(vocab_size+1, W2V_SIZE, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)(inputs)
    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(1, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
        loss='binary_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )
    return model
# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()


model = make_or_restore_model()
callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-loss={loss:.2f}",
            #save_weights_only=True,
            #monitor='val_acc',
            #mode='max',
            save_freq=100,
            #save_best_only=False
            ),
        
]
model.fit(X_train_padded, y_train,
                    batch_size=8,
                    epochs=1,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=callbacks)
model.save('newTest')