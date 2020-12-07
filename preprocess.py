import pandas as pd
import gensim
import os
import pickle
import numpy as np
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences




class CustomModel(keras.Model):
    def train_step(self, data):
         # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def preprocess_data():
    data = pd.read_csv('drugsCom_raw/drugsComTest_raw.tsv', sep="\t")
    data['rating'] = [1 if int(x)>5 else 0 for x in data['rating']]
    X = data.iloc[:,[3]]
    Y = data.iloc[:,4]
    TAG_CLEANING_RE = "@\S+"
    # Remove @tags
    X['review'] = X['review'].map(lambda x: re.sub(TAG_CLEANING_RE, ' ', x))
    X['review'] = X['review'].map(lambda x: x.lower())
    X['review'] = X['review'].map(lambda x: re.sub(r'\d+', ' ', x))
    TEXT_CLEANING_RE = "https?:\S+|http?:\S|[^A-Za-z0-9]+"
    X['review'] = X['review'].map(lambda x: re.sub(TEXT_CLEANING_RE, ' ', x))
    X['review'] = X['review'].map(lambda x: re.sub("i ve", 'i have', x))
    X['review'] = X['review'].map(lambda x: re.sub("i m", 'i am', x))
    X['review'] = X['review'].map(lambda x: x.strip())
    
    return X, Y


def prepare_glove_model(W2V_SIZE, W2V_WINDOW, W2V_EPOCH, W2V_MIN_COUNT, X):
    file_directory="w2v_model.model"
    if os.path.isfile(file_directory):
        print("############## word2vec model is already exists##############")
        model = gensim.models.Word2Vec.load("w2v_model.model")
        vocab_size = model.wv.vocab.keys()
        return model, len(vocab_size) 
    print("###########Creating word2vec model#################")
    documents = [_text.split() for _text in X.review]
    w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE, 
                                            window=W2V_WINDOW, 
                                            min_count=W2V_MIN_COUNT,
                                            workers=8)
    print("###########Building word2vec model#################")
    w2v_model.build_vocab(documents)
    words = w2v_model.wv.vocab.keys()
    vocab_size = len(words)
    print("###########Training word2vec model#################")
    w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
    w2v_model.init_sims(replace=True)
    w2v_model.save("w2v_model.model")
    return w2v_model, vocab_size
    


def tokenzie_and_embedding_matrix(X, MAX_SEQUENCE_LENGTH, W2V_SIZE, w2v_model):
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(X.review)
    word_index = tokenizer.word_index
    vocab_size = len(word_index)
     
    # Convert the data to padded sequences
    X_train_padded = tokenizer.texts_to_sequences(X.review)
    X_train_padded = pad_sequences(X_train_padded, maxlen=MAX_SEQUENCE_LENGTH)
    
    
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    # Embedding matrix for the embedding layer
    embedding_matrix = np.zeros((vocab_size+1, W2V_SIZE))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    print(embedding_matrix.shape)
    return embedding_matrix, vocab_size, X_train_padded


def get_uncompiled_model(vocab_size, W2V_SIZE, embedding_metrix, MAX_SEQUENCE_LENGTH): 
    inputs = keras.Input(shape=(vocab_size+1,), name="digits")
    x = layers.Embedding(vocab_size+1, W2V_SIZE, weights=[embedding_metrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)(inputs)
    #x = layers.Dropout(0.5)(x)
    x = layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2)(x)
    #x = layers.Dense(64, activation="relu", name="dense_1")(x)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
    
def get_compiled_model(vocab_size, W2V_SIZE, embedding_matrix, MAX_SEQUENCE_LENGTH):
    model = get_uncompiled_model(vocab_size, W2V_SIZE, embedding_matrix, MAX_SEQUENCE_LENGTH)
    model.compile(
        loss='binary_crossentropy',
        optimizer="adam",
        metrics=['accuracy']
    )
    return model
    
def make_or_restore_model(checkpoint_dir, vocab_size, W2V_SIZE, embedding_matrix, MAX_SEQUENCE_LENGTH):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model(vocab_size, W2V_SIZE, embedding_matrix, MAX_SEQUENCE_LENGTH)
    #return get_uncompiled_model()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


preprocess_data()