import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
import os
import tensorflow as tf


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

checkpoint_dir = "checkpoints/ckpt"
checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
latest_checkpoint = max(checkpoints, key=os.path.getctime)

x_test = pad_sequences(tokenizer.texts_to_sequences(["this is movie is not good as it should be i did not like the movie worst movie"]), maxlen=300)

model = load_model(latest_checkpoint)
score = model.predict([x_test])


