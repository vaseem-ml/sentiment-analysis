import argparse
import os
import keras
import h5py
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout



from preprocess import preprocess_data, prepare_glove_model, tokenzie_and_embedding_matrix, make_or_restore_model


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', default="checkpoints/ckpt")
parser.add_argument('--w2v_size', default=300)
parser.add_argument('--w2v_window', default=7)
parser.add_argument('--w2v_epoch', default=32)
parser.add_argument('--w2v_min_count', default=10)
parser.add_argument('--max_sequence_length', default=300)
parser.add_argument('--embedding-dim', default=300)
parser.add_argument('--epochs', default=10)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--validation_split', default=0.1)





args = parser.parse_args()


callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=args.checkpoint_dir + "/ckpt-loss={loss:.2f}",
            #save_weights_only=True,
            #monitor='val_acc',
            #mode='max',
            save_freq=100,
            #save_best_only=False
            ),
        
]

if __name__ == '__main__':
    
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    X, Y = preprocess_data()
    
    
    #print(args.w2v_epoch)
    w2v_model, vocab_sizee=prepare_glove_model(args.w2v_size, args.w2v_window, args.w2v_epoch, args.w2v_min_count, X)
    #print('this is vocab size', vocab_size)
    embedding_metrix, vocab_size, X_train_padded = tokenzie_and_embedding_matrix(X, args.max_sequence_length, args.w2v_size, w2v_model)

    model = make_or_restore_model(args.checkpoint_dir, vocab_size, args.w2v_size, embedding_metrix, args.max_sequence_length)
    
    history = model.fit(X_train_padded, Y,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    validation_split=args.validation_split,
                    verbose=1,
                    callbacks=callbacks)

    