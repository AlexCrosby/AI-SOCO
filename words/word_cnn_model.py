import sys

sys.path.append('..')
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from tensorflow.keras.layers import Embedding, Dense, LSTM, Conv1D, MaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from tools.datatools import load_data

train_x, train_y, dev_x, dev_y, _, test_x, _, _ = load_data('../data_dir')

# Word CNN model.

tokenizer = Tokenizer(num_words=2048)  # Tokenize source codes using top 2048 words
tokenizer.fit_on_texts(train_x)
train_y = np.array(train_y)
dev_y = np.array(dev_y)
train_x = tokenizer.texts_to_sequences(train_x)
dev_x = tokenizer.texts_to_sequences(dev_x)

train_x = pad_sequences(train_x, maxlen=512)  # Pad sequences to uniform length
dev_x = pad_sequences(dev_x, maxlen=512)
callback_list = [EarlyStopping(monitor='val_acc', patience=5),
                 ModelCheckpoint(filepath='word_model2.h5', monitor='val_acc', save_best_only=True),
                 ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=3)]
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 32,
                    input_length=512))  # Embedding layer -> words converted to 32-D embeddings
model.add(Conv1D(kernel_size=3, filters=32))  # Convolution layer to identify patterns
model.add(MaxPool1D(pool_size=2)) # Pool to half size.
model.add(LSTM(128, dropout=0.3)) # LSTM layer
model.add(Dense(1000, activation='softmax'))
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
plot_model(model, 'cnn_model.png', show_shapes=True)
history = model.fit(train_x, train_y, epochs=10000, batch_size=64, validation_data=(dev_x, dev_y),
                    callbacks=callback_list)
model.load_weights('word_model2.h5')
model.evaluate(dev_x, dev_y)
