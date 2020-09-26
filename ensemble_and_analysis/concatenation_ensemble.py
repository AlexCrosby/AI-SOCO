import sys

import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, LayerNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2

sys.path.append('..')
from tools.datatools import load_all_labels
from tools.generators import GeneratorHydra

# Concatenation ensemble. This model overfits too much due to its size and was discontinued.
def main():
    # Load all precalculated vector representations for each model.
    train_y, dev_y, _ = load_all_labels('../data_dir')
    train_ngram = np.memmap('../ngrams/vectors/train.mm', dtype='float32', mode='r', shape=(50000, 20000))
    dev_ngram = np.memmap('../ngrams/vectors/dev.mm', dtype='float32', mode='r', shape=(25000, 20000))
    train_ast = np.memmap('../asts/vectors/train.mm', dtype='float32', mode='r', shape=(50000, 20000))
    dev_ast = np.memmap('../asts/vectors/dev.mm', dtype='float32', mode='r', shape=(25000, 20000))
    train_style = np.memmap('../stylometry/vectors/train.mm', dtype='float32', mode='r', shape=(50000, 136))
    dev_style = np.memmap('../stylometry/vectors/dev.mm', dtype='float32', mode='r', shape=(25000, 136))
    train_bert = np.memmap('../bert/vectors/train_bert.mm', dtype='float32', mode='r',
                           shape=(50000, 768))
    dev_bert = np.memmap('../bert/vectors/dev_bert.mm', dtype='float32', mode='r', shape=(25000, 768))
    train = GeneratorHydra(train_ngram, train_ast, train_style, train_bert, train_y, 256)
    dev = GeneratorHydra(dev_ngram, dev_ast, dev_style, dev_bert, dev_y, 256)
    ####################################################################################################################
    # Sets up input layer for each head.
    input_layers = {}
    input_layers['ngram'] = Input(shape=(20000,), dtype='float32')
    input_layers['ast'] = Input(shape=(20000,), dtype='float32')
    input_layers['style'] = Input(shape=(136,), dtype='float32')
    input_layers['bert'] = Input(shape=(768,), dtype='float32')
    ####################################################################################################################
    # n-gram section of model.
    ngram1 = Dense(3000, activation='relu')(input_layers['ngram'])
    ngram2 = Dropout(0.5)(ngram1)
    ngram3 = Dense(2000, activation='relu')(ngram2)
    ngram4 = Dropout(0.5)(ngram3)

    ####################################################################################################################
    # AST section of model.
    ast1 = Dense(1000, activation='relu')(input_layers['ast'])
    ast2 = Dropout(0.5)(ast1)
    ast3 = Dense(1000, activation='tanh')(ast2)
    ast4 = Dropout(0.5)(ast3)

    ####################################################################################################################
    # Stylometry section of model.
    style1 = Dense(500, activation='relu')(input_layers['style'])
    style2 = Dropout(0.3)(style1)
    style3 = Dense(500, activation='relu')(style2)
    style4 = Dropout(0.3)(style3)
    ####################################################################################################################
    #Bert section of model.
    bert1 = Dropout(0.3)(input_layers['bert'])
    ####################################################################################################################
    concatenated = Concatenate()([ngram4, bert1, ast4, style4]) # Concatenate final layer of above models
    batch = LayerNormalization()(concatenated)
    final2 = Dropout(0.8)(batch) # Overfitting is really high on this model.
    output = Dense(1000, activation='softmax', kernel_regularizer=l2(0.001))(final2)
    model = Model(inputs=input_layers, outputs=output)

    opt = RMSprop(learning_rate=0.001)
    loss = SparseCategoricalCrossentropy()
    model.compile(optimizer=opt, loss=loss, metrics=['acc'])
    model.summary()
    callback_list = [EarlyStopping(monitor='val_acc', patience=5),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)]
    model.fit(train, epochs=10000, validation_data=dev, callbacks=callback_list)


if __name__ == '__main__':
    main()
