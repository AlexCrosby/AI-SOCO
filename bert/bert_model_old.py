import sys

sys.path.append('..')
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import TFRobertaModel

from tools.datatools import load_all_labels
from tools.generators import GeneratorBERT

length = 200
config_path = "../data_dir/codebert/config.json"
model_path = "../data_dir/codebert/pytorch_model.bin"
dev_path = "vectors/dev_{}.mm".format(length)
train_path = "vectors/train_{}.mm".format(length)
# label_path = "../ensemble/labels.joblib"

mode = 3
batch_size = 256

train_y, dev_y, _ = load_all_labels("../data_dir")

dev_batch = np.memmap(dev_path, dtype='int32', mode='r', shape=(25000, 3, length))
train_batch = np.memmap(train_path, dtype='int32', mode='r', shape=(50000, 3, length))

dev = GeneratorBERT(dev_batch, dev_y, batch_size, mode)
train = GeneratorBERT(train_batch, train_y, batch_size, mode)

bert_model = TFRobertaModel.from_pretrained(model_path, config='roberta-base', from_pt=True)

input_mode = {}
input_mode['input_ids'] = Input(shape=(length,), dtype='int32')
if mode >= 2:
    input_mode['attention_mask'] = Input(shape=(length,), dtype='int32')
if mode >= 3:
    input_mode['token_type_ids'] = Input(shape=(length,), dtype='int32')

bert_layer = bert_model(input_mode)[1]
bert_model.trainable = False
# There are two different outputs 0 = last_hidden_state, 1 = pooled

dense1_layer = Dense(1000, activation='softmax')(bert_layer)
# dense2_layer = Dense(1000, activation='softmax')(dense1_layer)
model = Model(inputs=input_mode, outputs=dense1_layer)
model.summary()

loss = SparseCategoricalCrossentropy(from_logits=True)
callback_list = [
    ModelCheckpoint(filepath='my_modelv2_acc_{epoch:02d}_{val_loss:.2f}_{val_acc:.2f}.h5', monitor='val_acc',
                    save_best_only=True),
    ModelCheckpoint(filepath='my_modelv2_loss_{epoch:02d}_{val_loss:.2f}_{val_acc:.2f}.h5',
                    monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)]
opt = Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss=loss, metrics=['acc'])
model.fit(train, epochs=1000000, validation_data=dev)
