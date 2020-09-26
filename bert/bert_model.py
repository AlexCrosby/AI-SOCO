import sys

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from transformers import TFRobertaForSequenceClassification

sys.path.append('..')
from tools.datatools import load_all_labels
from tools.generators import GeneratorBERT

# This version of codeBERT uses transformers own classifier model.

# This is the bert script for running on AWS. Token representations for each file need to be pre-generated and saved before using this.

length = 200  # 200 is used due to: 1, it was used in the codeBert paper, 2, 512 results in OOM errors and, 3, 512
# doesn't confer better accuracy anyway.
model_path = "../data_dir/codebert/pytorch_model.bin"
dev_path = f"vectors/dev_{length}.mm"
train_path = f"vectors/train_{length}.mm"

mode = 3  # Use tokens, attention mask, and node types.
batch_size = 16
# Load tokens and labels and set up generators to feed into NN.
train_y, dev_y, _ = load_all_labels("../data_dir")
dev_batch = np.array(np.memmap(dev_path, dtype='int32', mode='r', shape=(25000, 3, length)))
train_batch = np.array(np.memmap(train_path, dtype='int32', mode='r', shape=(50000, 3, length)))
dev = GeneratorBERT(dev_batch, dev_y, batch_size, mode)
train = GeneratorBERT(train_batch, train_y, batch_size, mode)
# Saving model callback
callback_list = [ModelCheckpoint(filepath='bert_model_{epoch:02d}_{val_loss:.2f}_{val_acc:.2f}.h5',
                                 monitor='val_loss', save_best_only=True)]
#Load codeBERT model weights from file into RoBERTa model.
model = TFRobertaForSequenceClassification.from_pretrained(model_path, config='roberta-base', from_pt=True,
                                                           num_labels=1000)
# model.load_weights('bert_model.h5') For continuing party fine-tunes model.
loss = SparseCategoricalCrossentropy(from_logits=True)
opt = Adam(learning_rate=2e-5)
model.summary()
model.compile(optimizer=opt, loss=loss, metrics=['acc'])
model.fit(train, epochs=10, validation_data=dev, callbacks=callback_list, shuffle=True)
