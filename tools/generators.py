import math
import numpy as np
from tensorflow.keras.utils import Sequence

# Various iterations of generators for different uses.
class Generator(Sequence):
    def __init__(self, data_x, data_y, batch_size):
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.max = len(data_x)

    def __len__(self):
        return math.ceil(len(self.data_x) / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.max)
        return np.array(self.data_x[start:end]), np.array(self.data_y[start:end])



class GeneratorBERT(Generator):
    def __init__(self, data_x, data_y, batch_size, mode):
        super(GeneratorBERT, self).__init__(data_x, data_y, batch_size)
        self.mode = mode

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.max)
        # if end > self.max:
        #     end = self.max
        if self.mode == 1:
            # return np.array(self.data_x[start:end, 0, :]), np.array(self.data_y[start:end])
            return {'input_ids': np.array(self.data_x[start:end, 0, :])}, np.array(self.data_y[start:end])
        if self.mode == 2:
            # return [np.array(self.data_x[start:end, 0, :]), np.array(self.data_x[start:end, 1, :])], np.array(
            #     self.data_y[start:end])
            return {'input_ids': np.array(self.data_x[start:end, 0, :]),
                    'attention_mask': np.array(self.data_x[start:end, 1, :])}, np.array(self.data_y[start:end])
        if self.mode == 3:
            # return [np.array(self.data_x[start:end, 0, :]), np.array(self.data_x[start:end, 1, :]), np.array(
            #     self.data_x[start:end, 2, :])], np.array(self.data_y[start:end])
            return {'input_ids': np.array(self.data_x[start:end, 0, :]),
                    'attention_mask': np.array(self.data_x[start:end, 1, :]),
                    'token_type_ids': np.array(self.data_x[start:end, 2, :])}, np.array(self.data_y[start:end])


class GeneratorX(Generator): # Single output generator when labels not needed.
    def __init__(self, data_x, batch_size):
        super(GeneratorX, self).__init__(data_x, None, batch_size)
        self.data_x = data_x
        self.batch_size = batch_size
        self.max = len(data_x)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.max)
        return self.data_x[start:end]


class GeneratorHydra(Generator): # Generator for multi headed concatenation ensemble.
    def __init__(self, ngram_x, ast_x, style_x,bert_x, data_y, batch_size):
        super(GeneratorHydra, self).__init__(ngram_x, data_y, batch_size)
        self.ast_x = ast_x
        self.style_x = style_x
        self.bert_x=bert_x

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.max)
        dict = {}
        dict['ngram'] = np.array(self.data_x[start:end])
        dict['ast'] = np.array(self.ast_x[start:end])
        dict['style'] = np.array(self.style_x[start:end])
        dict['bert'] = np.array(self.bert_x[start:end])
        return dict, np.array(self.data_y[start:end])
