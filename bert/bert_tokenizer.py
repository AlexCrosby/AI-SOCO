import sys
import time

sys.path.append('..')
import numpy as np
from transformers import RobertaTokenizerFast
from tools.datatools import load_data
from multiprocessing import Process


# This generates tokens for each file in the corpus. RobertaTokenizerFast is used to speed up the process and is
# performed in batches of 1000 to prevent OOM errors.
# The tokens can then be used in the bert model.

# config_path = "../data_dir/codebert/config.json"
# model_path = "../data_dir/codebert/pytorch_model.bin"
def main():
    train_x, _, dev_x, _, _, test_x, _, _ = load_data(r'../data_dir/', bytes=False, preprocess=True)
    del _

    print("Tokenizing.")
    length = 100
    start = time.time()
    processes = [Process(target=tokenize, args=(train_x, 'train', length)),
                 Process(target=tokenize, args=(dev_x, 'dev', length)),
                 Process(target=tokenize, args=(test_x, 'test', length))]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    print("Finalising writing tokens to file.")
    print(time.time() - start)


def tokenize(data_x, name, length):
    print(f"Started tokenizer for {name} at length {length}")
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    mm = np.memmap(f'vectors/{name}_{length}.mm', dtype='int32', mode='w+', shape=(len(data_x), 3, length))
    for i in range((int(len(data_x) / 1000))):  # This is run in batches of 1000 due to memory. Slow tokenizer has
        # issues with some files which causes it to take hours even though batching isn't needed
        tokens = tokenizer.batch_encode_plus(data_x[(i * 1000):((i + 1) * 1000)], add_special_tokens=True,
                                             pad_to_max_length=True, truncation=True, max_length=length,
                                             return_attention_mask=True, return_token_type_ids=True,
                                             return_tensors='np')
        mm[(i * 1000):((i + 1) * 1000), 0, :] = np.array(tokens.get('input_ids'))
        mm[(i * 1000):((i + 1) * 1000), 1, :] = np.array(tokens.get('attention_mask'))
        mm[(i * 1000):((i + 1) * 1000), 2, :] = np.array(tokens.get('token_type_ids'))
    print(f"Finished tokenizer for {name} at length {length}")


if __name__ == '__main__':
    main()
