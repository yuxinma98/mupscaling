# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
from datasets import load_dataset # huggingface datasets
from transformers import AutoTokenizer

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 32

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = 16

_tok = None
_eot_id = None
def get_tokenizer():
    global _tok, _eot_id
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained("nanchennn/gpt2-tokenizer-8192", 
                                             use_fast=True)
        _eot_id = _tok.eos_token_id
    return _tok, _eot_id

# enc = AutoTokenizer.from_pretrained("nanchennn/gpt2-tokenizer-8192", 
#                                     use_fast=True)

if __name__ == '__main__':
    # takes 54GB in huggingface .cache dir, about 11M documents
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2013-20", num_proc=num_proc_load_dataset)


    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    # tokenize using the custom Hub tokenizer
    def process(example):
        tok, eot_id = get_tokenizer()
        ids = tok.encode(example['text'],
                        add_special_tokens=False)
        ids.append(eot_id)
        return {'ids': ids, 'len': len(ids)}

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin is ~22GB, val.bin ~11MB
    # train has ~11.8B tokens
    # val has ~5.5M tokens

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
