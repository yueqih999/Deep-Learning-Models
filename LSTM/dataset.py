import torch
import os
from torch.utils.data import DataLoader, Dataset
import collections

class PTBDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data) - 1
    
    def __getitem__(self, index):
        return self.data[index], self.data[index + 1]
    

def build_vocab(filename):
    data = read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))

    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def read_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().replace("\n", "<eos>").split()


def load_data(data_path, batch_size):

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = build_vocab(train_path)

    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)

    vocab_size = len(word_to_id)

    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
    
    train_dataset = PTBDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataset = PTBDataset(valid_data)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataset = PTBDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, valid_loader, test_loader, vocab_size

if __name__ == "__main__":
    data_path = 'LSTM/ptb_data'
    load_data(data_path, 32)