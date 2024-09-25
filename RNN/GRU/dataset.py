import torch
import os
from torch.utils.data import DataLoader, Dataset
import collections

class PTBDataset(Dataset):
    def __init__(self, data, seq_length=20):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, index):
        input_data = self.data[index: index + self.seq_length]
        target_data = self.data[index + 1:index + self.seq_length + 1]
        return torch.tensor(input_data, dtype=torch.long), torch.tensor(target_data, dtype=torch.long)


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


def load_data(data_path, batch_size, seq_length=20):

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = build_vocab(train_path)

    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)

    vocab_size = len(word_to_id)

    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
    
    train_dataset = PTBDataset(train_data, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataset = PTBDataset(valid_data, seq_length)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataset = PTBDataset(test_data, seq_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, valid_loader, test_loader, vocab_size

if __name__ == "__main__":
    data_path = 'RNN/ptb_data'
    train_loader, valid_loader, test_loader, vocab_size = load_data(data_path, batch_size=20)

    train_dataset_size = len(train_loader.dataset)

    for batch_idx, (data, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}: Input data shape: {data.shape}, Target data shape: {targets.shape}")
        
        # For testing, break after first batch to check the shape
        if batch_idx == 0:
            break