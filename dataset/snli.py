from torch.utils.data import Dataset
from datasets import load_dataset
import nltk
import torch
nltk.download("punkt", quiet=True)


class SNLIDataset(Dataset):
    def __init__(self, data_split, glove_mapping):
        self.data_split = data_split
        self.dataset = load_dataset("snli")[self.data_split]
        self.dataset = [data for data in self.dataset if data["label"] != -1]
        
        self.max_sentence_length = 0
        self.glove_vocab_index_mapping = glove_mapping

        for data in self.dataset:
            premise = nltk.word_tokenize(data["premise"])
            hypothesis = nltk.word_tokenize(data["hypothesis"])

            self.max_sentence_length = max(
                self.max_sentence_length, len(premise))

            self.max_sentence_length = max(
                self.max_sentence_length, len(hypothesis))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        premise, hypothesis, label = self.dataset[idx].values()

        premise = nltk.word_tokenize(premise.lower())
        premise = [
            self.glove_vocab_index_mapping[token]
            if token in self.glove_vocab_index_mapping else 0
            for token in premise
        ]

        hypothesis = nltk.word_tokenize(hypothesis.lower())
        hypothesis = [
            self.glove_vocab_index_mapping[token]
            if token in self.glove_vocab_index_mapping else 0
            for token in hypothesis
        ]

        premise = torch.squeeze(torch.tensor([
            premise + [1] * (self.max_sentence_length - len(premise))
        ]))

        hypothesis = torch.squeeze(torch.tensor([
            hypothesis + [1] * (self.max_sentence_length - len(hypothesis))
        ]))

        label = torch.tensor(label)

        return premise, hypothesis, label
