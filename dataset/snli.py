from torch.utils.data import Dataset
from datasets import load_dataset
import nltk
from dataset.dataset_utils import get_glove_vocab_index_mapping
import torch
nltk.download("punkt", quiet=True)


class SNLIDataset(Dataset):
    def __init__(self, data_split, glove_mapping):
        self.data_split = data_split
        self.dataset = load_dataset("snli")[self.data_split]

        self.premises = []
        self.hypotheses = []
        self.labels = []
        self.max_sentence_length = 0

        self.glove_vocab_index_mapping = glove_mapping

        # tokenize + lowercase
        for i, data in enumerate(self.dataset):
            premise = nltk.word_tokenize(data["premise"].lower())

            # map tokens to vocab indices
            premise = [
                self.glove_vocab_index_mapping[token]
                if token in self.glove_vocab_index_mapping else 0
                for token in premise
            ]

            hypothesis = nltk.word_tokenize(data["hypothesis"].lower())

            # map tokens to vocab indices
            hypothesis = [
                self.glove_vocab_index_mapping[token]
                if token in self.glove_vocab_index_mapping else 0
                for token in hypothesis
            ]

            label = data["label"]

            self.premises.append(premise)
            self.hypotheses.append(hypothesis)
            self.labels.append(label)

            self.max_sentence_length = max(
                self.max_sentence_length, len(premise))

            self.max_sentence_length = max(
                self.max_sentence_length, len(hypothesis))

            # if i == 3:
                # break

        # add padding
        self.premises = [
            premise + [1] * (self.max_sentence_length - len(premise))
            for premise in self.premises
        ]

        self.hypotheses = [
            hypothesis + [1] * (self.max_sentence_length - len(hypothesis))
            for hypothesis in self.hypotheses
        ]

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        premise = self.premises[idx]
        hypothesis = self.hypotheses[idx]
        label = self.labels[idx]

        premise = torch.tensor(premise)
        hypothesis = torch.tensor(hypothesis)
        label = torch.tensor(label)

        return premise, hypothesis, label
