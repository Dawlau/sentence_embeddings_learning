import torch
import senteval
from dataset.dataset_utils import get_glove_vocab_index_mapping
from dataset.snli import SNLIDataset
import os
import argparse
from torch.utils.data import DataLoader
from train_module.train_utils import validation_step
import torch.nn as nn
import logging
import warnings
warnings.filterwarnings("ignore")

device = "cpu"
GLOVE_PATH = os.path.join("glove", "glove.840B.300d.txt")
glove_mapping = get_glove_vocab_index_mapping(GLOVE_PATH)
logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.DEBUG)


def prepare(params, samples):
    params.word_vec = glove_mapping
    params.embedding_dim = 300
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ["."] for sent in batch]
    embeddings = torch.tensor([]).to(device)

    for sentence in batch:
        sentence_idx = torch.tensor([
            params.word_vec[token]
            if token in params.word_vec else 0
            for token in sentence
        ]).to(device).unsqueeze(0)

        with torch.no_grad():
            embedding = encoder(sentence_idx)
            embeddings = torch.cat([embeddings, embedding])

    return embeddings.cpu().numpy()


def get_nli_accuracy(batch_size, num_workers):
    test_dataset = SNLIDataset("test", glove_mapping)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    _, test_acc = validation_step(
        encoder, classifier, test_loader, nn.CrossEntropyLoss(), device)

    return test_acc


def senteval_results(params_senteval):
    se = senteval.engine.SE(params_senteval, batcher, prepare)

    transfer_tasks = ["MR", "CR", "MPQA", "SUBJ", "SST2", "TREC",
                      "MRPC", "SICKEntailment"]

    senteval_results = se.eval(transfer_tasks)

    num_dev_samples = sum([
            senteval_results[task]["ndev"] for task in transfer_tasks])

    dev_accs = [
        senteval_results[task]["devacc"] for task in transfer_tasks]
    macro_acc = sum(dev_accs) / len(dev_accs)

    micro_acc = sum([
        senteval_results[task]["devacc"] *
        (senteval_results[task]["devacc"]) / num_dev_samples
        for task in transfer_tasks
    ])

    return micro_acc, macro_acc


parser = argparse.ArgumentParser(
    description="Hyperparameters used for training")

parser.add_argument("--encoder_checkpoint_path", type=str, default=None,
                    help="Pre-trained encoder path")

parser.add_argument("--classifier_checkpoint_path", type=str, default=None,
                    help="Pre-trained encoder path")

parser.add_argument("--batch_size", type=int, default=64,
                    help="Pre-trained encoder path")

parser.add_argument("--num_workers", type=int, default=6,
                    help="Pre-trained encoder path")

parser.add_argument("--senteval_kfolds", type=int, default=10,
                    help="Pre-trained encoder path")

parser.add_argument("--sent_eval_data_path", type=str, default=None,
                    help="Pre-trained encoder path")

args = parser.parse_args()

encoder = torch.load(args.encoder_checkpoint_path, map_location=device)
encoder.to(device)

classifier = torch.load(args.classifier_checkpoint_path, map_location=device)
classifier.to(device)

nli_acc = get_nli_accuracy(
    batch_size=args.batch_size,
    num_workers=args.num_workers
)

sent_eval_data_path = args.sent_eval_data_path
senteval_kfolds = args.senteval_kfolds
batch_size = args.batch_size

params_senteval = {
    "task_path": sent_eval_data_path,
    "usepytorch": False,
    "kfold": senteval_kfolds
}

params_senteval["classifier"] = {
    "optim": "adam",
    "batch_size": batch_size,
    "nhid": 0
}

micro_acc, macro_acc = senteval_results(params_senteval)

print("NLI task test accuracy: ", round(nli_acc.item() * 100, 2))
print("Senteval micro accuracy: ", round(micro_acc * 100, 2))
print("Senteval macro accuracy: ", round(macro_acc, 2))