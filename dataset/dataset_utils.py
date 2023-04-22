import numpy as np


def get_glove_vocab_index_mapping(glove_path):
    """
    Returns a dictionary mapping words in the GloVe file to their corresponding
    index in the vocabulary. The index of the padding token is 1, and the index
    of the unknown token is 0.

    Args:
        glove_path (str): Path to the file containing GloVe embeddings.

    Returns:
        vocab (dict): Dictionary mapping words in the GloVe file to their
            corresponding index in the vocabulary.
    """
    with open(glove_path, "r") as r:
        lines = r.readlines()
        vocab = {
            line.split()[0]: i + 2
            for i, line in enumerate(lines)
        }

    # add tokens for unknown tokens and padding
    vocab["<UNK>"] = 0
    vocab["<PAD>"] = 1

    return vocab


def get_glove_embeddings(glove_path):
    """
    Returns a numpy array of GloVe embeddings.

    Args:
        glove_path (str): Path to the GloVe file.

    Returns:
        embeddings (numpy array): A numpy array of GloVe embeddings.
    """
    embeddings = []

    with open(glove_path, "r") as r:
        for line in r.readlines():
            embedding = line.split(" ")[1:]
            embedding = np.array(embedding, dtype=np.float64)
            embeddings.append(embedding)

    embedding_size = len(embeddings[0])
    embedding = [0] * embedding_size

    # insert dummy embedding for padding token
    embeddings.insert(0, embedding)

    # insert dummy embedding for unknown token
    embeddings.insert(0, embedding)

    return np.array(embeddings)
