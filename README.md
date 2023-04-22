# sentence_embeddings_learning

## Installation requirements

Running the project requires that you install the necessary dependencies which are provided in the `env.yaml` in the form of a ananconda env file. Creating the anaconda environment using this file is as easy as running:

`conda env create -f env.yaml`

In order to activate the environment run:

`conda activate atcs`

## Running the project

The project provides 2 main scripts namely `train.py` which is used to train a type of model and `evaluate.py` which loads a pre-trained model and print the results on 2 tasks. The first result is the accuracy on the test data on the SNLI dataset for the NLI task. The other 2 metrics "micro" and "macro" are the micro and macro SentEval dev accuracies computed as described in [Conneau et al., 2018](https://arxiv.org/pdf/1705.02364.pdf).

The other code directories in the project are the following:

- dataset: contains functions for loading GLOVE embeddings from a file and a PyTorch dataset object for loading the SNLI dataset
- models: the PyTorch implementation of the models used in the project and a utils script for instantiating a model by the name
- train_module: contains training logic

## Example of running the project (keep in mind that additional arguments can be specified)

Train a model:

```
python train.py \
--encoder_name MaxPoolingBiLSTM \
--hidden_size 2048 \
--lstm_bidirectional True \
--encoder_output_size 16384
```

Evaluate a model:

```
python evaluate.py \
--encoder_checkpoint_path <PATH_TO_PRETRAINED_ENCODER> \
--classifier_checkpoint_path <PATH_TO_PRETRAINED_CLASSIFIER> \
--sent_eval_data_path <PATH_TO_SENTEVAL_DATA>
```

The pre-trained models and the tensorboard logs can be found at https://huggingface.co/AndreiBlahovici/sentence_embeddings/tree/main