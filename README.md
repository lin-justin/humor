# Humor Detection and Generation with Deep Learning

## 1. Detecting Humor

The [data](https://www.kaggle.com/moradnejad/200k-short-texts-for-humor-detection) consists of 200,000 formal short texts (100k humorous, 100k not humorous). Here is the [paper](https://arxiv.org/abs/2004.12765).

The purpose of this repository is to practice scripting, learn about deep learning in NLP, and completing a full project cycle of:

- collecting data
- preprocessing and cleaning text
- structuring directories and scripts
- implementing models
- training and evaluating models
- making it reproducible for others

### Usage

Arguments and default hyperparameters:

```
python train.py --model="rnn" \
    --train_data_path="./data/train_clean.csv" \
    --test_data_path="./data/test_clean.csv" \
    --seed=1234 \
    --vectors="fasttext.simple.300d" \
    --max_vocab_size=750 \
    --batch_size=32 \
    --bidirectional=True \
    --dropout=0.5 \
    --hidden_dim=64 \
    --output_dim=1 \
    --n_layers=2 \
    --lr=1e-3 \
    --n_epochs=5 \
    --n_filters=[3,4,5] \
    --filter_sizes=100
```

The script was ran in a virtual environment using Miniconda3 on WSL Ubuntu and on a CPU.

Available models are `rnn`, `cnn`, `fasttext`, `bilistm`, and `distilbert`. **Please run** `distilbert` **on a GPU, one epoch on a CPU took 1 hour and 36 mins.**

### Requirements

Python 3.7

- pytorch==1.5.1
- torchtext==0.6.0
- transformers==3.0.2
- texthero==1.0.9

```
pip install requirements.txt
```

To use spaCy tokenizer, make sure to run `python -m spacy download en` in the terminal.

### Results

The models were trained for 5 epochs, if I had more powerful hardware, I would (and should) train for longer.

| Model | Test Loss | Test Acc | Train Loss | Train Acc | Val Loss | Val Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---:
| RNN | 0.138 | 95.05% |
| BiLSTM | 0.139 | 95.27% |
| FastText | 0.163 | 94.95% |
| CNN | 0.128 | 95.41% |
| DistilBERT | | | 0.095 | 96.55% | 0.096 | 96.46% |

From the results, we can see that a Convolutional Neural Network model with default hyperparameters had the highest test accuracy as well as the lowest test loss. 

The DistilBERT model was trained for **one epoch** due to hardware limitations, I did not use Google Colab to train these models because I need to learn good software/coding practice since notebooks can get messy and irreproducible. Additionally, using Google Colab's free GPU strains my internet's memory capacity. 

### Reference

[Sentiment Analysis](https://github.com/bentrevett/pytorch-sentiment-analysis)

## 2. Humorous Text Generation

Using only the 100k texts labelled humorous (see the last few lines of the preprocess.py script), I leveraged a pretrained OpenAI [GPT-2](https://openai.com/blog/better-language-models/) from [Huggingface](https://huggingface.co/) (a great NLP library).

### Usage

```
python generate.py
```

The script only takes a sentence from the humorous text only dataset and generates 2 different sentences.

### Results

Input sentence: I think my iphone is broken i keep pressing the home button but i am still at work

Output:

- Sentence 1: I think my iphone is broken i keep pressing the home button but i am still at work and need a new one. I haven't changed my phone on all my work devices. I've been told it needs a new phone and

- Sentence 2: I think my iphone is broken i keep pressing the home button but i am still at work in the morning on my iPhone. i just found out after reading this that its working OK. Also, i have another broken phone with 3

I referred to this [tutorial](https://github.com/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb).