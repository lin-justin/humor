import torch
import torch.optim as optim
import torch.nn as nn
from torchtext.data import Field, TabularDataset, BucketIterator, LabelField

from model import RNN, BiLSTM
from utils import get_embedding_dim, binary_accuracy, train, evaluate, epoch_time, generate_bigrams

import random
import time

import spacy
spacy.load("en_core_web_sm")

import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type = str, default = 'rnn', 
                        help = "Available models are: 'rnn', 'cnn', 'bilstm', 'fasttext', and 'distilbert'\nDefault is 'rnn'")
    parser.add_argument('--train_data_path', type = str, default = "./data/train_clean.csv",
                        help = "Path to the training data")
    parser.add_argument('--test_data_path', type = str, default = "./data/dev_clean.csv",
                        help = "Path to the test data")
    parser.add_argument('--seed', type = int, default = 1234)
    parser.add_argument('--vectors', type = str, default = 'fasttext.simple.300d',
                        help = """
                                Pretrained vectors:
                                Visit 
                                https://github.com/pytorch/text/blob/9ce7986ddeb5b47d9767a5299954195a1a5f9043/torchtext/vocab.py#L146
                                for more 
                                """)
    parser.add_argument('--max_vocab_size', type = int, default = 750)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--bidirectional', type = bool, default = True)
    parser.add_argument('--dropout', type = float, default = 0.5)
    parser.add_argument('--hidden_dim', type = int, default = 64)
    parser.add_argument('--output_dim', type = int, default = 1)
    parser.add_argument('--n_layers', type = int, default = 2)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--n_epochs', type = int, default = 5)
    parser.add_argument('--n_filters', type = int, default = 100)
    parser.add_argument('--filter_sizes', type = list, default = [3,4,5])

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##########  BILSTM ##########

    if args.model == "bilstm":
        print('\nBiLSTM')
        TEXT = Field(tokenize = 'spacy')
        LABEL = LabelField(dtype = torch.float)
        data_fields = [("text", TEXT), ("label", LABEL)]

        train_data = TabularDataset(args.train_data_path,
                                    format = 'csv',
                                    fields = data_fields,
                                    skip_header = True,
                                    csv_reader_params = {'delimiter': ","})

        test_data = TabularDataset(args.test_data_path,
                                   format = 'csv',
                                   fields = data_fields,
                                   skip_header = True,
                                   csv_reader_params = {'delimiter': ","})

        train_data, val_data = train_data.split(split_ratio = 0.8, random_state = random.seed(args.seed))

        TEXT.build_vocab(train_data,
                         max_size = args.max_vocab_size,
                         vectors = args.vectors,
                         unk_init = torch.Tensor.normal_)
        LABEL.build_vocab(train_data)

        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, val_data, test_data),
            batch_size = args.batch_size,
            sort_key = lambda x: len(x.text),
            device = device
        )

        input_dim = len(TEXT.vocab)
        embedding_dim = get_embedding_dim(args.vectors)
        pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
        unk_idx = TEXT.vocab.stoi[TEXT.unk_token]

        model = BiLSTM(input_dim,
                       embedding_dim,
                       args.hidden_dim,
                       args.output_dim,
                       args.n_layers,
                       args.bidirectional,
                       args.dropout,
                       pad_idx
                       )
        
        pretrained_embeddings = TEXT.vocab.vectors

        model.embedding.weight.data.copy_(pretrained_embeddings)
        model.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
        model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)

        optimizer = optim.Adam(model.parameters(), lr = args.lr)
        criterion = nn.BCEWithLogitsLoss()

        model.to(device)
        criterion.to(device)

        best_valid_loss = float('inf')

        print("\nTraining...")
        print("===========")
        for epoch in range(1, args.n_epochs+1):

            start_time = time.time()

            train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), './checkpoints/{}-model.pt'.format(args.model))

            print(f'[Epoch: {epoch:02}] | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        model.load_state_dict(torch.load('./checkpoints/{}-model.pt'.format(args.model)))

        test_loss, test_acc = evaluate(model, test_iterator, criterion)

        print('\nEvaluating...')
        print("=============")
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%') # Test Loss: 0.139, Test Acc: 95.27%

    ##########  VANILLA RNN ##########

    else:
        print('\nVanilla RNN')
        TEXT = Field(tokenize = 'spacy')
        LABEL = LabelField(dtype = torch.float)
        data_fields = [("text", TEXT), ("label", LABEL)]

        train_data = TabularDataset(args.train_data_path,
                                    format = 'csv',
                                    fields = data_fields,
                                    skip_header = True,
                                    csv_reader_params = {'delimiter': ","})

        test_data = TabularDataset(args.test_data_path,
                                   format = 'csv',
                                   fields = data_fields,
                                   skip_header = True,
                                   csv_reader_params = {'delimiter': ","})

        train_data, val_data = train_data.split(split_ratio = 0.8, random_state = random.seed(args.seed))

        TEXT.build_vocab(train_data,
                         max_size = args.max_vocab_size,
                         vectors = args.vectors)
        LABEL.build_vocab(train_data)

        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, val_data, test_data),
            batch_size = args.batch_size,
            sort_key = lambda x: len(x.text),
            device = device
        )

        input_dim = len(TEXT.vocab)
        embedding_dim = get_embedding_dim(args.vectors)

        model = RNN(input_dim,
                    embedding_dim,
                    args.hidden_dim,
                    args.output_dim
                    )
        
        pretrained_embeddings = TEXT.vocab.vectors

        model.embedding.weight.data.copy_(pretrained_embeddings)

        optimizer = optim.Adam(model.parameters(), lr = args.lr)
        criterion = nn.BCEWithLogitsLoss()

        model.to(device)
        criterion.to(device)

        best_valid_loss = float('inf')

        print("\nTraining...")
        print("===========")
        for epoch in range(1, args.n_epochs+1):

            start_time = time.time()

            train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), './checkpoints/{}-model.pt'.format(args.model))

            print(f'[Epoch: {epoch:02}] | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        model.load_state_dict(torch.load('./checkpoints/{}-model.pt'.format(args.model)))

        test_loss, test_acc = evaluate(model, test_iterator, criterion)

        print('\nEvaluating...')
        print("=============")
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%') # Test Loss: 0.138, Test Acc: 95.05%

if __name__ == "__main__":
    main()
