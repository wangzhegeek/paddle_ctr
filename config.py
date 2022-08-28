
import os
import sys
import argparse


# class Config(object):
#     def __init__(self,):
#         pass

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--use_gpu', type=int, default=1, help='whether using gpu')
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
    parser.add_argument("--epochs", type=int, default=10, help="epochs")
    parser.add_argument('--raw_train_data', type=str, default='../data_files/train_data.csv', help='train_path')
    parser.add_argument('--raw_test_data', type=str, default='../data_files/test_data.csv', help='train_path')
    parser.add_argument('--train_data', type=str, default='../data_files/train_data', help='train_data_path')
    parser.add_argument('--test_data', type=str, default='../data_files/test_data', help='test_data_path')
    parser.add_argument('--model_dir', type=str, default='../model_dir', help='models')
    parser.add_argument('--hash_size', type=int, default=10000, help='hash_size')
    parser.add_argument('--lr', type=float, default=0.01, help='hash_size')
    parser.add_argument('--embed_dim', type=int, default=16, help='embed_dim')
    parser.add_argument('--layer_sizes', type=list, default=[75, 50, 25], help='layer_sizes')
    parser.add_argument('--act', type=list, default="relu", help='act')
    parser.add_argument('--optimizer', type=str, default='adagrad', help='optimizer')

    args = parser.parse_args()

    return args