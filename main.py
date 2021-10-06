'''
Jack Goetz et al., Active Federated Learning, 2019

paper:
    https://arxiv.org/pdf/1909.12641.pdf

unofficial implementation:
    https://github.com/euphoria0-0
'''
import os
import wandb
import argparse
import torch
from torch.utils.data import DataLoader

from data.reddit import RedditDataset
from model.BLSTM import BLSTM
from FL_core.trainer import Trainer
from FL_core.server import Server


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu cuda index')
    parser.add_argument('--dataset', type=str, default='Reddit', help='dataset', choices=['Reddit'])
    parser.add_argument('--data_dir', type=str, default='D:/data/Reddit/', help='dataset directory')
    parser.add_argument('--model', type=str, default='BLSTM', help='model', choices=['BLSTM'])
    parser.add_argument('--method', type=str, default='Random', choices=['Random', 'AFL'], help='client selection')
    parser.add_argument('--fed_algo', type=str, default='FedAvg', choices=['FedAvg', 'FedAdam'],
                        help='Federated algorithm for aggregation')

    parser.add_argument('--client_optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='client optim')
    parser.add_argument('--lr_local', type=float, default=0.01, help='learning rate for optim')
    parser.add_argument('--lr_global', type=float, default=0.001, help='learning rate for optim')
    parser.add_argument('--wdecay', type=float, default=0, help='weight decay for optim')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for Adam')

    parser.add_argument('--num_epoch', type=int, default=2, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of each client data')
    parser.add_argument('--num_round', type=int, default=20, help='total number of rounds')

    parser.add_argument('--total_num_client', type=int, default=7668, help='total number of clients')
    parser.add_argument('--num_clients_per_round', type=int, default=200, help='number of participated clients')
    parser.add_argument('--test_num_clients', type=int, default=None, help='number of participated clients for test')

    parser.add_argument('--maxlen', type=int, default=400, help='maxlen for NLP dataset')

    #parser.add_argument('--comment', type=str, default='', help='comment')
    args = parser.parse_args()
    return args


def load_data(args):
    if args.dataset == 'Reddit':
        return RedditDataset(args.data_dir, args)


def create_model(args):
    if args.model == 'BLSTM':
        return BLSTM(vocab_size=args.maxlen, num_classes=args.num_classes)


if __name__ == '__main__':
    # set up
    args = get_args()
    wandb.init(
        project=f'AFL-{args.dataset}',
        name=f"{args.method}-{args.fed_algo}-{args.num_clients_per_round}/{args.total_num_client}",#-{args.comment}",
        config=args
    )
    args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device)  # change allocation of current GPU
    print('Current cuda device: {}'.format(torch.cuda.current_device()))

    # set data
    data = load_data(args)
    args.num_classes = data.num_classes
    dataset = data.dataset

    # set model
    model = create_model(args)
    trainer = Trainer(model, args)

    # set federated optim algorithm
    FedAPI = Server(dataset, model, args)

    # train
    FedAPI.train()