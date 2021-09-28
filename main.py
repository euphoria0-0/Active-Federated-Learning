'''
Jack Goetz et al., Active Federated Learning, 2019

paper:
    https://arxiv.org/pdf/1909.12641.pdf

unofficial implementation:
    https://github.com/euphoria0-0
'''
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
    parser.add_argument('--dataset', type=str, default='Reddit', help='dataset')
    parser.add_argument('--data_dir', type=str, default='D:/data/Reddit/', help='dataset directory')
    parser.add_argument('--model', type=str, default='BLSTM', help='model')

    parser.add_argument('--client_optimizer', type=str, default='sgd', help='optimizer for client')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--wdecay', type=float, default=5e-4, help='learning rate')

    parser.add_argument('--num_epoch', type=int, default=100, help='learning rate')
    args = parser.parse_args()
    return args


def load_data(dataset, data_dir):
    if dataset == 'Reddit':
        return RedditDataset(data_dir).dataset


def create_model(model):
    if model == 'BLSTM':
        return BLSTM()


if __name__ == '__main__':
    # set up
    args = get_args()
    '''wandb.init(
        project=f'AFL-{args.data}',
        name=f"AFL-lr{args.lr}-{args.comment}",
        config=args
    )'''
    args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device)  # change allocation of current GPU
    print('Current cuda device: {}'.format(torch.cuda.current_device()))

    # set data
    dataset = load_data(args.dataset, args.data_dir)

    # set model
    model = create_model(args.model)
    print(model)
    trainer = Trainer(model, args)

    # set federated optim algorithm
    FedAPI = Server(dataset, args.device, method='random')

    # train
    FedAPI.train()