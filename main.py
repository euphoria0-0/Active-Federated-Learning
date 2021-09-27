'''
Jack Goetz et al., Active Federated Learning, 2019

paper:
    https://arxiv.org/pdf/1909.12641.pdf

unofficial implementation:
    https://github.com/euphoria0-0
'''
import wandb
import argparse

from data.reddit import RedditDataset
from model.BLSTM import BLSTM
from FL_core.trainer import Trainer
from FL_core.server import Server


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu cuda index')
    args = parser.parse_args()
    return args


def load_data(dataset):
    if dataset == 'Reddit':
        return RedditDataset()

def create_model(model):
    if model == 'BLSTM':
        return BLSTM() # 64-dim character level embedding, 32-dim BLSTM, and MLP with one 64-dim hidden layer

def get_fed_algo(algo, args):
    if algo == 'AFL':

        return Server()


if __name__ == '__main__':
    # set up
    args = get_args()
    '''wandb.init(
        project=f'AFL-{args.data}',
        name=f"AFL-lr{args.lr}-{args.comment}",
        config=args
    )'''

    # set data
    data = load_data()

    # set model
    model = create_model()
    trainer = Trainer()

    # set federated optim algorithm
    FedAPI = get_fed_algo()

    # training
    FedAPI.train()