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

from data import reddit, federated_emnist
from model import BLSTM, CNN
from FL_core.server import *
from FL_core.client_selection import *
from FL_core.federated_algorithm import *


<<<<<<< HEAD
=======

>>>>>>> beeafa95854742b23768dd6e6bc67a60d6998df0
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu cuda index')
    parser.add_argument('--dataset', type=str, default='FederatedEMNIST', help='dataset', choices=['Reddit','FederatedEMNIST'])
    parser.add_argument('--data_dir', type=str, default='../dataset/FederatedEMNIST/', help='dataset directory')
    parser.add_argument('--model', type=str, default='CNN', help='model', choices=['BLSTM','CNN'])
<<<<<<< HEAD
    parser.add_argument('--method', type=str, default='AFL', help='client selection',
                        choices=['Random','AFL','Cluster1','Cluster2','Pow-d','AFL_1_p','MaxEntropy','MaxEntropySampling'])
    parser.add_argument('--AL_method', type=str, default=None, choices=[None, 'Random', 'MaxEntropy'],
                        help='local active learning method')
=======
    parser.add_argument('--method', type=str, default='AFL', choices=['Random', 'AFL'],
                        help='client selection')
>>>>>>> beeafa95854742b23768dd6e6bc67a60d6998df0
    parser.add_argument('--fed_algo', type=str, default='FedAvg', choices=['FedAvg', 'FedAdam'],
                        help='Federated algorithm for aggregation')

    parser.add_argument('--client_optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='client optim')
    parser.add_argument('--lr_local', type=float, default=0.1, help='learning rate for client optim')
    parser.add_argument('--lr_global', type=float, default=0.001, help='learning rate for server optim')
    parser.add_argument('--wdecay', type=float, default=0, help='weight decay for optim')

    parser.add_argument('--momentum', type=float, default=0, help='momentum for SGD')

    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for Adam')

    parser.add_argument('--alpha1', type=float, default=0.75, help='alpha1 for AFL')
    parser.add_argument('--alpha2', type=float, default=0.01, help='alpha2 for AFL')
    parser.add_argument('--alpha3', type=float, default=0.1, help='alpha3 for AFL')

    parser.add_argument('-E', '--num_epoch', type=int, default=1, help='number of epochs')
    parser.add_argument('-B', '--batch_size', type=int, default=64, help='batch size of each client data')
    parser.add_argument('-R', '--num_round', type=int, default=2000, help='total number of rounds')
    parser.add_argument('-A', '--num_clients_per_round', type=int, default=10, help='number of participated clients')
    parser.add_argument('--total_num_clients', type=int, default=None, help='total number of clients')

    parser.add_argument('--maxlen', type=int, default=400, help='maxlen for NLP dataset')
<<<<<<< HEAD
    parser.add_argument('--distance_type', type=str, default='L1', help='distance type for clustered sampling 2')
    parser.add_argument('--d', type=int, default=None, help='d of power-of-choice')

    parser.add_argument('--labeled_ratio', type=float, default=1.0, help='ratio of unlabeled data')
    parser.add_argument('--nQuery', type=int, default=10, help='number of query data for local active learning')
=======
>>>>>>> beeafa95854742b23768dd6e6bc67a60d6998df0

    parser.add_argument('--comment', type=str, default='', help='comment')

    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix random seed')
    parser.add_argument('--parallel', action='store_true', default=False, help='use multi GPU')
    parser.add_argument('--use_mp', action='store_true', default=False, help='use multiprocessing')
    parser.add_argument('--nCPU', type=int, default=None, help='number of CPU cores for multiprocessing')
<<<<<<< HEAD
    parser.add_argument('--save_probs', action='store_true', default=False, help='save probs')
=======
>>>>>>> beeafa95854742b23768dd6e6bc67a60d6998df0

    args = parser.parse_args()
    return args


def load_data(args):
    if args.dataset == 'Reddit':
        return reddit.RedditDataset(args.data_dir, args)
    elif args.dataset == 'FederatedEMNIST':
        return federated_emnist.FederatedEMNISTDataset(args.data_dir, args)


def create_model(args):
    if args.model == 'BLSTM':
        model = BLSTM.BLSTM(vocab_size=args.maxlen, num_classes=args.num_classes)
    elif args.model == 'CNN':
        model = CNN.CNN_DropOut(False)
    if args.parallel:
        model = torch.nn.DataParallel(model, output_device=0)
    return model


<<<<<<< HEAD
=======
def client_selection_method(args, dataset):
    if args.method == 'AFL':
        return ActiveFederatedLearning(args.total_num_client, args.device, args)
    else:
        return RandomSelection(args.total_num_client, args.device)


>>>>>>> beeafa95854742b23768dd6e6bc67a60d6998df0
def federated_algorithm(dataset, model, args):
    train_sizes = dataset['train']['data_sizes']
    if args.fed_algo == 'FedAdam':
        return FedAdam(train_sizes, model, args=args)
    else:
        # FedAvg
        return FedAvg(train_sizes, model)


<<<<<<< HEAD
def client_selection_method(args):
    if args.method == 'AFL':
        return ActiveFederatedLearning(args.total_num_client, args.device, args)
    elif args.method == 'Cluster1':
        return ClusteredSampling1(args.total_num_client, args.device)
    elif args.method == 'Cluster2':
        return ClusteredSampling2(args.total_num_client, args.device, args)
    elif args.method == 'Pow-d':
        return PowerOfChoice(args.total_num_client, args.device)
    elif args.method == 'AFL_1_p':
        return ActiveFederatedLearning_1_p(args.total_num_client, args.device, args)
    elif args.method == 'MaxEntropy':
        return MaxEntropy(args.total_num_client, args.device)
    elif args.method == 'MaxEntropySampling':
        return MaxEntropySampling(args.total_num_client, args.device, args)
    else:
        return RandomSelection(args.total_num_client, args.device)


=======
>>>>>>> beeafa95854742b23768dd6e6bc67a60d6998df0

if __name__ == '__main__':
    # set up
    args = get_args()
<<<<<<< HEAD
    if args.labeled_ratio < 1:
        args.comment = f'-L{args.labeled_ratio}{args.comment}'
    wandb.init(
        project=f'AFL-{args.dataset}',
        name=f"{args.method}-{args.fed_algo}-{args.num_clients_per_round}/{args.total_num_clients}{args.comment}",
=======
    wandb.init(
        project=f'AFL-{args.dataset}',
        name=f"{args.method}-{args.fed_algo}-{args.num_clients_per_round}{args.comment}",
>>>>>>> beeafa95854742b23768dd6e6bc67a60d6998df0
        config=args,
        dir='.',
        save_code=True
    )
    args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device)  # change allocation of current GPU
    print('Current cuda device: {}'.format(torch.cuda.current_device()))

    # set data
    data = load_data(args)
    args.num_classes = data.num_classes
    args.total_num_client, args.test_num_clients = data.train_num_clients, data.test_num_clients
    dataset = data.dataset

    # set model
    model = create_model(args)
<<<<<<< HEAD
    client_selection = client_selection_method(args)
=======
    client_selection = client_selection_method(args, dataset)
>>>>>>> beeafa95854742b23768dd6e6bc67a60d6998df0
    fed_algo = federated_algorithm(dataset, model, args)

    # set federated optim algorithm
    ServerExecute = Server(dataset, model, args, client_selection, fed_algo)

    # train
<<<<<<< HEAD
    if args.save_probs:
        os.makedirs('./results/', exist_ok=True)
        results = open(f'./results/probs_{args.method}-{args.fed_algo}-{args.num_clients_per_round}-{args.total_num_clients}{args.comment}.txt', 'w')
    else:
        results = None
    ServerExecute.train(results)


    # save code
    '''from glob import glob
    code = wandb.Artifact(f'AFL-{args.dataset}', type='code')
    for path in glob('**/*.py', recursive=True):
        code.add_file(path)
    wandb.run.use_artifact(code)'''
=======
    ServerExecute.train()


    # save code
    from glob import glob
    code = wandb.Artifact(f'AFL-{args.dataset}', type='code')
    for path in glob('**/*.py', recursive=True):
        code.add_file(path)
    wandb.run.use_artifact(code)
>>>>>>> beeafa95854742b23768dd6e6bc67a60d6998df0
