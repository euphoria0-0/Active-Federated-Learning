import torch
import wandb
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import sys
from collections import OrderedDict

from FL_core.client import Client
from FL_core.trainer import Trainer
from FL_core.client_selection import ActiveFederatedLearning
from FL_core.federated_algorithm import FedAvg, FedAdam


class Server(object):
    def __init__(self, data, init_model, args):
        self.train_data = data['train']['data']
        self.train_sizes = data['train']['data_sizes']
        self.test_data = data['test']['data']
        self.test_sizes = data['test']['data_sizes']
        self.test_clients = data['test']['data_sizes'].keys()
        self.device = args.device
        self.args = args

        self.total_num_client = args.total_num_client
        self.num_clients_per_round = args.num_clients_per_round
        self.total_round = args.num_round

        self.trainer = Trainer(init_model, args)

        self._init_clients(init_model)
        self._init_fed_algo(args.fed_algo)
        self._client_selection(args.method)

    def _init_clients(self, init_model):
        self.client_list = []
        for client_idx in range(self.total_num_client):
            local_test_data = np.array([]) if client_idx not in self.test_clients else self.test_data[client_idx]
            c = Client(client_idx, self.train_data[client_idx], local_test_data, init_model, self.args)
            self.client_list.append(c)

    def _init_fed_algo(self, fed_algo='FedAvg'):
        global_model = self.trainer.get_model_params()
        if fed_algo == 'FedAdam':
            federated_method = FedAdam(self.train_sizes, global_model, args=self.args)
        else: # FedAvg
            federated_method = FedAvg(self.train_sizes, global_model)
        self.federated_method = federated_method

    def _client_selection(self, method='AFL'):
        if method == 'AFL':
            self.selection_method = ActiveFederatedLearning(self.total_num_client, self.device)
        else:
            self.selection_method = None

    def _aggregate_local_models(self, local_models, client_indices, global_model):
        update_model = self.federated_method.update(local_models, client_indices, global_model)
        return update_model


    def train(self):
        # get global model
        global_model = self.trainer.get_model_params()
        for round_idx in range(self.total_round):
            print(f'>> round {round_idx}')
            # set clients
            if self.selection_method is None:
                client_indices = torch.randint(self.total_num_client, (self.num_clients_per_round,)).tolist()
            else:
                client_indices = [*range(self.total_num_client)]

            # local training
            local_models, local_losses, accuracy = [], [], 0
            for client_idx in tqdm(client_indices, desc='>> Local training'):
                client = self.client_list[client_idx]
                local_model, local_acc, local_loss = client.train(deepcopy(global_model), tracking=False)
                local_models.append(deepcopy(local_model))
                local_losses.append(local_loss)
                accuracy += local_acc

            wandb.log({
                'Train/Loss': sum(local_losses) / len(client_indices),
                'Train/Acc': accuracy / len(client_indices)
            })

            # client selection
            if self.selection_method is not None:
                selected_client_indices = self.selection_method.select(self.num_clients_per_round,
                                                                       local_losses, round_idx)
                local_models = [local_models[i] for i in selected_client_indices]
                client_indices = np.array(client_indices)[selected_client_indices].tolist()

            # update global model
            global_model = self._aggregate_local_models(local_models, client_indices, global_model)
            self.trainer.set_model_params(global_model)


            # test
            self.test()


    def test(self):
        num_test_clients = len(self.test_clients)

        metrics = {'loss': [], 'acc': [], 'auc': []}
        for client_idx in tqdm(range(num_test_clients), desc='>> Local testing'):
            client = self.client_list[client_idx]
            result = client.test('test')
            metrics['loss'].append(result['loss'])
            metrics['acc'].append(result['acc'])
            metrics['auc'].append(result['auc'])
            sys.stdout.write('\rClient {}/{} TestLoss {:.6f} TestAcc {:.4f} TestAUC {:.4f}'.format(client_idx, num_test_clients,
                                                                                                   result['loss'], result['acc'], result['auc']))

        wandb.log({
            'Test/Loss': sum(metrics['loss']) / num_test_clients,
            'Test/Acc': sum(metrics['acc']) / num_test_clients,
            'Test/AUC': sum(metrics['auc']) / num_test_clients
        })