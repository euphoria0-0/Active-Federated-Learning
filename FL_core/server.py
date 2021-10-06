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


class Server(object):
    def __init__(self, data, model, args):
        self.train_data = data['train']['data']
        self.train_sizes = data['train']['data_sizes']
        self.test_data = data['test']['data']
        self.test_sizes = data['test']['data_sizes']
        self.test_clients = data['test']['data_sizes'].keys()
        self.method = args.method
        self.fed_algo = args.fed_algo
        self.model = model
        self.device = args.device
        self.args = args

        self.client_list = []
        self.total_num_client = args.total_num_client
        self.num_clients_per_round = args.num_clients_per_round
        self.test_num_clients = args.test_num_clients if args.test_num_clients is not None else args.total_num_client
        self.total_round = args.num_round

        if self.fed_algo == 'FedAdam':
            self.m = OrderedDict()
            self.v = OrderedDict()
            for k in model.state_dict():
                self.m[k], self.v[k] = 0., 0.

        self.trainer = Trainer(model, args)

        self._init_clients()

    def _init_clients(self):
        for client_idx in range(self.total_num_client):
            local_test_data = np.array([]) if client_idx not in self.test_clients else self.test_data[client_idx]
            c = Client(client_idx, self.train_data[client_idx], local_test_data, self.model, self.args)
            self.client_list.append(c)

    def _client_selection(self, method='AFL'):
        if method == 'AFL':
            selection_method = ActiveFederatedLearning
        return selection_method

    def _aggregate_local_models(self, local_models, client_indices, global_model):
        if self.fed_algo == 'FedAvg':
            update_model = deepcopy(local_models[0].cpu().state_dict())
            for k in update_model.keys():
                for idx in range(len(local_models)):
                    local_model = deepcopy(local_models[idx])
                    num_local_data = self.train_sizes[client_indices[idx]]
                    weight = num_local_data / sum(self.train_sizes)
                    if idx == 0:
                        update_model[k] = weight * local_model.cpu().state_dict()[k]
                    else:
                        update_model[k] += weight * local_model.cpu().state_dict()[k]

        elif self.fed_algo == 'FedAdam':
            gradient_update = OrderedDict()
            for k in global_model.keys():
                for idx in range(len(local_models)):
                    local_model = deepcopy(local_models[idx]).state_dict()
                    num_local_data = self.train_sizes[client_indices[idx]]
                    weight = num_local_data / sum(self.train_sizes)
                    if idx == 0:
                        gradient_update[k] = weight * local_model[k]
                    else:
                        gradient_update[k] -= weight * local_model[k]

            update_model = OrderedDict()
            for k in gradient_update.keys():
                g = gradient_update[k]
                self.m[k] = self.args.beta1 * self.m[k] + (1 - self.args.beta1) * g
                self.v[k] = self.args.beta2 * self.v[k] + (1 - self.args.beta2) * torch.mul(g, g)
                m_hat = self.m[k] / (1 - self.args.beta1)
                v_hat = self.v[k] / (1 - self.args.beta2)
                update_model[k] = global_model[k] - self.args.lr_global * m_hat / (np.sqrt(v_hat) + self.args.epsilon)

        return update_model

    def train(self):
        # get global model
        global_model = self.trainer.get_model_params()
        for round_idx in range(self.total_round):
            print(f'>> round {round_idx}')
            # set clients
            if self.method == 'Random':
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
            if self.method != 'Random':
                selection_method = self._client_selection()
                selection_method = selection_method(local_models, local_losses,
                                                    self.total_num_client, self.num_clients_per_round, self.device)
                selected_client_indices = selection_method.select(round_idx)
                local_models = [local_models[i] for i in selected_client_indices]
                client_indices = np.array(client_indices)[selected_client_indices.astype(int)].tolist()

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