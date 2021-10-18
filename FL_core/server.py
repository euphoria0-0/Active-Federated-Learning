import torch
import wandb
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import sys

from FL_core.client import Client
from FL_core.trainer import Trainer



class Server(object):
    def __init__(self, data, init_model, args, selection, fed_algo):
        self.train_data = data['train']['data']
        self.train_sizes = data['train']['data_sizes']
        self.test_data = data['test']['data']
        self.test_sizes = data['test']['data_sizes']
        self.test_clients = data['test']['data_sizes'].keys()
        self.device = args.device
        self.args = args
        self.global_model = init_model
        self.selection_method = selection
        self.federated_method = fed_algo

        self.total_num_client = args.total_num_client
        self.num_clients_per_round = args.num_clients_per_round
        self.total_round = args.num_round

        self.trainer = Trainer(init_model, args)

        self._init_clients(init_model)


    def _init_clients(self, init_model):
        self.client_list = []
        for client_idx in range(self.total_num_client):
            local_test_data = np.array([]) if client_idx not in self.test_clients else self.test_data[client_idx]
            c = Client(client_idx, self.train_data[client_idx], local_test_data, init_model, self.args)
            self.client_list.append(c)

    def train(self):
        # get global model
        self.global_model = self.trainer.get_model()
        for round_idx in range(self.total_round):
            print(f'>> round {round_idx}')
            # set clients
            client_indices = [*range(self.total_num_client)]
            if self.selection_method is None:
                client_indices = np.random.choice(client_indices, size=self.num_clients_per_round, replace=False)
            print(f'Selected clients: {sorted(client_indices)[:10]}')

            # local training
            local_models, local_losses, accuracy = [], [], 0
            for client_idx in tqdm(client_indices, desc='>> Local training', leave=True):
                client = self.client_list[client_idx]
                local_model, local_acc, local_loss = client.train(self.global_model, tracking=False)
                local_models.append(deepcopy(local_model))
                local_losses.append(local_loss)
                accuracy += local_acc

                torch.cuda.empty_cache()

            wandb.log({
                'Train/Loss': sum(local_losses) / len(client_indices),
                'Train/Acc': accuracy / len(client_indices)
            })
            print('{} Clients TrainLoss {:.6f} TrainAcc {:.4f}'.format(len(client_indices),
                                                                       sum(local_losses) / len(client_indices),
                                                                       accuracy / len(client_indices)))
            # client selection
            if self.selection_method is not None:
                selected_client_indices = self.selection_method.select(self.num_clients_per_round,
                                                                       local_losses, round_idx)
                local_models = [local_models[i] for i in selected_client_indices]
                client_indices = np.array(client_indices)[selected_client_indices].tolist()

            # aggregate local models and update global model
            global_model_params = self.federated_method.update(local_models, client_indices, self.global_model)
            self.global_model.load_state_dict(global_model_params)
            self.trainer.set_model(self.global_model)

            # test
            self.test()

            torch.cuda.empty_cache()


    def test(self, test_on_training_data=False):
        if test_on_training_data:
            metrics = {'loss': [], 'acc': []}
            for client_idx in tqdm(range(self.total_num_client), desc='>> Local test for train', leave=True):
                client = self.client_list[client_idx]
                result = client.test(self.global_model, 'train')
                metrics['loss'].append(result['loss'])
                metrics['acc'].append(result['acc'])
                sys.stdout.write(
                    '\rClient {}/{} TrainLoss {:.6f} TrainAcc {:.4f}'.format(client_idx, self.total_num_client,
                                                                             result['loss'], result['acc']))

            wandb.log({
                'Train/Loss': sum(metrics['loss']) / self.total_num_client,
                'Train/Acc': sum(metrics['acc']) / self.total_num_client
            })
            print('ALL Clients TrainLoss {:.6f} TrainAcc {:.4f}'.format(sum(metrics['loss']) / self.total_num_client,
                                                                        sum(metrics['acc']) / self.total_num_client))


        num_test_clients = len(self.test_clients)

        metrics = {'loss': [], 'acc': [], 'auc': []}
        for client_idx in tqdm(range(num_test_clients), desc='>> Local test', leave=True):
            client = self.client_list[client_idx]
            result = client.test(self.global_model, 'test')
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

        print('ALL Clients TestLoss {:.6f} TestAcc {:.4f}'.format(sum(metrics['loss']) / self.total_num_client,
                                                                  sum(metrics['acc']) / self.total_num_client))