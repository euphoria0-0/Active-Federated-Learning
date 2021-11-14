import torch
import wandb
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import sys
import multiprocessing as mp

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

        self.nCPU = mp.cpu_count() // 2 if args.nCPU is None else args.nCPU

        self.total_num_client = args.total_num_client
        self.num_clients_per_round = args.num_clients_per_round
        self.total_round = args.num_round

        self.trainer = Trainer(init_model, args)
        self.test_on_training_data = False

        self._init_clients(init_model)

        if self.args.method == 'Cluster1':
            self.selection_method.setup(self.num_clients_per_round, self.train_sizes)


    def _init_clients(self, init_model):
        self.client_list = []
        for client_idx in range(self.total_num_client):
            local_test_data = np.array([]) if client_idx not in self.test_clients else self.test_data[client_idx]
            c = Client(client_idx, self.train_data[client_idx], local_test_data, deepcopy(init_model), self.args)
            self.client_list.append(c)

    def train(self):
        for round_idx in range(self.total_round):
            print(f'>> round {round_idx}')
            # get global model
            self.global_model = self.trainer.get_model()
            # set clients
            client_indices = [*range(self.total_num_client)]

            # pre-client-selection
            if self.args.method in ['Random', 'Cluster1']:
                client_indices = self.selection_method.select(self.num_clients_per_round)
                print(f'Selected clients: {sorted(client_indices)[:10]}')

            # local training
            local_models, local_losses, accuracy = [], [], []
            if self.args.use_mp:
                iter = 0
                with mp.pool.ThreadPool(processes=self.nCPU) as pool:
                    iter += 1
                    result = list(tqdm(pool.imap(self.local_training, client_indices), desc='>> Local training'))
                    #result = pool.map(self.local_training, client_indices)
                    result = np.array(result)
                    local_model, local_loss, local_acc = result[:,0], result[:,1], result[:,2]
                    local_models.extend(local_model.tolist())
                    local_losses.extend(local_loss.tolist())
                    accuracy.extend(local_acc.tolist())
                    sys.stdout.write(
                        '\rClient {}/{} TrainLoss {:.6f} TrainAcc {:.4f}'.format(len(local_losses), len(client_indices),
                                                                                 local_loss.mean().item(),
                                                                                 local_acc.mean().item()))
            else:
                for client_idx in tqdm(client_indices, desc='>> Local training', leave=True):
                    local_model, local_loss, local_acc = self.local_training(client_idx)
                    local_models.append(local_model)
                    local_losses.append(local_loss)
                    accuracy.append(local_acc)
                    sys.stdout.write(
                        '\rClient {}/{} TrainLoss {:.6f} TrainAcc {:.4f}'.format(len(local_losses), len(client_indices),
                                                                                 local_loss, local_acc))
            print()

            # client selection
            if self.args.method == 'AFL':
                selected_client_indices = self.selection_method.select(self.num_clients_per_round,
                                                                       local_losses, round_idx)
                local_models = np.take(local_models, selected_client_indices).tolist()
                client_indices = np.take(client_indices, selected_client_indices).tolist()
                local_losses = np.take(local_losses, selected_client_indices)
                accuracy = np.take(accuracy, selected_client_indices)
                torch.cuda.empty_cache()
                print(len(selected_client_indices), len(client_indices))

            variance = 0
            for k in local_models[0].state_dict().keys():
                tmp = []
                for local_model_param in local_models:
                    tmp.extend(torch.flatten(local_model_param.cpu().state_dict()[k]).tolist())
                variance += torch.var(torch.tensor(tmp), dim=0)
            variance /= len(local_models)
            print('variance of model weights {:.8f}'.format(variance))

            wandb.log({
                'Train/Loss': sum(local_losses) / len(client_indices),
                'Train/Acc': sum(accuracy) / len(client_indices)
            })
            print('{} Clients TrainLoss {:.6f} TrainAcc {:.4f}'.format(len(client_indices),
                                                                       sum(local_losses) / len(client_indices),
                                                                       sum(accuracy) / len(client_indices)))

            # aggregate local models and update global model
            global_model_params = self.federated_method.update(local_models, client_indices, self.global_model)
            self.global_model.load_state_dict(global_model_params)
            self.trainer.set_model(self.global_model)

            # test
            if self.test_on_training_data:
                self.test(self.total_num_client, phase='Train')

            self.test_on_training_data = False
            self.test(len(self.test_clients), phase='Test')

            torch.cuda.empty_cache()


    def local_training(self, client_idx):
        client = self.client_list[client_idx]
        local_model, local_acc, local_loss = client.train(deepcopy(self.global_model), tracking=False)
        torch.cuda.empty_cache()
        return deepcopy(local_model.cpu()), local_loss, local_acc # / self.train_sizes[client_idx]

    def local_testing(self, client_idx):
        client = self.client_list[client_idx]
        phase = 'train' if self.test_on_training_data else 'test'
        result = client.test(self.global_model, phase)
        torch.cuda.empty_cache()
        return result

    def test(self, num_clients_for_test, phase='Test'):
        metrics = {'loss': [], 'acc': []}
        if self.args.use_mp:
            iter = 0
            with mp.pool.ThreadPool(processes=self.nCPU) as pool:
                iter += 1
                result = pool.map(self.local_testing, [*range(num_clients_for_test)])
                losses, accs = [x['loss'] for x in result], [x['acc'] for x in result]
                metrics['loss'].extend(losses)
                metrics['acc'].extend(accs)
                sys.stdout.write(
                    '\rClient {}/{} {}Loss {:.6f} {}Acc {:.4f}'.format(len(result) * iter, num_clients_for_test,
                                                                       phase, sum(losses) / len(result),
                                                                       phase, sum(accs) / len(result)))

        else:
            for client_idx in tqdm(range(num_clients_for_test), desc=f'>> Local test on {phase} set', leave=True):
                result = self.local_testing(client_idx)
                metrics['loss'].append(result['loss'])
                metrics['acc'].append(result['acc'])
                sys.stdout.write(
                    '\rClient {}/{} {}Loss {:.6f} {}Acc {:.4f}'.format(client_idx, num_clients_for_test,
                                                                       phase, result['loss'], phase, result['acc']))
        wandb.log({
            f'{phase}/Loss': sum(metrics['loss']) / num_clients_for_test,
            f'{phase}/Acc': sum(metrics['acc']) / num_clients_for_test
        })
        print('\nALL Clients {}Loss {:.6f} {}Acc {:.4f}'.format(phase, sum(metrics['loss']) / num_clients_for_test,
                                                              phase, sum(metrics['acc']) / num_clients_for_test))