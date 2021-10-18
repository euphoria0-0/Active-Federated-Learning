import os
import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset



class FederatedEMNISTDataset:
    def __init__(self, data_dir, args):
        self.num_classes = 62
        self.train_num_clients = 3400
        self.test_num_clients = 3400
        self.batch_size = args.batch_size # local batch size for local training

        self._init_data(data_dir)
        print(f'Total number of users: {self.train_num_clients}')

    def _init_data(self, data_dir):
        file_name = os.path.join(data_dir, 'FederatedEMNIST_preprocessed_.pickle')
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f)
        else:
            dataset = preprocess(data_dir, self.batch_size)
        self.dataset = dataset




def preprocess(data_dir, batch_size=128):
    train_data = h5py.File(os.path.join(data_dir, 'fed_emnist_train.h5'), 'r')
    test_data = h5py.File(os.path.join(data_dir, 'fed_emnist_test.h5'), 'r')

    train_ids = list(train_data['examples'].keys())
    test_ids = list(test_data['examples'].keys())
    num_clients_train = len(train_ids)
    num_clients_test = len(test_ids)
    print(num_clients_train, num_clients_test)

    # local dataset
    train_data_local_dict, train_data_local_num_dict = {}, {}
    test_data_local_dict, test_data_local_num_dict = {}, {}

    for client_idx in range(num_clients_train):
        client_id = train_ids[client_idx]

        train_x = train_data['examples'][client_id]['pixels'][()]
        train_y = train_data['examples'][client_id]['label'][()]

        local_data = _batch_data({'pixels':train_x, 'label':train_y}, batch_size=batch_size)
        train_data_local_dict[client_idx] = local_data
        train_data_local_num_dict[client_idx] = len(train_x)

        test_x = test_data['examples'][client_id]['pixels'][()]
        test_y = test_data['examples'][client_id]['label'][()]
        local_data = _batch_data({'pixels': test_x, 'label': test_y}, batch_size=batch_size)
        test_data_local_dict[client_idx] = local_data
        test_data_local_num_dict[client_idx] = len(test_x)
        if len(test_x) == 0:
            print(client_idx)

    train_data.close()
    test_data.close()

    dataset = {}
    dataset['train'] = {
        'data_sizes': train_data_local_num_dict,
        'data': train_data_local_dict,
    }
    dataset['test'] = {
        'data_sizes': test_data_local_num_dict,
        'data': test_data_local_dict,
    }

    #with open(os.path.join(data_dir, 'FederatedEMNIST_preprocessed_.pickle'), 'wb') as f:
    #    pickle.dump(dataset, f)

    return dataset


def _batch_data(data, batch_size=128):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = np.array(data['pixels'])
    data_y = np.array(data['label'])

    # randomly shuffle data
    np.random.seed(0)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_x = np.expand_dims(batched_x, axis=1)
        batched_x = torch.tensor(batched_x)
        batched_y = data_y[i:i + batch_size]
        batched_y = torch.tensor(batched_y, dtype=torch.long)
        batch_data.append((batched_x, batched_y))
    return batch_data