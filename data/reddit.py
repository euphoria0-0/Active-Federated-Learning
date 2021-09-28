import os
import bz2
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class RedditDataset:
    def __init__(self, data_dir, device):
        self.num_classes = 2
        self.train_size = 124638 # messages
        self.test_size = 15568 # messages
        self.train_num_clients = 7656 # 7527 # users
        self.test_num_clients = 3440 # users
        self.batch_size = 64
        self.maxlen = 500
        self.device = device

        self._init_data(data_dir)
        print(f'Total number of users: {self.train_num_clients}')


    '''def _update_data(self, additional_idx):
        self.current_idx += additional_idx
        self.data = self.train_data[self.current_idx]'''

    def _init_data(self, data_dir):
        if os.path.isfile(os.path.join(data_dir, 'Reddit_preprocessed_7656_.json')):
            #print('>> get preprocessed Reddit dataset ...')
            with open(os.path.join(data_dir, 'Reddit_preprocessed_7656_.json'), 'r') as f:
                dataset = json.load(f) # user_id, num_data, text, label

            train_data_num, test_data_num = 0, 0
            train_data_local_dict, test_data_local_dict = dict(), dict()
            train_data_local_num_dict = dict()
            train_data_global, test_data_global = list(), list()
            for client_idx in tqdm(range(self.train_num_clients), desc='>> Split data to clients'):
                user_train_data_num = dataset[str(client_idx)]['num_data']
                #user_test_data_num = dataset[client_idx]['num_data']
                train_data_num += user_train_data_num
                #test_data_num += user_test_data_num
                train_data_local_num_dict[client_idx] = user_train_data_num

                # transform to batches
                train_batch = self._batch_data(dataset[str(client_idx)])
                #test_batch = self._batch_data()

                # index using client index
                train_data_local_dict[client_idx] = train_batch
                #test_data_local_dict[client_idx] = test_batch
                train_data_global += train_batch
                #test_data_global += test_batch

            dataset = {}
            dataset['train'] = {
                'data_sizes': train_data_local_num_dict,
                'data': train_data_local_dict
            }
            dataset['test'] = {
                'data_sizes': train_data_num,
                'data': train_data_global
            }

            self.dataset = dataset
        else:
            self.dataset = self._preprocess(data_dir)

    def _batch_data(self, data):
        '''
        data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
        returns x, y, which are both numpy array of length: batch_size
        '''
        data_x = data['text']
        data_y = data['label']

        # randomly shuffle data
        np.random.seed(0)
        rng_state = np.random.get_state()
        np.random.shuffle(data_x)
        np.random.set_state(rng_state)
        np.random.shuffle(data_y)

        # loop through mini-batches
        batch_data = list()
        for i in range(0, len(data_x), self.batch_size):
            batched_x = data_x[i:i + self.batch_size]
            batched_y = data_y[i:i + self.batch_size]
            batched_x = self._process_x(batched_x)
            batched_y = torch.tensor(batched_y, dtype=torch.long)
            batch_data.append((batched_x, batched_y))
        return batch_data

    def _process_x(self, raw_x_batch):
        CHAR_VOCAB = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
        ALL_LETTERS = "".join(CHAR_VOCAB)

        x_batch = []
        for word in raw_x_batch:
            indices = torch.empty((0,), dtype=torch.long)
            for c in word:
                tmp = ALL_LETTERS.find(c)
                tmp = 0 if tmp == -1 else tmp
                tmp = torch.tensor([tmp], dtype=torch.long)
                indices = torch.cat((indices, tmp), dim=0)
            x_batch.append(indices)
        #maxlen = max([x.size(0) for x in x_batch])

        x_batch2 = torch.empty((0, self.maxlen), dtype=torch.long)
        for x in x_batch:
            x = torch.unsqueeze(F.pad(x, (0, self.maxlen-x.size(0)), value=0), 0)
            x_batch2 = torch.cat((x_batch2, x), dim=0)
        return x_batch2

    def _preprocess(self, data_dir):
        print('>> load and preprocess data ... WARNING ... it will take about 16 hours ...')
        users, dataset = {}, {}
        num_user = 0
        with bz2.BZ2File(os.path.join(data_dir, 'RC_2017-11.bz2'), 'r') as f:
            for i, line in tqdm(enumerate(f)):
                if i > 100000: break
                line = json.loads(line.rstrip())
                user = line['author']
                if user not in users.keys():
                    users[user] = num_user
                    user_idx = num_user
                    dataset[user_idx] = {
                        'num_data': 1,
                        'subreddit': [line['subreddit']],
                        'text': [line['body']],
                        'label': [int(line['controversiality'])]
                    }
                    num_user += 1
                else:
                    user_idx = users[user]
                    dataset[user_idx]['num_data'] += 1
                    dataset[user_idx]['subreddit'].append(line['subreddit'])
                    dataset[user_idx]['text'].append(line['body'])
                    dataset[user_idx]['label'].append(line['controversiality'])

        print(len(users.keys()), len(dataset.keys()))

        torch.manual_seed(0)
        select_users_indices = torch.randint(len(users.keys()), (8000,)).tolist()

        final_dataset = {}
        for user_id, user_idx in tqdm(users.items()):
            # preprocess 1
            if user_idx in select_users_indices:
                _data = dataset[user_idx]
                # preprocess 2
                if _data['num_data'] <= 100:
                    select_idx = []
                    for idx in range(_data['num_data']):
                        # preprocess 3-4
                        if user_id != _data['subreddit'][idx] or _data['text'] != '':
                            select_idx.append(idx)
                    final_dataset[user_idx] = {
                        'user_id': user_id,
                        'num_data': len(select_idx),
                        'text': np.array(_data['text'])[select_idx].tolist(),
                        'label': np.array(_data['label'])[select_idx].tolist()
                    }
        print(len(final_dataset.keys()))

        return final_dataset