import os
import bz2
import json
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset


class RedditDataset(Dataset):
    def __init__(self, data_dir, num_traindata=None):
        self.num_classes = 2
        self.train_size = 124638 # messages
        self.test_size = 15568 # messages
        self.train_num_clients = 7527 # users
        self.test_num_clients = 3440 # users

        self._init_data(data_dir, num_traindata)
        self._distributed()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data['text'][idx]
        y = self.data['label'][idx]
        return X, y

    def _update_data(self, additional_idx):
        self.current_idx += additional_idx
        self.data = self.train_data[self.current_idx]

    def _init_data(self, data_dir, num_traindata):
        if os.path.isfile(os.path.join(data_dir, 'Reddit_preprocessed.json')):
            print('>> get preprocessed Reddit dataset ...')

        else:
            torch.random.seed(0)
            selected_users_idx = torch.randint(84965681, (8000,)).tolist() # preprocessing1

            users, text, labels = [], [], []
            with bz2.BZ2File(os.path.join(data_dir,'RC_2017-11.bz2'), 'r') as f:
                for i, line in enumerate(f):
                    #if i > 10000: break  ##### just for test!
                    if i in selected_users_idx:
                        line = json.loads(line.rstrip())
                        if line['body'] != '': # preprocessing4
                            users.append(line['author'])
                            text.append(line['body'])
                            labels.append(int(line['controversiality']))
                    else:
                        continue

            # preprocessing3
            drop_users = []
            for user, count in Counter(users):
                if count > 100:
                    drop_users.append(user)
            selected_users_idx = [i for i,x in enumerate(users) if x not in drop_users]
            self.num_users = len(selected_users_idx)
            print(f'Total number of users: {self.num_users}')
            text = np.array(text)[selected_users_idx].tolist()
            labels = np.array(labels)[selected_users_idx].tolist()

        self.train_data = {
            'text': text,
            'labels': labels
        }
        self.current_idx = [*range(len(labels))]

        # random subset
        if num_traindata is not None:
            initial_idx = torch.randint(self.train_size, (num_traindata,))
            self._update_data(initial_idx)
        else:
            self.data = self.train_data

    def _distributed(self):
        local_data = {}
        for client_idx in range(self.train_num_clients):
            local_data[client_idx] = self.data[client_idx]
        self.local_data = local_data
