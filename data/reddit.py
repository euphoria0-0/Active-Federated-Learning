import os
import bz2
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class RedditDataset:
    def __init__(self, data_dir, args):
        self.num_classes = 2
        #self.train_size = 124638 # messages
        #self.test_size = 15568 # messages
        self.train_num_clients = args.total_num_client # 7668 # 7527 (paper)
        self.test_num_clients = args.test_num_clients # 2099
        self.batch_size = args.batch_size #128
        self.maxlen = args.maxlen #400

        self._init_data(data_dir)
        print(f'Total number of users: {self.train_num_clients}')


    '''def _update_data(self, additional_idx):
        self.current_idx += additional_idx
        self.data = self.train_data[self.current_idx]'''

    def _init_data(self, data_dir):
        file_name = os.path.join(data_dir, 'Reddit_preprocessed_7668.pickle')
        if os.path.isfile(file_name) and self.batch_size == 128 and self.maxlen == 400:
            with open(file_name, 'rb') as f:
                dataset = pickle.load(f) # user_id, num_data, text, label
        else:
            dataset = preprocess(data_dir)
        self.dataset = dataset


    def _batch_data(self, data, indices):
        '''
        data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
        returns x, y, which are both numpy array of length: batch_size
        '''
        data_x = np.array(data['text'])[indices]
        data_y = np.array(data['label'])[indices]

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
                tmp = len(ALL_LETTERS) if tmp == -1 else tmp
                tmp = torch.tensor([tmp], dtype=torch.long)
                indices = torch.cat((indices, tmp), dim=0)
            x_batch.append(indices)
        #maxlen = max([x.size(0) for x in x_batch])

        x_batch2 = torch.empty((0, self.maxlen), dtype=torch.long)
        for x in x_batch:
            x = torch.unsqueeze(F.pad(x, (0, self.maxlen-x.size(0)), value=0), 0)
            x_batch2 = torch.cat((x_batch2, x), dim=0)
        return x_batch2




def preprocess(data_dir):
    print('>> load and preprocess data ... WARNING ... it will take about 16 hours ...')
    users, dataset = {}, {}
    num_user = 0
    with bz2.BZ2File(os.path.join(data_dir, 'RC_2017-11.bz2'), 'r') as f:
        for i, line in tqdm(enumerate(f)):
            #if i > 100000: break
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