import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, N, episode_limit, batch_size):
        self.N = N
        self.episode_limit = episode_limit
        self.batch_size = batch_size
        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()
        # create a buffer (dictionary)

    def reset_buffer(self):
        self.buffer = {'obs_bm_n': np.empty([self.batch_size, self.episode_limit, self.N, 4, 5]),
                       'obs_ba_n': np.empty([self.batch_size, self.episode_limit, self.N, 7, 5]),
                       'obs_bts_n': np.empty([self.batch_size, self.episode_limit, self.N, 5, 1]),
                       's': np.empty([self.batch_size, self.episode_limit, self.N, 1, 128]),
                       'v_n': np.empty([self.batch_size, self.episode_limit + 1, self.N]),
                       'a_bm_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'a_ba_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'a_bm_logprob_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'a_ba_logprob_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'r_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'done_bm_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'done_ba_n': np.empty([self.batch_size, self.episode_limit, self.N])
                       }
        self.episode_num = 0

    def store_transition(self, episode_step, obs_bm_n, obs_ba_n, obs_bts_n, s, v_n, a_bm_n, a_ba_n, a_bm_logprob_n, a_ba_logprob_n, r_n, done_bm_n, done_ba_n):
        self.buffer['obs_bm_n'][self.episode_num][episode_step] = obs_bm_n
        self.buffer['obs_ba_n'][self.episode_num][episode_step] = obs_ba_n
        self.buffer['obs_bts_n'][self.episode_num][episode_step] = obs_bts_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['a_bm_n'][self.episode_num][episode_step] = a_bm_n
        self.buffer['a_ba_n'][self.episode_num][episode_step] = a_ba_n
        self.buffer['a_bm_logprob_n'][self.episode_num][episode_step] = a_bm_logprob_n
        self.buffer['a_ba_logprob_n'][self.episode_num][episode_step] = a_ba_logprob_n
        self.buffer['r_n'][self.episode_num][episode_step] = r_n
        self.buffer['done_bm_n'][self.episode_num][episode_step] = done_bm_n
        self.buffer['done_ba_n'][self.episode_num][episode_step] = done_ba_n

    def store_last_value(self, episode_step, v_n):
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.episode_num += 1

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch