import itertools
import json
import math
from os import rename

import numpy as np
import sys

import torch
from sympy import collect

from Incendio_model import BM_Actor, BA_Actor

sys.path.append("..")
from simulator.video_player import BITRATE_LEVELS
from simulator import mpc_module

from config_algorithm import VIDEO_BIT_RATE
from config_algorithm import alpha, beta, gamma, theta
MPC_FUTURE_CHUNK_COUNT = 5
PAST_BW_LEN = 5
TAU = 200.0  # ms
PLAYER_NUM = 5
MILLISECONDS_IN_SECOND = 1000.0
VIDEO_CHUNCK_LEN = 1000.0

USE_GPU = torch.cuda.is_available()
device = torch.device('cuda' if USE_GPU else 'cpu')


class Algorithm:
    def __init__(self):
        # 初始化参数
        self.sleep_time = None
        self.buffer_size = 0
        self.past_bandwidth = []
        self.past_bandwidth_ests = []
        self.past_errors = []
        # 记录经验池
        self.pre_download_video = None
        self.last_rebufs = [0.] * 100

        self.bm_actor = None
        self.ba_actor = None


    def Initialize(self, BM_model_path, BA_model_path):
        # 重置初始化
        self.buffer_size = 0
        self.past_bandwidth = list(np.zeros(PAST_BW_LEN))
        self.past_bandwidth_ests = list(np.zeros(PAST_BW_LEN))
        self.past_errors = list(np.zeros(PAST_BW_LEN))

        # 加载模型
        self.bm_actor = BM_Actor()
        self.ba_actor = BA_Actor()

        self.bm_actor.load_state_dict(torch.load(BM_model_path, map_location=device))
        self.ba_actor.load_state_dict(torch.load(BA_model_path, map_location=device))

        if USE_GPU:
            self.bm_actor.cuda()
            self.ba_actor.cuda()

    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
        # print(play_video_id)
        # 1. delay: the time cost of your last operation
        # 2. rebuf: the length of rebufferment
        # 3. video_size: the size of the last downloaded chunk
        # 4. end_of_video: if the last video was ended
        # 5. play_video_id: the id of the current video
        # 6. Players: the video data of a RECOMMEND QUEUE of 5 (see specific definitions in readme)
        # 7. first_step: is this your first step?

        if self.pre_download_video != None:
            self.last_rebufs[self.pre_download_video] = rebuf / 1000.

        if first_step:  # 第一步
            self.sleep_time = 0
            self.pre_download_video = play_video_id
            return play_video_id, 1, self.sleep_time

        if self.sleep_time == 0:
            self.past_bandwidth = np.roll(self.past_bandwidth, -1)
            self.past_bandwidth[-1] = (float(video_size) / 1000000.0) / (float(delay) / 1000.0)  # MB / s
        # print(self.past_bandwidth)
        # 1. 更新带宽估计
        self.update_bandwidth_estimate_()

        # 2. 计算保留概率和Max Buffer阈值
        retention_probs = self.calculate_retention_probabilities(Players)
        # print(retention_probs)

        # 3. 遍历视频，选择最优的比特率和视频
        video_id, bitrate, sleep_time = None, None, None
        with torch.no_grad():
            inputs = self.get_input_data(self.past_bandwidth, retention_probs, Players, play_video_id)
            inputs = torch.tensor(inputs).reshape(1,7,5).float()

            bts = []
            for input in inputs:
                bts.append(list(input[0]))
            bts = torch.tensor(bts).reshape(len(bts), 5, 1)

            states_bm = inputs[:,0:4,:].reshape(inputs.shape[0], 1, 4 ,5)
            states_ba = inputs
            bts = bts
            if USE_GPU:
                states_bm, states_ba, bts = states_bm.cuda(), states_ba.cuda(), bts.cuda()
            # print(states_bm.shape)
            # print(bts.shape)
            # print(input[:,0:4,:].shape)
            pi_video = self.bm_actor(states_bm, bts)
            self.sleep_time = 0
            # print(pi_video)
            # print(torch.argmax(pi_video))
            # 4. 决策输出，如果没有合适的块可供下载，则返回睡眠时间

            for i in range(len(Players)):
                if Players[i].get_remain_video_num() == 0:
                    # pi_video[0][5] += pi_video[0][i]
                    pi_video[0][i] = 0.
            if pi_video.sum() == 0:
                pi_video[0][5] = 1.

            pi_video = pi_video / pi_video.sum()

            if torch.argmax(pi_video) == 5:
                self.sleep_time = TAU
                return play_video_id, 0, self.sleep_time
            else:
                video_id = torch.argmax(pi_video) + play_video_id

            # 5. 若存在要下载的视频，则进行比特率决策，返回要下载视频及其比特率

            pi_bitrate = self.ba_actor(states_ba, bts)

            bitrate = torch.argmax(pi_bitrate).item()

            sleep_time = 0

            self.pre_download_video = video_id
            # print(video_id, bitrate, sleep_time)

        return video_id, bitrate, sleep_time

    def update_bandwidth_estimate_(self):
        # record the newest error
        curr_error = 0  # default assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if (len(self.past_bandwidth_ests) > 0) and self.past_bandwidth[-1] != 0:
            curr_error = abs(self.past_bandwidth_ests[-1] - self.past_bandwidth[-1]) / float(self.past_bandwidth[-1])
        self.past_errors.append(curr_error)
        # first get harmonic mean of last 5 bandwidths
        past_bandwidth = self.past_bandwidth[-5:]
        while past_bandwidth[0] == 0.0:
            past_bandwidth = past_bandwidth[1:]
        bandwidth_sum = 0
        for past_val in past_bandwidth:
            bandwidth_sum += (1 / float(past_val))
        harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidth))

        self.past_bandwidth_ests.append(harmonic_bandwidth)

    def calculate_retention_probabilities(self, Players):
        # 计算每个视频块的保留概率 p_{i,m}(mc)
        retention_probs = []
        for player in Players:
            # 根据当前播放时间 mc 和用户留存率模型 H_{i,m} 计算
            mc = math.ceil(player.play_timeline / 2000.)
            m = player.get_chunk_counter()
            p_i_m_mc = self.calculate_retention_probability(player, mc, m)
            retention_probs.append(p_i_m_mc)
        return retention_probs

    def calculate_retention_probability(self, player, mc, m):
        if m >= player.get_chunk_sum():
            return 0.0
        # 实现保留概率的计算逻辑

        user_time, user_retent_rate = player.get_user_model()

        # 如果用户已经看完了，则留存率为0，即不需要在考虑下载该视频
        if m + 1 <= mc:
            return 1.0
        else:
            return float(user_retent_rate[m + 1]) / float(user_retent_rate[mc])


    def get_input_data(self, past_bandwidth, retention_probs, Players, abs_cur_play_video_id):
        bt = [bd * 8. for bd in past_bandwidth] # Mb/s

        lj = [retention_probs[i] / float(Players[i].user_retent_rate[1]) for i in range(len(retention_probs))]

        gj = [Players[i].get_buffer_size() / 1000. / 5. for i in range(len(Players))] # norm 5s

        uj = []
        for i in range(len(Players)):
            if Players[i].get_remain_video_num() > 0:
                uj.append(np.average(Players[i].get_future_video_size(1)) * 8. / 1000000.) # Mb
            else:
                uj.append(0.)


        hj = self.last_rebufs[abs_cur_play_video_id: abs_cur_play_video_id + 5]

        qj = []
        for i in range(len(Players)):
            download_bitrate_i = Players[i].get_downloaded_bitrate()
            if len(download_bitrate_i) > 0:
                qj.append(VIDEO_BIT_RATE[download_bitrate_i[-1]] / max(VIDEO_BIT_RATE))
            else:
                qj.append(0.)

        fj = []
        for i in range(len(Players)):
            download_bitrate_i = Players[i].get_downloaded_bitrate()
            if len(download_bitrate_i) > 2:
                fj.append(abs(VIDEO_BIT_RATE[download_bitrate_i[-1]] - VIDEO_BIT_RATE[download_bitrate_i[-2]]) / max(VIDEO_BIT_RATE))
            else:
                fj.append(0.)


        return [bt, lj, gj, uj, hj, qj, fj]