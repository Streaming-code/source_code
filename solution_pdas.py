import itertools
import json
import math

import sys

sys.path.append("..")

# VIDEO_BIT_RATE = [750, 1200, 1850]
from config_algorithm import VIDEO_BIT_RATE
from config_algorithm import alpha, beta, gamma, theta
MPC_FUTURE_CHUNK_COUNT = 5
PAST_BW_LEN = 5
TAU = 500.0  # ms
PLAYER_NUM = 5
MILLISECONDS_IN_SECOND = 1000.0
VIDEO_CHUNCK_LEN = 1000.0

BM_exp_pool = './exp_pool/exp_pool_BM.txt'
BA_exp_pool = './exp_pool/exp_pool_BA.txt'

COLLECT_EXP = False

from simulator.video_player import Player
import numpy as np


class Algorithm:
    def __init__(self):
        # 初始化参数
        self.sleep_time = None
        self.buffer_size = 0
        self.past_bandwidth = []
        self.past_bandwidth_ests = []
        self.past_errors = []
        # 权重定义
        self.w1 = alpha
        self.w2 = gamma
        self.w3 = beta
        self.w4 = theta
        # 记录经验池
        self.pre_download_video = None
        self.last_rebufs = [0.] * 100


    def Initialize(self, chunklength):
        # 重置初始化
        self.buffer_size = 0
        self.past_bandwidth = list(np.zeros(PAST_BW_LEN))
        self.past_bandwidth_ests = list(np.zeros(PAST_BW_LEN))
        self.past_errors = list(np.zeros(PAST_BW_LEN))

        global VIDEO_CHUNCK_LEN
        VIDEO_CHUNCK_LEN = chunklength

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
        max_buffer_thresholds = self.calculate_max_buffer_thresholds(Players, retention_probs)
        # print(max_buffer_thresholds)

        # 3. 遍历视频，选择最优的比特率和视频
        best_video_id, best_bitrate, best_sleep_time = None, None, None
        best_Ui = float('-inf')

        res = []
        for i, player in enumerate(Players):
            remaining_chunks = player.get_remain_video_num()  # 计算剩余的块数
            P = min(5, remaining_chunks)  # 确保 P 不超过剩余块数
            if player.get_buffer_size() < max_buffer_thresholds[i] and player.get_chunk_sum() > player.get_chunk_counter():
                # 计算最优的比特率选择
                last_quality = 1
                downloaded_bitrate = player.get_downloaded_bitrate()
                if len(downloaded_bitrate) > 0:
                    last_quality = downloaded_bitrate[-1]
                # print(player.get_undownloaded_video_size(P))
                bit_rate = self.mpc(player.get_undownloaded_video_size(P), P, player.get_buffer_size(), last_quality)
                # 计算总的重缓冲时间
                total_rebuffering_time = self.calculate_total_rebuffering(player, Players, bit_rate)

                # 计算当前选择的QoE和Cost，并计算总的U_i
                current_qoe = self.calculate_qoe(bit_rate, retention_probs[i], player, total_rebuffering_time)
                current_cost = self.calculate_cost(bit_rate, player)
                current_Ui = current_qoe - self.w4 * current_cost
                res.append((bit_rate, current_qoe, current_Ui))
                # print(current_qoe, current_cost)

                # 如果当前的选择比之前的好，更新选择
                if current_Ui >= best_Ui:
                    best_Ui = current_Ui
                    best_video_id = i + play_video_id
                    best_bitrate = bit_rate
                    best_sleep_time = 0
        # print(max_buffer_thresholds)
        # print('best_video_id', best_video_id)
        # print('best_bitrate', best_bitrate)
        # print('best_sleep_time', best_sleep_time)
        # 4. 决策输出，如果没有合适的块可供下载，则返回睡眠时间
        self.pre_download_video = best_video_id

        if COLLECT_EXP:
            self.collect_exp(self.past_bandwidth, retention_probs, Players, best_video_id, best_bitrate, play_video_id)
        if best_video_id is not None:
            self.sleep_time = best_sleep_time
            return best_video_id, best_bitrate, best_sleep_time
        else:
            self.sleep_time = TAU
            return play_video_id, 0, self.sleep_time  # 睡眠时间设为50ms

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
            mc = math.ceil(player.play_timeline / VIDEO_CHUNCK_LEN)
            m = player.get_chunk_counter()
            p_i_m_mc = self.calculate_retention_probability(player, mc, m)
            retention_probs.append(p_i_m_mc)
        return retention_probs

    def calculate_max_buffer_thresholds(self, Players, retention_probs):
        # 计算每个视频的最大缓冲区阈值 b_{i,m}^{max}
        max_buffer_thresholds = []
        for i, player in enumerate(Players):
            max_buffer = self.calculate_max_buffer_threshold(i, player, retention_probs[i])

            max_buffer_thresholds.append(max_buffer)
        return max_buffer_thresholds

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

    def calculate_max_buffer_threshold(self, i, player, retention_prob):
        # 获取当前块的最大下载时间 T_{i,m}^{\text{max}}
        if player.get_chunk_counter() == player.get_chunk_sum():
            return player.get_buffer_size()

        # 计算最小缓冲阈值 b_{i}^{\text{th}}，使用指数衰减模型
        i_c = 0  # 当前播放的视频ID
        i = i  # 正在计算的视频ID
        C = self.past_bandwidth_ests[-1]  # 当前估计的带宽
        epsilon = 3.5  # 参数
        lambda1 = 0.3  # 参数
        lambda2 = 0.15  # 参数

        max_download_time = player.get_video_size(2) / 1000000. / C

        # 计算 b_{i}^{\text{th}} = ε * e^{-λ1*C - λ2*(i - i_c)}
        min_buffer_threshold = epsilon * np.exp(-lambda1 * C - lambda2 * (i - i_c))

        # 计算最大缓冲区阈值 b_{i,m}^{\text{max}} = max(p_{i,m}(m_c) * T_{i,m}^{\text{max}}, b_{i}^{\text{th}})
        max_buffer_threshold = max(min_buffer_threshold * 1000., retention_prob * max_download_time * 1000.)

        return max_buffer_threshold

    def calculate_total_rebuffering(self, player, Players, bit_rate):
        total_rebuffering_time = 0.0

        download_time = MILLISECONDS_IN_SECOND * (player.get_video_size(bit_rate) / 1000000.) / self.past_bandwidth_ests[-1]

        play_timeline = player.play_timeline

        for j in range(len(Players)):

            k = math.ceil(download_time / VIDEO_CHUNCK_LEN)
            z = math.ceil(play_timeline / VIDEO_CHUNCK_LEN) if j == 0 else 0

            mc = z
            m = z + k

            pj_zk = self.calculate_retention_probability(Players[j], mc, m)

            rebuffering_time_j = pj_zk * max(download_time - Players[j].get_buffer_size(), 0)

            pic_j = 1.0
            for l in range(0, j):
                z = math.ceil(play_timeline / VIDEO_CHUNCK_LEN) if l == 0 else 0
                mc = z
                m = z + k
                pic_j *= (1. - self.calculate_retention_probability(Players[l], mc, m))

            total_rebuffering_time += pic_j * rebuffering_time_j

        return total_rebuffering_time

    def calculate_qoe(self, bit_rate, retention_prob, player, total_rebuffering_time):
        # 计算QoE
        quality = retention_prob * self.calculate_quality(bit_rate, player)
        smoothness = retention_prob * self.calculate_smoothness(bit_rate, player)
        rebuffering = total_rebuffering_time / 1000.  # 累加即时重缓冲时间

        qoe = self.w1 * quality - self.w2 * smoothness - self.w3 * rebuffering
        return qoe

    def calculate_cost(self, bit_rate, player):
        # 计算下载视频块所需的带宽成本
        video_size = player.get_video_size(bit_rate)
        cost = video_size  * 8 / 1000000.

        return cost

    def calculate_quality(self, bit_rate, player):
        return VIDEO_BIT_RATE[bit_rate] / 1000.

    def calculate_smoothness(self, bit_rate, player):
        last_quality = player.get_downloaded_bitrate()[-1] if player.get_downloaded_bitrate() else bit_rate
        return abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_quality]) / 1000.

    def mpc(self, all_future_chunks_size, P, buffer_size, last_quality):

        CHUNK_COMBO_OPTIONS = []

        # make chunk combination options
        for combo in itertools.product(list(range(len(VIDEO_BIT_RATE))), repeat=P):
            CHUNK_COMBO_OPTIONS.append(combo)
        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors
        copy_past_errors = self.past_errors[-5:].copy()
        max_error = 0
        error_pos = -5
        if (len(copy_past_errors) < 5):
            error_pos = -len(copy_past_errors)
        max_error = float(max(copy_past_errors[error_pos:]))
        # print(self.past_errors[error_pos:])
        future_bandwidth = self.past_bandwidth_ests[-1] / (1. + max_error)  # robustMPC here
        # print(harmonic_bandwidth)
        # print(max_error)
        # print("future_bd:", future_bandwidth)

        # all possible combinations of 5 chunk bitrates (9^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = float('-inf')
        best_combo = ()
        start_buffer = buffer_size
        # print("start_buffer:", start_buffer)

        # start = time.time()
        for combo in CHUNK_COMBO_OPTIONS:
            # combo = full_combo[0:future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 1 seconds in that order)
            curr_rebuffer_time = 0
            curr_buffer = start_buffer  # ms
            bitrate_sum = 0
            smoothness_diffs = 0
            pre_quality = int(last_quality)
            cost_sum = 0
            # print(combo)
            for position in range(0, len(combo)):
                chunk_quality = combo[position]
                download_time = MILLISECONDS_IN_SECOND * (
                        all_future_chunks_size[chunk_quality][position] / 1000000.) / (
                                    future_bandwidth)  # this is MB/MB/s --> seconds
                # print("download time:", download_time)
                cost_sum += all_future_chunks_size[chunk_quality][position]
                if (curr_buffer < download_time):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += VIDEO_CHUNCK_LEN
                bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[pre_quality])
                pre_quality = chunk_quality
            # compute reward for this combination (one reward per 5-chunk combo)

            reward = self.w1 * (bitrate_sum / 1000.) - self.w2 * (smoothness_diffs / 1000.) - self.w3 * (
                        curr_rebuffer_time / 1000.) # - self.w4 * cost_sum * 8 / 1000000.

            # print(bitrate_sum, smoothness_diffs, curr_rebuffer_time, reward)

            if (reward >= max_reward):
                if (best_combo != ()) and best_combo[0] < combo[0]:
                    best_combo = combo
                else:
                    best_combo = combo
                max_reward = reward
                # send data to html side (first chunk of best combo)
                send_data = 0  # no combo had reward better than -1000000 (ERROR) so send 0
                if (best_combo != ()):  # some combo was good
                    send_data = best_combo[0]
                # print(bitrate_sum, smoothness_diffs, curr_rebuffer_time, reward)
        # print('max_reward', max_reward, send_data)

        bit_rate = send_data
        return bit_rate

    def collect_exp(self, past_bandwidth, retention_probs, Players, download_video_id, bit_rate, abs_cur_play_video_id):
        bt = [bd * 8. for bd in past_bandwidth] # Mb/s

        lj = [retention_probs[i] / float(Players[i].user_retent_rate[1]) for i in range(len(retention_probs))]

        gj = [Players[i].get_buffer_size() / 1000. / 5. for i in range(len(Players))] # norm 5s

        uj = []
        for i in range(len(Players)):
            if Players[i].get_remain_video_num() > 0:
                uj.append(np.average(Players[i].get_future_video_size(1)) * 8. / 1000000.) # Mb
            else:
                uj.append(0.)

        if download_video_id is None:
            with open(BM_exp_pool, 'a') as f:
                f.write(json.dumps([bt, lj, gj, uj, [5]]))
                f.write('\n')
            f.close()
            return
        else:
            with open(BM_exp_pool, 'a') as f:
                f.write(json.dumps([bt, lj, gj, uj, [download_video_id - abs_cur_play_video_id]]))
                f.write('\n')
            f.close()

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

        with open(BA_exp_pool, 'a') as f:
            f.write(json.dumps([bt, lj, gj, uj, hj, qj, fj, [bit_rate]]))
            f.write('\n')
        f.close()