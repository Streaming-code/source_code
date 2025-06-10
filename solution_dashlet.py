import sys
import copy
import os

sys.path.append("..")

MPC_FUTURE_CHUNK_COUNT = 5
PAST_BW_LEN = 5
TAU = 500.0  # ms
PLAYER_NUM = 5
MILLISECONDS_IN_SECOND = 1000.0
VIDEO_CHUNCK_LEN = 1000.0

COLLECT_EXP = False

from simulator.video_player import Player
import numpy as np

bitraterewards = [1143, 2064, 4449, 6086, 9193, 0]
CHUNKLENGTH = 5.0

penalty_weight = 700000
buffer_threshold = bitraterewards[0] / penalty_weight

class Algorithm:
    def __init__(self):
        # 初始化参数
        self.sleep_time = None
        self.buffer_size = 0
        self.past_bandwidth = []
        self.past_bandwidth_ests = []
        # 权重定义
        self.probability_map = {}

        self.estimate_bw = 0
        self.target_bitrate = 0


    def Initialize(self, foldername, chunklength):
        # 重置初始化
        self.buffer_size = 0
        self.past_bandwidth = list(np.zeros(PAST_BW_LEN))
        self.past_bandwidth_ests = list(np.zeros(PAST_BW_LEN))
        self.probability_map = self.get_probability_map(foldername)
        global CHUNKLENGTH
        CHUNKLENGTH = chunklength / 1000.

    def run(self, delay, rebuf, video_size, end_of_video, play_video_id, Players, first_step=False):
        # print('play_video_id:', play_video_id)
        # 1. delay: the time cost of your last operation
        # 2. rebuf: the length of rebufferment
        # 3. video_size: the size of the last downloaded chunk
        # 4. end_of_video: if the last video was ended
        # 5. play_video_id: the id of the current video
        # 6. Players: the video data of a RECOMMEND QUEUE of 5 (see specific definitions in readme)
        # 7. first_step: is this your first step?

        if self.sleep_time == 0 and not first_step:
            self.past_bandwidth = np.roll(self.past_bandwidth, -1)
            self.past_bandwidth[-1] = (float(video_size) / 1000000.0) / (float(delay) / 1000.0)  # MB / s
        # 1. 更新带宽估计
        if not first_step:
            self.update_bandwidth_estimate_()

        # 2. 遍历视频，选择最优的比特率和视频
        best_video_id, best_bitrate, best_sleep_time = None, None, None

        probability_weights = self.get_probability_weights(Players)
        bitrate = self.get_bitrate(Players)
        # print(bitrate)
        if first_step:  # 第一步
            throughput = 100.
        else:
            throughput = self.past_bandwidth_ests[-1] * 1000. * 8 # kb / s
        buffer_plan = self.dash_sv(Players, probability_weights, bitrate, throughput)

        # print(buffer_plan)
        for i in range(len(buffer_plan)):
            if buffer_plan[i] != -2:
                best_video_id = play_video_id + i
                best_bitrate = buffer_plan[i]
                best_sleep_time = self.sleep_time = 0.
                break

        # print(best_video_id, best_bitrate, best_sleep_time, self.sleep_time)

        if best_video_id is not None:
            return best_video_id, best_bitrate, best_sleep_time
        else:
            self.sleep_time = CHUNKLENGTH * 1000. - Players[0].play_timeline % (CHUNKLENGTH * 1000.)
            # self.sleep_time = 100.
            return play_video_id, 0, self.sleep_time  # 睡眠时间设为50ms

    def update_bandwidth_estimate_(self):
        # first get harmonic mean of last 5 bandwidths
        past_bandwidth = self.past_bandwidth[-5:]
        while past_bandwidth[0] == 0.0:
            past_bandwidth = past_bandwidth[1:]
        bandwidth_sum = 0
        for past_val in past_bandwidth:
            bandwidth_sum += (1 / float(past_val))
        harmonic_bandwidth = 1.0 / (bandwidth_sum / len(past_bandwidth))

        self.past_bandwidth_ests.append(harmonic_bandwidth)

    def get_probability_map(self, foldername):
        probability_map_f = {}

        filenames = os.listdir(foldername)

        for filename in filenames:
            # key = filename.strip(".txt")
            key = filename.strip()
            data = np.loadtxt(foldername + filename)

            probability_map_f[key] = data

        return probability_map_f

    def get_probability_weights(self, Players):
        '''
        该方法旨在获取视频列表中所有视频的用户聚合的滑动概率分布
        '''

        probability_weights = []
        for i in range(len(Players)):
            vid = Players[i].video_name
            probability_weights.append(self.probability_map[vid])

        return probability_weights

    def get_bitrate(self, Players):
        '''
        本方法旨在获取推荐列表中各个视频共有多少个比特率，各个比特率下块的大小又是多少
        '''
        bitrate_list = []

        for eidx in range(len(Players)):
            player = Players[eidx]
            bitrate_list.append([])
            for i in range(len(player.video_size[0])): # chunk位置
                bitrate_list[-1].append([])
                for j in range(len(player.video_size)): # 比特率
                    bitrate_list[eidx][i].append(player.video_size[j][i] * 8 / 1000.0)  # in kb

        return bitrate_list

    def parse_buffer_status(self, Players):
        '''
        这个方法的目的在于提取推荐列表里的视频的要buffer的视频块的序号,视频的时长,上一次的比特率以及对于当前正在观看的视频的观看进度
        '''
        buffer_length = []
        video_duration = []

        last_quality = []

        for i in range(len(Players)):
            player = Players[i]

            idx = player.video_chunk_counter
            buffer_length.append(idx)

            video_duration.append(player.video_len / 1000.)
            if len(player.download_chunk_bitrate) == 0:
                last_quality.append(-1)
            else:
                last_quality.append(player.download_chunk_bitrate[-1])

        # total buffered video in seconds - not played video in the buffer
        current_cursor = min(buffer_length[0] * CHUNKLENGTH, video_duration[0]) - Players[0].buffer_size / 1000.

        return buffer_length, video_duration, last_quality, current_cursor

    def dash_sv(self, players, probability_weights, bitrate_profile, estimate_throughput):
        '''
        这个方法负责根据推荐列表中的视频的相关信息，滑动概率分布以及预估的网络带宽找出最优的决策序列
        实质上是dashlet论文里核心代码的实现
        '''

        self.estimate_bw = estimate_throughput

        ret = [-2, -2, -2, -2, -2]
        # print(bitrate_profile)
        # 推荐列表里的视频的要buffer的视频块的序号,视频的时长,上一次的比特率以及对于当前正在观看的视频的观看进度
        buffer_length, video_duration, last_quality, current_cursor = self.parse_buffer_status(players)
        # print(buffer_length, video_duration, last_quality, current_cursor)
        look_forward_time = 25
        danger_zone_time = 5
        # print(video_duration)
        total_lengths = [int((vduration - 0.00000001) / CHUNKLENGTH) + 1 for vduration in
                         video_duration]  # 通过video_duration计算推荐列表里的每个视频的chunk总数
        # print(total_lengths)

        cursor_idx = int(current_cursor / CHUNKLENGTH) + 1

        # 获得当前正在播放的视频的播放进度的int表示
        current_playback_ts = int(current_cursor)

        # 获得推荐列表里视频的滑动概率分布，对于当前正在播放的视频，只统计还没有播放的视频的滑动概率分布
        update_weights = copy.deepcopy(probability_weights)

        update_weights[0] = update_weights[0][current_playback_ts:] / np.sum(update_weights[0][current_playback_ts:])

        nvideos = 5

        # 接下来的部分主要是计算从当前视频的观看的位置，一直看到第i个视频的第j个视频块的概率，即在第i个视频第j个视频块滑动的概率，结果会被记录到total_distribution中
        head_distribution = [np.array([1.0]) for i in range(nvideos)]

        # 这里时把概率分布的数据类型都变成浮点数
        for i in range(1, nvideos):
            head_distribution[i] = np.convolve(head_distribution[i - 1], update_weights[i - 1])

        total_distribution = {}

        danger_zone_dict = {}

        candidate_high = copy.deepcopy(buffer_length)

        for i in range(nvideos):
            for j in range(buffer_length[i], total_lengths[i]):
                # 这里会计算用户直接从i, j往后面接着看的概率,要更新滑动概率分布,被i,j之前的块,其发生滑动的概率会变成0,因为在这样的情况下,他们被视为已经看过或下载完了
                if i == 0:
                    shift_distance = j * int(CHUNKLENGTH) - int(current_cursor)
                else:
                    shift_distance = j * int(CHUNKLENGTH)
                shift_array = np.array([0.0 for ai in range(shift_distance)])

                total_distribution[(i, j)] = np.concatenate((shift_array, head_distribution[i])) * np.sum(update_weights[i][shift_distance:])
                # 然后，还会计算那些看到这些视频块的预计的penalty，然后会把那些penalty超过阈值的序号记录下来
                # penalty有两种，this_penalty和danger_penalty，实际就是论文里所说的预期rebuffer,他们之间的区别在于horizon不同，this_penalty是对于min(未来25s的，视频剩余时间)，danger_penalty是对于min(未来5s的，视频剩余时间)。
                # 对于较远的未来的预估的this_penalty的块，会将他们记为该视频高风险的块，如果该块的近期的预估的danger_penalty也较高，同时会将其记录到danger_zone_dict中
                this_penalty = 0
                danger_penalty = 0
                for tidx in range(min(look_forward_time, len(total_distribution[(i, j)]))):
                    this_penalty += (look_forward_time - tidx) * total_distribution[(i, j)][tidx]

                for tidx in range(min(danger_zone_time, len(total_distribution[(i, j)]))):
                    danger_penalty += (danger_zone_time - tidx) * total_distribution[(i, j)][tidx]

                if this_penalty > buffer_threshold:
                    candidate_high[i] = j + 1

                if danger_penalty > buffer_threshold:
                    danger_zone_dict[(i, j)] = 1

                # print((i, j, this_penalty))
        # 计算远期未来再缓冲风险较高的块的总数
        candidate_num = 0
        for i in range(nvideos):
            candidate_num += (candidate_high[i] - buffer_length[i])

        # 如果没有高风险块的话，就可以直接返回ret了，表示这个视频不需要继续下载以降低再缓冲的风险
        if candidate_num == 0:
            return ret

        # 如果有高风险的块的话，那么接下来要做的就是决定高风险块的比特率了
        # 首先，就是要估计可以接受的最大比特率target_bitrate
        target_bitrate = look_forward_time * estimate_throughput / candidate_num

        max_penalty = 0
        max_buffer_i = 0
        # print(total_distribution.keys())
        for i in range(nvideos):
            j = buffer_length[i]

            # 下面这段代码是说，虽然第i个视频的第j个块是高风险的，但是它属于是正在播放或已经播放的视频块，我们不需要考虑他的buffer问题，因为它不在total_distribution中
            if (i, j) not in total_distribution.keys():
                continue

            # 接下来,就计算这个视频会导致的this_penalty,和之前不同的是,观测的未来长度不一样了,在这里,高风险块越多,就预测更短的未来.
            look_forward_local = max(int(look_forward_time * 2 / candidate_num) + 1, 10)

            this_penalty = 0
            for tidx in range(min(look_forward_local, len(total_distribution[(i, j)]))):
                this_penalty += (look_forward_local - tidx) * total_distribution[(i, j)][tidx]

            if max_penalty < this_penalty:
                max_penalty = this_penalty
                max_buffer_i = i

        # 这样就得到了最需要进行处理的那个视频在推荐列表中的序号及其要buffer的视频块序号
        max_buffer_j = buffer_length[max_buffer_i]

        # 如果这个块在短的未来来看,也是很危险的,我们就降低其可接受的比特率上限target_bitrate
        if (max_buffer_i, max_buffer_j) in danger_zone_dict.keys():
            target_bitrate /= 2

        # 下面就开始对该视频块做比特率决策
        bitrate_choice = 0

        # 如果对这个视频来说最大的penalty实际上也很小的话,就不做预取了
        if max_penalty < 0.00001:
            return ret
        self.target_bitrate = target_bitrate
        # 这里就是一个比较了,我们只需要在该视频提供的可供选择的比特率里,选一个与target_bitrate最为接近的且小于它的比特率就可以了
        # print(len(bitrate_profile))
        # print(max_buffer_i, max_buffer_j)
        # print(len(bitrate_profile[max_buffer_i]))
        for i in range(1, len(bitrate_profile[max_buffer_i][max_buffer_j])):
            # chunk_duration = min(CHUNKLENGTH, video_duration[max_buffer_i] - max_buffer_j * CHUNKLENGTH)

            # print("=====================")
            # print(bitrate_profile[max_buffer_i][max_buffer_j][i] / CHUNKLENGTH)
            # print(target_bitrate)

            if bitrate_profile[max_buffer_i][max_buffer_j][i] < target_bitrate:
                bitrate_choice = i

        # 这里还考虑了平滑的事,为了防止前后两次决策的比特率差距过大
        # Take care of smoothness, reduce the bitrate to align with the formal chunk
        # if last_quality[max_buffer_i] != -1:
        #     if bitrate_choice != (len(bitrate_profile[max_buffer_i][max_buffer_j]) - 1):
        #         bitrate_choice = last_quality[max_buffer_i]

        # if bitrate_choice != (len(bitrate_profile[max_buffer_i][max_buffer_j]) - 1):
        #     print("change")
        # if bitrate_choice > 2:
        #     bitrate_choice = 2
        ret[max_buffer_i] = bitrate_choice

        return ret

        # 5 second danger zone
        # 10 second first chunk observe zone

