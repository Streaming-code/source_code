import sys, os
sys.path.append('./simulator/')
import argparse
import random
import numpy as np
from simulator import controller as env, short_video_load_trace

parser = argparse.ArgumentParser()
parser.add_argument('--trace', type=str, default='mixed', help='The network trace you are testing (fixed, high, low, medium, middle)')

parser.add_argument('--sub_dataset', type=str, default='', help='')
parser.add_argument('--video_size_dir', type=str, default='', help='')
parser.add_argument('--user_ret_dir', type=str, default='', help='')
parser.add_argument('--sample_user_dir', type=str, default='', help='')
parser.add_argument('--chunklength', type=float, default=1000., help='')

args = parser.parse_args()

# RANDOM_SEED = 42  # the random seed for user retention
# np.random.seed(RANDOM_SEED)
# seeds = np.random.randint(100, size=(7, 2))

# VIDEO_BIT_RATE = [750, 1200, 1850]  # Kbps
SUMMARY_DIR = 'logs'
LOG_FILE = 'logs/log.txt'
log_file = None

# QoE arguments

from config_algorithm import VIDEO_BIT_RATE
from config_algorithm import alpha, beta, gamma, theta

ALL_VIDEO_NUM = 100
# baseline_QoE = 600  # baseline's QoE
# TOLERANCE = 0.1  # The tolerance of the QoE decrease
MIN_QOE = -1e4
all_cooked_time = []
all_cooked_bw = []

# record the last chunk(which will be played) of each video to aid the calculation of smoothness
last_chunk_bitrate = [-1] * ALL_VIDEO_NUM
headtime = 5

# calculate the smooth penalty for an action to download:
# chunk:[chunk_id] of the video:[download_video_id] with bitrate:[quality]
def get_smooth(net_env, download_video_id, chunk_id, quality):
    if download_video_id == 0 and chunk_id == 0:  # is the first chunk of all
        return 0
    if chunk_id == 0:  # needs to find the last chunk of the last video
        last_bitrate = last_chunk_bitrate[download_video_id - 1]
        if last_bitrate == -1:  # the neighbour chunk is not downloaded
            return 0
    else:
        last_bitrate = net_env.players[download_video_id - net_env.get_start_video_id()].get_downloaded_bitrate()[chunk_id - 1]
    return abs(quality - VIDEO_BIT_RATE[last_bitrate])


def test(trace_id, user_sample_id, seeds):

    # print('------------trace ', trace_id, '--------------', file=log_file)

    from solution_pdas import Algorithm

    # start the test

    solution = Algorithm()
    solution.Initialize(args.chunklength)

    # all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(trace_path)
    net_env = env.Environment(user_sample_id, all_cooked_time[trace_id], all_cooked_bw[trace_id], ALL_VIDEO_NUM, seeds, args)

    # Decision variables
    download_video_id, bit_rate, sleep_time = solution.run(0, 0, 0, False, 0, net_env.players, True)  # take the first step

    # sum of wasted bytes for a user
    sum_wasted_bytes = 0
    QoE = 0

    quality_all = 0
    smooth_all = 0
    rebuffer_all = 0

    last_played_chunk = -1  # record the last played chunk
    bandwidth_usage = 0  # record total bandwidth usage

    real_time = 0
    pre_play_video_id = 0

    view_chunk_num = 0
    download_chunk_num = 0

    while True:
        # calculate the quality and smooth for this download step taken
        quality = 0
        smooth = 0
        if sleep_time == 0:
            # the last chunk id that user watched
            max_watch_chunk_id = net_env.user_models[
                download_video_id - net_env.get_start_video_id()].get_watch_chunk_cnt()
            # last downloaded chunk id
            download_chunk = net_env.players[download_video_id - net_env.get_start_video_id()].get_chunk_counter()
            if max_watch_chunk_id >= download_chunk:  # the downloaded chunk will be played
                if download_chunk == max_watch_chunk_id:  # maintain the last_chunk_bitrate array
                    last_chunk_bitrate[download_video_id] = bit_rate
                    rel_id = download_video_id - net_env.get_start_video_id()
                    if rel_id + 1 < len(net_env.user_models):  # If its not the last visible video
                        if net_env.players[rel_id + 1].get_chunk_counter() != 0:
                            # if the next video chunk has already been downloaded before this last chunk,
                            # we include the smooth penalty here.
                            next_bitrate = net_env.players[rel_id + 1].get_downloaded_bitrate()[0]
                            smooth += abs(quality - VIDEO_BIT_RATE[next_bitrate])
                quality = VIDEO_BIT_RATE[bit_rate]
                smooth += get_smooth(net_env, download_video_id, download_chunk, quality)
                # print("Causing smooth penalty: ", smooth, file=log_file)
                view_chunk_num += 1
            download_chunk_num += 1


        delay, rebuf, video_size, end_of_video, \
        play_video_id, waste_bytes = net_env.buffer_management(download_video_id, bit_rate, sleep_time)
        # print(delay, rebuf, video_size, end_of_video, play_video_id, waste_bytes)
        if pre_play_video_id == play_video_id:
            user_swipe = 0
        else:
            user_swipe = 1
        print(f'{real_time},{download_video_id},{bit_rate},{sleep_time},{delay},{rebuf},{user_swipe}', file=log_file)

        if sleep_time != 0:
            real_time += int(sleep_time)
        else:
            real_time += int(delay)
        pre_play_video_id = play_video_id

        # Update bandwidth usage
        bandwidth_usage += video_size

        # Update bandwidth wastage
        sum_wasted_bytes += waste_bytes  # Sum up the bandwidth wastage

        # play over all videos
        if len(net_env.players) < 5:
            for player in net_env.players:
                for i in range(len(player.download_chunk_bitrate)):
                    download_bitrate = player.download_chunk_bitrate[i]
                    download_size = player.video_size[download_bitrate][i]
                    sum_wasted_bytes += download_size
            break

        # Update QoE:

        one_step_QoE = alpha * quality / 1000. - beta * rebuf / 1000. - gamma * smooth / 1000.
        QoE += one_step_QoE

        quality_all += quality / 1000.
        smooth_all += smooth / 1000.
        rebuffer_all += rebuf / 1000.

        # Apply the participant's algorithm to decide the args for the next step
        download_video_id, bit_rate, sleep_time = solution.run(delay, rebuf, video_size, end_of_video, play_video_id, net_env.players, False)
        # print(download_video_id, bit_rate, sleep_time)

    # Score
    S = QoE - theta * bandwidth_usage * 8 / 1000000.
    print("Your score is: ", S)

    # QoE
    print("Your QoE is: ", QoE)
    # wasted_bytes
    print("Your sum of wasted bytes is: ", sum_wasted_bytes)
    print("Quality_all: ", quality_all)
    print("Smooth_all: ", smooth_all)
    print("Rebuffer_all: ", rebuffer_all)
    # end the test
    # print('------------trace ', trace_id, '--------------\n\n', file=log_file)
    return np.array([S, bandwidth_usage,  QoE, sum_wasted_bytes, quality_all / view_chunk_num, smooth_all, rebuffer_all])


def test_all_traces(trace, user_sample_id):

    LOG_DIR = 'logs/log_pdas/' + str(args.chunklength) + '/' + args.sub_dataset + '/user_' + str(user_sample_id) + '/'
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    avg = np.zeros(7)
    cooked_trace_folder = 'data/network_traces/' + trace + '/'
    global all_cooked_time, all_cooked_bw, ALL_VIDEO_NUM
    all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(cooked_trace_folder)

    user_swipe_dir = args.sample_user_dir
    user_swipe_trace = user_swipe_dir + '/user_' + str(user_sample_id) + '.txt'
    seeds = []
    with open(user_swipe_trace, 'r') as f:
        for line in f:
            seeds.append(float(line))
    ALL_VIDEO_NUM = len(seeds)
    # print(seeds)
    global log_file
    for i in range(len(all_cooked_time)):
        LOG_FILE = LOG_DIR + 'trace_' + str(i)
        log_file = open(LOG_FILE, 'w')
        print('------------trace ', i, '--------------')
        avg += test(i, user_sample_id, seeds)
        print('---------------------------------------\n\n')
    avg /= len(all_cooked_time)
    print("\n\nYour average indexes under [", trace, "] network is: ")
    print("Score: ", avg[0])
    print("Bandwidth Usage: ", avg[1])
    print("QoE: ", avg[2])
    print("Sum Wasted Bytes: ", avg[3])

    print("Quality_all: ", avg[4])
    print("Smooth_all: ", avg[5])
    print("Rebuffer_all: ", avg[6])

    return avg


def test_user_samples(trace, sample_cnt):  # test 50 user sample
    seed_for_sample = np.random.randint(10000, size=(1001, 1))
    avgs = np.zeros(7)
    for j in range(sample_cnt):
        print('------------sample user ', j, '--------------')
        # 这里把随机生成的seeds改成某用户的滑动轨迹
        # global seeds
        # np.random.seed(seed_for_sample[j])
        # seeds = np.random.randint(10000, size=(ALL_VIDEO_NUM, 2))  # reset the sample random seeds
        avgs += test_all_traces(trace, j)
    avgs /= sample_cnt
    print("\nScore: ", avgs[0])
    print("Bandwidth Usage: ", avgs[1])
    print("QoE: ", avgs[2])
    print("Sum Wasted Bytes: ", avgs[3])

    print("Quality_all: ", avgs[4])
    print("Smooth_all: ", avgs[5])
    print("Rebuffer_time: ", avgs[6])

    performance_under_sub_dataset = './pdas_' + str(args.chunklength/ 1000) + trace
    with open(performance_under_sub_dataset, 'a') as f:
        # f.write(args.sub_dataset + '\n')
        f.write(f'{args.sub_dataset},{avgs[0]},{avgs[1]},{avgs[2]},{avgs[3]},{avgs[4]},{avgs[5]},{avgs[6]}\n')


if __name__ == '__main__':
    # assert args.trace in ["mixed", "high", "low", "medium"]
        test_user_samples(args.trace, 5)
