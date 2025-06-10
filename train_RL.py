import sys, os
import time

import torch
from torch.distributions import Categorical

from Incendio_model import BM_Actor, BA_Actor, Critic, Trainer_RL
from replay_buffer import ReplayBuffer
from run_incendio import test_user_samples
from simulator import controller as env, short_video_load_trace

sys.path.append('./simulator/')
import argparse
import random
import numpy as np
import math
import multiprocess as mp

parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO")
parser.add_argument("--N", type=int, default=int(32), help=" number of agent")
parser.add_argument("--max_train_steps", type=int, default=int(2e3), help=" Maximum number of training steps")
parser.add_argument("--episode_limit", type=int, default=32, help="Maximum number of steps per episode")
parser.add_argument("--evaluate_freq", type=float, default=10, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--critic_pretrain", type=bool, default=True, help="train critic before RL train start")
parser.add_argument("--critic_pretrain_epoch", type=float, default=10, help="train critic before RL train start")

parser.add_argument("--batch_size", type=int, default=16, help="Batch size (the number of episodes)")
parser.add_argument("--mini_batch_size", type=int, default=4, help="Minibatch size (the number of episodes)")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
parser.add_argument("--K_epochs", type=int, default=2, help="GAE parameter")
parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
parser.add_argument("--use_value_clip", type=float, default=True, help="Whether to use value clip.")
parser.add_argument("--entropy_coef", type=float, default=0.03, help="Trick 5: policy entropy")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--use_lr_decay", type=float, default=False, help="")


parser.add_argument('--trace', type=str, default='mmgc-test', help='The network trace you are testing (fixed, high, low, medium, middle)')

parser.add_argument('--sub_dataset', type=str, default='./data/dataset_2s/', help='Is testing quickstart')
parser.add_argument('--video_size_dir', type=str, default='./data/dataset_2s/short_video_size/', help='Is testing quickstart')
parser.add_argument('--user_ret_dir', type=str, default='./data/dataset_2s/user_ret/', help='Is testing quickstart')
parser.add_argument('--sample_user_dir', type=str, default='./data/dataset_2s/sample_user', help='Is testing quickstart')
parser.add_argument('--chunklength', type=float, default=2000., help='')


args = parser.parse_args()

LOG_FILE = './RL_traing_logs'
nn_model_save_path= './model/RL/'

# QoE arguments
from config_algorithm import VIDEO_BIT_RATE
from config_algorithm import alpha, beta, gamma, theta

ALL_VIDEO_NUM = 100
all_cooked_time = []
all_cooked_bw = []

# For training a3c
NUM_AGENTS = args.N
S_INFO = 7
S_LEN = 5
A_DIM = 3
MODEL_SAVE_INTERVAL = args.evaluate_freq
TRAIN_SEQ_LEN = args.episode_limit
TRAIN_TRACES = './data/network_traces/sampled_4G_train/'
DEFAULT_ID = 0
DEFAULT_BITRATE = 0
DEFAULT_SLEEP = 0
PAST_BW_LEN = 5
TAU = 200.

b_IN_B = 8
b_IN_kb = 1000
# log file
log_file = open(LOG_FILE + '.txt', 'a')

# Random seeds settings
# RANDOM_SEED = 42  # the random seed for user retention
# np.random.seed(RANDOM_SEED)
# seeds = np.random.randint(100, size=(7, 2))

NN_MODEL_BM = './model/IL/bm_agent/bm_actor_epoch_1700.pth'
NN_MODEL_BA = './model/IL/ba_agent/ba_actor_epoch_1700.pth'
NN_MODEL_CRITIC = None

# NN_MODEL_BM = './model/RL/bm_agent/bm_actor_epoch_20.pth'
# NN_MODEL_BA = './model/RL/ba_agent/ba_actor_epoch_20.pth'
# NN_MODEL_CRITIC = './model/RL/critic/critic_epoch_20.pth'

start_epoch = 1

# NN_MODEL_BM = './model/RL/bm_agent/bm_actor_best.pth'
# NN_MODEL_BA = './model/RL/ba_agent/ba_actor_best.pth'
# NN_MODEL_CRITIC = './model/RL/critic/critic_best.pth'

USE_GPU = torch.cuda.is_available()

batch_size = args.batch_size

pre_score = -float('inf')
entropy_decay = False
score_decay_count = 0

def test_model(epoch, BM_MODEL_SAVE_PATH, BA_MODEL_SAVE_PATH):
    with torch.no_grad():
        stime = time.time()
        # origin_stdout = sys.stdout
        # sys.stdout = open('nul', 'w')
        avgs = test_user_samples('sampled_4G', 5, BM_MODEL_SAVE_PATH, BA_MODEL_SAVE_PATH)
        # sys.stdout = origin_stdout
        log_file.write(f'{epoch},{avgs[0]},{avgs[1]},{avgs[2]},{avgs[3]},{avgs[4]},{avgs[5]},{avgs[6]}\n')
        log_file.flush()
        etime = time.time()
        print('time cost is ', etime - stime)
        global pre_score, entropy_decay, score_decay_count

        if avgs[0] < pre_score:
            score_decay_count += 1
            if score_decay_count == int(100 / MODEL_SAVE_INTERVAL):
                score_decay_count = 0
                entropy_decay = True
        else:
            score_decay_count = 0
            pre_score = avgs[0]

def run_with_timeout(func, timeout, epoch, BM_MODEL_SAVE_PATH, BA_MODEL_SAVE_PATH):
    p = mp.Process(target=func, args=(epoch, BM_MODEL_SAVE_PATH, BA_MODEL_SAVE_PATH))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        print('=' * 20)
        print('time out')
        print('=' * 20)

def cul_reward(quality, rebuffer, smooth, bandwidth_usage, sleep_time):
    if sleep_time == 0 :
        reward = alpha * quality / 1000. - beta * rebuffer / 1000. - gamma * smooth / 1000. - theta * bandwidth_usage * 8. / 1000000.
    else:
        reward = - beta * rebuffer / 1000.
    return reward

def calculate_retention_probabilities(Players):
    # 计算每个视频块的保留概率 p_{i,m}(mc)
    retention_probs = []
    for player in Players:
        # 根据当前播放时间 mc 和用户留存率模型 H_{i,m} 计算
        mc = math.ceil(player.play_timeline / args.chunklength)
        m = player.get_chunk_counter()
        p_i_m_mc = calculate_retention_probability(player, mc, m)
        retention_probs.append(p_i_m_mc)
    return retention_probs

def calculate_retention_probability(player, mc, m):
    if m >= player.get_chunk_sum():
        return 0.0
    # 实现保留概率的计算逻辑

    user_time, user_retent_rate = player.get_user_model()

    # 如果用户已经看完了，则留存率为0，即不需要在考虑下载该视频
    if m + 1 <= mc:
        return 1.0
    else:
        return float(user_retent_rate[m + 1]) / float(user_retent_rate[mc])


def get_input_data(past_bandwidth, retention_probs, last_rebufs, Players, abs_cur_play_video_id):
    bt = [bd * 8. for bd in past_bandwidth] # Mb/s

    lj = [retention_probs[i] / float(Players[i].user_retent_rate[1]) for i in range(len(retention_probs))]

    gj = [Players[i].get_buffer_size() / 1000. / 5. for i in range(len(Players))] # norm 5s

    uj = []
    for i in range(len(Players)):
        if Players[i].get_remain_video_num() > 0:
            uj.append(np.average(Players[i].get_future_video_size(1)) * 8. / 1000000.) # Mb
        else:
            uj.append(0.)

    hj = last_rebufs[abs_cur_play_video_id: abs_cur_play_video_id + 5]

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

def store_work_agent_data(replay_buffer, s_bm_batchs, s_ba_batchs, bts_btachs, s_critic_batchs, a_bm_batchs, a_ba_batchs, r_batchs, v_batchs, bm_log_prob_batchs, ba_log_prob_batchs, dones_bm, dones_ba):
    # print(s_bm_batchs, s_ba_batchs, s_critic_batchs, a_bm_batchs, a_ba_batchs, r_batchs, v_batchs, bm_log_prob_batchs, ba_log_prob_batchs, dones)
    s_bm_batchs = torch.tensor(np.array(s_bm_batchs)).permute(1, 0, 2, 3).tolist()
    s_ba_batchs = torch.tensor(np.array(s_ba_batchs)).permute(1, 0, 2, 3).tolist()
    bts_btachs = torch.tensor(np.array(bts_btachs)).permute(1, 0, 2, 3).tolist()
    # print(bts_btachs)
    s_critic_batchs = torch.tensor(np.array(s_critic_batchs)).permute(1, 0, 2, 3).tolist()
    a_bm_batchs = torch.tensor(np.array(a_bm_batchs)).permute(1, 0).tolist()
    a_ba_batchs = torch.tensor(np.array(a_ba_batchs)).permute(1, 0).tolist()
    r_batchs = torch.tensor(np.array(r_batchs)).permute(1, 0).tolist()
    v_batchs = torch.tensor(np.array(v_batchs)).permute(1, 0).tolist()
    bm_log_prob_batchs = torch.tensor(np.array(bm_log_prob_batchs)).permute(1, 0).tolist()
    ba_log_prob_batchs = torch.tensor(np.array(ba_log_prob_batchs)).permute(1, 0).tolist()
    dones_bm = torch.tensor(np.array(dones_bm)).permute(1, 0).tolist()
    dones_ba = torch.tensor(np.array(dones_ba)).permute(1, 0).tolist()
    # print(len(s_bm_batchs), len(r_batchs), len(v_batchs),len(dones))
    for i in range(len(s_bm_batchs)):
        replay_buffer.store_transition(i, s_bm_batchs[i], s_ba_batchs[i], bts_btachs[i], s_critic_batchs[i],
                                       v_batchs[i],
                                       a_bm_batchs[i], a_ba_batchs[i],
                                       bm_log_prob_batchs[i], ba_log_prob_batchs[i],
                                       r_batchs[i],
                                       dones_bm[i],
                                       dones_ba[i])
    replay_buffer.store_last_value(len(v_batchs) - 1, v_batchs[-1])

    print('data restore, ', replay_buffer.episode_num)

def central_agent(net_params_queues, exp_queues, args):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    bm_actor = BM_Actor()
    ba_actor = BA_Actor()

    critic = Critic()

    trainer_rl = Trainer_RL(args.lr, bm_actor, ba_actor, critic)

    if NN_MODEL_BM is not None and NN_MODEL_BA is not None:
        trainer_rl.load_model(NN_MODEL_BM, NN_MODEL_BA)
        print('BM BA AGENT MODEL LOAD')

    if NN_MODEL_CRITIC is not None:
        trainer_rl.load_critic(NN_MODEL_CRITIC)
        args.critic_pretrain = False
        print('Critic MODEL LOAD')

    trainer_rl.Initial(args)

    if USE_GPU:
        trainer_rl.bm_actor = trainer_rl.bm_actor.cuda()
        trainer_rl.ba_actor = trainer_rl.ba_actor.cuda()
        trainer_rl.critic = trainer_rl.critic.cuda()

    # synchronize the network parameters of work agent
    bm_actor_net_params, ba_actor_net_params, critic_net_params = trainer_rl.get_network_params()
    for i in range(NUM_AGENTS):
        net_params_queues[i].put([bm_actor_net_params, ba_actor_net_params, critic_net_params])

    replay_buffer = ReplayBuffer(NUM_AGENTS, TRAIN_SEQ_LEN, batch_size)
    replay_buffer.reset_buffer()

    # restore neural net parameters
    epoch = start_epoch
    # assemble experiences from agents, compute the gradients
    if args.critic_pretrain:
        CRITIC_PRETRAIN_EPOCH = args.critic_pretrain_epoch
    else:
        CRITIC_PRETRAIN_EPOCH = -1

    while True:
        # record average reward and td loss change
        # in the experiences from the agents

        s_bm_batchs = []
        s_ba_batchs = []
        bts_btachs = []
        s_critic_batchs = []
        a_bm_batchs = []
        a_ba_batchs = []
        r_batchs = []
        v_batchs = []
        bm_log_prob_batchs = []
        ba_log_prob_batchs = []
        dones_bm = []
        dones_ba = []
        for i in range(NUM_AGENTS):
            s_bm_batch, s_ba_batch, bts_btach, s_critic_batch, a_bm_batch, a_ba_batch, r_batch, v_batch, bm_log_prob_batch, ba_log_prob_batch, done_bm, done_ba = exp_queues[i].get()

            s_bm_batchs.append(s_bm_batch)
            s_ba_batchs.append(s_ba_batch)
            bts_btachs.append(bts_btach)
            s_critic_batchs.append(s_critic_batch)
            a_bm_batchs.append(a_bm_batch)
            a_ba_batchs.append(a_ba_batch)
            r_batchs.append(r_batch)
            v_batchs.append(v_batch)
            bm_log_prob_batchs.append(bm_log_prob_batch)
            ba_log_prob_batchs.append(ba_log_prob_batch)
            dones_bm.append(done_bm)
            dones_ba.append(done_ba)

        store_work_agent_data(replay_buffer, s_bm_batchs, s_ba_batchs, bts_btachs, s_critic_batchs, a_bm_batchs,
                              a_ba_batchs, r_batchs, v_batchs, bm_log_prob_batchs, ba_log_prob_batchs, dones_bm, dones_ba)
        if replay_buffer.episode_num == batch_size:
            print('=' * 20, 'training of epoch ', epoch, '=' * 20)
            # 抽样，更新
            trainer_rl.train(epoch, replay_buffer)
            # 更新完毕，清楚缓冲
            replay_buffer.reset_buffer()

            # log training information

            if epoch % MODEL_SAVE_INTERVAL == 0 and epoch >= CRITIC_PRETRAIN_EPOCH:
                print("---------epoch %d--------" % epoch)
                # Save the neural net parameters to disk.
                BM_MODEL_SAVE_PATH = nn_model_save_path + 'bm_agent/bm_actor_epoch_' + str(epoch) + '.pth'
                BA_MODEL_SAVE_PATH = nn_model_save_path + 'ba_agent/ba_actor_epoch_' + str(epoch) + '.pth'

                CRITIC_SAVE_PATH = nn_model_save_path + 'critic/critic_epoch_' + str(epoch) + '.pth'

                trainer_rl.save_model(BM_MODEL_SAVE_PATH, BA_MODEL_SAVE_PATH)

                trainer_rl.save_critic(CRITIC_SAVE_PATH)

                run_with_timeout(test_model, 900, epoch, BM_MODEL_SAVE_PATH, BA_MODEL_SAVE_PATH)
                global entropy_decay
                if entropy_decay:
                    trainer_rl.entropy_decay()
                    entropy_decay = False

            # 分发模型参数
            if epoch > CRITIC_PRETRAIN_EPOCH:
                bm_actor_net_params, ba_actor_net_params, critic_net_params = trainer_rl.get_network_params()
            else:
                _, _, critic_net_params = trainer_rl.get_network_params()

            for i in range(NUM_AGENTS):
                net_params_queues[i].put([bm_actor_net_params, ba_actor_net_params, critic_net_params])

            epoch += 1

        if epoch == args.max_train_steps:
            sys.exit(0)


def work_agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue, args):

    # Initial the a3c
    bm_actor = BM_Actor()
    ba_actor = BA_Actor()
    critic = Critic()

    trainer = Trainer_RL(args.lr, bm_actor, ba_actor, critic)
    trainer.Initial(args)

    bm_actor_net_params, ba_actor_net_params, critic_net_params = net_params_queue.get()

    trainer.set_network_params(bm_actor_net_params, ba_actor_net_params, critic_net_params)

    if USE_GPU:
        trainer.bm_actor = trainer.bm_actor.cuda()
        trainer.ba_actor = trainer.ba_actor.cuda()
        trainer.critic = trainer.critic.cuda()

    # Initial the first step
    play_video_id = DEFAULT_ID
    download_video_id = DEFAULT_ID
    bit_rate = DEFAULT_BITRATE
    sleep_time = DEFAULT_SLEEP
    current_video_id = 0

    # Initial the state, action, reward batch

    s_bm_batch = [list(np.zeros((4, 5)))]
    s_ba_batch = [list(np.zeros((7, 5)))]
    bts_batch = [list(np.zeros(5))]
    s_critic_batch = [list(np.zeros((S_INFO, S_LEN)))]
    a_bm_batch = [download_video_id]
    a_ba_batch = [bit_rate]
    bm_log_prob_batch = [0]
    ba_log_prob_batch = [0]
    r_batch = []
    v_batch = [0]
    dones_batch_bm = [0]
    dones_batch_ba = [0]

    pre_download_video = None
    last_rebufs = [0.] * 100
    past_bandwidth = list(np.zeros(PAST_BW_LEN))

    # sum of wasted bytes for a user
    sum_wasted_bytes = 0
    QoE = 0
    last_played_chunk = -1  # record the last played chunk
    last_bitrate = DEFAULT_BITRATE
    bandwidth_usage = 0  # record total bandwidth usage

    user_sample_id = random.randint(0, 4)

    network_trace_idx = random.randint(0, len(all_cooked_time) - 1)
    # network_trace_idx = 0

    user_swipe_dir = args.sample_user_dir
    user_swipe_trace = user_swipe_dir + '/user_' + str(user_sample_id) + '.txt'
    seeds = []

    with open(user_swipe_trace, 'r') as f:
        for line in f:
            seeds.append(float(line))

    # Initial the environment
    net_env = env.Environment(user_sample_id=user_sample_id,
                              all_cooked_time=all_cooked_time[network_trace_idx],
                              all_cooked_bw=all_cooked_bw[network_trace_idx],
                              video_num=ALL_VIDEO_NUM,
                              seeds=seeds,
                              args=args)

    send_data_count = 0

    while True:
        if len(net_env.players) < 5:

            # print('env changed')

            user_sample_id = random.randint(0, 4)
            user_swipe_dir = args.sample_user_dir
            user_swipe_trace = user_swipe_dir + '/user_' + str(user_sample_id) + '.txt'
            seeds = []

            with open(user_swipe_trace, 'r') as f:
                for line in f:
                    seeds.append(float(line))

            # network_trace_idx = (network_trace_idx + 1) % len(all_cooked_time)
            network_trace_idx = random.randint(0, len(all_cooked_time) - 1)
            # Initial the environment
            net_env = env.Environment(user_sample_id=user_sample_id,
                                      all_cooked_time=all_cooked_time[network_trace_idx],
                                      all_cooked_bw=all_cooked_bw[network_trace_idx],
                                      video_num=ALL_VIDEO_NUM,
                                      seeds=seeds,
                                      args=args)

            s_bm_batch = [list(np.zeros((4, 5)))]
            s_ba_batch = [list(np.zeros((7, 5)))]
            bts_batch = [list(np.zeros(5))]
            s_critic_batch = [list(np.zeros((S_INFO, S_LEN)))]
            a_bm_batch = [download_video_id]
            a_ba_batch = [bit_rate]
            bm_log_prob_batch = [0]
            ba_log_prob_batch = [0]
            r_batch = []
            v_batch = [0]
            dones_batch_bm = [0]
            dones_batch_ba = [0]

            past_bandwidth = list(np.zeros(PAST_BW_LEN))
            last_rebufs = [0.] * 100


            play_video_id = DEFAULT_ID
            download_video_id = DEFAULT_ID
            bit_rate = DEFAULT_BITRATE
            sleep_time = DEFAULT_SLEEP
        else:
            dones_batch_bm.append(0)
            if sleep_time == 0:
                dones_batch_ba.append(0)
            else:
                dones_batch_ba.append(1)

        # culculate action reward
        # 计算上一步的reward

        reward = 0
        # print('=========================')
        # print(len(net_env.players), download_video_id, last_play_video)
        # print('=========================')
        player = net_env.players[download_video_id - play_video_id]
        # Get the current downloaded chunk number and play chunk number
        download_chunk_num = player.video_chunk_counter # n

        user_leave_chunk_id = math.floor(net_env.user_models[download_video_id - play_video_id].sample_playback_duration / args.chunklength) # get last play chunk id

        user_rets = calculate_retention_probabilities(net_env.players)
        user_ret = user_rets[download_video_id - play_video_id]

        quality = 0
        smooth = 0
        bitrate_usage = 0
        if user_leave_chunk_id >= download_chunk_num:
            quality = user_ret * VIDEO_BIT_RATE[bit_rate]
            if len(player.get_downloaded_bitrate()) > 0:
                last_download_bitrate = player.get_downloaded_bitrate()[-1]
                smooth = user_ret * abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_download_bitrate])
            else:
                smooth = 0.

        # print(download_video_id, bit_rate, sleep_time)
        # Take action on and get the states from the env
        delay, rebuf, video_size, end_of_video, \
        play_video_id, waste_bytes = net_env.buffer_management(download_video_id, bit_rate, sleep_time)

        bitrate_usage = video_size

        reward = cul_reward(quality, rebuf, smooth, bitrate_usage, sleep_time)

        r_batch.append(reward)

        if len(net_env.players) < 5:
            while len(r_batch) < TRAIN_SEQ_LEN + 1:
                s_bm_batch.append(list(np.zeros((4, 5))))
                s_ba_batch.append(list(np.zeros((7, 5))))
                bts_batch.append(list(np.zeros((5, 1))))
                s_critic_batch.append(list(np.zeros((1, 128))))
                a_bm_batch.append(0)
                a_ba_batch.append(0)
                r_batch.append(0)
                v_batch.append(0)
                bm_log_prob_batch.append(0)
                ba_log_prob_batch.append(0)
                dones_batch_bm.append(1)
                dones_batch_ba.append(1)
            v_batch.append(0)
            exp_queue.put([s_bm_batch[1:],  # ignore the first chuck
                           s_ba_batch[1:],
                           bts_batch[1:],
                           s_critic_batch[1:],
                           a_bm_batch[1:],  # since we don't have the
                           a_ba_batch[1:],
                           r_batch[1:],  # control over it
                           v_batch[1:],
                           bm_log_prob_batch[1:],
                           ba_log_prob_batch[1:],
                           dones_batch_bm[1:],
                           dones_batch_ba[1:]])
            send_data_count += 1
            if send_data_count == batch_size:
                # synchronize the network parameters from the coordinator
                bm_actor_net_params, ba_actor_net_params, critic_net_params = net_params_queue.get()
                trainer.set_network_params(bm_actor_net_params, ba_actor_net_params, critic_net_params)
                send_data_count = 0
            continue

        if len(r_batch) >= TRAIN_SEQ_LEN + 1 : # or len(net_env.players) == 0: # 可以在这里加判定，没有视频就加上dones=1，把数据补全到一个batch

            user_rets = calculate_retention_probabilities(net_env.players)
            state_actor = get_input_data(past_bandwidth, user_rets, last_rebufs, net_env.players, play_video_id)
            # print(state_actor)
            inputs = torch.tensor(state_actor).reshape(1, 7, 5).float()
            bts = []
            for input in inputs:
                bts.append(list(input[0]))
            bts = torch.tensor(bts).reshape(len(bts), 5, 1)

            states_bm = inputs[:, 0:4, :].reshape(inputs.shape[0], 1, 4, 5)
            states_ba = inputs
            bts = bts

            if USE_GPU:
                states_bm, states_ba, bts = states_bm.cuda(), states_ba.cuda(), bts.cuda()

            # Decide the actions for the next step
            trainer.bm_actor(states_bm, bts)
            trainer.ba_actor(states_ba, bts)

            bm_hidden_state = bm_actor.last_hidden_output
            ba_hidden_state = ba_actor.last_hidden_output

            # print(bm_hidden_state.shape, ba_hidden_state.shape)
            critic_input = torch.concat([bm_hidden_state, ba_hidden_state], dim=1)
            # print(critic_input.shape)

            value = trainer.critic(critic_input)

            v_batch.append(value.item())

            exp_queue.put([s_bm_batch[1:],  # ignore the first chuck
                           s_ba_batch[1:],
                           bts_batch[1:],
                           s_critic_batch[1:],
                           a_bm_batch[1:],  # since we don't have the
                           a_ba_batch[1:],
                           r_batch[1:],  # control over it
                           v_batch[1:],
                           bm_log_prob_batch[1:],
                           ba_log_prob_batch[1:],
                           dones_batch_bm[1:],
                           dones_batch_ba[1:]])

            del s_bm_batch[:]
            del s_ba_batch[:]
            del bts_batch[:]
            del s_critic_batch[:]
            del a_bm_batch[:]
            del a_ba_batch[:]
            del r_batch[:]
            del v_batch[:]
            del bm_log_prob_batch[:]
            del ba_log_prob_batch[:]
            del dones_batch_bm[:]
            del dones_batch_ba[:]

            send_data_count += 1
            if send_data_count == batch_size:
                # synchronize the network parameters from the coordinator
                bm_actor_net_params, ba_actor_net_params, critic_net_params = net_params_queue.get()
                trainer.set_network_params(bm_actor_net_params, ba_actor_net_params, critic_net_params)
                send_data_count = 0

        # 计算下一步的state
        pre_download_video = download_video_id

        if pre_download_video != None:
            last_rebufs[pre_download_video] = rebuf / 1000.

        if sleep_time == 0:
            past_bandwidth = np.roll(past_bandwidth, -1)
            past_bandwidth[-1] = (float(video_size) / 1000000.0) / (float(delay) / 1000.0)  # MB / s

        user_rets = calculate_retention_probabilities(net_env.players)

        state_actor = get_input_data(past_bandwidth, user_rets, last_rebufs, net_env.players, play_video_id)

        inputs = torch.tensor(state_actor).reshape(1, 7, 5).float()

        bts = []
        for input in inputs:
            bts.append(list(input[0]))
        bts = torch.tensor(bts).reshape(len(bts), 5, 1)

        states_bm = inputs[:, 0:4, :].reshape(inputs.shape[0], 1, 4, 5)
        states_ba = inputs
        bts = bts
        # print(states_bm.shape, states_ba.shape, bts.shape)

        s_bm_batch.append(states_bm.tolist()[0][0])
        # print(states_bm.shape,states_ba.shape, bts.shape)
        # print(bts.tolist())
        # print(s_bm_batch)
        s_ba_batch.append(states_ba.tolist()[0])
        # print(s_ba_batch)
        bts_batch.append(bts.tolist()[0])
        # print(bts.shape)

        if USE_GPU:
            states_bm, states_ba, bts = states_bm.cuda(), states_ba.cuda(), bts.cuda()

        # Decide the actions for the next step
        pi_video = trainer.bm_actor(states_bm, bts)

        pi = pi_video.clone()
        dist_pi = Categorical(pi)
        # print(pi_video)
        # print(torch.argmax(pi_video))
        # 4. 决策输出，如果没有合适的块可供下载，则返回睡眠时间
        a_bm = -1
        a_bm_logprob = -1
        sleep_time = 0.

        # IAM 无效动作掩码
        for i in range(5):
            if net_env.players[i].get_remain_video_num() == 0:
                # pi_video[0][5] += pi_video[0][i]
                pi_video[0][i] = 0.
        if pi_video.sum() == 0.:
            pi_video[0][5] = 1.
        pi_video = pi_video / pi_video.sum()

        dist = Categorical(probs=pi_video)

        a_bm = dist.sample()

        if a_bm.item() == 5:
            a_bm_logprob = dist_pi.log_prob(a_bm)
            sleep_time = TAU
        else:
            a_bm_logprob = dist_pi.log_prob(a_bm)
            download_video_id = a_bm.item() + play_video_id

        a_bm_batch.append(a_bm.item())
        # print(pi_video)
        # print(pi_video.shape)
        bm_log_prob_batch.append(a_bm_logprob.item())
        # print(a_bm, bm_log_prob_batch)

        # 5. 若存在要下载的视频，则进行比特率决策，返回要下载视频及其比特率

        pi_bitrate = trainer.ba_actor(states_ba, bts)

        dist = Categorical(probs=pi_bitrate)
        bit_rate = dist.sample()
        a_ba_logprob = dist.log_prob(bit_rate)

        bit_rate = bit_rate.item()

        a_ba_batch.append(bit_rate)
        # print(pi_bitrate.shape)
        ba_log_prob_batch.append(a_ba_logprob.item())
        # print(bit_rate, ba_log_prob_batch)

        bm_hidden_state = bm_actor.last_hidden_output
        ba_hidden_state = ba_actor.last_hidden_output

        # print(bm_hidden_state.shape, ba_hidden_state.shape)
        critic_input = torch.concat([bm_hidden_state, ba_hidden_state], dim=1)
        # print(critic_input.shape)

        s_critic_batch.append(critic_input.tolist())

        value = trainer.critic(critic_input)

        v_batch.append(value.item())


def main(args):
    # np.random.seed(RANDOM_SEED)
    # assert len(MODIFY_BIT_RATE) == A_DIM

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues, args))
    coordinator.start()

    all_cooked_time, all_cooked_bw = short_video_load_trace.load_trace(TRAIN_TRACES)
    work_agents = []
    for i in range(NUM_AGENTS):
        work_agents.append(mp.Process(target=work_agent,
                                      args=(i % 5, all_cooked_time, all_cooked_bw, net_params_queues[i], exp_queues[i], args)))
    for i in range(NUM_AGENTS):
        work_agents[i].start()
    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main(args)