import os.path
from modulefinder import Module

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SequentialSampler



FEATURE_NUM = 256
ACTION_EPS = 1e-4
ENTROPY_EPS = 1e-6
GAMMA = 0.99
USE_GPU = torch.cuda.is_available()
device = "cuda" if USE_GPU else "cpu"

class BM_Actor(nn.Module):
	def __init__(self):
		super(BM_Actor, self).__init__()
		# Actor network
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=5, padding=2)
		self.conv3 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=5, padding=2)

		self.gru = nn.GRU(input_size=1, hidden_size=64, batch_first=True)

		self.fc1 = nn.Linear(144, 64)
		self.fc2 = nn.Linear(64, 6)

		self.last_hidden_output = None

	def forward(self, states, bts):
		# print(states.shape)
		output_conv1 = F.leaky_relu(self.conv1(states))
		# print(output_conv1.shape)
		output_conv2 = F.leaky_relu(self.conv2(output_conv1))
		# print(output_conv2.shape)
		output_conv3 = F.leaky_relu(self.conv3(output_conv2))
		# print(output_conv3.shape)

		out, hidden = self.gru(bts)
		# print(bts.shape)
		# print(out)
		# print(out[:,-1,:].shape)
		out_gru = F.leaky_relu(out[:,-1,:])
		# print(out_gru.shape)

		flatten_conv3 = torch.flatten(output_conv3, start_dim=1)
		# print(flatten_conv3.shape)

		merge_input = torch.concat([flatten_conv3, out_gru], dim=1)
		# print(merge_input.shape)

		output_fc1 = F.leaky_relu(self.fc1(merge_input))

		self.last_hidden_output = output_fc1

		pi_video = F.softmax(self.fc2(output_fc1), dim=1)
		# print(pi_video.shape)
		# print(pi_video)
		return pi_video

class BA_Actor(nn.Module):
	def __init__(self):
		super(BA_Actor, self).__init__()
		# Actor network

		self.fc1 = nn.Linear(35, 128)

		self.fc2 = nn.Linear(128, 256)

		self.fc3 = nn.Linear(256, 80)

		self.gru = nn.GRU(input_size=1, hidden_size=64, batch_first=True)

		self.fc4 = nn.Linear(144, 64)
		self.fc5 = nn.Linear(64, 6)

		self.last_hidden_output = None

	def forward(self, states, bts):

		flatten_inputs = torch.flatten(states, start_dim=1)
		# print(flatten_inputs.shape)

		output_fc1 = F.leaky_relu(self.fc1(flatten_inputs))
		# print(output_fc1.shape)

		output_fc2 = F.leaky_relu(self.fc2(output_fc1))
		# print(output_fc2.shape)

		output_fc3 = F.leaky_relu(self.fc3(output_fc2))
		# print(output_fc3.shape)

		out, hidden = self.gru(bts)
		out_gru = F.leaky_relu(out[:,-1,:])
		# print(out_gru.shape)

		merge_input = torch.concat([output_fc3, out_gru], dim=1)
		# print(merge_input.shape)

		output_fc4 = F.leaky_relu(self.fc4(merge_input))

		self.last_hidden_output = output_fc4

		pi_bitrate = F.softmax(self.fc5(output_fc4), dim=1)

		return pi_bitrate

class Critic(nn.Module):
	def __init__(self):
		super(Critic, self).__init__()
		# Actor network
		self.fc = nn.Linear(128, 1)

	def forward(self, inputs):

		value = self.fc(inputs)

		return value

class Trainer_RL:
	def __init__(self,learning_rate, bm_actor, ba_actor, critic):

		super(Trainer_RL, self).__init__()
		self.lr = learning_rate

		self.bm_actor = bm_actor
		self.ba_actor = ba_actor
		self.critic = critic

	def Initial(self, args):
		self.N = args.N
		self.episode_limit = args.episode_limit

		self.batch_size = args.batch_size
		self.mini_batch_size = args.mini_batch_size
		self.max_train_steps = args.max_train_steps
		self.lr = args.lr
		self.gamma = args.gamma
		self.lamda = args.lamda
		self.epsilon = args.epsilon
		self.K_epochs = args.K_epochs
		self.entropy_coef = args.entropy_coef
		self.set_adam_eps = args.set_adam_eps
		self.use_grad_clip = args.use_grad_clip
		self.use_adv_norm = args.use_adv_norm
		self.use_value_clip = args.use_value_clip
		self.use_lr_decay = args.use_lr_decay

		if args.critic_pretrain:
			self.critic_pretrain_epoch = args.critic_pretrain_epoch
		else:
			self.critic_pretrain_epoch = -1

		if args.set_adam_eps:
			self.ac_optimizer = torch.optim.Adam(list(self.bm_actor.parameters()) + list(self.ba_actor.parameters()),
												 lr=self.lr,
												 eps=1e-5)
			self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
													 lr=self.lr,
													 eps=1e-5)
		else:
			self.ac_optimizer = torch.optim.Adam(list(self.bm_actor.parameters()) + list(self.ba_actor.parameters()),
												 lr=self.lr)
			self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
													 lr=self.lr)

	def entropy_decay(self):
		"""
        Decay entropy.
        """
		self.entropy_coef = max(self.entropy_coef - 0.01 , 0.01)

	def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
		lr_now = self.lr * (1 - total_steps / self.max_train_steps)
		for p in self.ac_optimizer.param_groups:
			p['lr'] = lr_now

	def get_network_params(self):
		return self.bm_actor.state_dict(), self.ba_actor.state_dict(), self.critic.state_dict()

	def set_network_params(self, input_network_params_BM, input_network_params_BA, input_network_params_CRITIC):
		self.bm_actor.load_state_dict(input_network_params_BM)
		self.ba_actor.load_state_dict(input_network_params_BA)
		self.critic.load_state_dict(input_network_params_CRITIC)

	def get_inputs(self, batch):
		bm_actor_inputs = batch['obs_bm_n'].reshape(self.batch_size,batch['obs_bm_n'].shape[1] ,self.N ,1 ,4 ,5).tolist()
		ba_actor_inputs = batch['obs_ba_n'].reshape(self.batch_size,batch['obs_ba_n'].shape[1] ,self.N ,7 ,5).tolist()
		bts = batch['obs_bts_n'].reshape(self.batch_size, batch['obs_bts_n'].shape[1] ,self.N ,5 ,1).tolist()
		critic_inputs = batch['s'].tolist()

		return bm_actor_inputs, ba_actor_inputs, bts, critic_inputs

	def train(self, epoch, replay_buffer):
		batch = replay_buffer.get_training_data()  # get training data
		# Calculate the advantage using GAE
		adv_bm = []
		gae_bm = 0

		adv_ba = []
		gae_ba = 0
		with torch.no_grad():  # adv_bm and td_target have no gradient
			deltas_bm = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_bm_n']) - batch['v_n'][:, :-1]  # deltas_bm.shape=(batch_size,episode_limit,N)
			for t in reversed(range(self.episode_limit)):
				gae_bm = deltas_bm[:, t] + self.gamma * self.lamda * gae_bm
				adv_bm.insert(0, gae_bm)
			adv_bm = torch.stack(adv_bm, dim=1)  # adv_bm.shape(batch_size,episode_limit,N)
			v_target_bm = adv_bm + batch['v_n'][:, :-1]  # v_target_bm.shape(batch_size,episode_limit,N)
			# print(adv_bm.shape, v_target_bm.shape)
			if self.use_adv_norm:  # Trick 1: advantage normalization
				adv_bm = ((adv_bm - adv_bm.mean()) / (adv_bm.std() + 1e-5))
			if USE_GPU:
				adv_bm = adv_bm.cuda()
				v_target_bm = v_target_bm.cuda()

			deltas_ba = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_ba_n']) - batch['v_n'][:,
																									 :-1]  # deltas_bm.shape=(batch_size,episode_limit,N)
			for t in reversed(range(self.episode_limit)):
				gae_ba = deltas_ba[:, t] + self.gamma * self.lamda * gae_ba
				adv_ba.insert(0, gae_ba)
			adv_ba = torch.stack(adv_ba, dim=1)  # adv_bm.shape(batch_size,episode_limit,N)
			v_target_ba = adv_ba + batch['v_n'][:, :-1]  # v_target_bm.shape(batch_size,episode_limit,N)
			# print(adv_bm.shape, v_target_bm.shape)
			if self.use_adv_norm:  # Trick 1: advantage normalization
				adv_ba = ((adv_ba - adv_ba.mean()) / (adv_ba.std() + 1e-5))
			if USE_GPU:
				adv_ba = adv_ba.cuda()
				v_target_ba = v_target_ba.cuda()

		"""
			Get actor_inputs and critic_inputs
			actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
			critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
		"""
		bm_actor_inputs, ba_actor_inputs, bts, critic_inputs = self.get_inputs(batch)

		# Optimize policy for K epochs:
		for _ in range(self.K_epochs):
			for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
				"""
                    get probs_now and values_now
                    probs_now.shape=(mini_batch_size, episode_limit, N, action_dim_bm)
                    values_now.shape=(mini_batch_size, episode_limit, N)
                """
				bm_states_batch = torch.tensor(bm_actor_inputs)[index]
				ba_states_batch =  torch.tensor(ba_actor_inputs)[index]
				bts_batch = torch.tensor(bts)[index]

				cr_states_batch = torch.tensor(critic_inputs)[index]

				probs_bm_now = []
				probs_ba_now = []
				values_now = []

				for i in range(len(index)):
					bm_states, ba_states, bts_state, cr_states = bm_states_batch[i], ba_states_batch[i], bts_batch[i], cr_states_batch[i]
					probs_bm = []
					probs_ba = []
					values = []
					for bm_state, ba_state, bt_state, cr_state in zip(bm_states, ba_states, bts_state, cr_states):
						if USE_GPU:
							bm_state = bm_state.cuda()
							ba_state = ba_state.cuda()
							bt_state = bt_state.cuda()
							cr_state = cr_state.cuda()

						prob_bm = self.bm_actor(bm_state, bt_state)
						prob_ba = self.ba_actor(ba_state, bt_state)
						value = self.critic(cr_state)

						probs_bm.append(prob_bm)
						probs_ba.append(prob_ba)
						values.append(value)

					probs_bm = torch.stack(probs_bm)
					probs_ba = torch.stack(probs_ba)
					values = torch.stack(values)

					probs_bm_now.append(probs_bm)
					probs_ba_now.append(probs_ba)
					values_now.append(values)

				probs_bm_now = torch.stack(probs_bm_now)
				probs_ba_now = torch.stack(probs_ba_now)
				values_now = torch.stack(values_now)

				if USE_GPU:
					probs_bm_now = probs_bm_now.cuda()
					probs_ba_now = probs_ba_now.cuda()
					values_now = values_now.cuda()

				dist_now_bm = Categorical(probs_bm_now)
				# print(probs_bm_now)
				# print(batch['a_bm_n'].shape)
				# print(batch['a_bm_n'])
				dist_entropy_bm = dist_now_bm.entropy()  # dist_entropy_bm.shape=(mini_batch_size, episode_limit, N)
				# batch['a_n'][index].shape=(mini_batch_size, episode_limit, N)
				a_bm_n_batch = batch['a_bm_n'][index]
				if USE_GPU:
					a_bm_n_batch = a_bm_n_batch.cuda()
				a_bm_logprob_n_now = dist_now_bm.log_prob(a_bm_n_batch)  # a_bm_logprob_n_now.shape=(mini_batch_size, episode_limit, N)
				# a/b=exp(log(a)-log(b))
				# print(a_bm_logprob_n_now)

				a_bm_logprob_n_batch = batch['a_bm_logprob_n'][index].detach()
				if USE_GPU:
					a_bm_logprob_n_batch = a_bm_logprob_n_batch.cuda()
				ratios_bm = torch.exp(a_bm_logprob_n_now - a_bm_logprob_n_batch)  # ratios_bm.shape=(mini_batch_size, episode_limit, N)
				surr1_bm = ratios_bm
				surr2_bm = torch.clamp(ratios_bm, 1 - self.epsilon, 1 + self.epsilon)
				bm_actor_loss = -torch.min(surr1_bm, surr2_bm) * adv_bm[index] + self.entropy_coef * dist_entropy_bm
				# print(surr1_bm)

				dist_now_ba = Categorical(probs_ba_now)
				dist_entropy_ba = dist_now_ba.entropy()  # dist_entropy_bm.shape=(mini_batch_size, episode_limit, N)
				# batch['a_n'][index].shape=(mini_batch_size, episode_limit, N)
				a_ba_n_batch = batch['a_ba_n'][index]
				if USE_GPU:
					a_ba_n_batch = a_ba_n_batch.cuda()
				a_ba_logprob_n_now = dist_now_ba.log_prob(a_ba_n_batch)  # a_bm_logprob_n_now.shape=(mini_batch_size, episode_limit, N)
				# a/b=exp(log(a)-log(b))
				a_ba_logprob_n_batch = batch['a_ba_logprob_n'][index].detach()
				if USE_GPU:
					a_ba_logprob_n_batch = a_ba_logprob_n_batch.cuda()
				ratios_ba = torch.exp(a_ba_logprob_n_now - a_ba_logprob_n_batch)  # ratios_bm.shape=(mini_batch_size, episode_limit, N)
				surr1_ba = ratios_ba
				surr2_ba = torch.clamp(ratios_ba, 1 - self.epsilon, 1 + self.epsilon)
				ba_actor_loss = -torch.min(surr1_ba, surr2_ba) * adv_ba[index] + self.entropy_coef * dist_entropy_ba
				# print(ratios_ba, torch.clamp(ratios_ba, 1 - self.epsilon, 1 + self.epsilon))

				if self.use_value_clip:
					values_old = batch["v_n"][index, :-1].detach().unsqueeze(-1).unsqueeze(-1).detach()
					if USE_GPU:
						values_old = values_old.cuda()
					values_error_clip = torch.clamp(values_now, values_old - self.epsilon, values_old + self.epsilon) - v_target_bm[index].unsqueeze(-1).unsqueeze(-1)
					values_error_original = values_now - v_target_bm[index].unsqueeze(-1).unsqueeze(-1)
					critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
					# print(values_now.shape, values_old.shape, v_target_bm[index].unsqueeze(-1).unsqueeze(-1).shape)
					# print(values_error_clip.shape, values_error_original.shape)
				else:
					critic_loss = (values_now - v_target_bm[index].unsqueeze(-1).unsqueeze(-1)) ** 2

				if epoch > self.critic_pretrain_epoch:
					self.ac_optimizer.zero_grad()
					ac_loss = bm_actor_loss.mean() + ba_actor_loss.mean()
					ac_loss.backward()
					print(ac_loss, bm_actor_loss.mean(), ba_actor_loss.mean())
					if self.use_grad_clip:  # Trick 7: Gradient clip
						torch.nn.utils.clip_grad_norm_(list(self.bm_actor.parameters()) + list(self.ba_actor.parameters()), 10.0)
					self.ac_optimizer.step()

				self.critic_optimizer.zero_grad()
				critic_loss.mean().backward()
				print(critic_loss.mean())
				if self.use_grad_clip:  # Trick 7: Gradient clip
					torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()), 10.0)
				self.critic_optimizer.step()

		if self.use_lr_decay:
			self.lr_decay(epoch)

	def predict(self, input):
		with torch.no_grad():
			pi, value = self.actor_critic.forward(input)
			return pi

	def load_model(self, nn_model_bm, nn_model_ba):
		self.bm_actor.load_state_dict(torch.load(nn_model_bm, map_location=device))
		self.ba_actor.load_state_dict(torch.load(nn_model_ba, map_location=device))

	def load_critic(self, nn_model):
		self.critic.load_state_dict(torch.load(nn_model, map_location=device))

	def save_critic(self, nn_model):
		torch.save(self.critic.state_dict(), nn_model)

	def save_model(self, nn_model_bm, nn_model_ba):
		model_params = self.bm_actor.state_dict()
		torch.save(model_params, nn_model_bm)

		model_params = self.ba_actor.state_dict()
		torch.save(model_params, nn_model_ba)

class Trainer_IL(nn.Module):
	def __init__(self,learning_rate, weight_decay, model):
		super(Trainer_IL, self).__init__()

		self.learning_rate = learning_rate
		self.weight_decay = weight_decay

		self.model = model

		self.loss_function = nn.CrossEntropyLoss(reduction='mean')

		self.optimizer = torch.optim.Adam(
			self.model.parameters(),
			lr=self.learning_rate
		)

	def forward(self, states, bts, labels):

		pi = self.model(states, bts)
		# print(pi)
		log_pi = torch.log(pi + 1e-8)

		# loss = self.loss_function(pi, labels)
		# print(log_pi * labels)
		loss = -(log_pi * labels).mean()
		# print(loss)

		return loss

	def predict(self, input):
		with torch.no_grad():
			pi = self.model.forward(input)
			return pi

	def load_model(self, nn_model):
		self.model.load_state_dict(torch.load(nn_model))

	def save_model(self, nn_model):
		torch.save(self.model.state_dict(), nn_model)