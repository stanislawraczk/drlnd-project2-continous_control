import random
import copy
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

BUFFER_SZIE = int(1e5) # memory size
BATCH_SIZE = 128
GAMMA = .99 # discount parameter
TAU = 1e-3 # soft update parameter
LR_ACTOR = 1e-4 # actor learning rate
LR_CRITIC = 3e-4 # critic learning rate
WEIGHT_DECAY = 0 # weight decay turned off
SIGMA = 0.3 # OU Noise sigma parameter
THETA = 0.15 # OU Noise theta parameter
LEARN_EVERY = 20 # number of steps between learning
LEARNS_NUM = 10 # number of times the networks are optimized each learning step


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Agent:
    '''
    ddpg agent class
    '''
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        '''
        Initializing local and target networks for actor and critic
        '''
        self.actor_network_local = Actor(state_size, action_size, seed).to(device)
        self.actor_network_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optim = optim.Adam(self.actor_network_local.parameters(), lr=LR_ACTOR)

        self.critic_network_local = Critic(state_size, action_size, seed).to(device)
        self.critic_network_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optim = optim.Adam(self.critic_network_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.noise = OUNoise(action_size, seed)
        self.t_step = 0
        '''
        Initializing replay buffer
        '''
        self.memory = ReplayBuffer(BUFFER_SZIE, BATCH_SIZE, seed)

    '''
    every step adds experience tuple to memory and runs learning if applicable in current step
    '''
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        if len(self.memory) > BATCH_SIZE and self.t_step % LEARN_EVERY == 0:
            for i in range(LEARNS_NUM):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def reset(self):
        self.noise.reset()

    '''
    selects action from actor local network, adds noise if applicable and returns it
    '''
    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_network_local.eval()
        with torch.no_grad():
            action = self.actor_network_local(state).cpu().data.numpy()
        self.actor_network_local.train()
        if add_noise:
            action = action + self.noise.sample()

        return np.clip(action, -1, 1)

    '''
    Calculate TD for critic and optimize critic network, then calculate actor loss and optimize actor network
    '''
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        next_actions = self.actor_network_target(next_states)
        Q_targets_next = self.critic_network_target(next_states, next_actions)
        Q_targets = rewards + (Q_targets_next * gamma * (1 - dones))

        Q_expected = self.critic_network_local(states, actions)

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network_local.parameters(), 1)
        self.critic_optim.step()

        actions_pred = self.actor_network_local(states)
        actor_loss = -self.critic_network_local(states, actions_pred).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network_local.parameters(), 1)
        self.actor_optim.step()
        self.soft_update(self.critic_network_local, self.critic_network_target, TAU)
        self.soft_update(self.actor_network_local, self.actor_network_target, TAU)

    '''
    Update target network weights using tau parameter
    '''
    def soft_update(self, local_model, target_model, tau):
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    def __init__(self, size, seed, mu=0., theta=THETA, sigma=SIGMA):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        np.random.seed(seed)
        self.reset()
    
    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.uniform(low=-1, high=1, size=len(x))
        self.state = x + dx
        return self.state


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)