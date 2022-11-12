import warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings("ignore")
import game_physics
import Environment

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, Sequential, Conv2d,Flatten, Module
from torch.optim import Adam, SGD
from collections import deque 
from skimage import transform
from PIL import Image
import gym
import matplotlib.pyplot as plt
import cv2
import random,time , datetime, os, copy
from pathlib import Path
from gym.spaces import Box
from gym.wrappers import FrameStack
import glob
import subprocess as sp
import shlex

stack_size = 4 # We stack 4 frames 
save_dir = os.path.dirname(os.path.realpath(__file__))

if not os.path.exists(f'{save_dir}\\plots'):
    os.makedirs(f'{save_dir}\\plots')
if not os.path.exists(f'{save_dir}\\log'):
    os.makedirs(f'{save_dir}\\log')

if not os.path.exists(f'{save_dir}\\weights'):
    os.makedirs(f'{save_dir}\\weights')

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class InputPreprocessing(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = Image.fromarray(observation).convert('L')
        observation = np.asanyarray(observation).astype(np.uint8)
        observation = cv2.adaptiveThreshold(observation, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        observation = observation / 255
        observation = transform.resize(observation, [84, 84])
        return observation
  


class Flappy:
    def __init__(self, state_dim, action_dim, save_dir, gamma = 0.9, lr = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.net = FlappyNet(self.state_dim, self.action_dim).float()



        self.save_every = 5e5  # no. of experiences between saving Flaapybird Net

        self.memory = deque(maxlen=100000)
        self.BATCH_SIZE = 32
        self.gamma = gamma
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.HuberLoss() #loss function
        self.burnin = 1e3  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
        self.epsilon = 1.0
        self.DECAY_RATE = 0.999975
        self.min_epsilon = 0.1
        self.current_step = 0

    def act(self, state):
        """Given a state, choose an epsilon-greedy action"""
        if np.random.uniform(0,1) < self.epsilon:
            action_index = np.random.choice([0,1], size =1, p=[0.5,0.5])[0]
        # EXPLOIT
        else:
            state = state.__array__()
            state = torch.tensor(state)
            #add fourth dim to state
            state = state.unsqueeze(0)
            state = state.to(torch.float)
            action_values = self.net(state, model="online")
            action_index = torch.argmax(action_values, axis=1).item()

        # decrease epsilon
        self.epsilon = self.epsilon * self.DECAY_RATE if self.epsilon > self.min_epsilon else self.min_epsilon

        # increment step
        self.current_step += 1
        return action_index


    def remember(self, state, next_state, action, reward, done):
        """Add the experience to memory"""
        state = state.__array__()
        next_state = next_state.__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))


    def replay(self):
        """Sample experiences from memory"""
        batch = random.sample(self.memory, self.BATCH_SIZE)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        state = state.to(torch.float)
        currentQ = self.net(state, model="online")[
            np.arange(0, self.BATCH_SIZE), action.long()
        ]  # Q_online(s,a)
        return currentQ

    @torch.no_grad()
    def td_target(self, reward, nextState, done):
        nextState = nextState.to(torch.float)
        nextStateQ = self.net(nextState, model="online")
        bestAction = torch.argmax(nextStateQ, axis=1)
        next_Q = self.net(nextState, model="target")[
            np.arange(0, self.BATCH_SIZE), bestAction.long()
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    #update q_values weights
    def update_estimated_Q(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
        f"{self.save_dir}\\weights\\flappy_net_{int(self.current_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.epsilon),
            save_path,
        )
        print(f"FlappyNet saved to {save_path} at step {self.current_step}")


    def load(self):
        latest_file = max(f"{self.save_dir}\\weights", key=os.path.getctime)
        load_path= (latest_file)
        torch.load(load_path)
        

    def update(self):
        """Update online action value (Q) function with a batch of experiences"""

        if self.current_step % self.save_every == 0:
            self.save()
            
        if self.current_step % self.sync_every == 0:
            self.sync_Q_target()
            
        #current step < update every then don't update
        if self.current_step < self.burnin:
            return 0, 0

        if self.current_step % self.learn_every != 0:
            return 0, 0

        # Sample from memory
        state, next_state, action, reward, done = self.replay()
        # Get TD Estimate
        td_est = self.td_estimate(state, action)
        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)
        # Backpropagate loss through Q_online
        loss = self.update_estimated_Q(td_est, td_tgt)

        return (td_est.mean().item(), loss)
    
class FlappyNet(Module):
    """mini cnn structure
  input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
  """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        channels, height, width = input_dim

        if height != 84:
            raise ValueError(f"Expecting input height: 84, got: {height}")
        if width != 84:
            raise ValueError(f"Expecting input width: 84, got: {width}")

        self.online = Sequential(
            Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4),
            ReLU(),
            Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            ReLU(),
            Flatten(),
            Linear(3136, 512),
            ReLU(),
            Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        



def save_plot(title, episode_list, y_values, y_label, x_label, file_name, show_plot):
    plt.figure(figsize=(16,10))
    plt.plot(episode_list, y_values)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if y_label == 'cumulative train loss':
        amax = np.argmin(np.array(y_values))
    else:
        amax = np.argmax(np.array(y_values))
    xlim,ylim = plt.xlim(), plt.ylim()
    plt.plot([episode_list[amax], episode_list[amax], xlim[0]], [xlim[0], y_values[amax], y_values[amax]],
            linestyle="--")
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.savefig(save_dir + f"\\plots\\{file_name}.jpg")
    if show_plot:
        plt.show()
    plt.close('all')


def write_logs_files(gamma, lr):
    
    save_train_log = save_dir + f"\\log\\train with gamma {gamma} and rl {lr}.LOG"
    save_test_log = save_dir + f"\\log\\test with gamma {gamma} and rl {lr}.LOG"

    with open(save_train_log, "w") as f:
        f.write(f"Episode\tStep\tEpsilon\tCumulative Reward\tLoss\tQValue\tTime Taken\tTime at finish\n")

    with open(save_test_log, "w") as f:
        f.write(f"Episode\tStep\tEpsilon\tCumulative Reward\tTime Taken\tTime at finish\n")
    
    return save_train_log, save_test_log



def print_train_info(gamma, lr, episode, step, epsilon, reward, loss, q_vals, record_time):
    last_record_time = record_time
    record_time = time.time()
    time_since_last_record = np.round(record_time - last_record_time, 3)

    print(
            f"Train info:- gamma: {gamma} and lr: {lr}: "
            f"Episode {episode} - "
            f"{step} Steps Taken - "
            f"Epsilon {round(epsilon,3)} - "
            f"Cumulative Reward {reward} - "
            f"Loss {round(loss,3)} - "
            f"Q Value {round(q_vals,3)} - "
            f"Time Taken {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}"
    )
    with open(save_train_log, "a") as f:
        f.write(f"{episode}\t{step}\t{round(epsilon,3)}\t{round(reward,3)}\t\t\t{round(loss,3)}\t{round(q_vals,3)}\t{time_since_last_record}\t\t{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n")

def print_test_info(gamma, lr, episode, step, epsilon, reward, record_time):
    last_record_time = record_time
    record_time = time.time()
    time_since_last_record = np.round(record_time - last_record_time, 3)

    print(
            f"Test info:- gamma: {gamma} and lr: {lr}: "
            f"Episode {episode} - "
            f"{step} Steps Taken - "
            f"Epsilon {round(epsilon,3)} - "
            f"Mean cumulative Reward {round(reward,2)} - "
            f"Time Taken {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}"
    )
    with open(save_test_log, "a") as f:
        f.write(f"{episode}\t{step}\t{round(epsilon,3)}\t{round(reward,3)}\t\t\t{time_since_last_record}\t\t{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n")
            
    
    
def train(current_state,flappy_object,env):
    # Run agent on the state
        action = flappy_object.act(current_state)
        # Agent performs action
        next_state, reward, done, _ = env.step(action)
        # Remember
        flappy_object.remember(current_state, next_state, action, reward, done)
        # update
        q, loss = flappy_object.update()
        # Logging
        # Update state
        return action, next_state,loss, q, reward, done
         
def test(current_state,flappy_object,env):
        # Run agent on the state
        action = flappy_object.act(current_state)
        # Agent performs action
        next_state, reward, done, _ = env.step(action)
        # Logging
        return action, next_state, reward, done


# 1 trial with gamma = 0.9, lr = 0.001


def depth_HYP_search(gamma,lr,number_train_episodes):

    env = gym.make('FlappyBird-v0')
    env = SkipFrame(env, skip=stack_size)
    env = InputPreprocessing(env)
    env = FrameStack(env, num_stack=stack_size)

    flappy = Flappy(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir,gamma=gamma,lr=lr)

    train_episodes = number_train_episodes  #episods training
    test_episodes = 5 #after 10 episods of training run estimated policy for 5 times  #CONSTANT
    test_e = 0
    train_steps = []
    test_steps = []
    train_actions = []
    test_actions = []
    train_total_rewards = []
    test_total_rewards = []
    q_list = []
    loss_list = []


    train_episodes_lst = []

    test_episode_lst = []
    cumulative_train_step = 0
    cumulative_test_step = 0
    cumulative_train_step_list = []
    cumulative_test_step_list = []
    for e in range(1,train_episodes+1):
        train_step = 0
        total_train_reward = 0
        total_q = 0
        total_loss = 0
        state = env.reset()
        # Play the game!
        train_record_time = time.time()
        while True:        
            # Update state
            env.render(mode='human')

            train_action, state,loss ,q, train_reward, done = train(state,flappy,env)
            total_train_reward += train_reward  
            total_q += q
            total_loss += loss 
            train_actions.append(train_action)
            cumulative_train_step += 1
            train_step = train_step + 1
            cumulative_train_step_list.append(cumulative_train_step)
            # Check if flappy died
            if done:
                train_episodes_lst.append(e)
                train_total_rewards.append(total_train_reward)
                q_list.append(total_q)
                loss_list.append(total_loss)
                train_steps.append(train_step)
                break
        save_plot(f"cumulative train reward per episode with gamma {flappy.gamma} and learning rate {flappy.lr}", train_episodes_lst, train_total_rewards,'cumulative train reward' ,'Number of episode', f'train cumulative reward with gamma {flappy.gamma} and learning rate {flappy.lr}', False)
        save_plot(f"cumulative train huber loss per episode with gamma {flappy.gamma} and learning rate {flappy.lr}", train_episodes_lst, loss_list, 'cumulative train loss' ,'Number of episode' , f'train cumulative huber loss with gamma {flappy.gamma} and learning rate {flappy.lr}', False)
        save_plot(f"cumulative train q_values per episode with gamma {flappy.gamma} and learning rate {flappy.lr}", train_episodes_lst, q_list, 'cumulative train q_values' ,'Number of episode', f'train cumulative q_values with gamma {flappy.gamma} and learning rate {flappy.lr}', False)
        save_plot(f"number of steps at each train episode with gamma {flappy.gamma} and learning rate {flappy.lr}", train_episodes_lst, train_steps, 'number of steps' ,'Number of episode' , f'Number of steps per train episode with gamma {flappy.gamma} and learning rate {flappy.lr}', False)
        if e % 10 == 0 and e != 0: #for every 10 episode
            print_train_info(gamma, lr, e, cumulative_train_step, flappy.epsilon, total_train_reward, total_loss, total_q, train_record_time)
            test_record_time = time.time()
            
            for i in range(1, test_episodes+1):
                test_step = 0
                total_test_reward = []
                state = env.reset()
                done = False
                test_e += 1
                test_episode_lst.append(test_e)
                while not done:
                    test_action, state, test_reward, done = test(state,flappy,env)
                    total_test_reward.append(test_reward)
                    test_step = test_step + 1
                    cumulative_test_step += 1
                    cumulative_test_step_list.append(cumulative_test_step)
                    test_actions.append(test_action)
                    if done:
                        break
                mean_reward = np.mean(total_test_reward)
                test_total_rewards.append(mean_reward)
                test_steps.append(test_step)
                save_plot(f"cumulative mean test reward per episode with gamma {flappy.gamma} and learning rate {flappy.lr}", test_episode_lst, test_total_rewards,'cumulative test reward' ,'Number of episode', f'test cumulative reward with gamma {flappy.gamma} and learning rate {flappy.lr}', False)
                save_plot(f"number of steps at each test episode with gamma {flappy.gamma} and learning rate {flappy.lr}", test_episode_lst, test_steps, 'number of steps' ,'Number of episode' , f'Number of steps per test episode with gamma {flappy.gamma} and learning rate {flappy.lr}', False)

                if i % 5 == 0:
                    print_test_info(gamma, lr, test_e, cumulative_test_step, flappy.epsilon, mean_reward, test_record_time)

            

    env.close()


    env = gym.make('FlappyBird-v0')

    del train_episodes_lst
    del test_episode_lst
    del loss_list
    del q_list
    del train_steps
    del test_steps
    del train_total_rewards
    del test_total_rewards




save_train_log, save_test_log = write_logs_files(0.8, 0.01)

depth_HYP_search(gamma=0.8,lr=0.01,number_train_episodes=1000)
