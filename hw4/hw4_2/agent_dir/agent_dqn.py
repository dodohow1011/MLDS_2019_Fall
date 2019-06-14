from agent_dir.agent import Agent
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import sys, os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
checkpoint = "checkpoints"

# hyperparameters
episodes = 6000
learning_rate = 1.5e-4
batch_size = 32
reward_discount = 0.99
minimum_EPSILON = 0.025
replay_memory_size = 10000

class Net(nn.Module):
	def __init__(self, Nstates, NActions):
		super(Net, self).__init__()
		self.linear1 = nn.Linear(Nstates*Nstates*4, 256)
		self.linear2 = nn.Linear(256, NActions)
	
	def forward(self, state):
		x = F.relu(self.linear1(state))
		x = self.linear2(x)
		x = F.softmax(x,dim=1)
		return x

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################
        self.env = env
        self.Nstates = env.observation_space.shape[0]
        self.NActions = env.action_space.n
		# define two neural network for target & evaluation
        self.eval_net = Net(self.Nstates, self.NActions).cuda()
        self.target_net = Net(self.Nstates,self.NActions).cuda()
        self.EPSILON = 1			
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr = learning_rate)
        self.loss_history = []
        self.reward_history = []
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.loss_function = nn.MSELoss()
		



    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

	
    def train_Q_network(self):
        #if len(self.replay_memory) < replay_memory_size:
        #   return
        if self.EPSILON > 0.025:
            self.EPSILON = self.EPSILON - 0.00001
        batch_memory = random.sample(self.replay_memory, batch_size)
        current_state, next_state, batch_action, batch_reward, batch_done = zip(*batch_memory)

        current_state = torch.stack(current_state).type(torch.FloatTensor).squeeze().cuda()
        next_state = torch.stack(next_state).type(torch.FloatTensor).squeeze().cuda()
        batch_action = torch.stack(batch_action).type(torch.LongTensor).squeeze().cuda()
        batch_reward = torch.stack(batch_reward).type(torch.FloatTensor).squeeze().cuda()
        batch_done = torch.stack(batch_done).type(torch.FloatTensor).squeeze().cuda()
        current_state = current_state.view(batch_size,-1)
        next_state = current_state.view(batch_size,-1)
        Q_predict = self.eval_net(current_state).gather(1,batch_action.unsqueeze(1)).squeeze(1)
        Q_target = self.target_net(next_state).detach()
        Q_target = batch_reward + reward_discount * Q_target.max(1)[0]
        loss = self.loss_function(Q_predict, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print ("loss: {:4f}".format(loss.item())) 
        return loss.item()




    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        step = 0

        for episode in range(episodes):
            observation = self.env.reset()
            self.total_reward = 0
            while(True):
                action = self.make_action(observation , test=False)
                next_observation, reward,done,info = self.env.step(action)
                self.total_reward += reward
                step += 1
                self.replay_memory.append((
                    torch.ByteTensor([observation]), 
                    torch.ByteTensor([next_observation]), 
                    torch.ByteTensor([action]), 
                    torch.ByteTensor([reward]), 
                    torch.ByteTensor([done])
                ))

                observation = next_observation

                if step%4 == 0:
                    self.loss_history.append(self.train_Q_network())
                if step%1000 == 0:
                    self.target_net.load_state_dict(self.eval_net.state_dict())

                if done:
                     print ("Episode {} done! Reward: {:3f}".format(episode+1, self.total_reward), " ")
                     self.reward_history.append(self.total_reward)
                     break
            if (episode+1)%50 == 0:
                print("save checkpoints")
                torch.save(self.eval_net.state_dict(), os.path.join(checkpoint, "_{}.pt".format(episode+1)))
        loss = np.array(self.loss_history)
        reward = np.array(self.reward_history)
        np.save('loss.npy', loss)
        np.save('reward.npy', reward)

    


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        observation = torch.FloatTensor(observation).cuda()
        observation = observation.view(1,-1)
        if np.random.uniform() < self.EPSILON:
            action = self.eval_net.forward(observation).max(-1)[1].data[0]
            action = action.item()
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

