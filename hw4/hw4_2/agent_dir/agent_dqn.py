from agent_dir.agent import Agent
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys, os

# hyperparameters
episodes = 6000
learning_rate = 1.5e-4
batch_size = 32
EPSILON = 1			
Naction = 3
replay_memory_size = 10000

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.linear1 = nn.Linear(80*80, 256)
		self.linear2 = nn.linear(256, Naction)
	
	def forward(self, state):
		x = F.relr(self.linear1(state))
		x = self.liner2(x)
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
		# define two neural network for target & evaluation
		self.env = env
		self.eval_net, self.target_net = Net(), Net() 
		self.update_target_counter = 0
		self.optimizer = optim.Adam(self.eval_net.paramters(), lr =
				learning_rate)
		self.loss_history = []
		self.reward_history = []
		self.replay_memory = deque()
		self.replay_start = 5000
		



    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

	def choose_action(self,x):
		x = torch.unsqueeze(torch.FloatTensor(x), 0)
		if np.random.uniform() < EPSILON:
			action_value = self.eval_net.forward(x)
			action = torch.max(actions_value, 1)[1].data.numpy()[0, 0]
		else:
			action = np.random.randint(0, N_ACTIONS)
		return action
	
	def train_Q_network(self):
		if self.update_target_counter % 1000 == 0:
			self.target_net = load_state_dict(self.eval_net.state_dict())
		sample_batch = np.random.choice(replay_memory_size,	batch_size)
		batch_memory = self.replay_memory[sample_batch, :]
		for data in batch_memory:
			current_state = torch.stack(data[0]).type(torch.FloatTensor).squeeze() / 255
			next_stat = torch.stack(data[1]).type(torch.FloatTensor).squeeze() /255


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################



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
        return self.env.get_random_action()

