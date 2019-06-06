from agent_dir.agent import Agent
import scipy.misc
import numpy as np
import torch
import torch.nn.Functional as F
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torch.distributions import Categorical
# from torchsummary import summary

episodes = 1000

def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = o.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)

def RGB2Gray(image):
    # remove score board
    # some margins still remain
    image = image[23:,:,:]
    gray = np.zeros((image.shape[0], image.shape[1]))
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            r = image[row][col][0] 
            g = image[row][col][1] 
            b = image[row][col][2] 
            gray[row][col] = 0.2126*r + 0.7125*g + 0.0722*b
    return gray

class AgentModel(nn.Module):
    def __init__(self):
        super(AgentModel, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        return x

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################
        self.env = env
        self.policy = AgentModel()

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        total_reward = 0
        for episode in range(episodes):
            state = env.reset()
            for step in range(10000):
                state = prepo(state)
                action = make_action(state)
                
                state, reward = self.env.step(action)
                total_reward = total_reward + reward



    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        probs = policy(observation)
        m = Categorical(probs)
        action = m.sample()

        return self.env.get_random_action()

