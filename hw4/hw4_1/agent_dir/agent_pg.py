from agent_dir.agent import Agent
import scipy.misc
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
# from torchvision import transforms
import torch.optim as optim
from torch.distributions import Categorical
import sys
# from torchsummary import summary

episodes = 1000
checkpoint = "checkpoints"

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
    resized = y.thumbnail()
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
        self.linear = nn.Linear(32*187*160, 3)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = x.reshape(1, -1)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
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
        self.rewards = []
        self.loss = []

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
        optimizer = optim.RMSprop(self.policy.parameters(), lr=1e-4, weight_decay=0.99)
        total_reward = 0
        for episode in range(episodes):
            state = self.env.reset()
            while(True):
                state = RGB2Gray(state)
                state = state.reshape(1, state.shape[0], -1) # size = (1, 187, 160)
                action = self.make_action(state)

                action = action + 1 # action space {1, 2, 3}
                
                state, reward, done, _ = self.env.step(action)
                total_reward = total_reward + reward

                self.rewards.append(reward) # save for update

                if done:
                    print ("Episode{} done! Reward: {:3f}".format(episode+1, total_reward))
                    total_reward = 0
                    break

            if  (episode+1) % 50 == 0:
                print ("Save checkpoint")
                torch.save(self.policy.state_dict(), os.path.join(checkpoints, "_{}.pt".format(episode+1)))



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
        state = torch.from_numpy(observation).float().unsqueeze(0) # size = (1, 1, 187, 160)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.loss.append(m.log_prob(action)) # save for update

        return action.item()

