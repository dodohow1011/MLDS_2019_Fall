from agent_dir.agent import Agent
import scipy.misc
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim
from torch.distributions import Categorical
import sys, os
from torchsummary import summary

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

episodes = 1000
checkpoint = "checkpoints"
trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor()])

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

def RGB2Gray(image, trans=trans):
    # remove score board
    image = image[24:,:,:]
    image = trans(image)
    return image

class AgentModel(nn.Module):
    def __init__(self):
        super(AgentModel, self).__init__()

        self.pool  = nn.MaxPool2d(2, stride=2)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(256*23*20, 3)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x

    def initWeight(self, mean=0.0, std=0.02):
        for m in self._modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        self.env = env
        self.policy = AgentModel().cuda()
        self.data = []

        # summary(self.policy, (1, 186, 160))
        # sys.exit()

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
        else:
            self.policy.initWeight()

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
        for episode in range(episodes):
            self.play()
            self.update()

            if  (episode+1) % 50 == 0:
                print ("Save checkpoint")
                torch.save(self.policy.state_dict(), os.path.join(checkpoints, "_{}.pt".format(episode+1)))

    def play(self):
        # clear memory
        self.data = []
        state = self.env.reset()
        total_reward = 0
        while(True):
            state = RGB2Gray(state)
            action = self.make_action(state)

            action = action + 1 # action space {1, 2, 3}
               
            next_state, reward, done, _ = self.env.step(action)
            total_reward = total_reward + reward

            self.data.append((state, action, reward)) # save for update
            print ('action', action)

            state = next_state

            if done:
                print ("Episode {} done! Reward: {:3f}".format(episode+1, total_reward))
                print (len(self.data))
                sys.exit()
                break

    def update(self):
        pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model

        There are six actions: 0 ~ 5
        0,1: Stop
        2,4: Up
        3,5: Down
        """
        ##################
        # YOUR CODE HERE #
        ##################
        state = observation.unsqueeze(0) # size = (1, 1, 187, 160)
        state = state.cuda()
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        # self.loss.append(m.log_prob(action)) # save for update
        return action.item()

