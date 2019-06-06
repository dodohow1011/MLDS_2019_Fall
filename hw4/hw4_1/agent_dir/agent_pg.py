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

        self.policy_history = []
        self.reward_episode = []
        self.discount = 0.99

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

        self.loss_history = []
        self.reward_history = []

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
        print ('Start training!')
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        for episode in range(episodes):
            self.play(episode)
            self.update()

            if  (episode+1) % 50 == 0:
                print ("Save checkpoint")
                torch.save(self.policy.state_dict(), os.path.join(checkpoint, "_{}.pt".format(episode+1)))

    def play(self, episode):
        # clear memory
        state = self.env.reset()
        state = RGB2Gray(state)
        self.total_reward = 0
        self.pre_state = None
        while(True):
            action = self.make_action(state)
            action = action + 1 # action space {1, 2, 3}

            state, reward, done, _ = self.env.step(action)
            self.policy.reward_episode.append(reward)
            self.total_reward = self.total_reward + reward
            state = RGB2Gray(state)

            if done:
                print ("Episode {} done! Reward: {:3f}".format(episode+1, self.total_reward))
                break

    def update(self):
        optimizer = optim.RMSprop(self.policy.parameters(), lr=1e-2, weight_decay=0.99)

        R = 0
        rewards = []
        reward_episode = np.array(self.policy.reward_episode)
        reward_episode = reward_episode[::-1]
        for i in range(len(reward_episode)):
            for j in range(len(reward_episode)-i):
                R = reward_episode[j] + self.policy.discount * R
            rewards.append(R)
            R = 0
        rewards = rewards[1:]

        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        loss = (torch.sum(torch.mul(Variable(torch.FloatTensor(self.policy.policy_history), requires_grad=True).cuda(), Variable(rewards, requires_grad=True).cuda()).mul(-1), -1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.policy.policy_history = []
        self.policy.reward_episode = []

        self.loss_history.append(loss.item())
        self.reward_history.append(self.total_reward)
    
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
        if self.pre_state is None:
            self.pre_state = observation
            return 0

        residual_state = observation - self.pre_state
        residual_state = observation.unsqueeze(0) # size = (1, 1, 187, 160)
        residual_state = residual_state.cuda()
        probs = self.policy(residual_state)
        m = Categorical(probs)
        action = m.sample()
        self.pre_state = observation

        # save logprob
        self.policy.policy_history.append(m.log_prob(action).data.cpu())

        return action.item()

