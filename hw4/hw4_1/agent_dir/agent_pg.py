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

episodes = 6000
checkpoint = "checkpoints"
trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize([80, 80]),
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
    image = image[34:,:,:]
    image = trans(image)
    return image

class AgentModel(nn.Module):
    def __init__(self):
        super(AgentModel, self).__init__()

        self.linear1 = nn.Linear(80*80, 256)
        self.linear2 = nn.Linear(256, 3)

        self.policy_history = []
        self.reward_episode = []
        self.discount = 0.99

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        x = F.softmax(x, dim=1)
        return x

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        self.env = env
        self.env.seed(101011387)
        self.policy = AgentModel().cuda()

        self.loss_history = []
        self.reward_history = []

        summary(self.policy, (1, 80*80))

        if args.test_pg:
            #you can load your model here
            print('loading trained model')

        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=5e-3)

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
            self.play(episode)# ; sys.exit()
            if (episode+1) % 10 == 0: self.update()

            if  (episode+1) % 50 == 0:
                print ("Save checkpoint")
                torch.save(self.policy.state_dict(), os.path.join(checkpoint, "_{}.pt".format(episode+1)))

        loss = np.array(self.loss_history)
        reward = np.array(self.reward_history)
        np.save('loss.npy', loss)
        np.save('reward.npy', reward)

    def play(self, episode):
        state = self.env.reset()
        self.total_reward = 0
        self.pre_state = 0
        while(True):
            state = RGB2Gray(state)
            action = self.make_action(state, test=False)

            state, reward, done, _ = self.env.step(action)
            self.policy.reward_episode.append(reward)
            self.total_reward = self.total_reward + reward

            if done:
                print ("Episode {} done! Reward: {:3f}".format(episode+1, self.total_reward), " ")
                break

    def update(self):

        R = 0
        rewards = []
        loss = 0
        reward_episode = np.array(self.policy.reward_episode)
        reward_episode = reward_episode[::-1]

        for i in reward_episode:
            R = i + self.policy.discount * R
            rewards.insert(0, R)

        rewards = torch.FloatTensor(rewards).cuda()
        rewards = (rewards - rewards.mean(-1)) / (rewards.std(-1))
        
        logp = torch.stack(self.policy.policy_history)
        loss = - torch.sum(torch.mul(logp, rewards), -1) / 10

        # print (loss.requires_grad)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print ("loss: {:4f}".format(loss.item()))

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
        residual_state = observation - self.pre_state
        # residual_state = observation.unsqueeze(0) # size = (1, 1, 187, 160)
        residual_state = residual_state.cuda()
        # observation = observation.view(1, 80*80).cuda()
        probs = self.policy(residual_state.view(1, 80*80))[0]
        m = Categorical(probs)
        action = m.sample()
        self.pre_state = observation

        act = [0, 2, 3]

        # print (probs)

        # save logprob
        self.policy.policy_history.append(m.log_prob(action))

        return act[action.item()]

