
# coding: utf-8

# In[ ]:


import numpy as np
from math import exp 

class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """
    
        self.epsilon = 1 
        self.epsilon_bis=0.02
        self.position_x=170
        self.vitesse_vx=50
        self.obs = (-120, -1)  
        self.activ=-1
        self.Reward=1
        self.poids = np.random.rand(self.position_x*self.vitesse_vx, 3)
        self.count=0
       
        

        minus_dmin=-170
        self.s=np.array([[minus_dmin-(i*minus_dmin/(self.position_x-1.0)) , -20.0+(j*40.0/(self.vitesse_vx-1.0))] for i in range(self.position_x) for j in range(self.vitesse_vx)])

        
    def reset(self, x_range):
        """Reset the state of the agent for the start of new game.
        Parameters of the environment do not change when starting a new 
        episode of the same game, but your initial location is randomized.
        x_range = [xmin, xmax] contains the range of possible values for x
        range for vx is always [-20, 20]
        """
        self.epsilon = (self.epsilon * 0.94)
        self.count += 1
        pass

    def getQ(self, S_t,Phi):
        return np.dot(Phi, self.poids)

    def getPhi(self,S_t):
        return np.exp(-((S_t[0] - self.s[:, 0]) ** 2)) * np.exp(-(S_t[1] - self.s[:, 1]) ** 2)
    
    def act(self, observation):
        """Acts given an observation of the environment.
        Takes as argument an observation of the current state, and
        returns the chosen action.
        observation = (x, vx)
        """
        Phi_s=self.getPhi(observation)
        Q_p=self.getQ(observation, Phi_s)
        self.epsilon = max(self.epsilon_bis, self.epsilon)
        
        if np.random.random()<self.epsilon :
            return np.random.choice([-1, 0, 1])
        else:
            return np.argmax(Q_p) - 1


    def reward(self, observation, action, reward):
        """Receive a reward for performing given action on
        given observation. This is where your agent can learn.
        """
        observation=self.obs
        action=self.activ
        reward=self.Reward
        pass

Agent = RandomAgent

