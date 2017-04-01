#!/usr/bin/python

import gym
import numpy
import time
import sys



env=gym.make('Humanoid-v1')
env.reset()
env.render()

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        print(self.action_space)

    def act(self, observation, reward, done):
        ac=[0 for i in range(17)];
        ac=numpy.array(ac);
        
        #ac[5+4]=500
        ac[int(sys.argv[1])] = 500*int(sys.argv[2])
        return ac;
        #return self.action_space.sample()

agent = RandomAgent(env.action_space)

episode_count=100
reward=0
done=False

for i in range(episode_count):
    ob = env.reset()
    for j in range(40+40):
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.1)
        #if done:
                #break
        
