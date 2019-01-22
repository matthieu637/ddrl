import gym, roboschool
from agent import *
import configparser
import time
import numpy as np


def str2bool(v):
  return v.lower() in ("yes", "true", "1")

config = configparser.ConfigParser()
config.read('config.ini')

total_max_steps=int(config['simulation']['total_max_steps'])
testing_trials=int(config['simulation']['testing_trials'])
testing_each=int(config['simulation']['testing_each'])
dump_log_each=int(config['simulation']['dump_log_each'])
display_log_each=int(config['simulation']['display_log_each'])
env_name=config['simulation']['env_name']

gamma=float(config['agent']['gamma'])
iter_by_episode=1

def run_episode(env, ag, learning, episode):
    observation = env.reset()
    transitions = []
    totalreward = 0
    undiscounted_rewards = 0
    cur_gamma = 1.0
    sample_steps=0
    max_steps=env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    ag.start_ep(observation, learning)
    reward=0 #not taken into account
    for step in range(max_steps):
        action = ag.run(reward, observation, learning, False, False)
        observation, reward, done, info = env.step(action)
#            env.render()
        totalreward += cur_gamma * reward
        cur_gamma = gamma * cur_gamma
        undiscounted_rewards += reward
        sample_steps+=1
        if done:
            break
    
    ag.run(reward, observation, learning, done, True)
    ag.end_ep(learning)
    
    if episode % dump_log_each == 0:
        ag.dump(learning, episode, sample_steps, totalreward)
    if episode % display_log_each == 0:
        ag.display(learning, episode, sample_steps, totalreward)
    
    return totalreward, transitions, undiscounted_rewards, sample_steps

def train(env, ag, episode):
    sample_steps=0
    for _ in range(iter_by_episode):
        _, tr, _, _sample_steps = run_episode(env, ag, True, episode)
        sample_steps += _sample_steps

    return sample_steps

def testing(env, ag, episode):
    undiscounted_reward=0.
    sample_steps=0
    for _ in range(testing_trials):
        _, _, reward, sample_step, = run_episode(env, ag, False, episode)
        undiscounted_reward += reward
        sample_steps += sample_step
    undiscounted_reward /= float(testing_trials)
    sample_steps /= float(testing_trials)

    return undiscounted_reward, sample_steps


env = gym.make(env_name)
observation = env.reset()
nb_sensors = env.observation_space.shape[0]

ag = NFACAg(env.action_space.shape[0], nb_sensors)

start_time = time.time()

sample_steps_counter=0
results=[]
testing_sample_steps=[]
testing_action_sum=[]
sample_steps=[]
sample_steps_counter=0
weights=[]
weightsQ=[]
weightsV=[]
used_samples=[]
local_training_info=[]
episode=0

max_steps=env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

while sample_steps_counter < total_max_steps + testing_each * max_steps:
    if episode % display_log_each == 0:
        print('episode', episode, 'total steps', sample_steps_counter, 'last perf', results[-1] if len(results) > 0 else 0)
    
    sample_step = train(env, ag, episode)
    sample_steps_counter += sample_step

    if episode % testing_each == 0:
        reward, testing_sample_step = testing(env, ag, episode)
        results.append(reward)
        testing_sample_steps.append(testing_sample_step)
        sample_steps.append(sample_steps_counter)
    episode+=1

elapsed_time = (time.time() - start_time)/60.
with open('time_elapsed', 'w') as f:
    f.write('%d\n' % elapsed_time)

