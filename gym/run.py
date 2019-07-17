import gym, roboschool
import configparser
import argparse
import time
import numpy as np
import sys
import os
from agent import load_so_libray

#import custom ContinuousBandit env if founded
try:
    import CASCD
except:
    pass

#for ddrl with perf.data gziped
#find . -type f -name 'perf.data' | sed -r 's|/[^/]+$||' | sort | uniq | sed -e 's/_[0-9]*$//' | sort | uniq | xargs -I P -n 1 bash -c 'LANG=en_US.UTF-8 printf "%08.2f" $(find P_* -name "perf.data" | xargs -I G zcat G | jq -s add/length) ; echo -n "($(find P_* -name "perf.data" | wc -l))" ; echo : P ' |  sort -g -r
#to display cpu time
#find . -type f -name 'time_elapsed' | sed -r 's|/[^/]+$||' | sort | uniq | sed -e 's/_[0-9]*$//' | sort | uniq | xargs -I P -n 1 bash -c 'LANG=en_US.UTF-8 printf "%08.2f" $(find P_* -name "time_elapsed" | xargs -I G cat G | jq -s add/length) ; echo -n "($(find P_* -name "time_elapsed" | wc -l))" ; echo : P ' |  sort -g -r
#print number of trials generated
#find . -name 'x.learning.data' |  xargs -I % tail -1 %
#specific column
#find . -type f -name '0.learning.data' | sed -r 's|/[^/]+$||' | sort | uniq | sed -e 's/_[0-9]*$//' | sort | uniq | xargs -I P -n 1 bash -c 'LANG=en_US.UTF-8 printf "%08.2f" $(find P_* -name "0.learning.data" | xargs -I G zcat G | cut -f13 -d" " | jq -s add/length) ; echo -n "($(find P_* -name "0.learning.data" | wc -l))" ; echo : P ' |  sort -g -r

#for baseline with uncompressed monitor
#find . -type f -name '0.1.monitor.csv' | sed -r 's|/[^/]+$||' | sort | uniq | sed -e 's/_[0-9]*$//' | sort | uniq | xargs -I P -n 1 bash -c 'LANG=en_US.UTF-8 printf "%08.2f" $(find P_* -name "0.1.monitor.csv" | xargs -I G tail -50 G | grep -v r | cut -f1 -d',' | jq -s add/length) ; echo -n "($(find P_* -name "0.1.monitor.csv" | wc -l))" ; echo : P ' |  sort -g -r
#display problematic datas
#find . -name "0.1.monitor.csv" | xargs -I G bash -c "tail -50 G | cut -f1 -d',' | jq -s add/length || echo G"

#for baseline with compressed monitor
#find . -type f -name '0.1.monitor.csv' | sed -r 's|/[^/]+$||' | sort | uniq | sed -e 's/_[0-9]*$//' | sort | uniq | xargs -I P -n 1 bash -c 'LANG=en_US.UTF-8 printf "%08.2f" $(find P_* -name "0.1.monitor.csv" | xargs -I G -n 1 bash -c "zcat G | tail -50 | grep -v r" | cut -f1 -d',' | jq -s add/length) ; echo -n "($(find P_* -name "0.1.monitor.csv" | wc -l))" ; echo : P ' |  sort -g -r

#uncompress monitor for baseline with mix of compress/uncompress monitor
#paste result and rerun it
#find . -name "0.1.monitor.csv" | xargs -I G bash -c "file G | grep gzip > /dev/null && echo mv 'G G.gz ; gunzip G.gz'"
#if ddrl:
#find . -name "x.learning.data" | xargs -I G bash -c "file G | grep gzip > /dev/null && echo mv 'G G.gz ; gunzip G.gz'"

np.seterr(all='raise')

parser = argparse.ArgumentParser()
parser.add_argument('--save-best', action='store_true', default=False)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--view', action='store_true', default=False)
parser.add_argument('--test-only', action='store_true', default=False)
parser.add_argument('--capture', action='store_true', default=False)
parser.add_argument('--goal-based', action='store_true', default=False)
parser.add_argument('--gpu', type=int, default=None) #used in cpp
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--config', type=str, default='config.ini')
clargs = parser.parse_args()
clparams = vars(clargs)
print(clparams)

askrender = clparams['render'] or clparams['view']
if clparams['capture'] and askrender:
    print("can only render or capture (not both at the same time)")
    exit()

config = configparser.ConfigParser()
if not os.path.isfile(clparams['config']) :
    print("cannot find the config.ini file at ", clparams['config'])
    exit()
config.read(clparams['config'])

total_max_steps=int(config['simulation']['total_max_steps'])
testing_trials=int(config['simulation']['testing_trials'])
testing_each=int(config['simulation']['testing_each'])
dump_log_each=int(config['simulation']['dump_log_each'])
display_log_each=int(config['simulation']['display_log_each'])
env_name=config['simulation']['env_name']

gamma=float(config['agent']['gamma'])
iter_by_episode=1

def str2bool(v):
  return v.lower() in ("yes", "true", "1")

def plot_exploratory_actions(observation, ag):
    print(observation)
    p=[np.array(ag.run(0, observation, True, False, False)) for _ in range(5000)]
    p=np.concatenate(p)
    
    print(p)
    import matplotlib.pyplot as plt
    from pylab import figure
    fig = figure()
    ax=fig.add_subplot(111)
    ax.hist(p, 50)
    plt.show()

def run_episode(env, ag, learning, episode):
    env.seed(0)
    observation = env.reset()
    transitions = []
    totalreward = 0
    undiscounted_rewards = 0
    cur_gamma = 1.0
    sample_steps=0
    max_steps=env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    ag.start_ep(observation, learning)
    #plot_exploratory_actions(observation, ag)
    reward=0 #not taken into account
    for step in range(max_steps):
        action = ag.run(reward, observation, learning, False, False)
        #be carreful action is a pointer so it can be changed
        action = np.array(action)
        observation, reward, done, info = env.step(action * action_scale)
        totalreward += cur_gamma * reward
        cur_gamma = gamma * cur_gamma
        undiscounted_rewards += reward
        sample_steps+=1
        if done:
            break

    ag.run(reward, observation, learning, done, True)
    ag.end_ep(learning)

    if episode % dump_log_each == 0:
        ag.dump(learning, episode, sample_steps, undiscounted_rewards)
    if episode % display_log_each == 0:
        ag.display(learning, episode, sample_steps, undiscounted_rewards)

    t=(time.time() - start_time)
    datalogged=str(undiscounted_rewards)+','+str(sample_steps)+','+str(t)
    if clparams['goal_based']:
        datalogged+=','+str(info['is_success'])
    if learning:
        training_monitor.write(datalogged+'\n')
        training_monitor.flush()
    else:
        testing_monitor.write(datalogged+'\n')
        testing_monitor.flush()

    return totalreward, transitions, undiscounted_rewards, sample_steps

def run_episode_displ(env, ag):
    observation = env.reset()
    if clparams['capture']: 
        envm = gym.wrappers.Monitor(env, 'test_monit', video_callable=lambda x: True, force=True)
        observation = envm.reset()
    undiscounted_rewards = 0
    max_steps=env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    ag.start_ep(observation, False)
    reward=0 #not taken into account
    for step in range(max_steps):
        action = ag.run(reward, observation, False, False, False)
        #be carreful action is a pointer so it can be changed
        action = np.array(action)
        if clparams['capture']:
            observation, reward, done, info = envm.step(action * action_scale)
        else:#render
            observation, reward, done, info = env.step(action * action_scale)
            env.render()
        undiscounted_rewards += reward
        if done:
            break
        
    print(undiscounted_rewards)

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

if (clparams['goal_based']) and not isinstance(env.observation_space, gym.spaces.Dict):
    print("goal_based algorithms only works with goal based goal-oriented environment")
    exit(1)

#for goal-oriented environment (https://openai.com/blog/ingredients-for-robotics-research/)
if isinstance(env.observation_space, gym.spaces.Dict):
    goal_size=env.observation_space.spaces.get('desired_goal').shape[0]
    print("Goal space:", env.observation_space.spaces.get('desired_goal'))
    print("Goal observed space:", env.observation_space.spaces.get('achieved_goal'))
    print("Observation space:", env.observation_space.spaces.get('observation'))
    #the following might be false for some env
    goal_achieved_start=0
    goal_start=goal_achieved_start+goal_size
    keys = env.observation_space.spaces.keys()
    env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

observation = env.reset()
nb_sensors = env.observation_space.shape[0]

print("State space:", env.observation_space)
print("- low:", env.observation_space.low)
print("- high:", env.observation_space.high)

print("Action space:", env.action_space)
print("- low:", env.action_space.low)
print("- high:", env.action_space.high)

for i in range(env.action_space.shape[0]):
    assert env.action_space.low[i] == - env.action_space.high[i]
action_scale=env.action_space.high

print("Create agent with (nb_motors, nb_sensors) : ", env.action_space.shape[0], nb_sensors)

if isinstance(nb_sensors, np.generic):
    nb_sensors=np.asscalar(nb_sensors)
DDRLAg=load_so_libray(config)

if not (clparams['goal_based']) or "libddrl-hpenfac" not in config.get('simulation', 'library'):
    ag = DDRLAg(env.action_space.shape[0], nb_sensors, sys.argv)
else:
    ag = DDRLAg(env.action_space.shape[0], nb_sensors, goal_size, goal_start, goal_achieved_start, sys.argv)
print("main algo : " + ag.name())

if clparams['load'] is not None:
    ag.load(int(clparams['load']))

if not clparams['test_only']:
    #classic learning run
    start_time = time.time()

    results=[]
    sample_steps_counter=0
    episode=0
    
    #comptatibility with openai-baseline logging
    training_monitor = open('0.0.monitor.csv','w')
    testing_monitor = open('0.1.monitor.csv','w')
    xlearning_monitor = open('x.learning.data','w')
    training_monitor.write('# { "t_start": '+str(start_time)+', "env_id": "'+env_name+'"} \n')
    testing_monitor.write('# { "t_start": '+str(start_time)+', "env_id": "'+env_name+'"} \n')
    if not clparams['goal_based']:
        training_monitor.write('r,l,t\n')
        testing_monitor.write('r,l,t\n')
    else:
        training_monitor.write('r,l,t,g\n')
        testing_monitor.write('r,l,t,g\n')
    
    max_steps=env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    bestTestingScore=float("-inf")
    while sample_steps_counter < total_max_steps + testing_each * max_steps:
        if episode % display_log_each == 0:
            print('episode', episode, 'total steps', sample_steps_counter, 'last perf', results[-1] if len(results) > 0 else 0)
        
        sample_step = train(env, ag, episode)
        sample_steps_counter += sample_step
    
        if episode % testing_each == 0 and episode != 0:
            xlearning_monitor.write(str(sample_steps_counter)+'\n')
            xlearning_monitor.flush()
            reward, testing_sample_step = testing(env, ag, episode)
            results.append(reward)
            if clparams['save_best'] and reward >= bestTestingScore:
                bestTestingScore=reward
                ag.save(episode)
        episode+=1
        
    #write logs
    results=np.array(results)
    lastPerf = results[int(results.shape[0]*0.9):results.shape[0]-1]
    np.savetxt('y.testing.data', results)
    np.savetxt('perf.data',  [np.mean(lastPerf)-np.std(lastPerf)])
    training_monitor.close()
    testing_monitor.close()
    xlearning_monitor.close()
    
    #comptatibility with lhpo
    elapsed_time = (time.time() - start_time)/60.
    with open('time_elapsed', 'w') as f:
        f.write('%d\n' % elapsed_time)

elif not clparams['capture'] and not askrender:
    #testing without log
    reward, testing_sample_step = testing(env, ag, episode)
    print(reward)
else:
    #testing with display
    run_episode_displ(env, ag)


