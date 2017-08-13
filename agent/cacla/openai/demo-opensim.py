from osim.env import RunEnv
from agent import *
import ConfigParser
import time

def str2bool(v):
  return v.lower() in ("yes", "true", "1")

config = ConfigParser.ConfigParser()
config.readfp(open('config.ini'))

max_episode=int(config.get('simulation', 'max_episode'))
testing_only=str2bool(config.get('simulation', 'testing_only'))
learning=(not testing_only)

env = RunEnv(visualize=False)
observation = env.reset(difficulty = 2)
nb_sensors = env.observation_space.shape[0]
if config.get('simulation', 'agent_type') == 'cacla':
    ag = CaclaAg(env.action_space.shape[0], nb_sensors);
else:
    ag = OffNFACAg(env.action_space.shape[0], nb_sensors);
if testing_only:
    ag.load(int(config.get('simulation', 'load_episode')))
start_time = time.time()
for ep in range(max_episode):
    #env stoch but testing only on one episode
    if ep % 100 == 0 and ep > 0:
        learning=False
    elif not testing_only:
        learning=True
    ag.start_ep(env.current_state, learning)
    ac = ag.run(0, env.current_state, learning, False, False)
    total_reward = 0.0
    step = 0
    while True:
        #-1 1 to 0 1
        ac=(ac+1.)/2.;
        observation, reward, done, info = env.step(ac)
        ac = ag.run(reward, observation, learning, done, False)
        total_reward += reward
        step += 1
#debug only
        if done or step >= 10:
#        if done:
            env.reset()
            break

    if not learning and not testing_only:
        ag.save(ep)
    ag.end_ep(learning)
    ag.dumpdisplay(learning, ep, step, total_reward)

if not testing_only:
    ag.save(max_episode)

elapsed_time = (time.time() - start_time)/60.
with open('time_elapsed', 'w') as f:
    f.write('%d\n' % elapsed_time)

