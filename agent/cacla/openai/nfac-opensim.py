from osim.env import RunEnv
from agent import OffNFACAg
import ConfigParser

config = ConfigParser.ConfigParser()
config.readfp(open('config.ini'))

max_episode=int(config.get('simulation', 'max_episode'))
learning=True

env = RunEnv(visualize=False)
observation = env.reset(difficulty = 0)
nb_sensors = env.observation_space.shape[0]
ag = OffNFACAg(env.action_space.shape[0], nb_sensors);
for ep in range(max_episode):
    #env stoch but testing only on one episode
    if ep % 100 == 0 and ep > 0:
        learning=False
    else:
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
        if done:
                env.reset()
        	break

    ag.end_ep(learning)
    ag.dumpdisplay(learning, ep, step)
    

