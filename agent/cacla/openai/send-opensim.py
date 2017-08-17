from osim.env import RunEnv
from agent import *
import ConfigParser
from osim.http.client import Client

def str2bool(v):
  return v.lower() in ("yes", "true", "1")

remote_base = 'http://grader.crowdai.org:1729'
client = Client(remote_base)

config = ConfigParser.ConfigParser()
config.readfp(open('config.ini'))

learning=False

env = RunEnv(visualize=False)
observation = client.env_create('2060a86df422ef5a67dea16c5320c8ad')
nb_sensors = env.observation_space.shape[0]
if config.get('simulation', 'agent_type') == 'cacla':
    ag = CaclaAg(env.action_space.shape[0], nb_sensors);
else:
    ag = OffNFACAg(env.action_space.shape[0], nb_sensors);
ag.load(int(config.get('simulation', 'load_episode')))
stop=False
while not stop:
    #env stoch but testing only on one episode
    ag.start_ep(observation, learning)
    ac = ag.run(0, observation, learning, False, False)
    total_reward = 0.0
    step = 0
    while True:
        #-1 1 to 0 1
        ac=(ac+1.)/2.;
        observation, reward, done, info = client.env_step(ac.tolist())
        ac = ag.run(reward, observation, learning, done, False)
        total_reward += reward
        step += 1
#debug only
#        if done or step >= 10:
        if done:
            observation = client.env_reset()
	    if not observation:
	        stop=True
            break

    ag.dumpdisplay(learning, 0, step, total_reward)

client.submit()


