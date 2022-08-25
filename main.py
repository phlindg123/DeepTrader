""" 
Algorithm
Randomly initiate critic network Q(s, a|theta_Q) and actor u(s | theta_u) with weights theta_Q, theta_u
Initialize target netowrks Q' and u' with weights theta_Q' <- theta_Q and theta_u' <- theta_u
Initialzie replay buffer R
for episode = 1 to M do:
    initialize random process N for action exporation
    Receive initial observation state s_1
    for t=1 to T do:
        Select action a_t = u(s_t | theta_u) + N_t according to current policy and exploration noice
        Execute action a_t and observe reward r_t and observe new state s_t
        store transition (s_t, a_t, r_t, s_(t+1)) in R
        Sample a random minibatch of K transitions (s_i, a_i, r_i, s_(i+1)) from R
        set y_i = r_i + gamma*Q'(s_(i+1), u'(s_i+1 | theta_u') | theta_Q')
        Update critic by minimizing the loss L = (1/K)sum((y_i - Q(s_i, a_i | theta_Q)**2)
        Update the actor policy using the sampled gradient
            ...
        Update target networks
            theta_Q' = tau*theta_Q + (1-tau)*theta_Q'
            theta_u' = tao*theta_u + (1-tao)*theta_u'
    end for
end for     
LÄNKAR:
    https://github.com/yanpanlau/DDPG-Keras-Torcs/blob/master/ddpg.py
    https://github.com/stevenpjg/RDPG/blob/master/critic_net.py
    https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter10-policy/policygradient-car-10.1.1.py
    https://github.com/openai/gym/wiki/Leaderboard
"""
from Agents import DeepTraderAgent, GaussianAgent
from Envs import PortfolioEnv, MinEnv, Simple
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


csv_dir = "C:/Users/Phili/Desktop/fond/data/"
eric = pd.read_csv(csv_dir + "ERIC.csv", header =0, sep=";", parse_dates = True, index_col = 0,
names= ["datetime", "bid","ask","open", "high", "low", "close", "avg_price", "volume", "turnover", "trades"])
sand = pd.read_csv(csv_dir + "SAND.csv", header =0, sep=";", parse_dates = True, index_col = 0,
names= ["datetime", "bid","ask","open", "high", "low", "close", "avg_price", "volume", "turnover", "trades"])
volv = pd.read_csv(csv_dir + "VOLV.csv", header =0, sep=";", parse_dates = True, index_col = 0,
names= ["datetime", "bid","ask","open", "high", "low", "close", "avg_price", "volume", "turnover", "trades"])
hm = pd.read_csv(csv_dir + "HM.csv", header =0, sep=";", parse_dates = True, index_col = 0,
names= ["datetime", "bid","ask","open", "high", "low", "close", "avg_price", "volume", "turnover", "trades"])


data = pd.DataFrame({"sand": sand["close"], "eric": eric["close"], "hm": hm["close"], "volv":volv["close"]}, index=sand.index).T
data = data[data.index > "2018-12-31"]

def sharpe(info):
    rets = [x["cur_ret"] for x in info]
    return np.mean(rets)/np.std(rets)

def run():
    batch_size = 128
    episodes = 500
    #env = gym.make("MountainCarContinuous-v0")
    #env = gym.make("Pendulum-v0")
    #env = gym.make("LunarLanderContinuous-v2")
    #env = PortfolioEnv(data, 10)
    env = Simple(100)
    state_size = env.observation_space.shape
    action_dim = env.action_space.shape
    agent = DeepTraderAgent(state_size, action_dim, env, "OU")
    print(action_dim)
    print(state_size)
    
    for episode in range(episodes):
        state = env.reset()
        agent.reset()
        done = False
        tot_reward = 0
        while not done:
            state = state.reshape((1,state_size[0]))
            action = agent.act(state, np.float(episode/episodes))
            #print(state)
            next_state, reward, done, info = env.step(action)
            tot_reward += reward
            next_state = next_state.reshape((1,state_size[0]))
            #env.render()
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > batch_size*3:
                agent.replay(batch_size)
        print("Episode {}/{}, Total reward: {}, Loss: {}".format(episode+1, episodes, tot_reward, agent.loss))
        #print("Final Weights: {}, Total return: {}".format(state, info["cur_val"]))
        #print("Sharpe: {}".format(sharpe(env.infos)))

def main():
    np.random.seed(12)
    env = MinEnv(5, 100, 1000, 10)
    state_size = env.observation_space.spaces["prices"].shape
    action_size = env.action_space.shape
    print(state_size)
    print(action_size)
    agent = GaussianAgent(state_size, action_size, env)
    agent.run(20)
    plt.plot(agent.losses)
    plt.show()
run()
    



