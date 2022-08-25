


from .price_generator import make_stocks
import gym
import numpy as np

class MinEnv(gym.Env):
    def __init__(self, num_stocks, length, initial_capital, lookback):

        self.num_stocks = num_stocks
        self.length = length
        self.initial_capital = initial_capital
        self.lookback = lookback
        prices = make_stocks(length, num_stocks)
        self.prices = np.concatenate([prices, np.ones([length+1,1])], axis=1)
        self.portfolio = num_stocks*[0] + [1] #portfolio weights
        self.real_portfolio = num_stocks*[0] + [self.initial_capital] #what we actually have
        self.time = 0
        self.account_value = initial_capital
        self.observation_space = gym.spaces.Dict({
            "prices": gym.spaces.Box(0,2000, shape=(num_stocks, lookback)),
            "portfolio" : gym.spaces.Box(0.0, 1.0, shape=(num_stocks+1,1))
        })
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(num_stocks+1,1))

        self.cost_per_share = 0.01
    def _get_obs(self):
        prices = self.prices[self.time - self.lookback: self.time,:-1]
        return {"prices": prices, "portfolio": self.portfolio} 
    def reset(self):
        self.time = self.lookback
        self.portfolio = np.array([0]*self.num_stocks + [1])
        self.real_portfolio = self.num_stocks*[0] + [self.initial_capital]
        self.account_value = self.initial_capital
        return self._get_obs()
    def step(self, action):
        self.time += 1
        action = action.reshape((len(self.portfolio),))
        cur_val = self.account_value
        self._update_account(action)
        new_val = self.accont_value
        reward = new_val - cur_val
        
        next_state = self._get_obs()

        done = self.time >= self.length-1
        return next_state, reward, done, None
    
    def _update_account(self, new_port):
        current_share_value = self.real_portfolio*self.prices[self.time,:]
        current_account_value = np.sum(current_share_value)

        current_port = current_share_value / current_account_value
        cash_change = (new_port - current_port)*current_account_value
        shares_change = np.floor(cash_change / self.prices[self.time,:])

        self.real_portfolio += shares_change
        new_account_value = np.sum(self.real_portfolio * self.prices[self.time,:])
        missing_cash = current_account_value - new_account_value
        transaction_cost = np.sum(np.abs(shares_change[:-1])*self.cost_per_share)
        self.real_portfolio[-1] += missing_cash - transaction_cost

        self.portfolio = np.array(self.real_portfolio/ np.sum(self.real_portfolio))
        self.accont_value = np.sum(self.real_portfolio*self.prices[self.time,:])

