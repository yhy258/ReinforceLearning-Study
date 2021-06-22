import numpy as np
import random
import matplotlib.pyplot as plt
from bandit_frame import Environment, Agent




class StationaryEnviron(Environment):
    def __init__(self):
        # a : 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
        self.q_star = np.random.normal(loc = 0, scale = 1.0, size=10)


    def play(self, a): # action : a -> {1,2,3,4,5,6,7,8,9.10}
        idx = a-1
        mean = self.q_star[idx]
        reward = np.random.normal(loc = mean, scale = 0.01)
        self.q_star[idx] += np.random.normal(loc=0, scale=0.01)
        return reward




class StationaryAgent(Agent):
    def __init__(self, eps, target_eps, eps_discount = 0.001):

        super().__init__(eps, target_eps, eps_discount)
        self.Env = StationaryEnviron()
        self.action_name = [1,2,3,4,5,6,7,8,9,10]
        self.action_init()

    def action(self):
        self.play_time += 1
        if self.target_eps != self.eps:
            self.eps = max(self.target_eps, self.eps - self.eps_discount * (self.eps - self.target_eps))
        rand = random.random()
        if rand > self.eps:
            values = self.get_values()
            max_value_idx = np.argmax(values)
            act = max_value_idx + 1
            r = self.Env.play(act)
        else :
            act = random.choice(self.action_name)
            r = self.Env.play(act)
        self.update_val(act, r)

    def action_init(self):
        for name in self.action_name:
            self.value_dict[name] = 0

    def update_val(self, a, r):
        self.value_dict[a] += (r - self.value_dict[a])/self.play_time
        self.last_val += (r - self.value_dict[a])/self.play_time
        self.vals.append(self.last_val)

    def get_values(self):
        values = []
        for name in self.action_name:
            values.append(self.value_dict[name])
        return values

    def __repr__(self):
        return "Action에 따른 value Dictionary : {}, Action Name : {}, 각 Action에 대한 Reward의 mean. (Gaussian Distribution) : {}".format(self.value_dict, self.action_name,self.Env.q_star)

    def __len__(self):
        return len(self.action_name)

if __name__ == '__main__' :
    agent = StationaryAgent(eps = 0.3, target_eps = 0.00)
    print(agent)

    iteration = 1000

    for i in range(iteration):
        agent.action()
        if (i+1)%100 == 0 :
            print("현재 play time : {}, Action에 따른 value dict : {}, Q STAR : {}".format(agent.play_time,agent.value_dict, agent.Env.q_star))

    plt.plot(agent.vals)
    plt.show()