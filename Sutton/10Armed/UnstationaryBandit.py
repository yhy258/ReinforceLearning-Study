import numpy as np
import random
from bandit_frame import Environment, Agent

import matplotlib.pyplot as plt

class UnstationaryEnviron(Environment):
    def __init__(self):
        self.q_star = np.random.normal(loc =0, scale = 1, size = 10)
        self.opt_action = np.argmax(self.q_star) + 1
    def play(self,a):
        mean = self.q_star[a-1]
        reward = np.random.normal(loc = mean, scale= 0.01)

        self.q_star += np.random.randn(10) * 0.01
        return reward

    def opt_action(self):
        idx = np.argmax(self.q_star)
        return idx + 1

class UnstationaryAgent(Agent):
    def __init__(self, eps, target_eps, fix_discount_factor = 0.1, eps_discount = 0.001, initial_value=0.):
        super().__init__(eps, target_eps, eps_discount)
        self.Env = UnstationaryEnviron()
        self.action_name = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.initial_value = initial_value
        self.fix_discount_factor = fix_discount_factor

        self.action_init()
        self.actnum = [0] * 10


    def action(self):
        self.play_time += 1
        if self.target_eps != self.eps :
            self.eps = max(self.target_eps, self.eps - self.eps_discount * (self.eps - self.target_eps))
        rand = random.random()
        if rand > self.eps:
            values = self.get_values()

            max_value = max(values)

            indices = [i for i, value in enumerate(values) if value == max_value]
            max_value_idx=random.choice(indices)
            act = max_value_idx + 1
            r = self.Env.play(act)
        else :
            act = random.choice(self.action_name)
            r = self.Env.play(act)

        self.rewards.append(r)
        self.opt_acts.append(1 if act == self.Env.opt_action else 0)
        self.update_val( act, r)

    def update_val(self, a, r):
        self.actnum[a-1] +=1

        if self.fix_discount_factor == None :
            self.value_dict[a] += (r - self.value_dict[a])/ self.actnum[a-1]
        else :
            self.value_dict[a] += (r - self.value_dict[a])*self.fix_discount_factor




    def action_init(self):
        for name in self.action_name:
            self.value_dict[name] = self.initial_value

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

    agent = UnstationaryAgent(eps = 0.1, target_eps = 0.1)
    print(agent)

    iteration = 1000

    for i in range(iteration):
        agent.action()
        if (i+1)%100 == 0 :
            print("현재 play time : {}, Action에 따른 value dict : {}, Q STAR : {}".format(agent.play_time,agent.value_dict, agent.Env.q_star))

    plt.plot(agent.vals)
    plt.show()