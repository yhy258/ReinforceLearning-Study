import numpy as np
class Environment:
    def __init__(self):
        pass
    def play(self, a):
        pass

class Agent:
    def __init__(self, eps, target_eps, eps_discount = 0.001):
        self.play_time = 0
        self.eps = eps
        self.target_eps = target_eps
        self.eps_discount = eps_discount
        self.value_dict = {} # action을 어떻게 할지 보기 위함.
        self.Env = None
        self.opt_acts = []

        self.rewards = []

    def action_init(self):
        pass

    def update_val(self, a, r):
        pass

    def get_values(self):
        pass


    def reset(self):
        self.rewards = []
        self.value_dict = {}
        self.Env.q_star = np.random.normal(loc =0, scale = 1, size = 10)
        self.opt_acts = []
        self.action_init()

