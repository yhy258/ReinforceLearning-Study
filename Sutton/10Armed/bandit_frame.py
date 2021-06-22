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

        self.last_val = 0
        self.vals = [] # plot 용.

    def action_init(self):
        pass

    def update_val(self, a, r):
        pass

    def get_values(self):
        pass


