from StationaryBandit import *
from UnstationaryBandit import *

cfg = {"example" : '2.5'}


def main(example_name):
    if example_name == "2.5":
        stationaryagent = StationaryAgent(eps=0.1, target_eps=0.1)
        unstationaryagent = UnstationaryAgent(eps=0.1, target_eps=0.1)
        print("Stationary State : {}".format(stationaryagent))
        print("Unstationary State : {}".format(unstationaryagent))

        iteration = 10000

        for i in range(iteration):
            stationaryagent.action()

            unstationaryagent.action()

            if (i + 1) % 100 == 0:
                print("<Stationary> \n 현재 play time : {}, Action에 따른 value dict : {}, Q STAR : {}\n".format(stationaryagent.play_time,
                                                                                          stationaryagent.value_dict,
                                                                                          stationaryagent.Env.q_star))

                print("<Unstationary> \n 현재 play time : {}, Action에 따른 value dict : {}, Q STAR : {}\n".format(
                    unstationaryagent.play_time,
                    unstationaryagent.value_dict,
                    unstationaryagent.Env.q_star))

    plt.plot(stationaryagent.vals, label ="time factor")
    plt.plot(unstationaryagent.vals, label ="fix factor")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main(cfg['example'])