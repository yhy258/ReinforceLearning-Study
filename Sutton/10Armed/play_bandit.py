from StationaryBandit import *
from UnstationaryBandit import *
import pickle
cfg = {"example" : '2.5'}


def main(example_name):
    if example_name == "2.5":
        stationaryagent = StationaryAgent(eps=0.1, target_eps=0.1)
        unstationaryagent = UnstationaryAgent(eps=0.1, target_eps=0.1)
        print("Stationary State : {}".format(stationaryagent))
        print("Unstationary State : {}".format(unstationaryagent))

        iteration = 2000
        plays = 10000
        st_reward_lists = []
        unst_reward_lists = []

        st_opt_action = []
        unst_opt_action = []
        for i in range(iteration):

            stationaryagent.reset()
            unstationaryagent.reset()
            print("==================\nNow Iteration : {}/{}\n==================".format(i+1, iteration))
            for play in range(plays):
                stationaryagent.action()
                unstationaryagent.action()


                if (play + 1) % 5000 == 0:
                    print("<Stationary> \n 현재 play time : {}, Action에 따른 value dict : {}, Q STAR : {}\n".format(stationaryagent.play_time,
                                                                                              stationaryagent.value_dict,
                                                                                              stationaryagent.Env.q_star))

                    print("<Unstationary> \n 현재 play time : {}, Action에 따른 value dict : {}, Q STAR : {}\n\n".format(
                        unstationaryagent.play_time,
                        unstationaryagent.value_dict,
                        unstationaryagent.Env.q_star))
            st_reward_lists.append(stationaryagent.rewards)
            unst_reward_lists.append(unstationaryagent.rewards)

            st_opt_action.append(stationaryagent.opt_acts)
            unst_opt_action.append(unstationaryagent.opt_acts)


    st_reward_lists = np.mean(st_reward_lists, axis =0)
    unst_reward_lists = np.mean(unst_reward_lists, axis=0)



    st_opt_action= np.mean(st_opt_action, axis = 0)
    unst_opt_action=np.mean(unst_opt_action, axis = 0)

    plt.plot(st_reward_lists, label ="time factor")
    plt.plot(unst_reward_lists, label ="fix factor")
    plt.legend()
    plt.show()

    plt.plot(st_opt_action, label="time factor")
    plt.plot(unst_opt_action, label="fix factor")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main(cfg['example'])