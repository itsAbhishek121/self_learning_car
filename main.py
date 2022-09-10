import gym
import numpy as np
env = gym.make("MountainCar-v0")
env.reset()

learning_rate = 0.1
Discount = 0.95
Episodes = 2500
temp = None
show = 50

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
DISCRETE_OS_SIZE = np.ndarray(DISCRETE_OS_SIZE,dtype=object)


discrete_os_win_size = (env.observation_space.high - env.observation_space.low) /DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state-env.observation_space.low)/ discrete_os_win_size
    return tuple(discrete_state.astype(np.int))

for ep in range(Episodes):
    if ep % show ==0:
        render = True
        print(ep)

    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    done = False

    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)

        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1-learning_rate)*current_q + learning_rate*(reward+ Discount*max_future_q)
            q_table[discrete_state+(action, )] = new_q

        elif new_state[0] >= env.goal_position:
            q_table[discrete_state+(action, )] = 0
            print(" we made it : ", ep)

        discrete_state = new_discrete_state



env.close()
