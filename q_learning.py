import argparse
import numpy as np
import matplotlib.pyplot as plt

from environment import MountainCar, GridWorld

"""
Please read: THE ENVIRONMENT INTERFACE

In this homework, we provide the environment (either MountainCar or GridWorld) 
to you. The environment returns states, represented as 1D numpy arrays, rewards, 
and a Boolean flag indicating whether the episode has terminated. The environment 
accepts actions, represented as integers.

The only file you need to modify/read is this one. We describe the environment 
interface below.

class Environment: # either MountainCar or GridWorld

    def __init__(self, mode, debug=False):
        Initialize the environment with the mode, which can be either "raw" 
        (for the raw state representation) or "tile" (for the tiled state 
        representation). The raw state representation contains the position and 
        velocity; the tile representation contains zeroes for the non-active 
        tile indices and ones for the active indices. GridWorld must be used in 
        tile mode. The debug flag will log additional information for you; 
        make sure that this is turned off when you submit to the autograder.

        self.state_space = an integer representing the size of the state vector
        self.action_space = an integer representing the range for the valid actions

        You should make use of env.state_space and env.action_space when creating 
        your weight matrix.

    def reset(self):
        Resets the environment to initial conditions. Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the initial state.
    
    def step(self, action):
        Updates itself based on the action taken. The action parameter is an 
        integer in the range [0, 1, ..., self.action_space). Returns:

            (1) state : A numpy array of size self.state_space, representing 
                        the new state that the agent is in after taking its 
                        specified action.
            
            (2) reward : A float indicating the reward received at this step.

            (3) done : A Boolean flag indicating whether the episode has 
                        terminated; if this is True, you should reset the 
                        environment and move on to the next episode.
    
    def render(self, mode="human"):
        Renders the environment at the current step. Only supported for MountainCar.


For example, for the GridWorld environment, you could do:

    env = GridWorld(mode="tile")

Then, you can initialize your weight matrix to all zeroes with shape 
(env.action_space, env.state_space+1) (if you choose to fold the bias term in). 
Note that the states returned by the environment do *not* have the bias term 
folded in.
"""

def parse_args() -> tuple:
    """
    Parses all args and returns them. Returns:

        (1) env_type : A string, either "mc" or "gw" indicating the type of 
                    environment you should use
        (2) mode : A string, either "raw" or "tile"
        (3) weight_out : The output path of the file containing your weights
        (4) returns_out : The output path of the file containing your returns
        (5) episodes : An integer indicating the number of episodes to train for
        (6) max_iterations : An integer representing the max number of iterations 
                    your agent should run in each episode
        (7) epsilon : A float representing the epsilon parameter for 
                    epsilon-greedy action selection
        (8) gamma : A float representing the discount factor gamma
        (9) lr : A float representing the learning rate
    
    Usage:
        env_type, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, lr = parse_args()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, choices=["mc", "gw"])
    parser.add_argument("mode", type=str, choices=["raw", "tile"])
    parser.add_argument("weight_out", type=str)
    parser.add_argument("returns_out", type=str)
    parser.add_argument("episodes", type=int)
    parser.add_argument("max_iterations", type=int)
    parser.add_argument("epsilon", type=float)
    parser.add_argument("gamma", type=float)
    parser.add_argument("learning_rate", type=float)

    args = parser.parse_args()

    return args.env, args.mode, args.weight_out, args.returns_out, args.episodes, args.max_iterations, args.epsilon, args.gamma, args.learning_rate


if __name__ == "__main__":

    env_type, mode, weight_out, returns_out, episodes, max_iterations, epsilon, gamma, lr = parse_args()

    if env_type == "mc":
        env = MountainCar(mode)  # Replace me!
    elif env_type == "gw":
        env = GridWorld(mode)  # Replace me!
    else:
        raise Exception(f"Invalid environment type {env_type}")
    r = np.zeros(episodes)
    w = np.zeros((env.action_space, env.state_space + 1))
    r_mean = []
    for episode in range(episodes):
        # Get the initial state by calling env.reset()
        state = env.reset()
        state = np.append(1, state)
        for iteration in range(max_iterations):
            # Select an action based on the state via the epsilon-greedy strategy
            cur_q = np.matmul(w, state)
            if np.random.uniform() > epsilon:
                a = np.argmax(cur_q)
            else:
                a = np.random.randint(0, env.action_space)

            # Take a step in the environment with this action, and get the 
            # returned next state, reward, and done flag
            next_state, reward, done = env.step(a)

            next_state = np.append(1, next_state)
            # Using the original state, the action, the next state, and 
            # the reward, update the parameters. Don't forget to update the 
            # bias term!
            next_q = np.matmul(w, next_state)

            w[a] = w[a] - lr * (cur_q[a] - (reward + gamma * np.max(next_q))) * state
            r[episode] += reward
            state = next_state
            # Remember to break out of this inner loop if the environment signals done!
            if done:
                print(r[episode])
                print(reward)
                break
        if (episode + 1) % 25 == 0:
            r_mean.append(np.mean(r[0:episode + 1]))
        else:
            r_mean.append(None)

    # Save your weights and returns. The reference solution uses 
    # np.savetxt(..., fmt="%.18e", delimiter=" ")
    np.savetxt(weight_out, w, fmt="%.18e", delimiter=" ")
    np.savetxt(returns_out, r, fmt="%.18e", delimiter=" ")

    r_mean_arr = np.array(r_mean).astype(np.double)
    print(r_mean_arr)
    x = np.arange(1, episodes + 1)
    # plt.plot(x, r, label="Return per Episode")
    # plt.plot(x, r_mean_arr, label="Rolling Mean of Return")
    s1mask = np.isfinite(r)
    s2mask = np.isfinite(r_mean_arr)
    plt.plot(x[s1mask], r[s1mask], linestyle='-', label="Return per Episode")
    plt.plot(x[s2mask], r_mean_arr[s2mask], linestyle='-', label="Rolling Mean of Return")
    plt.title('Return with Raw Features')
    plt.ylabel('Return')
    plt.xlabel('Episode')
    plt.legend()
    # plt.xticks(ticks=x, labels=x_labels, rotation=50)
    # plt.subplots_adjust(bottom=0.2)
    plt.savefig('5_1_raw.png')


