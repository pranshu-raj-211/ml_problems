import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# number of actions
k = 10
# number of trials
iterations = 1000
# values for creating the true mean distribution
true_mean_dist_mean = 0.0
true_mean_dist_stddev = 1.0
# defining optimal mean values for stationary reward probability distribution
true_means = np.random.normal(loc=true_mean_dist_mean, scale = true_mean_dist_stddev, size=k)
# initial estimate of mean of value of actions
estimated_means = np.zeros(k)


distributions = [np.random.normal(loc=mean, scale=1, size=1000) for mean in true_means]
reward_distribution = pd.DataFrame(distributions).transpose()
plt.figure(figsize=(8,6))
sns.violinplot(data=reward_distribution, orient='v', color='pink')
plt.title('Reward distributions')
plt.show()


def greedy_agent(estimated_means):
    max_estimate = np.max(estimated_means)
    max_indexes = np.where(max_estimate == estimated_means)[0]
    return np.random.choice(max_indexes)

optimal_action = np.argmax(true_means)


def get_reward(action):
    '''
    Returns the actual reward for the action taken.
    The reward is sample from a normal distribution having mean defined in true_means and variance 1'''
    return np.random.normal(loc=true_means[action], scale=1.0)


def update_mean(old_mean : float, reward : float, scaling_factor:float):
    new_mean= old_mean + (reward - old_mean)*scaling_factor
    return new_mean


def plot_progress(reward_hist,  agents:list):
    # TODO : Implement logic for plotting charts for various agents for comparison
    cumulative_mean = np.cumsum(reward_hist)/(np.arange(len(reward_hist))+1)
    plt.figure(figsize=(16,8))
    plt.plot(reward_hist, label = agents[0], color = 'blue')
    plt.plot(cumulative_mean, label='plot_mean', color = 'orange')
    plt.xlabel('Epochs')
    plt.ylabel('Rewards')
    plt.title('Reward vs epoch')
    plt.legend()
    plt.show()
    # TODO : Implement logic for optimal action plotting - how to regularize over time


def bandit(estimated_values, epsilon:float, epochs:int):
    action_hist = np.zeros(epochs, dtype=np.int8)
    reward_hist = np.zeros(epochs, dtype=np.float16)
    for epoch in range(epochs):
        # 0: explore, 1: exploit
        action_type = np.random.choice(a=np.array([0,1]), p=[epsilon, (1-epsilon)])
        greedy_choice = greedy_agent(estimated_means=estimated_values)
        if 1 == action_type:
            action =greedy_choice
        else:
            action = np.random.choice(np.arange(len(estimated_values)))
        
        reward = get_reward(action)
        action_hist[epoch] = action
        reward_hist[epoch] = reward
        estimated_values[action]=update_mean(estimated_values[action], reward, 1.0/(epoch+1))

    return action_hist, reward_hist



action_history_greedy, reward_history_greedy = bandit(estimated_means, epsilon=0, epochs=1000)
action_history_ep_01, reward_history_ep_01 = bandit(estimated_means,epsilon=0.1, epochs=1000)
action_history_ep_001, reward_history_ep_001 = bandit(estimated_means, epsilon=0.01, epochs=1000)

plot_progress(reward_history_greedy, agents=['greedy'])
plot_progress(reward_history_ep_01, agents=['epsilon 0.1'])
plot_progress(reward_history_ep_001, agents=['epsilon 0.01'])