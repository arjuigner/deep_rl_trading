# Deep Reinforcement Learning for Stock Trading

This is the git repository for my semester project (2025) at ETH Zürich.

## Goal

The goal is to train a RL agent to perform stock trading - with the goal of maximizing portfolio returns at the end of a N-days window. More specifically, the agent has access to daily stock data and rebalances its portfolio every day as it wants - as long as it has enough money to perform the desired rebalancing. The RL algorithm used here is **TD3** (**Twin-delayed deep deterministic policy gradient**).

## Results

* As discussed before, it was difficult to make the agent even learn at all, as it had the tendency to get stuck with extreme actions (for example: sell the max amount of everything). This was however "solved" by redesigning the environment such that there is no set (of non-zero Lebesgue measure...) of actions that do nothing. [Note from later: proper normalization of the state features makes a huge difference too!]
* The next issue was overfitting: I could easily get an agent to achieve unrealistically high returns (way beyond 10x initial cash in a year) in the trading environment, but then it would do no better than random trading on another environment with data from a few months later.  
* At that point, I decided to implement TD3 myself to have more control over the details (validation, early stopping, ...) and the logging, as well as an easier time debugging. I am therefore not using any RL library anymore.  
* The current code runs experiments with different agent/environment settings and different features in the state space. It performs early stopping and validation on a "test environment" during training, to avoid blatant overfitting.  
* Sadly, I am unable to get convincing results so far. It seems to be slightly better than random trading and sometimes buy-and-hold, but not even consistently so. As a consolation prize, on Ford data, where the test set spans a bearish period, the agents do lose less than random trading and buy-and-hold.

## Summary of the repo structure

### `environment.py` : The trading environment

This file contains the code for the environment which will be used for training and testing. The main class is called `TradingEnv` and follows the same structure as the usual gym environment. This property allows it to be used with usual RL libraries such as stable-baselines3. However, I ended up rewriting my own implementation of TD3, so I currently do not use any external libraries for the RL algorithm (excepted obviously for `torch`, `numpy`, etc for the implementation of TD3).

The environment has many parameters, to choose the maximum position size, the initial wealth, the transaction fees, etc. At each episode, a random point in time is chosen and the agent will have to trade for $N$ ($=20$) consecutive days. The reward at each step is the transaction fee plus either:
* the portfolio return on that day (if `reward_type = "relative returns")
* or the final portfolio return between the start and the end of the episode if we are at the last day, otherwise 0.  

Note: to encourage the agent to not try to rebalance in a way that would be too expensive for the current wealth it owns, there is a penalty if the action is too expensive.

The action is a numpy vector with as many components as stocks to trade, with values in $[0,1]$ which indicate how many shares to own for the given stock, where $0$ means 0 shares and $1$ means $K_{max}$ shares.

### `td3_impl.py` : TD3 implementation

This file contains my implementation of the TD3 algorithm.  
The main class is `TD3Agent`, where the training method is called `train`(unexpectedly, I know!). Most of the hyperparameters in the algorithm can be changed either at initialization of the `TD3Agent` or in the parameters of the `train` method. 

For logging during training, the `train` method has a parameter log_fn which is meant to be a function that will be executed at each iteration with the following dictionary as parameter: 
```
{
   'episode': episode number,
   'cumulative_reward': cumulative rewards in the current episode,
   'step': step number,
   'actor_loss': actor_loss in that iteration,
   'critic_loss': critic_loss in that iteration,
   'action_component': first component of the chosen action (to see what the actions typically look like),
   'eval_score': latest evaluation score on the validation environment (if not executed at each iteration, it puts the latest evaluation score)
}
```
By default, `log_fn = create_live_plot_logger` which is defined in the same file, and plots all these quantities "live".  

Other important classes in that file include ReplayBuffer (self-explanatory), Actor (implements the actor model, i.e. neural network), Critic (implements one critic, meaning that a TD3Agent has two instances of Critic updated at each step during train and two instances of Critic which are the "target networks"), and EarlyStopping (which encapsulates all the logic related to early stopping, based on the latest evaluation score of the agent on the validation environment).

### `learn.py`: Main script

Main script and entry point. Can simply be run with `python learn.py` It contains a function run_experiment that takes dictionaries of parameters for 
the environment, the TD3Agent and the TD3Agent.train method. It runs the experiment with these parameters multiple times and logs everything needed to reconstruct the experiment, as well as the results and the final model.  

Then, the rest of the code is extremely repetitive: it simply calls run_experiment a bunch of times with different configurations (right now, with different kind of features).

### `data.py`: Get the stock market data

Last but not least, this file contains everything related to the data. For now, I tried two stocks (separately): Apple (AAPL) (always goes up) and Ford (F) (always go up a bit less). The data for AAPL comes from `data/HistoricalQuotes.csv` and the data for F from yahoo-finance. This file contains the code to load the data into `pandas.DataFrame` object, as well as the code to create features and to normalize and clean the data.

### Extras: `useful_scripts` directory

* `useful_scripts/features.ipynb` is the notebook where I explored the DataFrame containing data + features, in particular to figure out by how much to scale each column in order to have roughly normalized features and data. Since I do not want to introduce look-ahead bias, normalizations are simply done by hand and sort of arbitrarily.
* `useful_scripts/visualize_results.ipynb` is a notebook to print out the results of each experiment, to quickly see which experiments worked better or not.

### `tests`: Unit testing
`tests` is a directory with (lacking) unit tests. For now, there is only one file to test the environment and make sure that it works as expected.
