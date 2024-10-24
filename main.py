from env import *
from models import *

import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchrl.data import ReplayBuffer, ListStorage

if __name__ == '__main__':
    # Get data and process it
    TICKERS = ['MSFT', 'QCOM']
    N_STOCKS = len(TICKERS)
    data = get_stock_data(TICKERS, '2011-01-01', '2017-01-01', '1d')
    data = timestamps_processing(data)
    data = make_adjusted(data)

    # Initialize environment
    INITIAL_CASH = 1000
    K_MAX = 10
    env = TradingEnvironment(data, cash=INITIAL_CASH, K_max=K_MAX, indicators=True, t0=14) # t0=14 because it is the size of the window for SMA and RSI
    state = env.start()
    state = torch.from_numpy(state).float()

    STATE_SIZE = 4*N_STOCKS + 1  # TODO: Change when adding new indicators
    ACTION_SIZE = N_STOCKS

    # Initialize Q-Networks
    Q_HIDDEN_LAYERS = [128, 128, 32]
    initial_prices = torch.from_numpy(env.stock_data['Adj Close'].iloc[env.t0].values).float()
    q1        = QNetwork(STATE_SIZE, ACTION_SIZE, INITIAL_CASH, initial_prices, K_MAX, hidden_layers=Q_HIDDEN_LAYERS)
    q2        = QNetwork(STATE_SIZE, ACTION_SIZE, INITIAL_CASH, initial_prices, K_MAX, hidden_layers=Q_HIDDEN_LAYERS)
    q_target1 = copy.deepcopy(q1)
    q_target2 = copy.deepcopy(q2)

    # Initialize Policy Networks
    POLICY_HIDDEN_LAYERS = [128, 128, 32]
    policy = PolicyNetwork(STATE_SIZE, ACTION_SIZE, INITIAL_CASH, initial_prices, K_MAX, hidden_layers=POLICY_HIDDEN_LAYERS)
    policy_target = copy.deepcopy(policy)

    # Optimizers
    Q_LR = 1e-4
    P_LR = 1e-3
    q1_optimizer = optim.Adam(q1.parameters(), lr=Q_LR)
    q2_optimizer = optim.Adam(q2.parameters(), lr=Q_LR)
    policy_optimizer = optim.Adam(policy.parameters(), lr=P_LR)

    # Some training related hyperparameters
    GAMMA = 0.99
    RHO = 0.005
    Q_UPDATE_INTERVAL = 1
    POLICY_UPDATE_INTERVAL = 2
    EXPLORATION_STEPS = 252 # Take random actions for the first year
    BATCH_SIZE = 128

    # Initialize replay buffer
    REPLAY_BUFFER_SIZE = 8000
    replay_buffer = ReplayBuffer(storage=ListStorage(REPLAY_BUFFER_SIZE), batch_size=BATCH_SIZE)

    # Training loop
    done = False
    i = 0
    q1_loss, q2_loss, policy_loss = None, None, None
    while not done:
        # Select action with noise
        action = policy(state.unsqueeze(0)).detach().squeeze(0) + torch.normal(0, 0.2, size=(N_STOCKS,)).detach() if i >= EXPLORATION_STEPS else 0.25 * torch.randn(N_STOCKS)
        action = torch.clamp(action, -1, 1)
        
        # Apply action and get next state
        next_state, reward, done = env.step(action.cpu().detach().numpy())
        next_state = torch.from_numpy(next_state).float()

        # Next line only if reward is the value gained/lost instead of the return
        #reward /= 20 # TODO: smarter normalization 

        # Add experience to replay buffer
        replay_buffer.add((state, action, reward, next_state))
        
        # Update state to current state
        state = next_state
        
        i += 1

        # Train networks occasionally
        if i % Q_UPDATE_INTERVAL == 0:
            # Sample batch
            state_, action_, reward_, next_state_ = replay_buffer.sample()
            reward_ = reward_.float().unsqueeze(1) # by default it is a float64/double tensor, since it is created from scalar values

            # Compute target Q values
            with torch.no_grad():
                noise = torch.normal(0, 0.2, size=(BATCH_SIZE, N_STOCKS)).clamp(-0.5, 0.5)
                target_action = policy_target(next_state_) + noise
                target_action = torch.clamp(target_action, -1, 1)
                target_q1 = q_target1(next_state_, target_action)
                target_q2 = q_target2(next_state_, target_action)
                target_q = reward_ + torch.minimum(target_q1, target_q2)
                #print(reward_.mean(), reward_.std(), target_q.mean(), target_q.std())

            # Compute Q loss
            q1_loss = F.mse_loss(q1(state_, action_), target_q)
            q2_loss = F.mse_loss(q2(state_, action_), target_q)

            # Optimize Q networks
            q1_optimizer.zero_grad()
            q1_loss.backward()
            q1_optimizer.step()    

            q2_optimizer.zero_grad()
            q2_loss.backward()
            q2_optimizer.step()

            # Update policy and target networks
            if i % POLICY_UPDATE_INTERVAL == 0 and i >= EXPLORATION_STEPS:
                # Compute policy loss
                policy_loss = -q1(state_, policy(state_)).mean()

                # Optimize policy network
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # Update target policy network
                with torch.no_grad():
                    for param, target_param in zip(policy.parameters(), policy_target.parameters()):
                        target_param.data = RHO * param.data + (1 - RHO) * target_param.data

                # Update target Q networks
                with torch.no_grad():
                    for param, target_param in zip(q1.parameters(), q_target1.parameters()):
                        target_param.data = RHO * param.data + (1 - RHO) * target_param.data
                    for param, target_param in zip(q2.parameters(), q_target2.parameters()):
                        target_param.data = RHO * param.data + (1 - RHO) * target_param.data

        if i == EXPLORATION_STEPS:
            print(f"Finished all {i} exploration steps")
        if i % 100 == 0:
            if i >= EXPLORATION_STEPS:
                print(f"Step {i}, Portfolio value: {env.get_portfolio_return()}, q1_loss: {q1_loss.item()}, q2_loss: {q2_loss.item()}, policy_loss: {   policy_loss.item()}")
            else:
                print(f"Step {i}, Portfolio value: {env.get_portfolio_return()}, q1_loss: {q1_loss.item()}, q2_loss: {q2_loss.item()}")
            # Check what typical actions look like
            state_, _, _, _ = replay_buffer.sample()
            actions = policy(state_).detach().numpy()
            print(f"Step {i}, Stats about recommended actions for a random batch of data: mean: {actions.mean()}, std: {actions.std()}, min: {actions.min()}, max: {actions.max()}")
    

    # Show logs
    print(env.log.to_string())