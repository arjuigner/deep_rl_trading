import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from evaluation import evaluate_agent_during_training
matplotlib.use('TkAgg')  # Use this backend to prevent freezing

def create_live_plot_logger(update_freq=20, steps_shown=10000):
    plt.ion()  # Turn on interactive mode
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    axs = axs.flatten()  # Make it a 1D list for easier indexing

    # Initialize data lists
    episodes = []
    rewards = []
    actor_losses = []
    critic_losses = []
    action_components = []
    steps = []
    eval_rewards = []

    # Create a line object for each subplot
    line_reward, = axs[0].plot([], [], 'b-')
    axs[0].set_title("Cumulative Reward (Train env)")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Reward")
    
    line_eval_reward, = axs[1].plot([], [], 'b-')
    axs[1].set_title("Cumulative Reward (Test env)")
    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Reward")
    
    line_actor, = axs[2].plot([], [], 'r-')
    axs[2].set_title("Actor Loss")
    axs[2].set_xlabel("Step")
    axs[2].set_ylabel("Loss")
    
    line_critic, = axs[3].plot([], [], 'g-')
    axs[3].set_title("Critic Loss")
    axs[3].set_xlabel("Step")
    axs[3].set_ylabel("Loss")
    
    line_action, = axs[4].plot([], [], 'm-')
    axs[4].set_title("First Action Component")
    axs[4].set_xlabel("Step")
    axs[4].set_ylabel("Action Value")
    
    plt.tight_layout()
    plt.show()

    def log_fn(metrics):
        """
        Expects metrics to be a dictionary with the following keys:
          - 'episode': episode number (x-axis)
          - 'cumulative_reward': total reward for the episode
          - 'actor_loss': loss value from the actor update
          - 'critic_loss': loss value from the critic update
          - 'action_component': the first component of the action taken
          - 'eval_score': evaluation score from the test environment
        """
        episodes.append(metrics['episode'])
        rewards.append(metrics['cumulative_reward'])
        actor_losses.append(metrics['actor_loss'])
        critic_losses.append(metrics['critic_loss'])
        action_components.append(metrics['action_component'])
        steps.append(metrics['step'])
        eval_rewards.append(metrics['eval_score'])

        if steps[-1] % update_freq == update_freq - 1:

            # Update cumulative reward plot
            line_reward.set_data(steps[-steps_shown:], rewards[-steps_shown:])
            axs[0].relim()
            axs[0].autoscale_view()
            
            # Update cumulative reward plot
            line_eval_reward.set_data(steps[-steps_shown:], eval_rewards[-steps_shown:])
            axs[1].relim()
            axs[1].autoscale_view()

            # Update actor loss plot
            line_actor.set_data(steps[-steps_shown:], actor_losses[-steps_shown:])
            axs[2].relim()
            axs[2].autoscale_view()

            # Update critic loss plot
            line_critic.set_data(steps[-steps_shown:], critic_losses[-steps_shown:])
            axs[3].relim()
            axs[3].autoscale_view()

            # Update first action component plot
            line_action.set_data(steps[-steps_shown:], action_components[-steps_shown:])
            axs[4].relim()
            axs[4].autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()

    return log_fn

# Usage Example:
# live_logger = create_live_plot_logger()
# Then pass live_logger as the log_fn parameter to your agent.train(...) method.

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_score = None
        self.counter = 0

    def step(self, score):
        """
        Return two booleans:
        - whether it is a new best
        - whether patience has been exceeded
        """
        if self.best_score is None:
            self.best_score = score
            return True, False

        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            return True, False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return False, True
            return False, False

class ReplayBuffer():
    def __init__(self, size, state_dim, action_dim):
        # Have separate tensors for each component of the transition:
        # x, a, x', r, done
        self.state      = torch.zeros(size, state_dim,  dtype=torch.float32)
        self.action     = torch.zeros(size, action_dim, dtype=torch.float32)
        self.next_state = torch.zeros(size, state_dim,  dtype=torch.float32)
        self.reward     = torch.zeros(size, 1,          dtype=torch.float32)
        self.done       = torch.zeros(size, 1,          dtype=torch.float32)
        
        self.size = size
        self.i = 0
        self.full = False
        
        
    def add(self, state, action, next_state, reward, done):
        '''
        state: torch.Tensor of shape (state_dim,)
        action: torch.Tensor of shape (action_dim,)
        next_state: torch.Tensor of shape (state_dim,)
        reward: float
        done: bool
        '''
        self.state[self.i] = state
        self.action[self.i] = action
        self.next_state[self.i] = next_state
        self.reward[self.i, 0] = reward
        self.done[self.i, 0] = done
        
        self.i += 1
        if self.i >= self.size and not self.full:
            self.full = True
        self.i %= self.size
        
    def sample_batch(self, batch_size):
        # Generate random indices and return the corresponding transitions
        idx = torch.randint(0, self.size if self.full else self.i, size=(batch_size,))
        return self.state[idx], self.action[idx], self.next_state[idx], self.reward[idx], self.done[idx]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action, layers=[256, 256]):
        super().__init__()
        
        # Initialize layers 
        if layers:
            self.layers = [nn.Linear(state_dim, layers[0]), nn.LeakyReLU()]
            for i in range(1, len(layers)):
                self.layers.append(nn.Linear(layers[i-1], layers[i]))
                self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Linear(layers[-1], action_dim))
            self.layers.append(nn.Sigmoid())
            self.model = nn.Sequential(*self.layers)
        else: # No hidden layers, i.e. logistic regression
            self.model = nn.Sequential(nn.Linear(state_dim, action_dim), nn.Sigmoid())
        
        self.min_action = torch.from_numpy(min_action).float()
        self.max_action = torch.from_numpy(max_action).float()
        
    def forward(self, state):
        return self.model(state) * (self.max_action - self.min_action) + self.min_action
    
class Critic(nn.Module):
    
    def __init__(self, state_dim, action_dim, layers=[256, 256]):
        super().__init__()
        
        # Initialize layers
        if layers:
            self.layers = [nn.Linear(state_dim+action_dim, layers[0]), nn.LeakyReLU()]
            for i in range(1, len(layers)):
                self.layers.append(nn.Linear(layers[i-1], layers[i]))
                self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Linear(layers[-1], 1))
            self.model = nn.Sequential(*self.layers)
        else: # No hidden layers, i.e. linear regression
            self.model = nn.Linear(state_dim+action_dim, 1)
        
    def forward(self, state, action):
        # Make sure there is a batch dimension and a features dimension
        assert state.dim() == 2 and action.dim() == 2
        assert state.shape[0] == action.shape[0]
        
        return self.model(torch.cat([state, action], dim=1))
    
class TD3Agent():
    
    def __init__(self, state_dim, action_dim, min_action, max_action, optim_constructor, polyak, actor_class=Actor, critic_class=Critic):
        self.actor = actor_class(state_dim, action_dim, min_action, max_action)
        self.q1 = critic_class(state_dim, action_dim)
        self.q2 = critic_class(state_dim, action_dim)
        
        self.actor_target = actor_class(state_dim, action_dim, min_action, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.requires_grad_(False)
        
        self.q1_target = critic_class(state_dim, action_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q1_target.requires_grad_(False)
        
        self.q2_target = critic_class(state_dim, action_dim)
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q2_target.requires_grad_(False)
        
        self.actor_optim = optim_constructor(self.actor.parameters())
        self.q_optim = optim_constructor(list(self.q1.parameters()) + list(self.q2.parameters()))
        self.polyak = polyak
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def act(self, state):
        # Check if state is a numpy array (otherwise torch Tensor expected)
        numpy = False
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
            numpy = True
            
        if state.dim() == 1:
            action = self.actor(state.unsqueeze(0)).squeeze(0)
        elif state.dim() == 2:
            action = self.actor(state)
        else:
            raise ValueError("State must have 1 or 2 dimensions")
        
        return action.detach().numpy() if numpy else action
    
    def critic(self, state, action):
        if state.dim() == 1 and action.dim() == 1:
            return self.q1(state.unsqueeze(0), action.unsqueeze(0)).squeeze(0), self.q2(state.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
        elif state.dim() == 2 and action.dim() == 2:
            return self.q1(state, action), self.q2(state, action)
        else:
            raise ValueError("State and action must have the same number of dimensions, which is 1 or 2")
        
    def actor_step(self, state):
        loss = -self.q1(state, self.actor(state)).mean()
        
        # Update actor
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()
        
        # loss.backward() also populated the gradients of self.q1
        self.q1.zero_grad()
        return loss.item()
    
    def critic_step(self, state, action, target):
        # Predicted q values
        preds1 = self.q1(state, action)
        preds2 = self.q2(state, action)
        
        # Loss
        mse1 = F.mse_loss(preds1, target)
        mse2 = F.mse_loss(preds2, target)
        loss = mse1 + mse2
        
        # Update critics
        self.q_optim.zero_grad()
        loss.backward()
        self.q_optim.step()
        return loss.item()
    
    def update_targets(self):
        for p, p_target in zip(self.actor.parameters(), self.actor_target.parameters()):
            p_target.data = p_target.data * self.polyak + p.data * (1 - self.polyak)
        
        for p, p_target in zip(self.q1.parameters(), self.q1_target.parameters()):
            p_target.data = p_target.data * self.polyak + p.data * (1 - self.polyak)
        
        for p, p_target in zip(self.q2.parameters(), self.q2_target.parameters()):
            p_target.data = p_target.data * self.polyak + p.data * (1 - self.polyak)
    
    def _action_noise(self, std, clip=None):
        noise = torch.normal(0., std, size=(self.action_dim,))
        if clip is not None:
            return noise.clamp(-clip, clip)
        else:
            return noise
    
    def train(self, env, eval_env, steps, batch_size, gamma, expl_noise_std, 
              policy_noise_std, policy_noise_clip, policy_delay,
              random_steps, memory_size, patience, eval_freq, save_path, log_fn):
        early_stopping = EarlyStopping(patience)
        
        assert self.state_dim == env.observation_space.shape[0]
        assert self.action_dim == env.action_space.shape[0]
        
        replay_buffer = ReplayBuffer(memory_size, self.state_dim, self.action_dim)
        
        state, _ = env.reset()
        state = torch.from_numpy(state).float()
        cum_reward = 0.0
        episode = 1
        eval_score = 0.0
        actor_loss, critic_loss = 0.0, 0.0
        for s in range(steps):
            # Sample an action to perform
            if s < random_steps:
                action = env.action_space.sample()
            else:
                action = self.act(state).detach() + self._action_noise(expl_noise_std)
                # Make sure we are in the allowed range of actions
                action = action.numpy().clip(env.action_space.low, env.action_space.high)
                
            # Perform the action and store transition
            next_state, reward, done, _, _ = env.step(action)
            cum_reward += reward
            next_state = torch.from_numpy(next_state).float()
            action = torch.from_numpy(action).float()
            reward = torch.tensor([reward], dtype=torch.float32).item()
            replay_buffer.add(state, action, next_state, reward, done)            
        
            # Train critics and sometimes actor
            # No issues with requires_grad since only target networks are used to compute target
            x, a, xp, r, d = replay_buffer.sample_batch(batch_size)
            ap = self.actor_target(xp) + self._action_noise(policy_noise_std, policy_noise_clip)
            target = r + gamma * (1-d) * torch.min(
                torch.stack((self.q1_target(xp, ap), self.q2_target(xp, ap))), 
                dim=0
            )[0]
            critic_loss = self.critic_step(x, a, target)
            
            # Delayed updates of target networks and actor to make 
            # training of critics easier ("moving goalpost issue")
            if s % policy_delay == 0:
                actor_loss = self.actor_step(x)
                self.update_targets()
                
            log_fn({
                'episode': episode,
                'cumulative_reward': cum_reward,
                'step': s,
                'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'action_component': action[0].item(),
                'eval_score': eval_score
            })
                
            if done: 
                print(f'Step {s}\nCumulative Reward at the end of episode #{episode}: {cum_reward}')
                print('------------------------------------------------')
                next_state, _ = env.reset()
                next_state = torch.from_numpy(next_state).float()
                episode += 1
                cum_reward = 0.0
                
                # Eval agent and potentially save it or stop training
                if s > random_steps and episode % eval_freq == 0:
                    with torch.no_grad():
                        eval_score = evaluate_agent_during_training(eval_env, self, n_iter=30)
                    is_new_best, stop_now = early_stopping.step(eval_score)
                    if is_new_best:
                        print(f"New best score: {eval_score}")
                        self.save(save_path)
                    if stop_now:
                        print("Early stopping triggered.")
                        break
            state = next_state
        
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        # Save model parameters
        torch.save({
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'q_optim': self.q_optim.state_dict(),
        }, os.path.join(path, 'models.pt'))

        # Save configuration
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump({
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'min_action': self.actor.min_action.numpy(),
                'max_action': self.actor.max_action.numpy(),
                'polyak': self.polyak,
            }, f)

    @classmethod
    def load(cls, path, optim_constructor, actor_class=Actor, critic_class=Critic):
        # Load configuration
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        agent = cls(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            min_action=config['min_action'],
            max_action=config['max_action'],
            optim_constructor=optim_constructor,
            polyak=config['polyak'],
            actor_class=actor_class,
            critic_class=critic_class
        )
        # Load model weights
        checkpoint = torch.load(os.path.join(path, 'models.pt'))
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.q1.load_state_dict(checkpoint['q1'])
        agent.q2.load_state_dict(checkpoint['q2'])
        agent.actor_target.load_state_dict(checkpoint['actor_target'])
        agent.q1_target.load_state_dict(checkpoint['q1_target'])
        agent.q2_target.load_state_dict(checkpoint['q2_target'])
        agent.actor_optim.load_state_dict(checkpoint['actor_optim'])
        agent.q_optim.load_state_dict(checkpoint['q_optim'])
        return agent
        