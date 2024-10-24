import torch
import torch.nn as nn
import torch.nn.functional as F

# Components of state:
# * 0: Cash in portfolio
# * 1 to n: Number of stocks we have for each stock
# * n+1 to 2n: Prices of each stock
# * 2n+1 to 3n: SMA
# * 3n+1 to 4n: RSI

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, initial_cash, initial_prices, K_max, hidden_layers=[64, 64]):
        '''
        state_size: int, dimension of each state
        action_size: int, dimension of each action
        initial_prices: torch tensor, initial prices of the stocks, used to normalize price inputs
        hidden_layers: list of ints, sizes of the hidden layers
        '''
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.c0 = initial_cash
        self.p0 = initial_prices
        self.K_max = K_max

        assert self.state_size == 4*initial_prices.shape[0] + 1, f"Expected state size 4*n+1 = {4*initial_prices.shape[0] + 1}, got {self.state_size}"
        assert action_size == initial_prices.shape[0], f"Expected action size to be size of initial prices {initial_prices.shape[0]}, got {action_size} instead"

        self.layers = nn.Sequential(*[nn.Linear(state_size + action_size, hidden_layers[0]), nn.ReLU()] + \
                                     [nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)] + \
                                     [nn.Linear(hidden_layers[-1], 1)])
        
    def forward(self, state, action):
        '''
        Forward pass of the network
        state: torch tensor, state of the environment. Shape: (batch_size, state_size)
        action: torch tensor, action to take, values in [-1, 1]
        '''
        assert state.dim() == 2, f"Expected state to have 2 dimensions, got {state.dim()} dimensions instead"
        assert state.shape[1] == self.state_size, f"Expected state size {self.state_size}, got {state.shape[1]}"

        n_stocks = self.p0.shape[0]
        x = torch.cat((state.clone(), action.clone()), dim=1)

        # Normalize inputs
        x[:,0] /= self.c0                           # Normalize cash
        x[:,1:n_stocks+1] /= self.K_max             # Normalize number of stocks in portfolio
        x[:,n_stocks+1:2*n_stocks+1] /= self.p0     # Normalize prices
        x[:,2*n_stocks+1:3*n_stocks+1] /= self.p0   # Normalize SMA
        x[:,3*n_stocks+1:4*n_stocks+1] /= 100       # Normalize RSI
        # ... add more if new indicators
        # No need to normalize actions, they are already in [-1, 1]

        #print("EXAMPLE INPUT TO Q-NN:\n",x[0])
        #print("CORRESPONDING OUTPUT:\n",self.layers(x)[0])

        return self.layers(x)
    

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, initial_cash, initial_prices, K_max, hidden_layers=[64, 64]):
        '''
        state_size: int, dimension of each state
        action_size: int, dimension of each action
        initial_prices: torch tensor, initial prices of the stocks, used to normalize price inputs
        hidden_layers: list of ints, sizes of the hidden layers
        '''
        super(PolicyNetwork, self).__init__()
        self.state_size = state_size
        self.c0 = initial_cash
        self.p0 = initial_prices
        self.K_max = K_max

        assert self.state_size == 4*initial_prices.shape[0] + 1, f"Expected state size 4*n+1 = {4*initial_prices.shape[0] + 1}, got {self.state_size}"
        assert action_size == initial_prices.shape[0], f"Expected action size to be size of initial prices {initial_prices.shape[0]}, got {action_size} instead"
        
        self.layers = nn.Sequential(*[nn.Linear(state_size, hidden_layers[0]), nn.ReLU()] + \
                                     [nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)] + \
                                     [nn.Linear(hidden_layers[-1], action_size), nn.Tanh()])
        
    def forward(self, state):
        '''
        Forward pass of the network
        state: torch tensor, state of the environment. Shape: (batch_size, state_size)
        '''
        assert state.dim() == 2, f"Expected state to have 2 dimensions, got {state.dim()} dimensions instead"
        assert state.shape[1] == self.state_size, f"Expected state size {self.state_size}, got {state.shape[1]}"

        n_stocks = self.p0.shape[0]
        x = state.clone()

        # Normalize inputs
        x[:,0] /= self.c0                           # Normalize cash
        x[:,1:n_stocks+1] /= self.K_max             # Normalize number of stocks in portfolio
        x[:,n_stocks+1:2*n_stocks+1] /= self.p0     # Normalize prices
        x[:,2*n_stocks+1:3*n_stocks+1] /= self.p0   # Normalize SMA
        x[:,3*n_stocks+1:4*n_stocks+1] /= 100       # Normalize RSI
        # ... add more if new indicators

        # Last layer is tanh, perfect to get actions in [-1, 1]
        return self.layers(x)

        