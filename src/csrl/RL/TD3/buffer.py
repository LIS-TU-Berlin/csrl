import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device, max_size=1e6):
        max_size = int(max_size)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.device = device

        self.state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.action = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.next_state = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.reward = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.not_done = torch.zeros((max_size, 1), dtype=torch.float32, device=device)

    def add(self, state, action, next_state, reward, done, prev_index=None):
        assert state.ndim == 2
        n = state.shape[0]
        if self.ptr + n > self.max_size:
            self.ptr = 0

        self.state[self.ptr : self.ptr + n] = state
        self.action[self.ptr : self.ptr + n] = action
        self.next_state[self.ptr : self.ptr + n] = next_state
        self.reward[self.ptr : self.ptr + n] = reward.view(n, 1)
        self.not_done[self.ptr : self.ptr + n] = 1.0 - done.view(n, 1).float()

        indices = torch.arange(self.ptr, self.ptr + n, dtype=torch.int32, device=self.device)
        self.ptr += n
        self.size = max(self.ptr, self.size)
        return indices

    def get_sub(self, indices):
        assert indices.ndim == 1
        return (
            self.state[indices],
            self.action[indices],
            self.next_state[indices],
            self.reward[indices],
            self.not_done[indices],
        )

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            indices,
            self.state[indices],
            self.action[indices],
            self.next_state[indices],
            self.reward[indices],
            self.not_done[indices],
        )


class TD3_CloneBuffer:
    def __init__(self, state, action, Return, max_size=int(1e6)):
        self.size = state.shape[0]

        self.state = state
        self.action = action
        self.Return = Return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.Return[ind]).to(self.device),
        )
