import torch


class LAP:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_size=1e6,
        max_action=1,
        normalize_actions=True,
        prioritized=True,
    ):

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
        self.prev_index = torch.full((max_size,), -1, dtype=torch.int32, device=device)

        self.prioritized = prioritized
        if prioritized:
            self.priority = torch.zeros(max_size, dtype=torch.float32, device=device)
            self.max_priority = torch.ones((), dtype=torch.float32, device=device)

        self.normalize_actions = max_action if normalize_actions else 1

    def add(self, state, action, next_state, reward, terminated, prev_index=None):
        assert state.ndim == 2
        n = state.shape[0]
        if self.ptr + n > self.max_size:
            self.ptr = 0

        self.state[self.ptr : self.ptr + n] = state
        self.action[self.ptr : self.ptr + n] = action / self.normalize_actions
        self.next_state[self.ptr : self.ptr + n] = next_state
        self.reward[self.ptr : self.ptr + n] = reward.view(n, 1)
        self.not_done[self.ptr : self.ptr + n] = 1.0 - terminated.view(n, 1).float()
        if prev_index is not None:
            self.prev_index[self.ptr : self.ptr + n] = prev_index
        else:
            self.prev_index[self.ptr : self.ptr + n] = -1

        if self.prioritized:
            self.priority[self.ptr : self.ptr + n] = self.max_priority  # initial priority: maximal

        indices = torch.arange(self.ptr, self.ptr + n, dtype=torch.int32, device=self.device)
        self.ptr += n
        self.size = max(self.ptr, self.size)
        return indices

    def get_episode_trace(self, i):
        assert i >= 0 and i < self.size
        # mt: this might be slow! how else?
        # tim: I also think that this may be quite slow. Maybe it makes sense to check how torchrl implements their buffers (or use theirs since they have some nice features): https://docs.pytorch.org/rl/stable/reference/generated/torchrl.data.ReplayBuffer.html#replaybuffer . I think that they store their episodes in contigous memory and retrieve it in one piece.
        indices = []
        while i != -1:
            indices.append(i)
            i = int(self.prev_index[i].item())
        indices.reverse()
        return torch.tensor(indices, dtype=torch.int32, device=self.device)

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
        if self.prioritized:
            csum = torch.cumsum(self.priority[: self.size], 0).to(self.device)
            val = torch.rand(size=(batch_size,), device=self.device) * csum[-1]
            indices = torch.searchsorted(csum, val).clamp_(max=self.size - 1)
        else:
            indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            indices,
            self.state[indices],
            self.action[indices],
            self.next_state[indices],
            self.reward[indices],
            self.not_done[indices],
        )

    def update_priority(self, priority, indices):
        prio = priority.reshape(-1).detach()
        self.priority[indices] = prio
        self.max_priority = torch.maximum(prio.max(), self.max_priority)

    def reset_max_priority(self):
        self.max_priority = self.priority[: self.size].max().detach().clone()


class CloneBuffer:
    def __init__(self, state, action, max_size=int(1e6)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.size = self.state.shape[0]

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.state[ind], self.action[ind]
