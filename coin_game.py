import torch


class CoinGameGPU:
    """
    Vectorized Coin Game environment.
    """

    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    MOVES = torch.stack(
        [
            torch.LongTensor([0, 1]),
            torch.LongTensor([0, -1]),
            torch.LongTensor([1, 0]),
            torch.LongTensor([-1, 0]),
        ],
        dim=0,
    )

    def __init__(self, grid_size=5, batch_size=16, max_steps=10086, device=torch.device("cuda:0")):

        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        # The 4 channels stand for 2 players and 2 coin positions
        self.time_step = None
        self.device = device
        self.MOVES = self.MOVES.to(self.device)

        self.blue_coin_pos = None
        self.red_coin_pos = None
        self.blue_pos = None
        self.red_pos = None

        self.red_rgb_state = None
        self.red_state = None
        self.blue_rgb_state = None
        self.blue_state = None

    def reset(self):
        self.time_step = 0

        red_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(self.device)
        self.red_pos = torch.stack((red_pos_flat // self.grid_size, red_pos_flat % self.grid_size), dim=-1).to(self.device)

        blue_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(self.device)
        self.blue_pos = torch.stack((blue_pos_flat // self.grid_size, blue_pos_flat % self.grid_size), dim=-1).to(self.device)

        red_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(self.device)
        blue_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(self.device)

        self.red_coin_pos = torch.stack((red_coin_pos_flat // self.grid_size, red_coin_pos_flat % self.grid_size), dim=-1).to(self.device)
        self.blue_coin_pos = torch.stack((blue_coin_pos_flat // self.grid_size, blue_coin_pos_flat % self.grid_size), dim=-1).to(self.device)

        self.blue_state, self.blue_rgb_state = self._generate_state("blue")
        self.red_state, self.red_rgb_state = self._generate_state("red")
        observations = [self.blue_state, self.red_state]
        infos = []
        return observations, infos

    def _generate_coins(self):
        mask_red = torch.logical_or(self._same_pos(self.red_coin_pos, self.blue_pos), self._same_pos(self.red_coin_pos, self.red_pos))
        red_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(self.device)[mask_red]
        self.red_coin_pos[mask_red] = torch.stack((red_coin_pos_flat // self.grid_size, red_coin_pos_flat % self.grid_size), dim=-1)

        mask_blue = torch.logical_or(self._same_pos(self.blue_coin_pos, self.blue_pos), self._same_pos(self.blue_coin_pos, self.red_pos))
        blue_coin_pos_flat = torch.randint(self.grid_size * self.grid_size, size=(self.batch_size,)).to(self.device)[mask_blue]
        self.blue_coin_pos[mask_blue] = torch.stack((blue_coin_pos_flat // self.grid_size, blue_coin_pos_flat % self.grid_size), dim=-1)

    def _same_pos(self, x, y):
        return torch.all(x == y, dim=-1)

    def _generate_state(self, agent):
        blue_pos_flat = self.blue_pos[:, 0] * self.grid_size + self.blue_pos[:, 1]
        blue_coin_pos_flat = self.blue_coin_pos[:, 0] * self.grid_size + self.blue_coin_pos[:, 1]
        red_pos_flat = self.red_pos[:, 0] * self.grid_size + self.red_pos[:, 1]
        red_coin_pos_flat = self.red_coin_pos[:, 0] * self.grid_size + self.red_coin_pos[:, 1]

        if agent == "blue":
            blue_state = torch.zeros((self.batch_size, 4, self.grid_size * self.grid_size)).to(self.device)
            blue_state[:, 0].scatter_(1, blue_pos_flat[:, None], 1)
            blue_state[:, 1].scatter_(1, red_pos_flat[:, None], 1)
            blue_state[:, 2].scatter_(1, blue_coin_pos_flat[:, None], 1)
            blue_state[:, 3].scatter_(1, red_coin_pos_flat[:, None], 1)

            blue_rgb_state = torch.zeros((self.batch_size, 3, self.grid_size * self.grid_size)).to(self.device)
            blue_rgb_state[:, 2].scatter_(1, blue_pos_flat[:, None], 1)
            blue_rgb_state[:, 0].scatter_(1, red_pos_flat[:, None], 1)
            blue_rgb_state[:, 2].scatter_(1, blue_coin_pos_flat[:, None], 1)
            blue_rgb_state[:, 0].scatter_(1, red_coin_pos_flat[:, None], 1)
            return blue_state.view(self.batch_size, 4, self.grid_size, self.grid_size), blue_rgb_state.view(self.batch_size, 3, self.grid_size, self.grid_size),

        else:
            red_state = torch.zeros((self.batch_size, 4, self.grid_size * self.grid_size)).to(self.device)
            red_state[:, 0].scatter_(1, red_pos_flat[:, None], 1)
            red_state[:, 1].scatter_(1, blue_pos_flat[:, None], 1)
            red_state[:, 2].scatter_(1, red_coin_pos_flat[:, None], 1)
            red_state[:, 3].scatter_(1, blue_coin_pos_flat[:, None], 1)

            red_rgb_state = torch.zeros((self.batch_size, 3, self.grid_size * self.grid_size)).to(self.device)
            red_rgb_state[:, 2].scatter_(1, red_pos_flat[:, None], 1)
            red_rgb_state[:, 0].scatter_(1, blue_pos_flat[:, None], 1)
            red_rgb_state[:, 2].scatter_(1, red_coin_pos_flat[:, None], 1)
            red_rgb_state[:, 0].scatter_(1, blue_coin_pos_flat[:, None], 1)
            return red_state.view(self.batch_size, 4, self.grid_size, self.grid_size), red_rgb_state.view(self.batch_size, 3, self.grid_size, self.grid_size)

    def step(self, actions):
        ac0, ac1 = actions
        self.time_step += 1
        self.blue_pos = (self.blue_pos + self.MOVES[ac0]) % self.grid_size
        self.red_pos = (self.red_pos + self.MOVES[ac1]) % self.grid_size

        # Compute rewards
        red_reward = torch.zeros(self.batch_size).to(self.device)
        red_red_matches = self._same_pos(self.red_pos, self.red_coin_pos)
        red_reward[red_red_matches] += 1
        red_blue_matches = self._same_pos(self.red_pos, self.blue_coin_pos)
        red_reward[red_blue_matches] += 1

        blue_reward = torch.zeros(self.batch_size).to(self.device)
        blue_red_matches = self._same_pos(self.blue_pos, self.red_coin_pos)
        blue_reward[blue_red_matches] += 1
        blue_blue_matches = self._same_pos(self.blue_pos, self.blue_coin_pos)
        blue_reward[blue_blue_matches] += 1
        red_reward[blue_red_matches] -= 2
        blue_reward[red_blue_matches] -= 2
        self._generate_coins()
        reward = [blue_reward.float(), red_reward.float()]
        self.blue_state, self.blue_rgb_state = self._generate_state("blue")
        self.red_state, self.red_rgb_state = self._generate_state("red")
        observations = [self.blue_state, self.red_state]
        if self.time_step >= self.max_steps:
            done = torch.ones(self.batch_size).to(self.device)
        else:
            done = torch.zeros(self.batch_size).to(self.device)
        infos = (blue_red_matches.sum(), blue_blue_matches.sum(), red_red_matches.sum(), red_blue_matches.sum())
        return observations, reward, done, infos
