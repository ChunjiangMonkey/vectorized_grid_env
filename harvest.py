import torch

from utils import intersection_1d


class CommonHarvestSimpleGPU:
    NUM_AGENTS = 2
    # up, right, down, left, stay
    NUM_ACTIONS = 5
    MOVES = torch.stack(
        [
            torch.tensor([-1, 0], dtype=torch.int64),  # up
            torch.tensor([0, 1], dtype=torch.int64),  # right
            torch.tensor([1, 0], dtype=torch.int64),  # down
            torch.tensor([0, -1], dtype=torch.int64),  # left
            torch.tensor([0, 0], dtype=torch.int64),  # stay
        ],
        dim=0,
    )
    # TURNS = {4: 1, 5: -1}
    REGROWTH_PROBABILITIES = 0.15

    def __init__(self, gird_size=5, batch_size=16, max_step=10086, device=torch.device("cuda:0")):
        self.device = device
        self.MOVES = self.MOVES.to(self.device)

        self.gird_size = gird_size
        self.batch_size = batch_size
        self.max_step = max_step
        self.time_step = 0

        # binary_tensor: self.batch_size, 4, self.height, self.width
        # rgb_tensor: self.batch, 3, self.height, self.width
        self.blue_state = None
        self.red_state = None

        self.blue_rgb_state = None
        self.red_rgb_state = None

        # self.batch_size, 2
        self.blue_agent_pos = None
        self.red_agent_pos = None

        # self.batch_size, 13
        self.apple_hidden = None

        # self.batch_size, 13, 2
        self.apple_pos = None

        self._apple_no_hidden_batch_index = None
        self._apple_hidden_batch_index = None

        self._central_apple_no_hidden_batch = None
        self._central_apple_hidden_batch = None

        self._red_agent_reward = None
        self._blue_agent_reward = None

        # self.batch_size,
        self._step_blue_agent_action = None
        self._step_red_agent_action = None

    def reset(self):
        self.time_step = 0
        # init actions: stay
        self._step_blue_agent_action = torch.full((self.batch_size,), fill_value=4, dtype=torch.int64).to(self.device)
        self._step_red_agent_action = torch.full((self.batch_size,), fill_value=4, dtype=torch.int64).to(self.device)

        # blue agent pos
        self.blue_agent_pos = torch.full((self.batch_size, 2), fill_value=0, dtype=torch.int64).to(self.device)
        # red agent pos
        self.red_agent_pos = torch.full((self.batch_size, 2), fill_value=self.gird_size - 1, dtype=torch.int64).to(self.device)

        # apple pos
        apple_init_pos = torch.tensor(self._generate_apple_init_pos()).to(self.device)
        self.apple_pos = torch.zeros((self.batch_size, apple_init_pos.shape[0], 2), dtype=torch.int64).to(self.device)
        self.apple_pos[:, :, :] = apple_init_pos
        self.apple_hidden = torch.zeros((self.batch_size, apple_init_pos.shape[0]), dtype=torch.int64).to(self.device)
        self._update_hidden_batch()

        self.blue_state, self.blue_rgb_state = self._generate_state("blue")
        self.red_state, self.red_rgb_state = self._generate_state("red")
        observations = [self.blue_state, self.red_state]
        infos = []
        return observations, infos

    def _update_hidden_batch(self):
        self._apple_hidden_batch_index = torch.nonzero(self.apple_hidden).squeeze(-1).to(self.device)
        self._apple_no_hidden_batch_index = torch.nonzero(self.apple_hidden == 0).squeeze(-1).to(self.device)
        self._central_apple_hidden_batch = torch.nonzero(self.apple_hidden[:, 0]).squeeze(-1).to(self.device)
        self._central_apple_no_hidden_batch = torch.nonzero(self.apple_hidden[:, 0] == 0).squeeze(-1).to(self.device)

    def _generate_apple_init_pos(self):
        # the apple is like:
        #  *
        # ***
        #  *
        apple_pos = []
        # center
        center_point = torch.tensor((self.gird_size // 2, self.gird_size // 2)).to(self.device)
        apple_pos.append(tuple(center_point.tolist()))
        # 4 directions
        for i in range(4):
            center_center_point = center_point + self.MOVES[i]
            apple_pos.append(tuple(center_center_point.tolist()))
        return list(apple_pos)

    def _generate_state(self, agent):

        if agent == "blue":
            self_agent_pos = self.blue_agent_pos
            opponent_agent_pos = self.red_agent_pos
        else:
            self_agent_pos = self.red_agent_pos
            opponent_agent_pos = self.blue_agent_pos

        # apple
        # apple_no_hidden_batch: N*2,
        # where N is the number of apple_hidden==False, 2 is the dim of (batch_i, number_i)
        # batch (dim 0 in self.apple_pos)
        apple_no_hidden_batch = self._apple_no_hidden_batch_index[:, 0]
        # index in one batch (dim 1 in self.apple_pos)
        apple_no_hidden_index = self._apple_no_hidden_batch_index[:, 1]
        apple_no_hidden_pos_x = self.apple_pos[apple_no_hidden_batch, apple_no_hidden_index, 0]
        apple_no_hidden_pos_y = self.apple_pos[apple_no_hidden_batch, apple_no_hidden_index, 1]

        all_batch = torch.tensor(range(self.batch_size))
        state = torch.zeros((self.batch_size, 3, self.gird_size, self.gird_size), dtype=torch.float32).to(self.device)
        # self agent

        state[all_batch, 0, self_agent_pos[:, 0], self_agent_pos[:, 1]] = 1

        state[all_batch, 1, opponent_agent_pos[:, 0], opponent_agent_pos[:, 1]] = 1
        # print(self_agent_pos[:, 0].shape)
        state[apple_no_hidden_batch, 2, apple_no_hidden_pos_x, apple_no_hidden_pos_y] = 1

        rgb_state = torch.zeros((self.batch_size, 3, self.gird_size, self.gird_size), dtype=torch.float32).to(
            self.device)
        # G: green agent
        rgb_state[apple_no_hidden_batch, 1, apple_no_hidden_pos_x, apple_no_hidden_pos_y] = 1
        # R: opponent agent
        rgb_state[all_batch, 1, opponent_agent_pos[:, 0], opponent_agent_pos[:, 1]] = 0
        rgb_state[all_batch, 0, opponent_agent_pos[:, 0], opponent_agent_pos[:, 1]] = 1
        # B: agent self
        rgb_state[all_batch, 1, self_agent_pos[:, 0], self_agent_pos[:, 1]] = 0
        rgb_state[all_batch, 2, self_agent_pos[:, 0], self_agent_pos[:, 1]] = 1

        conflict_agent_batch = torch.nonzero(torch.all(self_agent_pos == opponent_agent_pos, dim=1)).squeeze(-1).to(self.device)
        conflict_agent_pos_x = self_agent_pos[conflict_agent_batch, 0]
        conflict_agent_pos_y = self_agent_pos[conflict_agent_batch, 1]
        if conflict_agent_batch.numel() > 0:
            for i in range(3):
                if i != 1:
                    # 0.5R, 0.5 B
                    rgb_state[conflict_agent_batch, i, conflict_agent_pos_x, conflict_agent_pos_y] = 0.5
                else:
                    rgb_state[conflict_agent_batch, i, conflict_agent_pos_x, conflict_agent_pos_y] = 0
        return state, rgb_state

    def _respawn(self):
        # respawn the central apples
        probs = torch.rand(self._central_apple_hidden_batch.shape[0])
        central_respawn_batch = self._central_apple_hidden_batch[torch.nonzero(probs < self.REGROWTH_PROBABILITIES)]
        self.apple_hidden[central_respawn_batch, 0] = 0

        # respawn other apples
        probs = torch.rand((self._central_apple_no_hidden_batch.shape[0], 4))
        respawn_batch_index = torch.nonzero(probs < self.REGROWTH_PROBABILITIES)
        respawn_batch = respawn_batch_index[:, 0]
        respawn_index = respawn_batch_index[:, 1]
        respawn_batch = self._central_apple_no_hidden_batch[respawn_batch]

        self.apple_hidden[respawn_batch, respawn_index + 1] = 0
        self._update_hidden_batch()

    def _move(self):
        # blue agent
        self.blue_agent_pos += self.MOVES[self._step_blue_agent_action]
        self.blue_agent_pos[:, 0] = self.blue_agent_pos[:, 0] % self.gird_size
        self.blue_agent_pos[:, 1] = self.blue_agent_pos[:, 1] % self.gird_size

        # red agent
        self.red_agent_pos += self.MOVES[self._step_red_agent_action]
        self.red_agent_pos[:, 0] = self.red_agent_pos[:, 0] % self.gird_size
        self.red_agent_pos[:, 1] = self.red_agent_pos[:, 1] % self.gird_size

    def _eat_apple(self):
        # blue
        # (batch,index) for apple where there is an agent no hidden and some apple no hidden at the batch

        apple_no_hidden_batch = self._apple_no_hidden_batch_index[:, 0]
        apple_no_hidden_index = self._apple_no_hidden_batch_index[:, 1]

        # add a dim for agent_pos to make it's dim and apple pos 's dim equal to compare them
        agent_pos_compare = self.blue_agent_pos.unsqueeze(1)
        # find the batch where an agent's pos is equal to any apple's pos
        blue_agent_apple_match = ((self.apple_pos[apple_no_hidden_batch, apple_no_hidden_index, 0] == agent_pos_compare[apple_no_hidden_batch, 0, 0])
                                  & (self.apple_pos[apple_no_hidden_batch, apple_no_hidden_index, 1] == agent_pos_compare[apple_no_hidden_batch, 0, 1]))

        blue_agent_apple_match_batch = apple_no_hidden_batch[blue_agent_apple_match]
        blue_agent_apple_match_index = apple_no_hidden_index[blue_agent_apple_match]

        # hide apple
        self.apple_hidden[blue_agent_apple_match_batch, blue_agent_apple_match_index] = 1
        self._blue_agent_reward[blue_agent_apple_match_batch] += 1

        # red, all is the same as blue

        agent_pos_compare = self.red_agent_pos.unsqueeze(1)
        agent_apple_match = (
                (self.apple_pos[apple_no_hidden_batch, apple_no_hidden_index, 0] == agent_pos_compare[apple_no_hidden_batch, 0, 0])
                & (self.apple_pos[apple_no_hidden_batch, apple_no_hidden_index, 1] == agent_pos_compare[apple_no_hidden_batch, 0, 1]))
        # print(agent_apple_match)
        red_agent_apple_match_batch = apple_no_hidden_batch[agent_apple_match]
        red_agent_apple_match_index = apple_no_hidden_index[agent_apple_match]

        self.apple_hidden[red_agent_apple_match_batch, red_agent_apple_match_index] = 1
        self._red_agent_reward[red_agent_apple_match_batch] += 1

        # find overlapping agent
        overlapping_agent_apple_match_batch = intersection_1d(blue_agent_apple_match_batch, red_agent_apple_match_batch)
        conflict_agent_batch = torch.nonzero(torch.all(self.blue_agent_pos == self.red_agent_pos, dim=1)).squeeze(-1).to(self.device)
        overlapping_agent_apple_match_batch = intersection_1d(overlapping_agent_apple_match_batch, conflict_agent_batch)
        self._red_agent_reward[overlapping_agent_apple_match_batch] = 0.5
        self._blue_agent_reward[overlapping_agent_apple_match_batch] = 0.5

        self._update_hidden_batch()

    def step(self, actions):
        self.time_step += 1
        done = torch.full((self.batch_size,), fill_value=self.time_step == self.max_step).to(self.device)

        step_blue_agent_action, step_red_agent_action = actions
        self._step_blue_agent_action = step_blue_agent_action.to(self.device)
        self._step_red_agent_action = step_red_agent_action.to(self.device)

        self._blue_agent_reward = torch.zeros(self.batch_size).to(self.device)
        self._red_agent_reward = torch.zeros(self.batch_size).to(self.device)

        self._respawn()
        # move
        self._move()
        # eat apple
        self._eat_apple()

        self.blue_state, self.blue_rgb_state = self._generate_state("blue")
        self.red_state, self.red_rgb_state = self._generate_state("red")

        observations = [self.blue_state, self.red_state]
        rewards = [self._blue_agent_reward, self._red_agent_reward]
        dones = [done, done]
        infos = self._central_apple_hidden_batch.shape[0] / self.batch_size
        # print("blue hidden", blue_hidden)
        return observations, rewards, dones, infos
