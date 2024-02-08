import torch
import torch.nn.functional as F

from utils import intersection_1d


class GatheringGPU:
    NUM_AGENTS = 2
    # up, right, down, left, turn_clockwise, turn_counterclockwise, beam, stay
    NUM_ACTIONS = 8
    MOVES = torch.stack(
        [
            torch.tensor([-1, 0], dtype=torch.int64),  # up
            torch.tensor([0, 1], dtype=torch.int64),  # right
            torch.tensor([1, 0], dtype=torch.int64),  # down
            torch.tensor([0, -1], dtype=torch.int64),  # left
        ],
        dim=0,
    )
    TURNS = torch.tensor([1, -1], dtype=torch.int64)
    # TURNS = {4: 1, 5: -1}
    OBS_DIRECTION = ["center", "front"]

    def __init__(self, width=31, height=11, batch_size=16, max_steps=1000,
                 full_obs=False,
                 obs_direction="center",
                 view_size=5,
                 n_apple=10,
                 n_tagged=5,
                 device=torch.device("cpu")):

        self.device = device
        self.MOVES = self.MOVES.to(self.device)
        self.TURNS = self.TURNS.to(self.device)

        self.width = width
        self.height = height
        self.max_step = max_steps
        self.batch_size = batch_size
        self.full_obs = full_obs

        self.time_step = 0

        # apple respawn time
        self.n_apple = n_apple
        # agent hidden time
        self.n_tagged = n_tagged

        # rgb_state: self.batch, 3, self.height, self.width
        self.blue_state = None
        self.red_state = None
        # state: self.batch_size, 4, self.height, self.width
        self.blue_rgb_state = None
        self.red_rgb_state = None

        # rgb_tensor: self.batch, 3, view_size * 2 + 1, view_size * 2 + 1
        self._blue_agent_obs = None
        self._red_agent_obs = None

        # self.batch_size, 2
        self.blue_agent_pos = None
        self.red_agent_pos = None

        # self.batch_size,
        self.blue_agent_orientation = None
        self.red_agent_orientation = None

        # self.batch_size, 13
        self.apple_hidden = None

        # self.batch_size,
        self.blue_agent_hit = None
        self.red_agent_hit = None

        # self.batch_size,
        self.blue_agent_hidden = None
        self.red_agent_hidden = None

        #  self.batch_size, self.height, self.width
        self.beam_binary_pos = None

        # self.batch_size, 13, 2
        self.apple_pos = None

        self.view_size = view_size
        if obs_direction == "front":
            self.orientation_box = torch.stack(
                [torch.tensor([-view_size * 2, 0, -view_size, view_size]),
                 torch.tensor([-view_size, view_size, 0, view_size * 2]),
                 torch.tensor([0, view_size * 2, -view_size, view_size]),
                 torch.tensor([-view_size, view_size, -view_size * 2, 0])], dim=0).to(self.device)
        else:
            self.orientation_box = torch.stack(
                [torch.tensor([-view_size, view_size, -view_size, view_size]),
                 torch.tensor([-view_size, view_size, -view_size, view_size]),
                 torch.tensor([-view_size, view_size, -view_size, view_size]),
                 torch.tensor([-view_size, view_size, -view_size, view_size])], dim=0).to(self.device)

        self._apple_no_hidden_batch_index = None
        self._apple_hidden_batch_index = None
        self._red_agent_no_hidden_batch = None
        self._blue_agent_no_hidden_batch = None
        self._red_agent_hidden_batch = None
        self._blue_agent_hidden_batch = None

        self._red_agent_reward = None
        self._blue_agent_reward = None

        # self.batch_size,
        self._step_blue_agent_action = None
        self._step_red_agent_action = None

    def reset(self):
        self.time_step = 0
        # init actions: stay
        self._step_blue_agent_action = torch.full((self.batch_size,), fill_value=7, dtype=torch.int64).to(self.device)
        self._step_red_agent_action = torch.full((self.batch_size,), fill_value=7, dtype=torch.int64).to(self.device)

        # blue agent pos
        self.blue_agent_pos = torch.zeros((self.batch_size, 2), dtype=torch.int64).to(self.device)
        blue_agent_init_pos = torch.tensor([self.height // 2, 0], dtype=torch.int64).to(self.device)
        self.blue_agent_pos[:, :] = blue_agent_init_pos

        blue_agent_init_orientation = 1
        self.blue_agent_orientation = torch.zeros(self.batch_size, dtype=torch.int64).to(self.device)
        self.blue_agent_orientation[:] = blue_agent_init_orientation

        # red agent pos
        self.red_agent_pos = torch.zeros((self.batch_size, 2), dtype=torch.int64).to(self.device)

        red_agent_init_pos = torch.tensor([self.height // 2, self.width - 1], dtype=torch.int64).to(self.device)
        self.red_agent_pos[:, :] = red_agent_init_pos

        red_agent_init_orientation = 3
        self.red_agent_orientation = torch.zeros(self.batch_size, dtype=torch.int64).to(self.device)
        self.red_agent_orientation[:] = red_agent_init_orientation

        # apple pos
        apple_init_pos = torch.tensor(self._generate_apple_init_pos()).to(self.device)
        self.apple_pos = torch.zeros((self.batch_size, apple_init_pos.shape[0], 2), dtype=torch.int64).to(self.device)
        self.apple_pos[:, :, :] = apple_init_pos
        # agent hit or hidden
        self.blue_agent_hidden = torch.zeros(self.batch_size, dtype=torch.int64).to(self.device)
        self.blue_agent_hit = torch.zeros(self.batch_size, dtype=torch.int64).to(self.device)

        self.red_agent_hidden = torch.zeros(self.batch_size, dtype=torch.int64).to(self.device)
        self.red_agent_hit = torch.zeros(self.batch_size, dtype=torch.int64).to(self.device)
        self.apple_hidden = torch.zeros((self.batch_size, apple_init_pos.shape[0]), dtype=torch.int64).to(self.device)

        self._update_hidden_batch()
        self.beam_binary_pos = self._generate_beam_pos()

        self.blue_state, self.blue_rgb_state = self._generate_state("blue")
        self.red_state, self.red_rgb_state = self._generate_state("red")
        if not self.full_obs:
            self._blue_agent_obs, self._red_agent_obs = self._generate_obs("blue"), self._generate_obs("red")

        else:
            self._blue_agent_obs, self._red_agent_obs = self.blue_state, self.red_state
        observations = [self._blue_agent_obs, self._red_agent_obs]
        infos = []
        return observations

    def _update_hidden_batch(self):
        self._blue_agent_hidden_batch = torch.nonzero(self.blue_agent_hidden).squeeze(-1).to(self.device)
        self._red_agent_hidden_batch = torch.nonzero(self.red_agent_hidden).squeeze(-1).to(self.device)
        self._blue_agent_no_hidden_batch = torch.nonzero(self.blue_agent_hidden == 0).squeeze(-1).to(self.device)
        self._red_agent_no_hidden_batch = torch.nonzero(self.red_agent_hidden == 0).squeeze(-1).to(self.device)
        self._apple_hidden_batch_index = torch.nonzero(self.apple_hidden).squeeze(-1).to(self.device)
        self._apple_no_hidden_batch_index = torch.nonzero(self.apple_hidden == 0).squeeze(-1).to(self.device)

    def _generate_apple_init_pos(self):
        # the apple is like:
        #   *
        #  ***
        # *****
        #  ***
        #   *
        apple_pos = set()
        # center
        center_point = torch.tensor((self.height // 2, self.width // 2)).to(self.device)
        apple_pos.add(tuple(center_point.tolist()))
        # 4 directions
        for i in range(4):
            center_center_point = center_point + self.MOVES[i]
            apple_pos.add(tuple(center_center_point.tolist()))
            # 4 directions
            for j in range(4):
                center_center_center_point = center_center_point + self.MOVES[j]
                apple_pos.add(tuple(center_center_center_point.tolist()))
        return list(apple_pos)

    def _generate_beam_pos(self):
        beam_binary_pos = torch.zeros((self.batch_size, self.height, self.width), dtype=torch.int64).to(self.device)
        # blue agent
        beam_batch = torch.nonzero(self._step_blue_agent_action == 6).squeeze(-1).to(self.device)
        no_hidden_beam_batch = intersection_1d(beam_batch, self._blue_agent_no_hidden_batch)

        for i in no_hidden_beam_batch:
            if self.blue_agent_orientation[i] == 0:
                beam_end_pos_x = self.blue_agent_pos[i, 0]
                beam_start_pos_x = self.blue_agent_pos[i, 0] + self.MOVES[0, 0] * (self.blue_agent_pos[i, 0])
                beam_pos_y = self.blue_agent_pos[i, 1]
                beam_binary_pos[i, beam_start_pos_x:beam_end_pos_x, beam_pos_y] = 1
            elif self.blue_agent_orientation[i] == 1:
                beam_start_pos_y = self.blue_agent_pos[i, 1] + self.MOVES[1, 1]
                beam_end_pos_y = self.blue_agent_pos[i, 1] + self.MOVES[1, 1] * (self.width - 1 - self.blue_agent_pos[i, 1] + 1)
                beam_pos_x = self.blue_agent_pos[i, 0]
                beam_binary_pos[i, beam_pos_x, beam_start_pos_y:beam_end_pos_y] = 1
            elif self.blue_agent_orientation[i] == 2:
                beam_start_pos_x = self.blue_agent_pos[i, 0] + self.MOVES[2, 0]
                beam_end_pos_x = self.blue_agent_pos[i, 0] + self.MOVES[2, 0] * (self.height - 1 - self.blue_agent_pos[i, 0] + 1)
                beam_pos_y = self.blue_agent_pos[i, 1]
                beam_binary_pos[i, beam_start_pos_x:beam_end_pos_x, beam_pos_y] = 1
            else:
                beam_end_pos_y = self.blue_agent_pos[i, 1]
                beam_start_pos_y = self.blue_agent_pos[i, 1] + self.MOVES[3, 1] * (self.blue_agent_pos[i, 1])
                beam_pos_x = self.blue_agent_pos[i, 0]
                beam_binary_pos[i, beam_pos_x, beam_start_pos_y:beam_end_pos_y] = 1

        # self.step_red_agent_action[0] = 6

        beam_batch = torch.nonzero(self._step_red_agent_action == 6).squeeze(-1).to(self.device)
        # beam and no hidden
        no_hidden_beam_batch = intersection_1d(beam_batch, self._red_agent_no_hidden_batch)
        for i in no_hidden_beam_batch:
            if self.red_agent_orientation[i] == 0:
                beam_end_pos_x = self.red_agent_pos[i, 0]
                beam_start_pos_x = self.red_agent_pos[i, 0] + self.MOVES[0, 0] * (self.red_agent_pos[i, 0])
                beam_pos_y = self.red_agent_pos[i, 1]
                beam_binary_pos[i, beam_start_pos_x:beam_end_pos_x, beam_pos_y] = 1
            elif self.red_agent_orientation[i] == 1:
                beam_start_pos_y = self.red_agent_pos[i, 1] + self.MOVES[1, 1]
                beam_end_pos_y = self.red_agent_pos[i, 1] + self.MOVES[1, 1] * (self.width - 1 - self.red_agent_pos[i, 1] + 1)
                beam_pos_x = self.red_agent_pos[i, 0]
                beam_binary_pos[i, beam_pos_x, beam_start_pos_y:beam_end_pos_y] = 1
            elif self.red_agent_orientation[i] == 2:
                beam_start_pos_x = self.red_agent_pos[i, 0] + self.MOVES[2, 0]
                beam_end_pos_x = self.red_agent_pos[i, 0] + self.MOVES[2, 0] * (self.height - 1 - self.red_agent_pos[i, 0] + 1)
                beam_pos_y = self.red_agent_pos[i, 1]
                beam_binary_pos[i, beam_start_pos_x:beam_end_pos_x, beam_pos_y] = 1
            else:
                beam_end_pos_y = self.red_agent_pos[i, 1]
                beam_start_pos_y = self.red_agent_pos[i, 1] + self.MOVES[3, 1] * (self.red_agent_pos[i, 1])
                beam_pos_x = self.red_agent_pos[i, 0]
                beam_binary_pos[i, beam_pos_x, beam_start_pos_y:beam_end_pos_y] = 1
        return beam_binary_pos

    def _generate_state(self, agent):
        # only for test

        if agent == "blue":
            self_agent_pos = self.blue_agent_pos
            self_agent_hidden = self.blue_agent_hidden
            opponent_agent_pos = self.red_agent_pos
            opponent_agent_hidden = self.red_agent_hidden
        else:
            self_agent_pos = self.red_agent_pos
            self_agent_hidden = self.red_agent_hidden
            opponent_agent_pos = self.blue_agent_pos
            opponent_agent_hidden = self.blue_agent_hidden

        # self agent
        self_agent_no_hidden_batch = torch.nonzero(self_agent_hidden == 0).squeeze(-1).to(self.device)
        self_agent_no_hidden_batch_pos_x = self_agent_pos[self_agent_no_hidden_batch, 0]
        self_agent_no_hidden_batch_pos_y = self_agent_pos[self_agent_no_hidden_batch, 1]

        # opponent agent
        opponent_agent_no_hidden_batch = torch.nonzero(opponent_agent_hidden == 0).squeeze(-1).to(self.device)
        opponent_agent_no_hidden_batch_pos_x = opponent_agent_pos[opponent_agent_no_hidden_batch, 0]
        opponent_agent_no_hidden_batch_pos_y = opponent_agent_pos[opponent_agent_no_hidden_batch, 1]
        # apple
        # apple_no_hidden_batch: N*2,
        # where N is the number of apple_hidden==False, 2 is the dim of (batch_i, number_i)
        apple_no_hidden_batch_index = torch.nonzero(self.apple_hidden == 0).squeeze(-1).to(self.device)
        # batch (dim 0 in self.apple_pos)
        apple_no_hidden_batch = apple_no_hidden_batch_index[:, 0]
        # index in one batch (dim 1 in self.apple_pos)
        apple_no_hidden_index = apple_no_hidden_batch_index[:, 1]
        apple_non_hidden_pos_x = self.apple_pos[apple_no_hidden_batch, apple_no_hidden_index, 0]
        apple_non_hidden_pos_y = self.apple_pos[apple_no_hidden_batch, apple_no_hidden_index, 1]

        state = torch.zeros((self.batch_size, 4, self.height, self.width), dtype=torch.float32).to(self.device)
        # self agent
        state[self_agent_no_hidden_batch, 0, self_agent_no_hidden_batch_pos_x, self_agent_no_hidden_batch_pos_y] = 1
        state[opponent_agent_no_hidden_batch, 1, opponent_agent_no_hidden_batch_pos_x, opponent_agent_no_hidden_batch_pos_y] = 1
        state[apple_no_hidden_batch, 2, apple_non_hidden_pos_x, apple_non_hidden_pos_y] = 1
        # beam
        state[:, 3, :, :] = self.beam_binary_pos[:, :, :]

        # RGB state
        rgb_state = torch.zeros((self.batch_size, 3, self.height, self.width)).to(self.device)
        # green apple
        # G
        rgb_state[apple_no_hidden_batch, 1, apple_non_hidden_pos_x, apple_non_hidden_pos_y] = 1
        # self agent
        for i in range(3):
            # B
            if i == 2:
                rgb_state[self_agent_no_hidden_batch, i, self_agent_no_hidden_batch_pos_x, self_agent_no_hidden_batch_pos_y] = 1
            else:
                rgb_state[self_agent_no_hidden_batch, i, self_agent_no_hidden_batch_pos_x, self_agent_no_hidden_batch_pos_y] = 0
        # opponent agent
        for i in range(3):
            # R
            if i == 0:
                rgb_state[opponent_agent_no_hidden_batch, i, opponent_agent_no_hidden_batch_pos_x, opponent_agent_no_hidden_batch_pos_y] = 1
            else:
                rgb_state[opponent_agent_no_hidden_batch, i, opponent_agent_no_hidden_batch_pos_x, opponent_agent_no_hidden_batch_pos_y] = 0

        # conflict agent
        # pos: conflict agent
        conflict_agent_batch = torch.nonzero(torch.all(self_agent_pos == opponent_agent_pos, dim=1)).squeeze(-1).to(self.device)
        # pos: both agents no hidden
        both_agent_no_hidden = torch.nonzero((~self_agent_hidden) & (~opponent_agent_hidden)).squeeze(-1).to(self.device)
        # pos: intersection
        conflict_agent_batch = intersection_1d(conflict_agent_batch, both_agent_no_hidden)
        conflict_agent_pos_x = self_agent_pos[conflict_agent_batch, 0]
        conflict_agent_pos_y = self_agent_pos[conflict_agent_batch, 1]
        # print(conflict_agent_batch)
        # conflict agent: 0.5 red + 0.5 blue
        if conflict_agent_batch.numel() > 0:
            for i in range(3):
                if i == 1:
                    rgb_state[conflict_agent_batch, i, conflict_agent_pos_x, conflict_agent_pos_y] = 0
                else:
                    # 0.5R, 0.5 B
                    rgb_state[conflict_agent_batch, i, conflict_agent_pos_x, conflict_agent_pos_y] = 0.5

        # beam: gray
        beam_pos = torch.nonzero(self.beam_binary_pos)
        # print(beam_pos)
        for i in range(3):
            rgb_state[beam_pos[:, 0], i, beam_pos[:, 1], beam_pos[:, 2]] = 0.5
        return state, rgb_state

    def _generate_obs(self, agent):
        padding_width = self.view_size * 2
        if agent == "blue":
            state = self.blue_state
            orientation = self.blue_agent_orientation
            pos = self.blue_agent_pos
        else:
            state = self.red_state
            orientation = self.red_agent_orientation
            pos = self.red_agent_pos
        padding_state = F.pad(state, (padding_width, padding_width, padding_width, padding_width), value=0)
        x_min = padding_width + self.orientation_box[orientation][:, 0] + pos[:, 0]
        x_max = padding_width + self.orientation_box[orientation][:, 1] + pos[:, 0]
        y_min = padding_width + self.orientation_box[orientation][:, 2] + pos[:, 1]
        y_max = padding_width + self.orientation_box[orientation][:, 3] + pos[:, 1]
        observation = torch.zeros((self.batch_size, padding_state.shape[1], self.view_size * 2 + 1, self.view_size * 2 + 1)).to(self.device)
        # it seems only to iterate
        for i in range(self.batch_size):
            observation[i] = padding_state[i, :, x_min[i]:x_max[i] + 1, y_min[i]:y_max[i] + 1]
            observation[i] = torch.rot90(observation[i], k=orientation[i], dims=(1, 2))
        return observation

    def _respawn(self):
        # respawn
        self.blue_agent_hidden[self._blue_agent_hidden_batch] -= 1
        self.blue_agent_hidden = torch.clamp(self.blue_agent_hidden, min=0)
        self.red_agent_hidden[self._red_agent_hidden_batch] -= 1
        self.red_agent_hidden = torch.clamp(self.red_agent_hidden, min=0)

        apple_hidden_batch = self._apple_hidden_batch_index[:, 0]
        apple_hidden_index = self._apple_hidden_batch_index[:, 1]
        self.apple_hidden[apple_hidden_batch, apple_hidden_index] -= 1
        self.apple_hidden = torch.clamp(self.apple_hidden, min=0)

        self._update_hidden_batch()

    def _move(self):
        # blue agent
        blue_agent_move_batch = torch.nonzero(self._step_blue_agent_action < 4).squeeze(-1).to(self.device)
        move_no_hidden_batch = intersection_1d(blue_agent_move_batch, self._blue_agent_no_hidden_batch)

        # blue_move_no_hidden_batch
        self.blue_agent_pos[move_no_hidden_batch] += self.MOVES[(self._step_blue_agent_action[move_no_hidden_batch] + self.blue_agent_orientation[move_no_hidden_batch]) % 4]
        self.blue_agent_pos[:, 0] = torch.clamp(self.blue_agent_pos[:, 0], min=0, max=self.height - 1)
        self.blue_agent_pos[:, 1] = torch.clamp(self.blue_agent_pos[:, 1], min=0, max=self.width - 1)

        # red agent
        red_agent_move_batch = torch.nonzero(self._step_red_agent_action < 4).squeeze(-1).to(self.device)
        move_no_hidden_batch = intersection_1d(red_agent_move_batch, self._red_agent_no_hidden_batch)
        self.red_agent_pos[move_no_hidden_batch] += self.MOVES[(self._step_red_agent_action[move_no_hidden_batch] + self.red_agent_orientation[move_no_hidden_batch]) % 4]
        self.red_agent_pos[:, 0] = torch.clamp(self.red_agent_pos[:, 0], min=0, max=self.height - 1)
        self.red_agent_pos[:, 1] = torch.clamp(self.red_agent_pos[:, 1], min=0, max=self.width - 1)

    def _turn(self):
        # blue
        blue_agent_turn_batch = torch.nonzero((self._step_blue_agent_action >= 4) & (self._step_blue_agent_action < 6)).squeeze(-1).to(self.device)
        turn_no_hidden_batch = intersection_1d(blue_agent_turn_batch, self._blue_agent_no_hidden_batch)
        self.blue_agent_orientation[turn_no_hidden_batch] = (self.blue_agent_orientation[turn_no_hidden_batch] + self.TURNS[self._step_blue_agent_action[turn_no_hidden_batch] - 4]) % 4
        # red
        red_agent_turn_batch = torch.nonzero((self._step_red_agent_action >= 4) & (self._step_red_agent_action < 6)).squeeze(-1).to(self.device)
        turn_no_hidden_batch = intersection_1d(red_agent_turn_batch, self._red_agent_no_hidden_batch)
        self.red_agent_orientation[turn_no_hidden_batch] = (self.red_agent_orientation[turn_no_hidden_batch] + self.TURNS[self._step_red_agent_action[turn_no_hidden_batch] - 4]) % 4

    def _beam(self):
        self.beam_binary_pos = self._generate_beam_pos().to(self.device)

    def _eat_apple(self):
        # blue
        # (batch,index) for apple where there is an agent no hidden and some apple no hidden at the batch
        blue_apple_and_agent_no_hidden_batch_index = self._apple_no_hidden_batch_index[torch.isin(self._apple_no_hidden_batch_index[:, 0], self._blue_agent_no_hidden_batch)]
        # get batch from (batch,index)
        blue_apple_and_agent_no_hidden_batch = blue_apple_and_agent_no_hidden_batch_index[:, 0]
        blue_apple_and_agent_no_hidden_index = blue_apple_and_agent_no_hidden_batch_index[:, 1]

        # add a dim for agent_pos to make it's dim and apple pos 's dim equal to compare them
        agent_pos_compare = self.blue_agent_pos.unsqueeze(1)
        # find the batch where an agent's pos is equal to any apple's pos
        blue_agent_apple_match = ((self.apple_pos[blue_apple_and_agent_no_hidden_batch, blue_apple_and_agent_no_hidden_index, 0] == agent_pos_compare[blue_apple_and_agent_no_hidden_batch, 0, 0])
                                  & (self.apple_pos[blue_apple_and_agent_no_hidden_batch, blue_apple_and_agent_no_hidden_index, 1] == agent_pos_compare[blue_apple_and_agent_no_hidden_batch, 0, 1]))

        blue_agent_apple_match_batch = blue_apple_and_agent_no_hidden_batch[blue_agent_apple_match]
        blue_agent_apple_match_index = blue_apple_and_agent_no_hidden_index[blue_agent_apple_match]

        # agent_apple_match_batch = apple_and_agent_no_hidden_batch[torch.any(agent_apple_match, dim=1)]
        # hide apple
        self.apple_hidden[blue_agent_apple_match_batch, blue_agent_apple_match_index] = self.n_apple
        self._blue_agent_reward[blue_agent_apple_match_batch] += 1

        # red, all is the same as blue
        red_apple_and_agent_no_hidden_batch_index = self._apple_no_hidden_batch_index[torch.isin(self._apple_no_hidden_batch_index[:, 0], self._red_agent_no_hidden_batch)]
        red_apple_and_agent_no_hidden_batch = red_apple_and_agent_no_hidden_batch_index[:, 0]
        red_apple_and_agent_no_hidden_index = red_apple_and_agent_no_hidden_batch_index[:, 1]

        agent_pos_compare = self.red_agent_pos.unsqueeze(1)
        agent_apple_match = ((self.apple_pos[red_apple_and_agent_no_hidden_batch, red_apple_and_agent_no_hidden_index, 0] == agent_pos_compare[red_apple_and_agent_no_hidden_batch, 0, 0])
                             & (self.apple_pos[red_apple_and_agent_no_hidden_batch, red_apple_and_agent_no_hidden_index, 1] == agent_pos_compare[red_apple_and_agent_no_hidden_batch, 0, 1]))
        # print(agent_apple_match)
        red_agent_apple_match_batch = red_apple_and_agent_no_hidden_batch[agent_apple_match]
        red_agent_apple_match_index = red_apple_and_agent_no_hidden_index[agent_apple_match]

        self.apple_hidden[red_agent_apple_match_batch, red_agent_apple_match_index] = self.n_apple
        self._red_agent_reward[red_agent_apple_match_batch] += 1

        # find overlapping agent
        overlapping_agent_apple_match_batch = intersection_1d(blue_agent_apple_match_batch, red_agent_apple_match_batch)
        conflict_agent_batch = torch.nonzero(torch.all(self.blue_agent_pos == self.red_agent_pos, dim=1)).squeeze(-1).to(self.device)
        overlapping_agent_apple_match_batch = intersection_1d(overlapping_agent_apple_match_batch, conflict_agent_batch)
        self._red_agent_reward[overlapping_agent_apple_match_batch] = 0.5
        self._blue_agent_reward[overlapping_agent_apple_match_batch] = 0.5

        self._update_hidden_batch()

    def _hit(self):
        blue_agent_no_hidden_pos_x = self.blue_agent_pos[self._blue_agent_no_hidden_batch, 0]
        blue_agent_no_hidden_pos_y = self.blue_agent_pos[self._blue_agent_no_hidden_batch, 1]
        blue_agent_hit_batch = self._blue_agent_no_hidden_batch[self.beam_binary_pos[self._blue_agent_no_hidden_batch, blue_agent_no_hidden_pos_x, blue_agent_no_hidden_pos_y] == 1]
        self.blue_agent_hidden[blue_agent_hit_batch] = self.n_tagged

        red_agent_no_hidden_pos_x = self.red_agent_pos[self._red_agent_no_hidden_batch, 0]
        red_agent_no_hidden_pos_y = self.red_agent_pos[self._red_agent_no_hidden_batch, 1]
        red_agent_hit_batch = self._red_agent_no_hidden_batch[self.beam_binary_pos[self._red_agent_no_hidden_batch, red_agent_no_hidden_pos_x, red_agent_no_hidden_pos_y] == 1]
        self.red_agent_hidden[red_agent_hit_batch] = self.n_tagged

        self._update_hidden_batch()

    def step(self, actions):
        self.time_step += 1
        step_blue_agent_action, step_red_agent_action = actions
        self._step_blue_agent_action = step_blue_agent_action.to(self.device)
        self._step_red_agent_action = step_red_agent_action.to(self.device)

        done = torch.full((self.batch_size,), fill_value=self.time_step == self.max_step).to(self.device)
        self._blue_agent_reward = torch.zeros(self.batch_size).to(self.device)
        self._red_agent_reward = torch.zeros(self.batch_size).to(self.device)

        # apple respawn
        self._respawn()

        # agent move
        self._move()

        # turn
        self._turn()

        # beam
        self._beam()

        # eat apple
        self._eat_apple()

        # hit by beam
        self._hit()

        # self.blue_state, self.red_state = self._generate_state("blue"), self._generate_state("red")
        # self.blue_rgb_state, self.red_rgb_state = self._generate_rgb_state("blue"), self._generate_rgb_state("red")
        self.blue_state, self.blue_rgb_state = self._generate_state("blue")
        self.red_state, self.red_rgb_state = self._generate_state("red")

        if not self.full_obs:
            self._blue_agent_obs, self._red_agent_obs = self._generate_obs("blue"), self._generate_obs("red")
        else:
            self._blue_agent_obs, self._red_agent_obs = self.blue_state, self.red_state

        observations = [self._blue_agent_obs, self._red_agent_obs]
        rewards = [self._blue_agent_reward, self._red_agent_reward]
        dones = [done, done]
        infos = []
        return observations, rewards, dones, infos
