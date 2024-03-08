import random
import os
import sys
from pathlib import Path
CURRENT_PATH = str(Path(__file__).resolve().parent.parent.parent)
taxing_path = os.path.join(CURRENT_PATH)
sys.path.append(taxing_path)

import numpy as np
import importlib

from TaxAI.env.env_core import economic_society
from omegaconf import OmegaConf

from utils.box import Box
from env.simulators.game import Game

__all__ = ['Taxing_Household']

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.agent_name = 'Random'
    def __call__(self, observation):
        return [self.action_space[0].sample()]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.action_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = F.softplus(self.fc3(x))
        actions = torch.tanh(self.action_out(x))
        return actions
    
    def sample(self, state, num_actions=4):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
            if self.action_out.weight.is_cuda:
                state = state.cuda()
        state = state.unsqueeze(0)
        actions = []
        with torch.no_grad():
            action = self.forward(state)
            for _ in range(num_actions):
                actions.append(action.squeeze(0).cpu().numpy())
        return np.array(actions)
    
class Critic(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        # self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        # self.ln3 = nn.LayerNorm(hidden_size)
        self.q_out = nn.Linear(hidden_size, 1)
        # self.apply(weight_init)

        # self.initialize_weights()
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        # x = F.softplus(self.ln1(self.fc1(x)))
        # x = F.softplus(self.ln2(self.fc2(x)))
        # x = F.softplus(self.ln3(self.fc3(x)))
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = F.softplus(self.fc3(x))
        q_value = self.q_out(x)
        return q_value

param_dict = torch.load("/root/Competition_TaxingAI/agents/g/25.pth", map_location=torch.device('cpu'))
input_size = param_dict["fc1.weight"].size()[1]
output_size = param_dict["action_out.weight"].size()[0]
hidden_size = param_dict["fc1.weight"].size()[0]
gov_actor = Actor(input_size, output_size, hidden_size)
gov_actor.load_state_dict(param_dict)
gov_actor.eval()  # 设置网络为评估模式

class GovAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.agent_name = 'Random'
    def __call__(self, observation):
        global gov_actor
        
        raw_obs = observation['obs']['raw_obs']
        if self.action_space[0].shape==(4,2):
            raw_obs = np.concatenate(raw_obs, axis=-1)
            action = gov_actor.sample(raw_obs, num_actions=4)
            return [np.array(action)]
        else:
            action = gov_actor.sample(raw_obs, num_actions=1)
            return np.array(action)

class Taxing_Household(Game):
    def __init__(self, conf, seed=None):
        super(Taxing_Household, self).__init__(conf['n_player'], conf['is_obs_continuous'], conf['is_act_continuous'],
                                               conf['game_name'], conf['agent_nums'], conf['obs_type'])
        self.seed = seed
        self.set_seed()

        # yaml_path = os.path.join(CURRENT_PATH, '/root/Competition_TaxingAI/TaxAI/cfg/n4.yaml')
        # yaml_cfg = OmegaConf.load(yaml_path)
        yaml_cfg = OmegaConf.load("/root/Competition_TaxingAI/TaxAI/cfg/n4.yaml")
        self.env_core = economic_society(yaml_cfg.Environment)
        self.max_step = int(conf['max_step'])       #TODO: check max step in the env_core

        self.controllable_agent_id = 'Household'

        self.agent_id = list(self.env_core.action_spaces.keys())
        self.joint_action_space = {self.controllable_agent_id :self.set_action_space()[self.controllable_agent_id]}
        self.action_dim = self.joint_action_space
        self.each_gov_spaces = Box(np.array([-1., -1., -1, -1, -1]),
                                         np.array([1.,1., 1., 1., 1.]))
        self.n_households = self.env_core.households.n_households
        self.n_gov = 1
        
        self.total_r = 0
        self.gov_reward = 0

        # self.sub_controller = [RandomAgent([self.each_gov_spaces])]
        self.sub_controller = [GovAgent([self.each_gov_spaces])]

        self.reset()
        self.init_info = {'Controllable': self.controllable_agent_id,
                          'Opponent': [i.agent_name for i in self.sub_controller]}

    @staticmethod
    def create_seed():
        seed = random.randrange(1000)
        return seed

    def set_seed(self, seed=None):
        if not seed:        #use previous seed when no new seed input
            seed = self.seed
        else:               #update env global seed
            self.seed = seed
        random.seed(seed)
        np.random.seed(seed)


    def set_action_space(self):
        a_s = {}
        for aid, space in self.env_core.action_spaces.items():
            a_s[aid] = [space]

        return a_s

    def get_single_action_space(self, player_id):
        player_name = self.controllable_agent_id
        return self.joint_action_space[player_name]

    def reset(self):
        global_obs, private_obs = self.env_core.reset()
        self.step_cnt = 0
        self.done = False
        self.init_info = None
        self.won = {}
        self.n_return = [0]*self.n_player
        self.gov_reward = 0
        self.total_r = np.zeros(self.n_player)

        self.current_state = (global_obs, private_obs)
        self.all_observes = self.get_all_observes()
        return self.all_observes

    def step(self, joint_action):
        self.is_valid_action(joint_action)
        joint_action_decode = self.decode(joint_action)
        info_before = {"actions": joint_action_decode}

        global_obs, private_obs, gov_r, house_r, done = self.env_core.step(joint_action_decode)
        info_after = self.step_after_info()
        self.current_state = (global_obs, private_obs)
        self.all_observes = self.get_all_observes()

        self.total_r += house_r[:,0]

        self.step_cnt += 1
        self.done = done
        self.set_n_return()
        
        # # gcc：尝试奖励长期政府
        gov_r = 1 / (1 + np.exp(-gov_r))
        self.gov_reward += gov_r #dict(zip(self.agent_id, [gov_r, house_r]))
        
        if self.done:
            print('total gov_return = ', self.gov_reward)
            print('total house_return = ', self.n_return)
            with open("/root/Competition_TaxingAI/hh.csv", "a") as f:
                f.write(f"{self.gov_reward},{self.n_return[0]},{self.n_return[1]},{self.n_return[2]},{self.n_return[3]}" + "\n")

        return self.all_observes, gov_r, self.done, info_before, info_after

    def is_valid_action(self, joint_action):
        if len(joint_action) != self.n_player:          #check number of player
            raise Exception("Input joint action dimension should be {}, not {}".format(
                self.n_player, len(joint_action)))

    def decode(self, joint_action):
        joint_action_decode = {}
        joint_action_decode[self.controllable_agent_id] = joint_action[0][0]
        gov_action = None
        for gov_idx, house_obs in enumerate(self.gov_obs):
            _action = self.sub_controller[gov_idx](house_obs)[0]
            gov_action= _action
        # gov_action = np.array(gov_action)

        joint_action_decode['government'] = gov_action # self.sub_controller(self.household_obs[0])[0]
        return joint_action_decode


    def get_all_observes(self):
        all_observes = []
        global_obs, private_obs = self.current_state
        #global obs is for government, private_obs+global_obs are for each household
        gov_obs = []

        for idx, aid in enumerate(self.agent_id):
            if aid == 'government':
                _gov_obs = {'obs': {"agent_id": aid, "raw_obs": global_obs},
                           "controlled_player_index": idx}
                gov_obs.append(_gov_obs)
            elif aid == 'Household':
                for house_idx in range(private_obs.shape[0]):
                    each_private_obs = private_obs[house_idx]
                    _obs = {"obs": {"agent_id": aid, 'raw_obs': [global_obs, each_private_obs],
                                    "controlled_player_index": house_idx}}

                # _obs = {"obs": {"agent_id": aid, "raw_obs": [global_obs, private_obs]},
                #              "controlled_player_index": idx}
                    all_observes.append(_obs)
            else:
                raise NotImplementedError

        self.gov_obs = gov_obs

        return all_observes

    def step_after_info(self):
        current_step = self.env_core.step_cnt
        social_welfare = self.env_core.households_reward.mean()
        wealth_gini = self.env_core.wealth_gini
        income_gini = self.env_core.income_gini
        gdp = self.env_core.GDP

        tau = self.env_core.government.tau
        xi = self.env_core.government.xi
        tau_a = self.env_core.government.tau_a
        xi_a = self.env_core.government.xi_a
        Gt_prob = self.env_core.Gt_prob
        return {'step': current_step, 'social_welfare': social_welfare, 'wealth_gini': wealth_gini,
                'income_gini': income_gini, 'gdp': gdp, 'tau': tau, 'xi': xi, 'tau_a': tau_a, 'xi_a': xi_a,
                "Gt_prob": Gt_prob}
    def is_terminal(self):
        return self.done

    def check_win(self):
        return '-1'

    def set_n_return(self):
        self.n_return = list(self.total_r)

