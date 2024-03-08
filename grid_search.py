# -*- coding:utf-8  -*-
import os
import random
import time
import json
import numpy as np
import argparse
import sys

sys.path.append("./olympics_engine")

from env.chooseenv import make
from env.obs_interfaces.observation import obs_type

class JointActor:
    x = 0
    y = 0
    
    def set_params(x, y):
        JointActor.x = x
        JointActor.y = y
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def my_controller(observation, action_space, is_act_continuous=False):
        agent_action = [np.array([[JointActor.y, JointActor.y] for _ in range(4)], dtype=np.float32)]
        return agent_action


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_players_and_action_space_list(g):
    if sum(g.agent_nums) != g.n_player:
        raise Exception("agent number = %d 不正确，与n_player = %d 不匹配" % (sum(g.agent_nums), g.n_player))

    n_agent_num = list(g.agent_nums)
    for i in range(1, len(n_agent_num)):
        n_agent_num[i] += n_agent_num[i - 1]

    # 根据agent number 分配 player id
    players_id = []
    actions_space = []
    for policy_i in range(len(g.obs_type)):
        if policy_i == 0:
            players_id_list = range(n_agent_num[policy_i])
        else:
            players_id_list = range(n_agent_num[policy_i - 1], n_agent_num[policy_i])
        players_id.append(players_id_list)

        action_space_list = [g.get_single_action_space(player_id) for player_id in players_id_list]
        actions_space.append(action_space_list)

    return players_id, actions_space


def get_joint_action_eval(game, multi_part_agent_ids, policy_list, actions_spaces, all_observes):
    if len(policy_list) != len(game.agent_nums):
        error = "模型个数%d与玩家个数%d维度不正确！" % (len(policy_list), len(game.agent_nums))
        raise Exception(error)
    joint_action = []
    for policy_i in range(len(policy_list)):
        if game.obs_type[policy_i] not in obs_type:
            raise Exception("可选obs类型：%s" % str(obs_type))
        agents_id_list = multi_part_agent_ids[policy_i]
        action_space_list = actions_spaces[policy_i]
        for i in range(len(agents_id_list)):
            agent_id = agents_id_list[i]
            a_obs = all_observes[agent_id]
            each = JointActor.my_controller(a_obs, action_space_list[i], game.is_act_continuous)
            joint_action.append(each)
    return joint_action

def run_game(g, env_name, multi_part_agent_ids, actions_spaces, policy_list, render_mode):
    all_observes = g.all_observes
    while not g.is_terminal():
        if render_mode and hasattr(g, "env_core"):
            if hasattr(g.env_core, "render"):
                g.env_core.render()
        elif render_mode and hasattr(g, 'render'):
            g.render()
        joint_act = get_joint_action_eval(g, multi_part_agent_ids, policy_list, actions_spaces, all_observes)
        all_observes, reward, done, info_before, info_after = g.step(joint_act)
    return np.mean(g.n_return)


def timeit(func):
    """
    用于函数计时的装饰器
    """
    def wrapper(*args, **kwargs):
        nowTime = time.perf_counter_ns()
        ret = func(*args, **kwargs)
        duration = time.perf_counter_ns() - nowTime
        duration = round(duration / 1000000, 5)
        print(f'{func.__name__} 耗时 {duration} ms')
        return ret
    return wrapper

if __name__ == "__main__":
    import gc
    import copy
    import warnings
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=UserWarning, module="box.py", lineno=61)
    
    log_file = open("log", "a")
    
    env_type = "taxing_household"   #"taxing_household" "taxing_gov"
    # env_type = "taxing_gov"   #"taxing_household" "taxing_gov"
    # env_type = "olympics"   #"taxing_household" "taxing_gov"
    game = make(env_type, seed=None)

    render_mode = False

    _policy_list = list(range(4))
    policy_list = _policy_list[:len(game.agent_nums)]
    multi_part_agent_ids, actions_space = get_players_and_action_space_list(game)
    
    def step(x, y):
        scores = []
        for i in range(30):
            seed = random.randrange(1000)
            game.reset()
            game.set_seed(seed)
            JointActor.set_params(x, y)
            score = run_game(game, env_type, multi_part_agent_ids, actions_space, policy_list, render_mode)
            scores.append(score)
            time.sleep(0)
            # gc.collect()
        mean_score = np.mean(scores)
        # print(f"params:({x:.3f},{y:.3f}) step score={mean_score:.2f}")
        with open("log", "a") as log_file:
            log_file.write(f"params:({x:.3f},{y:.3f}) step score={mean_score:.2f}\n")
        return np.mean(mean_score)
    
    
    
    from skopt import Optimizer
    from skopt.space import Real

    # 定义目标函数
    def objective(params):
        x, y = params
        return -step(x, y)  # 这里使用 -step 是因为我们要最小化损失

    # 定义参数空间
    param_space = [Real(-1, 1, name='x'), Real(-1, 1, name='y')]

    # 创建贝叶斯优化器，初始点设为网格搜索得到的最优参数
    opt = Optimizer(dimensions=param_space, random_state=0, base_estimator="gp")
    opt.tell([(-1, -0.333)], objective([-1, -0.333]))  # 初始点

    # 进行贝叶斯优化
    for i in range(20):  # 迭代次数
        suggested_params = opt.ask()
        y = objective(suggested_params)
        opt.tell(suggested_params, y)

    # 输出最优参数
    print("Best parameters found: ", opt.Xi[np.argmin(opt.yi)])
