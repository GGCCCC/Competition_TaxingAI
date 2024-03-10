import numpy as np
# import pdb
# pdb.set_trace()

from env.env_core import economic_society
from agents.rule_based import rule_agent
# from agents.pg import pg_agent
from agents.independent_ppo import ppo_agent
from agents.calibration import calibration_agent
from agents.BMFAC import BMFAC_agent
from agents.MADDPG.MAAC import maddpg_agent as maddpg
from agents.MADDPG_block.MAAC import maddpg_agent as maddpg_block
from agents.MADDPG_attention.MAAC import maddpg_agent as maddpg_attention
from agents.MADDPG_attention2.MAAC import maddpg_agent as maddpg_attention2
from utils.seeds import set_seeds
from arguments import get_args
import os
import torch
import yaml
import argparse
from omegaconf import OmegaConf

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='default')
    parser.add_argument("--alg", type=str, default='ac', help="ac, rule_based, independent, maddpg-block")
    parser.add_argument("--task", type=str, default='gdp_gini', help="gini, social_welfare, gdp_gini")
    parser.add_argument('--device-num', type=int, default=1, help='the number of cuda service num')
    parser.add_argument('--n_households', type=int, default=100, help='the number of total households')
    parser.add_argument('--seed', type=int, default=42, help='the random seed')
    parser.add_argument('--hidden_size', type=int, default=128, help='[64, 128, 256]')
    parser.add_argument('--q_lr', type=float, default=3e-4, help='[3e-3, 3e-4, 3e-5]')
    parser.add_argument('--p_lr', type=float, default=3e-4, help='[3e-3, 3e-4, 3e-5]')
    parser.add_argument('--lr', type=float, default=0, help='none 0 for both q_lr & p_lr')
    parser.add_argument('--batch_size', type=int, default=32, help='[32, 64, 128, 256]')
    parser.add_argument('--update_cycles', type=int, default=50, help='[10,100,1000]')
    parser.add_argument('--update_freq', type=int, default=30, help='[10,20,30]')
    parser.add_argument('--initial_train', type=int, default=10, help='[10,100,200]')

    args = parser.parse_args()
    if args.lr > 0:
        args.__setattr__("q_lr", args.lr)
        args.__setattr__("p_lr", args.lr)
    return args



def tuning(cfg):
    # maddpg
    hidden_size_list = [64, 128, 256]
    lr_list = [3e-3, 3e-4, 3e-5]
    batch_size_list = [32, 64, 128, 256]
    for hidden_i in hidden_size_list:
        for lr_i in lr_list:
            for batch_i in batch_size_list:
                cfg.hidden_size=hidden_i
                cfg.q_lr = lr_i
                cfg.p_lr = lr_i
                cfg.batch_size = batch_i

    return cfg

if __name__ == '__main__':
    # set signle thread
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    args = parse_args()
    path = args.config
    yaml_cfg = OmegaConf.load("/root/Competition_TaxingAI/TaxAI/cfg/cc.yaml")
    yaml_cfg.Trainer["n_households"] = args.n_households
    yaml_cfg.Environment.Entities[1]["entity_args"].n = args.n_households
    yaml_cfg.Environment.env_core["env_args"].gov_task = args.task
    yaml_cfg.seed = args.seed
    
    '''tuning'''
    # tuning(yaml_cfg)
    yaml_cfg.Trainer["hidden_size"] = args.hidden_size
    yaml_cfg.Trainer["q_lr"] = args.q_lr
    yaml_cfg.Trainer["p_lr"] = args.p_lr
    yaml_cfg.Trainer["batch_size"] = args.batch_size
    
    set_seeds(yaml_cfg.seed, cuda=yaml_cfg.Trainer["cuda"])
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
    env = economic_society(yaml_cfg.Environment)

    if args.alg == "rule_based":    # rule_based
        trainer = rule_agent(env, yaml_cfg.Trainer)
    # if args.alg == "pg":            # rule_based policy-gradient
    #     trainer = pg_agent(env, yaml_cfg.Trainer)
    elif args.alg == "ppo":         # independent
        trainer = ppo_agent(env, yaml_cfg.Trainer)
    elif args.alg == "maddpg":      # add central critic
        trainer = maddpg(env, yaml_cfg.Trainer)
    elif args.alg == "bmfac":       # 2 batch share critic
        trainer = BMFAC_agent(env, yaml_cfg.Trainer)
    elif args.alg == "maddpgb":     # 4 batch (split by households wealth)
        trainer = maddpg_block(env, yaml_cfg.Trainer)
    elif args.alg == "maddpga":     # add attention layer
        trainer = maddpg_attention(env, yaml_cfg.Trainer)
    elif args.alg == "maddpga2":
        trainer = maddpg_attention2(env, yaml_cfg.Trainer)
    else:
        # undo agent
        trainer = calibration_agent(env, yaml_cfg.Trainer)
    # start to learn
    print("n_households: ", yaml_cfg.Trainer["n_households"])
    trainer.learn()
    # trainer.test()
    # # close the environment
    # env.close()


