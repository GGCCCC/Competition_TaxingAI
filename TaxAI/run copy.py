import itertools
from multiprocessing.pool import Pool
import os
import subprocess
import time

def dict_to_str(param_dict):
    return ' '.join([f'--{k} {v}' for k, v in param_dict.items()])

if __name__ == "__main__":
    task = ["gdp", "gdp_gini"]
    params = {
        "n_household": [10],
        "hidden_size": [128],
        "batch_size": [32],
        "task": [task[1]],
        "lr": [3e-4, 34-5]
    }
    algs = ["ppo", "bmfac", "maddpg", "maddpgb", "maddpga"]

    param_combinations = list(itertools.product(*params.values()))

    results = []
    with Pool(processes=12) as pool:
        for alg in algs:
            for param_combo in param_combinations:
                time.sleep(1)
                params_dict = dict(zip(params.keys(), param_combo))
                params_dict["alg"] = alg
                cmd = "python3 main.py " + dict_to_str(params_dict)
                print(cmd)
                result = pool.apply_async(os.system, (cmd,))
                results.append(result)
        pool.close()
        pool.join()

    # get results
    for result in results:
        result.get()