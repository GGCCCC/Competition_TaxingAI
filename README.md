<img src="imgs/Jidi%20logo.png" width='300px'> 

# CCF 2023 Taxing AI Competition 

This repo provide the source code for the [CCF 2023 Taxing AI Competition ](http://www.jidiai.cn/compete_detail?compete=42)



## Multi-Agent Game Evaluation Platform --- Jidi (及第)
Jidi supports online evaluation service for various games/simulators/environments/testbeds. Website: [www.jidiai.cn](www.jidiai.cn).

A tutorial on Jidi: [Tutorial](https://github.com/jidiai/ai_lib/blob/master/assets/Jidi%20tutorial.pdf)


## Environment
The competition adopts a Taxing simulator [TaxingAI](https://github.com/jidiai/git). A brief description can be found on [JIDI](http://www.jidiai.cn/env_detail?envid=99).
A complementary document is also presented in [docs](./docs/).





## Quick Start

You can use any tool to manage your python environment. Here, we use conda as an example.

```bash
conda create -n taxingai-venv python==3.7.5  #3.8, 3.9
conda activate taxingai-venv
```

Next, clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/jidiai/Competition_TaxingAI.git
cd Competition_TaxingAI
pip install -r requirements.txt
```

Finally, run the game by executing:
```bash
python run_log.py
```

## Navigation

```
|-- Competition_OvercookedAI               
	|-- agents                              // Agents that act in the environment
	|	|-- random                      // A random agent demo
	|	|	|-- submission.py       // A ready-to-submit random agent file
	|-- env		                        // scripts for the environment
	|	|-- config.py                   // environment configuration file
	|	|-- taxing_gov.py               // The environment wrapper for taxing_gov env	
	|   |-- taxing_household.py             // The environment wrapper for taxing_household env	      
	|-- utils               
	|-- run_log.py		                // run the game with provided agents (same way we evaluate your submission in the backend server)
```



## How to test submission

- You can train your own agents using any framework you like as long as using the provided environment wrapper. 

- For your ready-to-submit agent, make sure you check it using the ``run_log.py`` scrips, which is exactly how we 
evaluate your submission.

- ``run_log.py`` takes agents from path `agents/` and run a game. For example:

>python run_log.py --my_ai "random" --opponent "random"

set both agents as a random policy and run a game.

- You can put your agents in the `agent/` folder and create a `submission.py` with a `my_controller` function 
in it. Then run the `run_log.py` to test:

>python run_log.py --my_ai your_agent_name --opponent xxx

- If you pass the test, then you can submit it to the Jidi platform. You can make multiple submission and the previous submission will
be overwritten.


