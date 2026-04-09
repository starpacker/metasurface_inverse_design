这里是对于多智能体/单智能体的代码需要改的地方
1. train.py  parse_args   num_agents
2. env_core.py  init 初始化中action_dim 和 num_agents 都需要调整
3. algorithms ACTlayer  BOX action_dim = 10
这里直接改了util.py里面的act_shape

python train/train.py 可以直接运行
需要将fsp和lsf文件放置在另外一个文件夹中。