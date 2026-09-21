# 游戏规则

- 4 只鹅在 `7 × 11` 的环形棋盘上同时行动。
- 动作：`NORTH / EAST / SOUTH / WEST`。
- 棋盘上下、左右相连，例如从最上面向北会到最下面。
- 不能立即反向；例如上一回合向东，本回合不能向西。
- 吃到食物后身体增长，并补充食物，棋盘至少维持 2 个食物。
- 每 40 步额外缩短一节，长度减到 0 就饿死。
- 头撞到任何仍占用的身体格会死亡，包括自己。
- 多只鹅的头同时进入同一格会发生碰撞。两只鹅同时死亡。
- 最多 200 回合，目标首先是存活得更久，其次是保持更长。
- 每回合提供完整信息，并非部分可观测环境。

官方介绍见 [Hungry Geese Overview](https://www.kaggle.com/competitions/hungry-geese/overview)。

## Observation

Agent 的典型输入是：

```python
observation = {
    "index": 0,               # 当前 agent 对应的玩家编号
    "step": 37,               # 当前回合
    "geese": [
        [12, 13, 14],         # 玩家 0，首元素是头
        [42, 31],
        [],
        [70, 69, 68]
    ],
    "food": [5, 61],
    "remainingOverageTime": 60
}
```

位置采用一维整数编码：

```python
row = position // columns
col = position % columns
position = row * columns + col
```

在默认 11 列棋盘中：

```text
0  1  2 ... 10
11 12 13 ... 21
...
66 67 68 ... 76
```

需要特别注意：

- `geese[i][0]` 是第 `i` 只鹅的头。
- 数组最后一项是尾巴。
- `geese[i] == []` 表示该玩家已经死亡。
- 观测中没有可靠的“上一动作”字段，通常通过前后两帧头部位置推断方向。
- 因为棋盘是环面，推断移动必须使用取模运算。


## Agent

```python
from kaggle_environments import make, evaluate
from kaggle_environments.envs.hungry_geese.hungry_geese import *

class SimpleAgent:
    def __call__(self, observation, configuration): # no need to input params yourself, Kaggle will do it
        # code here
        return "NORTH" # ACTIONS = ["NORTH", "SOUTH", "EAST", "WEST"]

agents = [SimpleAgent(),SimpleAgent(),SimpleAgent(),SimpleAgent(),] 

env = make("hungry_geese", debug=True)

env.run(agents)

env.render(mode="ipython",width=800,height=700)
```

## to-do


step0不会用作训练，因为不是模型的决定。

bonus: eat food 1, collision/starve -20, move 0.05, around food +0.1, around head -2

generate: for each iteration, generate 128 games, max time 200. Just run 4 agents using the latest model

train: Stable-Baselines3 PPO

val: early stopping；checkpoints combat each other;

