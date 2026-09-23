# DOCS

## Rules

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

一些游戏信息由AIfenxi[Kaggle Environment 源码](https://github.com/Kaggle/kaggle-environments/tree/master)得到。

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
```


## Gameplay

Render a play
```python
#@title 跑一下
from kaggle_environments import make, evaluate

agents = [
    SimpleAgent(),
    SimpleAgent(),
    SimpleAgent(),
    SimpleAgent(),
]

env = make("hungry_geese", debug=True)

env.run(agents)

env.render(
    mode="ipython",
    width=800,
    height=700
)
```

Render a JSON
```python
import json
from kaggle_environments import make

path = "gameplays/random_init/game_000004.json"

with open(path, "r", encoding="utf-8") as file:
    replay = json.load(file)

replay_env = make(
    replay["name"],
    configuration=replay["configuration"],
    info=replay.get("info", {}),
    steps=replay["steps"],
)

replay_env.render(
    mode="ipython",
    width=800,
    height=700,
)
```

## JSON

```python
import json

env.run(agents)

with open("replay.json", "w", encoding="utf-8") as file:
    json.dump(env.toJSON(), file, ensure_ascii=False)
```

```json
{
  "name": "hungry_geese",
  "configuration": {},
  "info": {},
  "steps": [
    [
      {
        "action": "NORTH",
        "observation": {},
        "reward": 0,
        "status": "ACTIVE",
        "info": {}
      }
    ]
  ],
  "rewards": [],
  "statuses": [],
  "schema_version": 1
}
```
`steps[回合][玩家]`每回合每个玩家的完整状态。

reward = 存活步数 × (max_length + 1) + 当前身体长度

Status: 
    ACTIVE：仍在对局中。
    DONE：正常结束，包括死亡或对局结束。
    INVALID：返回了非法动作。
    ERROR：Agent 运行报错。
    TIMEOUT：Agent 超时。

## Files

`model.py`:
    model architecture
`utils.py`:
    long functions used in other files
`constants.py`:
    constants for coding
`agent.py`:
    load the model, let it read the observations, and return an action. The agent class is `Agent`. There's another class called `SimpleAgent` whose behavior is defined by code rather than by the model, to prevent `Agent` instances from cooperating with each other.
`generate.py`:
    generate PPO-ready gameplay for training.
    Format: gameplays/model_000002/game_000001.json
    The greatest numeric model ID is loaded. A checkpoint may be a raw state_dict or a dictionary containing its state_dict under "model_state_dict". If no valid checkpoint exists, random weights are saved as `models/model_000000.pt` before any game is generated. Existing replay numbers are never overwritten.
    Every player seat has a default 10% probability of using `SimpleAgent`, and every game keeps at least one model player.
    Every replay keeps the official Kaggle fields and adds a top-level `ppo` object:
        ppo
        schema_version
        model_id
        player_types
        trainable_players
        trajectories[player]
            step
            relative_action
            action_logit_vec
            value
            action_mask
            policy_sampled
    `relative_action` uses `FORWARD / LEFT / RIGHT` order. `action_logit_vec` contains the three raw model logits before masking, and `action_mask` uses the same order. `step == 0` and `SimpleAgent` decisions are not stored in PPO trajectories. When `policy_sampled` is false, all survival actions were masked and the action came from the random fallback; exclude that record from the actor loss, but it can still train the critic.
    command format: `python generate.py --games 16 --simple-agent-probability 0.1 --models-dir models --gameplays-dir gameplays --debug`


Checkpoint format:
models/model_000000.pt
models/model_000001.pt
models/model_000002.pt



## to-do

train: Stable-Baselines3 PPO, each iteration ? epochs
模型默认 CPU；外部调用 model.to("cuda"/"mps") 才使用 GPU。
Agent 会自动把所有模型输入移到模型所在设备。
生存模拟始终在 CPU。
单局小模型推理建议 CPU；批量训练或批量推理再用 GPU。
step0不会用作训练，因为不是模型的决定。
注意模型储存规则

reward = 存活步数 × (max_length + 1) + 当前身体长度
每步奖励 = delta_reward / 20000
赢过一个对手：+1/3（杀死时结算）
输给一个对手：-1/3（死亡时结算）
平局：0

