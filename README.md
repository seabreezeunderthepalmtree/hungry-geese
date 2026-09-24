# Hungry Geese PPO

使用 PyTorch PPO 训练 Hungry Geese Agent。

基本流程：

```text
generate.py 生成对局 JSON
→ train.py 读取对局并训练
→ 保存下一个模型
→ 重复
```

## 游戏规则

- 4 只鹅在 `7 × 11` 的环形棋盘上同时行动。
- 动作为 `NORTH / EAST / SOUTH / WEST`。
- 棋盘上下、左右相连。
- 不能立即选择当前方向的反方向。
- 吃到食物后身体增长；没有吃到时尾巴正常缩短。
- 每 40 步还会因饥饿额外缩短一节，长度变为 0 时死亡。
- 头撞到碰撞结算时仍被占用的身体格会死亡，包括自己的身体和本回合稍后才消失的尾巴。
- 多只鹅的头同时进入同一格会发生头部碰撞。
- 一局最多 200 步，首先比较存活时间，其次比较身体长度。
- 每回合提供完整棋盘信息，不是部分可观测环境。

官方累计 reward：

```text
reward = 存活步数 × (max_length + 1) + 当前身体长度
```

参考：[Hungry Geese overview](https://www.kaggle.com/competitions/hungry-geese/overview) 和 [Kaggle Environments source](https://github.com/Kaggle/kaggle-environments)。

## Kaggle API

### Observation

```python
observation = {
    "index": 0,              # 当前玩家编号
    "step": 37,              # 当前步数
    "geese": [
        [12, 13, 14],         # 第一个位置是头，最后一个位置是尾巴
        [42, 31],
        [],                   # 空数组表示已经死亡
        [70, 69, 68],
    ],
    "food": [5, 61],
    "remainingOverageTime": 60,
}
```

棋盘位置使用一维整数：

```python
row = position // configuration["columns"]
col = position % configuration["columns"]
position = row * configuration["columns"] + col
```

`remainingOverageTime` 是 Kaggle 的计算时间额度，不是游戏剩余步数。

### Agent 接口

Kaggle 自动传入 `observation` 和 `configuration`：

```python
class MyAgent:
    def __call__(self, observation, configuration):
        return "NORTH"
```

返回值必须是：

```text
NORTH / EAST / SOUTH / WEST
```

### 运行和渲染

```python
from kaggle_environments import make

agents = [
    SimpleAgent(),
    SimpleAgent(),
    SimpleAgent(),
    SimpleAgent(),
]

env = make("hungry_geese", debug=True)
env.run(agents)
env.render(mode="ipython", width=800, height=700)
```

### 保存 replay

```python
import json

with open("replay.json", "w", encoding="utf-8") as replay_file:
    json.dump(env.toJSON(), replay_file, ensure_ascii=False)
```

主要结构：

```text
replay
├── name
├── configuration
├── info
├── steps[step][player]
│   ├── action
│   ├── observation
│   ├── reward
│   ├── status
│   └── info
├── rewards
└── statuses
```

Status：

```text
ACTIVE  仍在游戏中
DONE    已死亡或对局正常结束
INVALID 返回了非法动作
ERROR   Agent 执行报错
TIMEOUT Agent 超时
```

### 回放 JSON

```python
import json

from kaggle_environments import make

with open("replay.json", "r", encoding="utf-8") as replay_file:
    replay = json.load(replay_file)

replay_env = make(
    replay["name"],
    configuration=replay["configuration"],
    info=replay.get("info", {}),
    steps=replay["steps"],
)

replay_env.render(mode="ipython", width=800, height=700)
```

回放时应创建新的 `replay_env`，不要继续 render 之前已经运行过的 `env`。

## 文件说明

### `utils.py`

提供两个主要工具。

#### Observation to tensor

`observation_to_tensor()` 将 Kaggle observation 转换为：

```text
tensor:    (11, 11, 13), float32
direction: NORTH / EAST / SOUTH / WEST 对应的整数
```

棋盘以自己的头为中心，并旋转到当前移动方向始终朝上。13 个通道为：

```text
0:    自己的头
1-3:  对手的头
4:    自己的尾巴
5-7:  对手的尾巴
8:    自己的身体，不含头尾
9-11: 对手的身体，不含头尾
12:   食物
```

方向通常由头和脖子推断。长度为 1 时使用 Agent 保存的上一方向；只有第一步可以默认 `NORTH`。

#### Survival mask

`get_survival_mask()` 返回四个绝对方向是否存在短期生存路径：

```text
(NORTH, EAST, SOUTH, WEST)
```

默认检查 2 步；如果全部被 mask，则退化为检查 1 步。模拟时禁止反向，并在尾巴消失前先进行碰撞判断。它不预测对手的未来动作，只过滤短期确定性死亡。

### `model.py`

定义 `HungryGeeseActorCritic`：

```text
(batch, 11, 11, 13)
→ 定制环形 padding
→ Conv2d(13 → 16, 3×3) + SiLU
→ 定制环形 padding
→ Conv2d(16 → 16, 3×3) + SiLU
→ Flatten
→ Linear(1936 → 15) + SiLU
```

将 15 维棋盘特征与 1 维归一化剩余时间拼接：

```text
Actor:  16 → 16 → 3 logits
Critic: 16 → 16 → 1 value
```

三个相对动作固定为：

```text
0 = FORWARD
1 = LEFT
2 = RIGHT
```

模型不使用 BatchNorm、LayerNorm 或 Dropout。

### `agent.py`

`Agent` 读取模型，根据 observation 得到相对动作，再转换为绝对方向：

```text
NORTH / EAST / SOUTH / WEST
```

第一步不使用模型，而是在安全方向中选择最接近食物的方向。之后的步骤：

```text
observation
→ tensor 和当前方向
→ model logits/value
→ survival mask
→ 采样 FORWARD/LEFT/RIGHT
→ 转换为绝对方向
```

该文件还包含 `SimpleAgent`。它使用 survival mask、避开对手头部可能到达的位置，并倾向靠近食物。生成对局时加入部分 `SimpleAgent`，用于打破完全同质的自博弈，降低多个相同模型学会合作的可能性。

### `constants.py`

保存整个项目共享的常量，包括棋盘、动作、张量通道、模型结构、文件名和 PPO 默认参数。

其他文件统一使用：

```python
from constants import *
```

不要在其他文件重复定义相同常量。

### `generate.py`

读取最新模型并生成训练对局：

```bash
python generate.py --games 32
```

如果没有模型，会创建随机初始化的：

```text
models/model_000000.pt
```

每个座位默认有 10% 概率使用 `SimpleAgent`，但每局至少保留一个模型玩家。对局保存为：

```text
gameplays/model_000000/game_000001.json
```

JSON 保留 Kaggle 原始 replay，并增加 `ppo`：

```text
ppo
├── schema_version
├── model_id
├── player_types ["model", "simple", "model", "model"]
├── trainable_players [0, 2, 3]
└── trajectories[player]
    ├── step
    ├── relative_action
    ├── action_logit_vec
    ├── value
    ├── action_mask
    └── policy_sampled
```

`policy_sampled = false` 表示所有动作都被 mask，最终使用了随机fallback。这类记录可训练 Critic，但不参与 Actor loss。

### `train.py`

读取最新模型对应的全部对局，完成一次 PPO iteration：

```bash
python train.py --device auto
```

流程：

```text
读取 replay
→ 重建 observation tensor
→ 计算每步 reward 和终局排名奖励
→ 按玩家计算 GAE 和 return
→ 合并所有模型决策
→ 分 mini-batch 训练多个 epochs
→ 保存下一个模型和训练日志
```

#### Reward

每一步先计算 Kaggle 累计 reward 的变化：

$$
r_t = (OfficialReward[t+1] - OfficialReward[t]) / 20000
$$

每名玩家最后一个模型决策再加入终局排名奖励：

```text
第一名 +1
第二名 +1/3
第三名 -1/3
第四名 -1
```

并列时取较低名次，例如两只鹅并列第一时都算第二名。

#### GAE

每局、每名模型玩家分别从后向前计算：

$$
\delta_t = r_t + \gamma × (1 - done_t) × V_{(t+1)} - V_t
$$
$$
A_t = \delta_t
    + \gamma × \lambda × (1 - done_t) × A_{(t+1)}
$$
$$
return_t = A_t + V_t
$$

- `V_t` 是生成对局时保存的旧模型 value；
- `done_t = 1` 时不使用下一状态的 value；
- 默认 `gamma = 0.99`；
- 默认 `lambda = 0.95`；
- 所有对局合并后，只标准化能够训练 Actor 的 advantage；
- `return_t` 是 Critic 的训练目标，不进行标准化。

#### PPO policy loss

先使用保存的 `action_mask`，分别计算旧模型和当前模型对已选择动作的 log probability：

$$
ratio_t = exp(NewLogProbability_t - OldLogProbability_t)
$$
$$
unclipped_t = ratio_t × A_t
$$
$$
clipped_t   = clip(ratio_t, 1-epsilon, 1+epsilon) × A_t
$$
$$
PolicyLoss = -mean(min(unclipped_t, clipped_t))
$$

mean是对一个batch的所有决策平均

默认 `epsilon = 0.2`。clip 限制一次训练对策略概率造成过大的改变。

#### Value loss

$$
value_loss = 0.5 × mean((V_new - return)^2)
$$

Critic 学习预测当前状态之后的折扣回报。

#### Entropy

$$
entropy = mean(H(masked_policy))
$$

entropy 由应用 `action_mask` 后的当前 logits 计算，用于保留探索。三个动作都可用时最大值为 `ln(3)`。

#### Total loss

$$
total_loss = policy_loss
           + 0.5 × value_loss
           - 0.01 × entropy
$$

之后执行反向传播，将 gradient norm 裁剪到 `0.5`，再调用一次 `optimizer.step()`。

`policy_sampled = false` 的全 mask fallback：

- 仍参与 GAE、return 和 value loss；
- 不参与 policy loss；
- 不参与 entropy。

默认每个 mini-batch 最多包含 256 个模型决策，每个 mini-batch 更新一次参数。全部数据训练 4 个 epochs，并在每个 epoch 开始时重新打乱。

输出：

```text
models/model_000001.pt
training_logs/model_000001.json
```

`--device auto` 按 `CUDA → MPS → CPU` 选择训练设备。

## 运行

```bash
python generate.py --games 32
python train.py --device auto
```

重复这两条命令即可继续训练。
