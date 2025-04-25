import json
from gym import spaces
from QSolver import *
import numpy as np
import gym
from tqdm import tqdm
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import (
    Compose, DoubleToFloat, ObservationNorm, StepCounter, TransformedEnv, GymWrapper)
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.collectors import SyncDataCollector
from torch import nn
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from torch import multiprocessing
import warnings
import os
from datetime import datetime

do_load_model = True
model_path = './qec_model_origin.pth'

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings("ignore")

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
# device = torch.device("cpu")
print(device)
num_cells = 64  # number of cells in each layer i.e. output dim.
lr = 4e-4
max_grad_norm = 1.0

frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 50_000*6

# cardinality of the sub-samples gathered from the current data in the inner loop
sub_batch_size = 64
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    # clip value for PPO loss: see the equation in the intro for more context.
    0.2  # 0.2
)
gamma = 0.99
lmbda = 0.98
entropy_eps = 1e-5


def break_even(t, dt):
    # 只有0.6倍点
    n = min(round(t/dt), 69)
    be = [1., 0.9804426,  0.96174158, 0.94385562, 0.92674504, 0.91037281,
          0.89470359, 0.87970371, 0.8653423, 0.85158866, 0.83841358, 0.82578996,
          0.81369203, 0.80209543, 0.79097694, 0.78031455, 0.77008738, 0.76027555,
          0.75086023, 0.74182347, 0.73314827, 0.72481843, 0.71681857, 0.70913405,
          0.70175096, 0.69465608, 0.68783679, 0.68128112, 0.67497767, 0.66891558,
          0.6630845, 0.6574746, 0.65207649, 0.64688124, 0.64188032, 0.63706562,
          0.6324294, 0.62796428, 0.62366322, 0.61951949, 0.6155267, 0.61167871,
          0.6079697, 0.60439409, 0.60094654, 0.59762197, 0.59441552, 0.59132254,
          0.58833859, 0.58545941, 0.58268095, 0.57999933, 0.57741081, 0.57491185,
          0.57249904, 0.57016911, 0.56791894, 0.56574555, 0.56364605, 0.56161771,
          0.5596579, 0.55776408, 0.55593384, 0.55416486, 0.55245491, 0.55080184,
          0.54920362, 0.54765826, 0.54616389, 0.54471868]
    return be[n]


class QECEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 动作空间：8个编码操作和7个解码操作
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(15,),  # 8 + 7 = 15维动作空间
            dtype=np.float32
        )

        # 观察空间：6个保真度数据,15个上次动作数据,15个最初动作数据,3个系统参数
        self.observation_space = spaces.Box(
            low=np.array([0]*6+[-1.0]*(15*2)+[0]*3),
            high=np.array([1.0]*6+[1.0]*(15*2)+[2.0]*3),
            dtype=np.float32
        )

        # 初始化量子系统参数
        self.initialize_quantum_system()

    def initialize_quantum_system(self):
        # 系统参数
        self.N = 8
        self.gamma_a = 2
        self.dt = 0.06/self.gamma_a

        # 时间演化参数
        self.t = 0
        self.total_time = 4.2/self.gamma_a
        self.tlist = [0, self.dt]

        # 记录上一次动作
        self.last_action = None
        self.init_action = None
        self.eps = 0
        self.fidelity = 0
        self.rho_flag = True

    def reset(self):
        self.t = 0
        # self.coef = np.array([np.random.uniform(0, 2, 1)[0],
        #                       np.random.uniform(0.55, 0.65, 1)[0],
        #                       np.random.uniform(0.35, 0.45, 1)[0]])
        self.coef = np.array([np.random.uniform(0.5, 2, 1)[0],
                              np.random.uniform(0.55, 0.65, 1)[0],
                              np.random.uniform(0.35, 0.45, 1)[0],])
        # self.coef = np.array([0.24, 0.6, 0.4])
        a2, b, g = self.coef
        self.gamma_a2 = 0.05*a2
        self.gamma_b = 500*b+1500
        self.g = 500+250*g
        self.lam = 8*self.g*self.g/self.gamma_a/self.gamma_b
        self.last_action = np.array([0, 0, 0, 0, -1, 0, 0, 1, 0, 0,
                                     0, 0, 0, 0, 0])
        self.init_action = np.array([0, 0, 0, 0, -1, 0, 0, 1, 0, 0,
                                     0, 0, 0, 0, 0])
        self.rho_flag = True
        initial_obs = np.concatenate([
            np.ones(6),  # 初始6维状态
            # np.array([0, 0, 0, 0, 1, 0, 0, -1, 0, 0,
            #          .25, .25, 0, .25, .25]),  # 初始15维动作
            self.last_action,
            self.init_action,
            np.array([a2, b, g])  # 初始3维系统参数
        ])
        return initial_obs, {}

    def step(self, action):
        # print('action',action)
        # 解码动作
        encode = np.array(action[0:8])
        encode = encode - np.mean(encode)
        en0 = np.array([encode[i] if encode[i] > 0 else 0 for i in range(8)])
        en1 = np.array([-encode[i] if encode[i] < 0 else 0 for i in range(8)])
        sum_en0 = np.sum(en0*en0)
        sum_en1 = np.sum(en1*en1)
        en0 = np.array([1, 0, 0, 0, 0, 0, 0, 0]
                       ) if sum_en0 < 1e-5 else en0/sum_en0
        en1 = np.array([0, 1, 0, 0, 0, 0, 0, 0]
                       ) if sum_en1 < 1e-5 else en1/sum_en1
        encode = en0-en1
        d1 = np.abs(action[8:15])
        sum_d1 = np.sum(d1*d1)
        d1 = d1/sum_d1 if sum_d1 > 1e-5 else np.ones(7)

        en0_in = np.array([np.arange(8), en0]).T
        en1_in = np.array([np.arange(8), en1]).T
        d1_in = np.array([np.arange(7), d1]).T

        if self.rho_flag:
            equ = MasterEqu(self.N, en0_in, en1_in, 2, [
                            1, self.gamma_a2], lam=self.lam, L_list=d1_in)
            equ.get_basic_states()
            self.current_states = equ.rhos.copy()
            self.rho_flag = False

        fid_list, self.current_states = equ.get_fid_basic_matrices(
            self.current_states, self.tlist)
        fidelities = np.array([fid_list[i][-1] for i in range(6)])

        action = np.concatenate([encode, d1])
        self.last_action = action
        if self.t == 0:
            self.init_action = action.copy()
        observation = np.concatenate([
            fidelities,  # 6维保真度数据
            self.last_action,      # 15维动作数据
            self.init_action,
            self.coef    # 3维系统参数
        ])

        self.t += self.dt

        # 计算奖励
        reward, fid_pos = self._get_reward(fidelities, action)

        # 更新状态
        self.last_action = action
        done1 = self.t >= self.total_time  # 到达最长模拟时间
        done2 = self.t > 6*self.dt and fid_pos  # 在模拟结束前低于breakeven
        # done = done1  # phase 1
        done = done1 or done2  # phase 2
        if done2:
            reward = -20

        return observation, reward, done, False, {}
    '''
    def get_current_states(self):
        encode = self.last_action[:8]
        d1 = self.last_action[8:]
        en0 = np.array([encode[i] if encode[i] > 0 else 0 for i in range(8)])
        en1 = np.array([-encode[i] if encode[i] < 0 else 0 for i in range(8)])
        c_ops = self._setup_operators(en0, en1, d1, d2=None)
        H = zero_ket(self.N)*zero_ket(self.N).dag()

        for k, rho0 in enumerate(self.initial_states):
            current_state = self.initial_states[k]
            tlist = np.arange(0,self.t+self.dt,self.dt)
            result = mesolve(H, current_state, tlist, c_ops,
                             options=Options(nsteps=10000))
            self.current_states[k] = result.states[-1]
    '''

    def _get_reward(self, fidelities, action):
        # 计算保真度提升奖励
        fidelity = np.mean(fidelities)
        # var_fidelity = np.max(fidelities)-np.min(fidelities)
        eps = fidelity-break_even(self.t, self.dt)
        # print('eps',eps)
        # last_eps = self.eps
        self.eps = eps
        # last_fidelity = self.fidelity
        self.fidelity = fidelity

        # 计算动作连续性奖励
        action_bonus = np.dot(
            action, self.last_action) if self.last_action is not None else 0
        action_bonus_sum = np.sqrt(
            np.sum(action*action))*np.sqrt(np.sum(self.last_action*self.last_action))
        action_bonus = action_bonus/action_bonus_sum if action_bonus_sum > 1e-5 else 0
        if action_bonus > 0.97 or self.t == self.dt:
            action_bonus = 1
        # delta_be = break_even(self.t, self.dt) - \
        #     break_even(self.t-self.dt, self.dt)
        # delta = min(delta_be, (fidelity-last_fidelity))
        # grThBe = 0

        # Be = break_even(self.t, self.dt)
        # for f in fidelities:
        #     if f >= Be:
        #         grThBe += 1

        # phase 1
        # if eps < 0:
        #     return 50 * eps + action_bonus*0.04, True
        # return 250 * eps + action_bonus*0.2, False

        # phase 2
        # return 250 * eps + action_bonus*2, False
        if eps < 0:
            # if np.any(fidelities < 0):
            return 50 * eps + action_bonus*0.4, True
        return 250 * eps + action_bonus*2, False
        '''
        # print('eps', eps)
        # print('action_bonus', action_bonus)
        # print('var_fidelity', var_fidelity)

        # return (1+eps)*20-9
        if self.eps < 0:
            return (1+eps)*20-10 + action_bonus*0.1-var_fidelity

        eps_bonus = 150
        if self.eps > last_eps:
            eps_bonus = 160

        # print('eps_bonus',eps_bonus)

        return eps * eps_bonus + action_bonus*5-var_fidelity*10
        '''


base_env = QECEnv()
base_env = GymWrapper(base_env, device=device)


class CustomObservationNorm(ObservationNorm):
    def _call(self, tensordict):
        obs = tensordict.get(self.in_keys[0])
        # 只对前6维(保真度数据)进行标准化
        fidelity_data = obs[..., :6]
        action_data = obs[..., 6:]

        # 标准化保真度数据
        normalized_fidelity = (
            fidelity_data - self.loc[..., :6]) / self.scale[..., :6]

        # 组合标准化后的保真度数据和原始动作数据
        normalized_obs = torch.cat([normalized_fidelity, action_data], dim=-1)

        tensordict.set(self.out_keys[0], normalized_obs)
        return tensordict


env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        # ObservationNorm(in_keys=["observation"]),
        CustomObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter(),
    ),
)

env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
'''
actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)
'''


class ActorNet(nn.Module):
    def __init__(self, num_cells, code_dim, proj_dim, device):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LazyLinear(num_cells, device=device),
            nn.LeakyReLU(),
            nn.LazyLinear(num_cells, device=device),
            nn.LeakyReLU(),
        )
        self.code_head = nn.Sequential(
            nn.LazyLinear(2*code_dim, device=device),
        )
        self.proj_head = nn.Sequential(
            nn.LazyLinear(2*proj_dim, device=device),
        )

        self.normal_extractor = NormalParamExtractor()

    def forward(self, x):
        features = self.backbone(x)
        code_params = self.code_head(features)
        proj_params = self.proj_head(features)
        combined_params = torch.cat([code_params, proj_params], dim=-1)
        return self.normal_extractor(combined_params)


actor_net = ActorNet(num_cells, code_dim=8, proj_dim=7, device=device)

policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
    },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
)

value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.LeakyReLU(),
    nn.LazyLinear(num_cells, device=device),
    nn.LeakyReLU(),
    nn.LazyLinear(num_cells, device=device),
    nn.LeakyReLU(),
    nn.LazyLinear(1, device=device),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)
print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))
collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

if do_load_model:
    # 加载模型
    checkpoint = torch.load(model_path)
    policy_module.load_state_dict(checkpoint['policy_state_dict'])
    value_module.load_state_dict(checkpoint['value_state_dict'])

logs = defaultdict(list)
pbar = tqdm(total=total_frames, disable=True)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    logs["max_reward"].append(tensordict_data["next", "reward"].max().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 5.5f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(
                eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval reward (max)"].append(
                eval_rollout["next", "reward"].max().item()
            )
            logs["eval step_count"].append(
                eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"eval maximum reward: {logs['eval reward (max)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(
        ", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()
    # 保存模型
    print(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
    if i % 10 == 0:
        torch.save({
            'policy_state_dict': policy_module.state_dict(),
            'value_state_dict': value_module.state_dict(),
        }, f'./output/tmp/qec_model_{i}.pth')

# 保存训练日志
torch.save({
    'policy_state_dict': policy_module.state_dict(),
    'value_state_dict': value_module.state_dict(),
}, f'./output/tmp/qec_model_end.pth')

# 将日志中的numpy数组转换为列表
log_dict = {
    "reward": [float(x) for x in logs["reward"]],
    "eval_reward": [float(x) for x in logs["eval reward"]],
    "eval_reward_max": [float(x) for x in logs["eval reward (max)"]],
    "eval_reward_sum": [float(x) for x in logs["eval reward (sum)"]],
    "step_count": [float(x) for x in logs["step_count"]],
    "eval_step_count": [float(x) for x in logs["eval step_count"]],
    "lr": [float(x) for x in logs["lr"]]
}

# 保存为json文件
with open('./output/tmp/training_logs.json', 'w') as f:
    json.dump(log_dict, f)

tmp_dir = './output/tmp'
now = datetime.now()
new_dir = os.path.join('./output', now.strftime("%y%m%d%H%M"))
os.rename(tmp_dir, new_dir)
