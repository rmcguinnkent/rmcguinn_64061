
from stable_baselines3 import DQN, PPO
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import CatanEnvironmentCoop, CatanEnvironmentComp, CatanEnvironmentDQNSettlements

env = DummyVecEnv([lambda: CatanEnvironmentCoop()])

model = DQN(DQNPolicy, env, verbose=1)
model.learn(total_timesteps=2_000_000_000)

model = PPO('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=100_000)

model.save("CatanModelNaive")

for i in range(4, 14):
    model_1 = PPO.load(f"CatanModel{i-3}")
    model_2 = PPO.load(f"CatanModel{i-2}")
    model_3 = PPO.load(f"CatanModel{i-1}")

    env = DummyVecEnv([lambda: CatanEnvironmentComp([model_3, model_2, model_1])])

    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=200_000)
    model.save(f"CatanModel{i}")


opp_model = PPO.load("CatanModel13")

env = DummyVecEnv([lambda: CatanEnvironmentDQNSettlements(opp_model, [opp_model] * 3)])

model = DQN(DQNPolicy, env, verbose=1)
model.learn(total_timesteps=3_000_000)
model.save("CatanModelDQN1")
