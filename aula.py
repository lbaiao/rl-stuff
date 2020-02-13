# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Treinamento de Reinforcement Learning
# %% [markdown]
# algum bla bla bonito

# %%
import gym

# %% [markdown]
# ## Descobrir ambientes gym disponíveis

# %%
from gym import envs
env_names = [spec.id for spec in envs.registry.all()]
for name in sorted(env_names):
    print(name)

# %% [markdown]
# ### Explorando o ambiente

# %%
env = gym.make('FrozenLake8x8-v0')
env.reset()
for _ in range(2000):
    env.render()
    env.step(env.action_space.sample())
env.close()

# %% [markdown]
# ### Verificando os espaços de observações e ações

# %%
print(f'Espaço de estados: {env.observation_space}')
print(f'Espaço de ações: {env.action_space}')

# %% [markdown]
# ### Veficando as saídas do ambiente

# %%
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)

print(f'Ação: {action}')
print(f'Próxima observação: {next_state}')
print(f'Recompensa: {reward}')
print(f'Finalizado: {done}')
print(f'Finalizado: {info}')

# %% [markdown]
# ### Hiperparâmetros

# %%
from parameters import TrainingParameters, LearningParameters, AgentParameters

EPSILON_MIN = 0.05
MAX_NUM_EPISODES = 50000
STEPS_PER_EPISODE = 64
max_num_steps = MAX_NUM_EPISODES * STEPS_PER_EPISODE
ALPHA = 0.1
GAMMA = 0.999
EPSILON_DECAY = 25 * EPSILON_MIN / max_num_steps

train_params = TrainingParameters(MAX_NUM_EPISODES, STEPS_PER_EPISODE)
learn_params = LearningParameters(ALPHA, GAMMA)
agent_params = AgentParameters(EPSILON_MIN, EPSILON_DECAY, 1)

# %% [markdown]
# ### Agente

# %%
from agent import Agent
from q_table import QTable
import numpy as np

agent = Agent(agent_params, env.action_space.n)
q_table = QTable(env.observation_space.n, env.action_space.n, learn_params)
agent.set_q_table(q_table)

# %% [markdown]
# ### Funções de treino e teste

# %%
def train(agent: Agent, env, params: TrainingParameters):
    best_reward = -float('inf')
    for episode in range(MAX_NUM_EPISODES):
        obs = env.reset()
        done = False
        total_reward = 0.0        
        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.q_table.learn(obs, action, reward, next_obs)
            obs = next_obs
            total_reward += reward
        if total_reward > best_reward:
            best_reward = total_reward
        print("Episode#:{} reward:{} best_reward:{} eps:{}".format(episode,
            total_reward, best_reward, agent.epsilon))
    return np.argmax(agent.q_table.table, axis=1)


def test(agent: Agent, env, policy):
    done = False
    obs = env.reset()
    total_reward = 0.0
    while not done:
        action = policy[obs]
        next_obs, reward, done, info = env.step(action)   
        env.render()     
        obs = next_obs
        total_reward += reward
    return total_reward

# %% [markdown]
# ### Treinamento

# %%
learned_policy = train(agent, env, train_params)

# %% [markdown]
# ### Teste

# %%
# gym_monitor_path = "./gym_monitor_output"
# env = gym.wrappers.Monitor(env, gym_monitor_path, force=True)
for _ in range(10):
    r = test(agent, env, learned_policy)
    print(r)
# env.close()

