class TrainingParameters:
    def __init__(self, max_episodes: int, steps_per_episode: int):  
        self.max_episodes = int(max_episodes)
        self.steps_per_episode = int(steps_per_episode)


class AgentParameters:
    def __init__(self, epsilon_min, epsilon_decay, start_epsilon):
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.start_epsilon = start_epsilon


class LearningParameters:
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma