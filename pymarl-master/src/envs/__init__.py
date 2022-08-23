from functools import partial
# from smac.env import MultiAgentEnv, StarCraft2Env, Matrix_game1Env, Matrix_game2Env, Matrix_game3Env, mmdp_game1Env
import sys
import os
import numpy as np

from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.football import Football_Env

from types import SimpleNamespace

class MultiAgentEnv(object):

    def step(self, actions):
        """ Returns reward, terminated, info """
        raise NotImplementedError

    def get_obs(self):
        """ Returns all agent observations in a list """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        """ Returns the shape of the state"""
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    def reset(self):
        """ Returns initial observations and states"""
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
        
def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {
    # "sc2": partial(env_fn, env=StarCraft2Env),
    # "matrix_game_1": partial(env_fn, env=Matrix_game1Env),
    # "matrix_game_2": partial(env_fn, env=Matrix_game2Env),
    # "matrix_game_3": partial(env_fn, env=Matrix_game3Env),
    # "mmdp_game_1": partial(env_fn, env=mmdp_game1Env)
}

class MPE(MultiAgentEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        args = SimpleNamespace(**kwargs)
        self.env = MPEEnv(args)

        self.episode_limit = self.env.world_length
        self.n_agents = len(self.env.agents)

        self.n_actions = self.env.action_space[0].n

    def reset(self):
        return self.env.reset()
    
    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        reward = np.array(reward)
        return reward.mean(), np.array(done).any(), {}
    
    def render(self, mode='human'):
        return self.env.render(mode=mode)
    
    def get_state_size(self):
        return self.env.share_observation_space[0].shape[0]
    
    def get_obs_size(self):
        return self.env.observation_space[0].shape[0]

    def get_total_actions(self):
        return self.n_actions
    
    def get_state(self):
        obs = self.get_obs()
        return obs.flatten()

    def get_obs(self):
        obs_n = []
        for i, agent in enumerate(self.env.agents):
            obs_n.append(self.env._get_obs(agent))
        return np.stack(obs_n)
        
    def get_avail_actions(self):
        return np.ones((self.n_agents, self.n_actions))

class Football(MultiAgentEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.env = Football_Env.FootballEnv(SimpleNamespace(**kwargs))

        self.episode_limit = self.env.max_steps
        self.n_agents = self.env.num_agents
        self.n_actions = self.env.action_space[0].n
    
    def reset(self):
        return self.env.reset()
    
    def step(self, actions):
        self.obs, reward, done, info = self.env.step(actions)
        return reward.mean(), done, info
    
    def get_state_size(self):
        return self.env.observation_space[0].shape[0] * self.n_agents
    
    def get_state(self):
        return self.obs.flatten()
    
    def get_obs(self):
        return self.obs
    
    def get_avail_actions(self):
        return np.ones((self.n_agents, self.n_actions))

REGISTRY["mpe"] = partial(env_fn, env=MPE)
REGISTRY["football"] = partial(env_fn, env=Football)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
