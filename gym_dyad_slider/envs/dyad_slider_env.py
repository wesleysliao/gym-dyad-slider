import gym
from gym import error, spaces, utils
from gym.utils import seeding


import numpy as np
import scipy as sp


class DyadSliderEnv(gym.Env):
    """
    Description:
        Two agents apply forces on an object to follow a target which moves in one dimension.
    Source:
    Observation:
        Observations of the state depend on the agent

    Actions:
        Applied force

    Reward:
        Reward is a combination of tracking error and effort expended.
        The tracking error is common to both agents, but effort will depend on individual actions.

    Starting State:

    Episode Termination:
        Episode length is greater than
        Force limits are exceeded
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30,
    }

    def __init__(self,
                 simulation_freq_Hz = 1200,
                 action_freq_Hz = 300,

                 episode_length_s = 30.0,

                 n_agents = 2,
                 agent_force_min = -1.0,
                 agent_force_max = 1.0,

                 force_net_limits = np.array([-1.0, 1.0]),
                 force_interaction_limits = np.array([-1.0, 1.0]),

                 slider_mass = 1.0,
                 slider_friction_coeff = 0.1,
                 slider_range = np.array([-1.0, 1.0]),

                 reference_trajectory_fn = lambda x: np.sin(x * np.pi),
    ):

        self.simulation_timestep_s = 1.0 / simulation_freq_Hz
        self.action_timestep_s = 1.0 / action_freq_Hz
        self.simsteps_per_action = int(simulation_freq_Hz / action_freq_Hz)
        self.max_episode_steps = int(simulation_freq_Hz * episode_length_s)
        self.episode_length_s = episode_length_s

        self.n_agents = n_agents

        if np.isscalar(agent_force_min):
            self.action_space = spaces.Box(low = agent_force_min,
                                           high = agent_force_max,
                                           shape = (n_agents,),
                                           dtype = np.float32)
        else:
            self.action_space = spaces.Box(low = agent_force_min,
                                           high = agent_force_max,
                                           dtype = np.float32)

        self.slider_mass = slider_mass
        self.slider_friction_coeff = slider_friction_coeff

        self.force_net_limits = force_net_limits
        self.force_interaction_limits = force_interaction_limits

        self.reference_trajectory_fn = reference_trajectory_fn


        self.reset()


    def step(self, action):

        done = False

        x_0, x_dot_0, r_0, r_dot_0, \
        force_net_0, force_net_dot_0, \
        force_interaction_0, force_interaction_dot_0 = self.state

        force_interaction_1 = np.min(action)
        force_net_1 = np.max(action) - force_interaction_1

        force_net_dot_1 = force_net_1 - force_net_0
        force_interaction_dot_1 = force_interaction_1 - force_interaction_0

        for t in np.linspace(self.t, self.t + self.action_timestep_s, self.simsteps_per_action):

            r_1 = self.reference_trajectory_fn(t)
            r_dot_1 = r_1 - r_0

            acceleration = force_net_1 / self.slider_mass

            x_dot_1 = x_dot_0 + (acceleration * self.simulation_timestep_s)
            x_1 = x_0 + (x_dot_1 * self.simulation_timestep_s)

            self.state = np.array([x_1, x_dot_1, r_1, r_dot_1,
                                  force_net_1, force_net_dot_1,
                                  force_interaction_1, force_interaction_dot_1])

            self.error -= abs(x_1 - r_1)

            if t >= self.episode_length_s:
                done = True
                self.t += t
                break


        reward = self.error

        return self.observe(self.state), reward, done


    def reset(self):

        self.t = 0.0
        self.error = 0.0

        #state = x, x_dot, r, r_dot, force_net, force_net_dot, force_interaction, force_interaction_dot
        self.state = np.zeros((8,), dtype=np.float32)

        return self.observe(self.state)

    def observe(self, state):
        x, x_dot, r, r_dot, \
        force_net, force_net_dot, \
        force_interaction, force_interaction_dot = state

        return np.array([x, x_dot, r, r_dot,
                         force_interaction, force_interaction_dot])

    def render(self, mode='human'):
        print("render lol", self.state)

    def close(self):
        pass
