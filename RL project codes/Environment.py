import gym
import game_physics
import pygame
from typing import Dict, Tuple, Optional, Union
import numpy as np
import matplotlib
class FlappyBirdEnvRGB(gym.Env):
    def __init__(self,screen_size):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0, 255, [*screen_size, 3])
        self._screen_size = screen_size
        self.game = game_physics.render_game(screen_size[0], screen_size[1])
        self.done = False
        self.surface = None
        
    def _get_observation(self):
        return pygame.surfarray.array3d(self.game.surface)
    
    def reset(self):
        """ 
        Resets the environment (starts a new game).
        """
        _, _, _ = self.game.reset_game()
        self.done = False
        return self._get_observation()
    
    def step(self, action: Union[game_physics.render_game.Actions, int],) -> Tuple[np.ndarray, float, bool, Dict]:
        
        alive = self.game.update_state(action)
        obs = self._get_observation()
        reward = 1
        self.done = not alive
        if self.done:
            reward = -10
        info = {"score": self.game.score}
        return obs, reward, self.done, info
    
    def render(self, mode):
        if mode == 'human':
            self.game.make_display(True,True)
            self.game.update_display(True)
            self.surface = None
            
        elif mode=='rgb-array':
            self.game.make_display(False,False)
            self.game.surface = self.game.update_display(False)

        else:
            raise ValueError("Mode is not included")    


env = FlappyBirdEnvRGB((864, 936-70))

env_dict = gym.envs.registration.registry.env_specs
for env in env_dict:
    if 'FlappyBird-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]

gym.envs.registration.register(
    id='FlappyBird-v0',
    entry_point=lambda screen_size: FlappyBirdEnvRGB(screen_size),
    kwargs={'screen_size': ((864, 936-100))},
    max_episode_steps=5_000,
)
