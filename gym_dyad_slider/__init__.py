from gym.envs.registration import register

register(
    id='DyadSlider-v0',
    entry_point='gym_dyad_slider.envs:DyadSliderEnv'
)
