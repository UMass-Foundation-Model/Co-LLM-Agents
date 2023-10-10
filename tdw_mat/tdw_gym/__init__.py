from gym.envs.registration import register

register(
    id='transport_challenge_MA',
    entry_point='tdw_gym.tdw_gym:TDW'
)