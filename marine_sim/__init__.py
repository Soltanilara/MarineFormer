from gym.envs.registration import register

USV_NAVIGATION_ENV_ID = "USVNavigation-v0"

register(
    id=USV_NAVIGATION_ENV_ID,
    entry_point="marine_sim.envs:MarineNavigationEnv",
)
