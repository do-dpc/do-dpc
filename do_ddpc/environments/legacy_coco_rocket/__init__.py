# mypy: ignore-errors
# pylint: skip-file
from gymnasium.envs.registration import register

from do_ddpc.environments.legacy_coco_rocket.rocketlander import RocketLander

register(
    id="do_ddpc_env/RocketLander-v0",
    entry_point=RocketLander,
    max_episode_steps=None,
)
