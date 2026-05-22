import logging
import argparse
import os
import shutil
import sys
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

from rl_marine.network.envs import make_vec_envs
from rl_marine.evaluation import evaluate


from rl_marine.network.model import Policy

import marine_sim


def _apply_scenario_config(env_config, scenario):
    if scenario == "s1":
        env_config.sim.dynamic_obstacle_num = 10
        env_config.sim.obstacle_num = 5
        env_config.sim.spawned_dynamic_obstacle_num = 10
        env_config.sim.spawned_obstacle_num = 5
    elif scenario == "s2":
        env_config.sim.dynamic_obstacle_num = 25
        env_config.sim.obstacle_num = 10
        env_config.sim.spawned_dynamic_obstacle_num = 25
        env_config.sim.spawned_obstacle_num = 10
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")


def main():
    parser = argparse.ArgumentParser("Parse configuration file")

    parser.add_argument("--model_dir", type=str, default="trained_models/ppo_s2")

    parser.add_argument("--visualize", default=False, action="store_true")
    parser.add_argument("--base_path", type=str, default="./")

    parser.add_argument("--test_case", type=int, default=500)

    parser.add_argument("--test_model", type=str, default="08000.pt")

    parser.add_argument("--render_traj", default=True, action="store_true")

    parser.add_argument("--save_slides", default=False, action="store_true")

    test_args, remaining_args = parser.parse_known_args()

    if test_args.save_slides:
        test_args.visualize = True

    if os.path.exists(
        os.path.join(
            test_args.base_path,
            test_args.model_dir,
            "social_eval",
            test_args.test_model[:-3],
        )
    ):
        shutil.rmtree(
            os.path.join(
                test_args.base_path,
                test_args.model_dir,
                "social_eval",
                test_args.test_model[:-3],
            )
        )
    sys.path.append(test_args.base_path)
    model_dir_temp = test_args.model_dir
    if model_dir_temp.endswith("/"):
        model_dir_temp = model_dir_temp[:-1]
    print("model_dir_temp", model_dir_temp)

    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining_args
    from rl_marine.ppo.arguments import get_args

    algo_args = get_args()
    from marine_nav.configs.default import Config

    sys.argv = original_argv

    env_config = config = Config()
    _apply_scenario_config(env_config, algo_args.scenario)
    _apply_scenario_config(config, algo_args.scenario)
    if algo_args.collision_penalty is not None:
        env_config.reward.collision_penalty = algo_args.collision_penalty
        config.reward.collision_penalty = algo_args.collision_penalty
    if hasattr(env_config.robot, "flow_grid_num"):
        algo_args.flow_grid_num = env_config.robot.flow_grid_num

    log_file = os.path.join(test_args.base_path, test_args.model_dir, "test")
    if not os.path.exists(log_file):
        os.mkdir(log_file)
    if test_args.visualize:
        log_file = os.path.join(
            test_args.base_path, test_args.model_dir, "test", "test_visual.log"
        )
    else:
        log_file = os.path.join(
            test_args.base_path,
            test_args.model_dir,
            "test",
            "test_" + test_args.test_model + ".log",
        )

    file_handler = logging.FileHandler(log_file, mode="w")
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    logging.basicConfig(
        level=level,
        handlers=[stdout_handler, file_handler],
        format="%(asctime)s, %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("robot FOV %f", config.robot.FOV)
    logging.info("dynamic obstacles FOV %f", config.dynamic_obstacles.FOV)

    torch.manual_seed(algo_args.seed)
    torch.cuda.manual_seed_all(algo_args.seed)
    if algo_args.cuda:
        if algo_args.cuda_deterministic:

            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:

            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    torch.set_num_threads(1)
    device = torch.device("cuda" if algo_args.cuda else "cpu")

    logging.info("Create other envs with new settings")

    if test_args.visualize:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-20.0, 20.0)
        ax.set_ylim(-20.0, 20.0)

        plt.ion()
        plt.show()
    else:
        ax = None

    load_path = os.path.join(
        test_args.base_path, test_args.model_dir, "checkpoints", test_args.test_model
    )

    env_name = algo_args.env_name

    eval_dir = os.path.join(test_args.base_path, test_args.model_dir, "eval")
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    env_config.render_traj = test_args.render_traj
    env_config.save_slides = test_args.save_slides
    env_config.save_path = os.path.join(
        test_args.base_path,
        test_args.model_dir,
        "social_eval",
        test_args.test_model[:-3],
    )
    if not os.path.exists(env_config.save_path):
        os.makedirs(env_config.save_path)
    envs = make_vec_envs(
        env_name,
        algo_args.seed,
        1,
        algo_args.gamma,
        eval_dir,
        device,
        allow_early_resets=True,
        config=env_config,
        ax=ax,
        test_case=test_args.test_case,
        pretext_wrapper=config.env.use_wrapper,
    )

    if config.robot.policy not in ["orca", "social_force"]:

        actor_critic = Policy(
            envs.observation_space.spaces,
            envs.action_space,
            base_kwargs=algo_args,
            base=config.robot.policy,
        )
        actor_critic = torch.compile(actor_critic)
        actor_critic.load_state_dict(torch.load(load_path, map_location=device))
        actor_critic.base.nenv = 1

        nn.DataParallel(actor_critic).to(device)
    else:
        actor_critic = None

    test_size = config.env.test_size

    import time

    st = time.time()

    evaluate(
        actor_critic,
        envs,
        1,
        device,
        test_size,
        logging,
        config,
        algo_args,
        test_args.visualize,
        algo_args.action_delay_k,
    )
    et = time.time()
    print(f"time duration {et-st}")


if __name__ == "__main__":
    main()
