import os
import shutil
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from rl_marine import ppo
from rl_marine.network import network_utils
from rl_marine.ppo.arguments import get_args
from rl_marine.network.envs import make_vec_envs
from rl_marine.network.model import Policy
from rl_marine.network.storage import RolloutStorage

from marine_nav.configs.default import Config
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
    """
    main function for training a robot policy network
    """

    algo_args = get_args()
    run_name = f"ppo_{algo_args.scenario}_k{algo_args.action_delay_k}"

    algo_args.output_dir = os.path.join("trained_models", run_name)

    if not os.path.exists(algo_args.output_dir):
        os.makedirs(algo_args.output_dir)

    elif not algo_args.overwrite:
        raise ValueError("output_dir already exists!")

    save_config_dir = os.path.join(algo_args.output_dir, "configs")
    if not os.path.exists(save_config_dir):
        os.makedirs(save_config_dir)
    shutil.copy("marine_nav/configs/default.py", save_config_dir)
    shutil.copy("marine_nav/configs/__init__.py", save_config_dir)
    shutil.copy("train.py", algo_args.output_dir)
    with open(os.path.join(algo_args.output_dir, "parsed_args.txt"), "w") as f:

        f.write(str(algo_args))

    env_config = config = Config()
    _apply_scenario_config(env_config, algo_args.scenario)
    _apply_scenario_config(config, algo_args.scenario)
    if algo_args.collision_penalty is not None:
        env_config.reward.collision_penalty = algo_args.collision_penalty
        config.reward.collision_penalty = algo_args.collision_penalty
    if hasattr(env_config.robot, "flow_grid_num"):
        algo_args.flow_grid_num = env_config.robot.flow_grid_num

    torch.manual_seed(algo_args.seed)
    torch.cuda.manual_seed_all(algo_args.seed)
    if algo_args.cuda:
        if algo_args.cuda_deterministic:

            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:

            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    torch.set_num_threads(algo_args.num_threads)
    device = torch.device("cuda" if algo_args.cuda else "cpu")

    env_name = algo_args.env_name

    if config.sim.render:
        algo_args.num_processes = 1
        algo_args.num_mini_batch = 1

    if config.sim.render:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel("x(m)", fontsize=16)
        ax.set_ylabel("y(m)", fontsize=16)
        plt.ion()
        plt.show()
    else:
        ax = None

    envs = make_vec_envs(
        env_name,
        algo_args.seed,
        algo_args.num_processes,
        algo_args.gamma,
        None,
        device,
        False,
        config=env_config,
        ax=ax,
        pretext_wrapper=config.env.use_wrapper,
    )

    actor_critic = Policy(
        envs.observation_space.spaces,
        envs.action_space,
        base_kwargs=algo_args,
        base=config.robot.policy,
    )

    actor_critic = torch.compile(actor_critic)

    rollouts = RolloutStorage(
        algo_args.num_steps,
        algo_args.num_processes,
        envs.observation_space.spaces,
        envs.action_space,
        algo_args.graph_hidden_size,
        algo_args.graph_edge_hidden_size,
    )

    if algo_args.resume:
        load_path = algo_args.load_path
        actor_critic.load_state_dict(torch.load(load_path), strict=True)
        print("Loaded the following checkpoint:", load_path)

    nn.DataParallel(actor_critic).to(device)

    agent = ppo.PPO(
        actor_critic,
        algo_args.clip_param,
        algo_args.ppo_epoch,
        algo_args.num_mini_batch,
        algo_args.value_loss_coef,
        algo_args.entropy_coef,
        lr=algo_args.lr,
        eps=algo_args.eps,
        max_grad_norm=algo_args.max_grad_norm,
    )

    obs = envs.reset()
    if isinstance(obs, dict):
        for key in obs:
            rollouts.obs[key][0].copy_(obs[key])
    else:
        rollouts.obs[0].copy_(obs)

    rollouts.to(device)

    episode_rewards = deque(maxlen=100)

    start = time.time()
    num_updates = (
        int(algo_args.num_env_steps) // algo_args.num_steps // algo_args.num_processes
    )

    initial_entropy_coef = 0.2
    action_delay_buffer = None

    for j in range(num_updates):
        if j >= 15000:
            agent.entropy_coef = initial_entropy_coef * (j / float(num_updates))

        if algo_args.use_linear_lr_decay:
            network_utils.update_linear_schedule(
                agent.optimizer, j, num_updates, algo_args.lr
            )

        for step in range(algo_args.num_steps):

            with torch.no_grad():

                rollouts_obs = {}
                for key in rollouts.obs:
                    rollouts_obs[key] = rollouts.obs[key][step]
                rollouts_hidden_s = {}
                for key in rollouts.recurrent_hidden_states:
                    rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][step]
                value, action, action_log_prob, recurrent_hidden_states = (
                    actor_critic.act(
                        rollouts_obs, rollouts_hidden_s, rollouts.masks[step]
                    )
                )

            if algo_args.action_delay_k > 0:
                if action_delay_buffer is None:
                    action_delay_buffer = deque(
                        [
                            torch.zeros_like(action)
                            for _ in range(algo_args.action_delay_k)
                        ],
                        maxlen=algo_args.action_delay_k,
                    )
                executed_action = action_delay_buffer.popleft()
                action_delay_buffer.append(action.detach().clone())
            else:
                executed_action = action

            if config.sim.render:
                envs.render()
            rew_thred = 0.2

            obs, reward, done, infos = envs.step(executed_action)

            if algo_args.action_delay_k > 0 and action_delay_buffer is not None:
                done_mask = torch.as_tensor(
                    done, dtype=torch.bool, device=action.device
                )
                if done_mask.any():
                    for i in range(len(action_delay_buffer)):
                        action_delay_buffer[i][done_mask] = 0.0

            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )
            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                value,
                reward,
                masks,
                bad_masks,
            )

        with torch.no_grad():
            rollouts_obs = {}
            for key in rollouts.obs:
                rollouts_obs[key] = rollouts.obs[key][-1]
            rollouts_hidden_s = {}
            for key in rollouts.recurrent_hidden_states:
                rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][-1]
            next_value = actor_critic.get_value(
                rollouts_obs, rollouts_hidden_s, rollouts.masks[-1]
            ).detach()

        rollouts.compute_returns(
            next_value,
            algo_args.use_gae,
            algo_args.gamma,
            algo_args.gae_lambda,
            algo_args.use_proper_time_limits,
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % algo_args.save_interval == 0 or j == num_updates - 1:
            save_path = os.path.join(algo_args.output_dir, "checkpoints")
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            torch.save(
                actor_critic.state_dict(), os.path.join(save_path, "%.5i" % j + ".pt")
            )

        if j % algo_args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * algo_args.num_processes * algo_args.num_steps
            end = time.time()
            print(
                "[{}] Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward "
                "{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    run_name,
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                    dist_entropy,
                    value_loss,
                    action_loss,
                )
            )

            df = pd.DataFrame(
                {
                    "misc/nupdates": [j],
                    "misc/total_timesteps": [total_num_steps],
                    "fps": int(total_num_steps / (end - start)),
                    "eprewmean": [np.mean(episode_rewards)],
                    "loss/policy_entropy": dist_entropy,
                    "loss/policy_loss": action_loss,
                    "loss/value_loss": value_loss,
                }
            )

            if (
                os.path.exists(os.path.join(algo_args.output_dir, "progress.csv"))
                and j > 20
            ):
                df.to_csv(
                    os.path.join(algo_args.output_dir, "progress.csv"),
                    mode="a",
                    header=False,
                    index=False,
                )
            else:
                df.to_csv(
                    os.path.join(algo_args.output_dir, "progress.csv"),
                    mode="w",
                    header=True,
                    index=False,
                )


if __name__ == "__main__":
    main()
