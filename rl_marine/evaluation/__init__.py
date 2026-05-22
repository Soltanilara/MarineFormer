import numpy as np
import torch
from collections import deque

from marine_sim.envs.utils.info import *


def evaluate(
    actor_critic,
    eval_envs,
    num_processes,
    device,
    test_size,
    logging,
    config,
    args,
    visualize=False,
    action_delay_k=0,
):
    """function to run all testing episodes and log the testing metrics"""

    eval_episode_rewards = []

    if config.robot.policy not in ["orca", "social_force"]:
        eval_recurrent_hidden_states = {}

        node_num = 1
        edge_num = actor_critic.base.dynamic_obstacle_num + 1
        eval_recurrent_hidden_states["graph_temporal_state"] = torch.zeros(
            num_processes,
            node_num,
            actor_critic.base.graph_hidden_size,
            device=device,
        )

        eval_recurrent_hidden_states["graph_edge_state"] = torch.zeros(
            num_processes,
            edge_num,
            actor_critic.base.graph_edge_hidden_size,
            device=device,
        )

    eval_masks = torch.zeros(num_processes, 1, device=device)

    success_times = []
    collision_times = []
    timeout_times = []

    success = 0
    collision = 0
    timeout = 0
    too_close_ratios = []
    min_dist = []

    collision_cases = []
    timeout_cases = []

    all_path_len = []

    if hasattr(eval_envs.venv, "envs"):
        baseEnv = eval_envs.venv.envs[0].env
    else:
        baseEnv = eval_envs.venv.unwrapped.envs[0].env
    time_limit = baseEnv.time_limit

    for k in range(test_size):
        baseEnv.episode_k = k
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0
        obs = eval_envs.reset()

        global_time = 0.0
        path_len = 0.0
        too_close = 0.0
        last_pos = obs["ego_state"][0, 0, :2].cpu().numpy()
        print(f"Rand seed {eval_envs.envs[0].rand_seed}")

        action_delay_buffer = None
        if action_delay_k > 0:

            action_delay_buffer = None

        while not done:
            stepCounter = stepCounter + 1
            if config.robot.policy not in ["orca", "social_force"]:

                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True,
                    )
            else:
                action = torch.zeros([1, 2], device=device)

            if action_delay_k > 0:
                if action_delay_buffer is None:
                    action_delay_buffer = deque(
                        [torch.zeros_like(action) for _ in range(action_delay_k)],
                        maxlen=action_delay_k,
                    )
                executed_action = action_delay_buffer.popleft()
                action_delay_buffer.append(action.detach().clone())
            else:
                executed_action = action

            if not done:
                global_time = baseEnv.global_time

            if visualize:
                eval_envs.render()

            obs, rew, done, infos = eval_envs.step(executed_action)

            rewards.append(rew)

            path_len = path_len + np.linalg.norm(
                obs["ego_state"][0, 0, :2].cpu().numpy() - last_pos
            )
            last_pos = obs["ego_state"][0, 0, :2].cpu().numpy()

            if isinstance(infos[0]["info"], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]["info"].min_dist)

            episode_rew += rew[0]

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device,
            )

            for info in infos:
                if "episode" in info.keys():
                    eval_episode_rewards.append(info["episode"]["r"])

        print("")
        print("Reward={}".format(episode_rew))
        print("Episode", k, "ends in", stepCounter)

        too_close_ratios.append(too_close / stepCounter * 100)

        if isinstance(infos[0]["info"], ReachGoal):
            success += 1
            success_times.append(global_time)
            all_path_len.append(path_len)
            print("Success")
        elif isinstance(infos[0]["info"], Collision):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            print("Collision")
        elif isinstance(infos[0]["info"], Timeout):
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(time_limit)
            print("Time out")
        elif isinstance(infos[0]["info"] is None):
            pass
        else:
            raise ValueError("Invalid end signal from environment")

    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size
    assert success + collision + timeout == test_size
    avg_nav_time = (
        sum(success_times) / len(success_times) if success_times else time_limit
    )

    logging.info(
        "Testing success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, "
        "nav time: {:.2f}, path length: {:.2f}, average intrusion ratio: {:.2f}%, "
        "average minimal distance during intrusions: {:.2f}".format(
            success_rate,
            collision_rate,
            timeout_rate,
            avg_nav_time,
            np.mean(all_path_len),
            np.mean(too_close_ratios),
            np.mean(min_dist),
        )
    )

    logging.info("Collision cases: " + " ".join([str(x) for x in collision_cases]))
    logging.info("Timeout cases: " + " ".join([str(x) for x in timeout_cases]))
    print(
        " Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(eval_episode_rewards), np.mean(eval_episode_rewards)
        )
    )

    eval_envs.close()
