import argparse

import torch

USV_NAVIGATION_ENV_ID = "USVNavigation-v0"


def get_args():
    parser = argparse.ArgumentParser(description="RL")

    parser.add_argument("--output_dir", type=str, default="trained_models/ppo")

    parser.add_argument("--resume", default=False, action="store_true")

    parser.add_argument(
        "--load-path",
        default="trained_models/ppo32_ln/checkpoints/13887.pt",
        help="path of weights for resume training",
    )
    parser.add_argument(
        "--overwrite",
        default=True,
        action="store_true",
        help="whether to overwrite the output directory in training",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="number of threads used for intraop parallelism on CPU",
    )

    parser.add_argument("--phase", type=str, default="test")

    parser.add_argument(
        "--cuda-deterministic",
        action="store_true",
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)",
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=425, help="random seed (default: 1)"
    )

    parser.add_argument(
        "--num-processes",
        type=int,
        default=128,
        help="how many training processes to use (default: 16)",
    )

    parser.add_argument(
        "--num-mini-batch",
        type=int,
        default=2,
        help="number of batches for ppo (default: 32)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=45,
        help="number of forward steps in A2C (default: 5)",
    )
    parser.add_argument(
        "--recurrent-policy",
        action="store_true",
        default=False,
        help="use a recurrent policy",
    )

    parser.add_argument(
        "--ppo-epoch", type=int, default=5, help="number of ppo epochs (default: 4)"
    )
    parser.add_argument(
        "--clip-param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.0,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--lr", type=float, default=4e-5, help="learning rate (default: 7e-4)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="RMSprop optimizer apha (default: 0.99)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="max norm of gradients (default: 0.5)",
    )

    parser.add_argument(
        "--num-env-steps",
        type=int,
        default=int(8e7),
        help="number of environment steps to train (default: 10e6)",
    )

    parser.add_argument(
        "--use-linear-lr-decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )
    parser.add_argument(
        "--algo", default="ppo", choices=["ppo"], help="algorithm to use"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=500,
        help="save interval, one save per n updates (default: 100)",
    )
    parser.add_argument(
        "--use-gae",
        action="store_true",
        default=True,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="log interval, one log per n updates (default: 10)",
    )
    parser.add_argument(
        "--use-proper-time-limits",
        action="store_true",
        default=False,
        help="compute returns taking into account time limits",
    )

    parser.add_argument(
        "--graph_hidden_size",
        type=int,
        default=384,
        help="spatio-temporal graph hidden size",
    )
    parser.add_argument(
        "--graph_edge_hidden_size",
        type=int,
        default=256,
        help="spatio-temporal graph edge hidden size",
    )
    parser.add_argument(
        "--aux-loss",
        action="store_true",
        default=False,
        help="auxiliary loss on graph outputs",
    )

    parser.add_argument(
        "--dynamic_obstacle_state_size",
        type=int,
        default=3,
        help="dynamic obstacle state feature dimension",
    )
    parser.add_argument(
        "--graph_edge_state_size",
        type=int,
        default=2,
        help="spatio-temporal graph edge feature dimension",
    )
    parser.add_argument(
        "--policy_feature_size",
        type=int,
        default=256,
        help="policy feature dimension",
    )

    parser.add_argument(
        "--graph_embedding_size",
        type=int,
        default=128,
        help="spatio-temporal graph embedding size",
    )
    parser.add_argument(
        "--graph_edge_embedding_size",
        type=int,
        default=64,
        help="spatio-temporal graph edge embedding size",
    )

    parser.add_argument("--attention_size", type=int, default=64, help="Attention size")

    parser.add_argument("--seq_length", type=int, default=45, help="Sequence length")

    parser.add_argument("--use_self_attn", type=bool, default=True)

    parser.add_argument(
        "--env-name", default=USV_NAVIGATION_ENV_ID, help="name of the environment"
    )

    parser.add_argument("--sort_dynamic_obstacles", type=bool, default=True)

    parser.add_argument(
        "--action-delay-k",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="control latency in env steps: action sampled at t is executed at t+k",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="s1",
        choices=["s1", "s2"],
        help="difficulty setting: s1=(10 dynamic, 5 static), s2=(25 dynamic, 10 static)",
    )
    parser.add_argument(
        "--collision-penalty",
        type=float,
        default=None,
        help="override environment collision penalty when set",
    )

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
