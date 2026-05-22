import numpy as np
from rl_marine.ppo.arguments import get_args


class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):

    args = get_args()

    training = BaseConfig()
    training.device = "cuda:0" if args.cuda else "cpu"

    env = BaseConfig()
    env.time_limit = 110
    env.time_step = 0.25
    env.val_size = 100
    env.test_size = 500

    env.randomize_attributes = True
    env.num_processes = args.num_processes

    reward = BaseConfig()
    reward.goal_arrival_reward = 12
    reward.collision_penalty = -20

    reward.encroachment_radius = 0.25
    reward.dynamic_obstacle_encroachment_penalty_factor = 10
    reward.static_obstacle_encroachment_penalty_factor = 10
    reward.gamma = 0.99

    sim = BaseConfig()
    sim.circle_radius = 20 * np.sqrt(2)
    sim.arena_size = 20
    sim.dynamic_obstacle_num = 25
    sim.obstacle_num = 10
    sim.spawned_dynamic_obstacle_num = 25
    sim.spawned_obstacle_num = 10

    sim.dynamic_obstacle_num_range = 0
    sim.predict_steps = 5

    sim.predict_method = "truth"

    sim.render = False

    render_traj = False
    save_slides = False
    save_path = None

    if sim.predict_method == "inferred":
        env.use_wrapper = True
    else:
        env.use_wrapper = False

    dynamic_obstacles = BaseConfig()
    dynamic_obstacles.visible = True

    dynamic_obstacles.policy = "orca_obstacles"
    dynamic_obstacles.radius = 0.3
    dynamic_obstacles.v_pref = 2
    dynamic_obstacles.sensor = "coordinates"

    dynamic_obstacles.FOV = 2.0

    dynamic_obstacles.random_goal_changing = True
    dynamic_obstacles.goal_change_chance = 0.5

    dynamic_obstacles.end_goal_changing = True
    dynamic_obstacles.end_goal_change_chance = 1.0

    dynamic_obstacles.random_radii = False
    dynamic_obstacles.random_v_pref = False

    dynamic_obstacles.random_unobservability = False
    dynamic_obstacles.unobservable_chance = 0.3

    dynamic_obstacles.random_policy_changing = False

    robot = BaseConfig()

    robot.visible = False
    robot.policy = "marineformer"
    robot.radius = 0.3
    robot.v_pref = 2
    robot.sensor = "coordinates"

    robot.FOV = 2

    robot.sensor_range = 5

    robot.flow_a = 5
    robot.flow_b = 5

    robot.flow_grid_num = 8

    action_space = BaseConfig()

    action_space.kinematics = "unicycle"

    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.15
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    sf = BaseConfig()
    sf.A = 2.0
    sf.B = 1
    sf.KI = 1

    data = BaseConfig()
    data.tot_steps = 40000
    data.render = False
    data.collect_train_data = False
    data.num_processes = 5
    data.data_save_dir = "gst_updated/datasets/orca_20dynamic_obstacles_no_rand"

    data.pred_timestep = 0.25
    pred = BaseConfig()
    pred.model_dir = "gst_updated/results/100-gumbel_social_transformer-faster_lstm-lr_0.001-init_temp_0.5-edge_head_0-ebd_64-snl_1-snh_8-seed_1000_rand/sj"

    lidar = BaseConfig()
    lidar.angular_res = 5
    lidar.range = 10

    if sim.predict_method == "inferred" and env.use_wrapper == False:
        raise ValueError("If using inferred prediction, you must wrap the envs!")
    if sim.predict_method != "inferred" and env.use_wrapper:
        raise ValueError(
            "If not using inferred prediction, you must NOT wrap the envs!"
        )
