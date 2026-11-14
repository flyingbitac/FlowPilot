#!/usr/bin/env python3
"""Collect one 30 Hz Flightmare trajectory with the Phase-2 controller.

Example (inside the simulation container):
    python scripts/collect_phase2_trajectory.py \
        --output runs/phase2_replay/trajectory.npz
"""

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from ruamel.yaml import YAML, RoundTripDumper, dump

import configs.runtime.rviz as cfg
from flowpilot.coords import body_to_world, get_body_frame
from inference.depth_aug import clip_and_noise_depth_from_cfg
from inference.runtime import FlowPilotRuntime
from inference.state import WorldState, build_state_goal
from ros_io import geometry
from simulation import flightmare_proc
from simulation.sim_env import SimEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('runs/phase2_replay/trajectory.npz'))
    parser.add_argument('--goal', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        help='World-frame goal; default is 25 m ahead in +x.')
    parser.add_argument('--duration', type=float, default=6.0,
                        help='Maximum trajectory duration in seconds (default: 6).')
    parser.add_argument('--scene-id', type=int, default=cfg.scene_id)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default=cfg.device)
    parser.add_argument('--phase2-ckpt', default=cfg.phase2_ckpt)
    parser.add_argument('--vae-pth', default=cfg.vae_pth)
    parser.add_argument('--no-launch-unity', action='store_true',
                        help='Connect to an already running Flightmare renderer.')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg.device = args.device
    cfg.phase2_ckpt = args.phase2_ckpt
    cfg.vae_pth = args.vae_pth
    cfg.scene_id = args.scene_id
    cfg.real_time = False

    flightmare_path = os.environ.get('FLIGHTMARE_PATH')
    if not flightmare_path:
        raise RuntimeError('FLIGHTMARE_PATH is not set; run inside the simulation container.')
    from flightgym import QuadrotorEnv_v1

    repo_root = Path(__file__).resolve().parent.parent
    env_cfg = YAML().load((repo_root / 'flightlib/configs/vec_env.yaml').read_text())
    env_cfg['env']['num_envs'] = 1
    env_cfg['env']['num_threads'] = 1
    env_cfg['env']['render'] = True
    env_cfg['env']['supervised'] = False
    env_cfg['env']['imitation'] = False
    env_cfg['env']['seed'] = args.seed
    env = SimEnv(QuadrotorEnv_v1(dump(env_cfg, Dumper=RoundTripDumper), False))
    unity = None
    runtime = None
    try:
        if not args.no_launch_unity:
            binary = Path(flightmare_path) / 'flightrender/RPG_Flightmare/flightmare.x86_64'
            unity = flightmare_proc.launch(str(binary))
        env.connect_unity()
        env.seed(args.seed)
        env.set_map_id(np.array([args.scene_id]))
        env.spawn_trees_and_save_ply(args.scene_id, spacing=cfg.spacing)
        env.render()
        obs = env.reset()
        pos, vel, acc = geometry.obs_to_world_pva(obs)
        cur = WorldState(pos=pos, vel=vel, acc=acc, jerk=np.zeros(3))

        wb = env.world_box.astype(np.float64)
        if args.goal is None:
            goal = cur.pos + np.array([25.0, 0.0, 0.0])
            goal[0] = np.clip(goal[0], wb[0] + 1.0, wb[3] - 1.0)
            goal[1] = np.clip(goal[1], wb[1] + 1.0, wb[4] - 1.0)
            goal[2] = np.clip(goal[2], max(1.0, wb[2] + 1.0), wb[5] - 1.0)
        else:
            goal = np.asarray(args.goal, dtype=np.float64)
        env.set_goal(goal)

        yaw = math.atan2(goal[1] - cur.pos[1], goal[0] - cur.pos[0])
        init_quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        state_goal, quat_wb, r_wb = build_state_goal(
            cur, goal, get_body_frame,
            quat_wb_override=init_quat, r_wb_override=geometry.quat_to_rot(init_quat),
        )
        pitch = math.radians(cfg.camera_pitch_deg)
        q_pitch = np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0])

        runtime = FlowPilotRuntime(cfg)
        runtime.warmup()
        print(f'[collect] start={cur.pos.tolist()} goal={goal.tolist()}', flush=True)

        max_frames = int(round(args.duration * 30.0)) + 1
        depths, waypoints, state_goals = [], [], []

        def capture(state, quat, sg):
            quat_render = geometry.safe_quat(geometry.quat_mul(quat, q_pitch))
            env.set_state(state.pos, state.vel, state.acc, quat_render)
            env.render()
            depth = clip_and_noise_depth_from_cfg(
                env.get_depth()[0].astype(np.float32), cfg, resize_to_net=True,
            )
            depths.append(depth.astype(np.float16))
            waypoints.append(np.concatenate([
                [len(waypoints) / 30.0], state.pos, state.vel, state.acc, state.jerk,
            ]).astype(np.float32))
            state_goals.append(sg.astype(np.float32))

        capture(cur, quat_wb, state_goal)
        while len(depths) < max_frames:
            result = runtime.infer(
                depths[-1].astype(np.float32), state_goal,
                r_wb=r_wb, pos_world=cur.pos,
            )
            if not np.all(np.isfinite(result.wp_body)):
                raise RuntimeError('Phase-2 inference produced NaN/Inf waypoints.')
            wp_world = body_to_world(result.wp_body, cur.pos, r_wb).astype(np.float64)
            wp_world[:, 0] = np.clip(wp_world[:, 0], wb[0] + 0.2, wb[3] - 0.2)
            wp_world[:, 1] = np.clip(wp_world[:, 1], wb[1] + 0.2, wb[4] - 0.2)
            wp_world[:, 2] = np.clip(wp_world[:, 2], max(0.05, wb[2] + 0.05), wb[5] - 0.1)

            exec_n = min(cfg.rollout_points, len(wp_world))
            for wp in wp_world[1:exec_n]:
                if len(depths) >= max_frames:
                    break
                nxt = WorldState(pos=wp[0:3].copy(), vel=wp[3:6].copy(),
                                 acc=wp[6:9].copy(), jerk=wp[9:12].copy())
                sg, quat, _ = build_state_goal(nxt, goal, get_body_frame)
                capture(nxt, quat, sg)

            last = wp_world[exec_n - 1]
            cur = WorldState(pos=last[0:3].copy(), vel=last[3:6].copy(),
                             acc=last[6:9].copy(), jerk=last[9:12].copy())
            state_goal, quat_wb, r_wb = build_state_goal(cur, goal, get_body_frame)
            dist = float(np.linalg.norm(goal - cur.pos))
            print(f'[collect] frames={len(depths)}/{max_frames} dist={dist:.2f} m',
                  flush=True)
            if dist < cfg.goal_reach_dist:
                break

        if len(depths) < 49:
            raise RuntimeError(
                f'Only {len(depths)} frames collected; at least 49 are required '
                'for a 1.6 s replay. Increase --duration or --goal distance.'
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            depths=np.stack(depths),
            waypoints=np.stack(waypoints), state_goals=np.stack(state_goals),
            goal=goal.astype(np.float32), dt=np.float32(1.0 / 30.0),
            frame_stride=np.int32(6), scene_id=np.int32(args.scene_id),
        )
        print(f'[collect] saved {len(depths)} frames to {args.output}', flush=True)
    finally:
        if runtime is not None:
            runtime.shutdown()
        env.disconnect_unity()
        if unity is not None:
            flightmare_proc.terminate(unity)


if __name__ == '__main__':
    main()
