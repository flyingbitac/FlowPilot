#!/usr/bin/env python3
"""Replay a collected window with Phase 2 and save a 2 x 9 depth comparison.

The columns are t=0, 0.2, ..., 1.6 s. Phase 2 jointly generates video and
action latents; t=0 is the observed condition frame decoded with the prediction.

Example:
    python scripts/replay_phase2_comparison.py \
        --trajectory runs/phase2_replay/trajectory.npz --start-time 0.0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

import configs.runtime.rviz as cfg
from inference.runtime import FlowPilotRuntime


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory', type=Path,
                        default=Path('runs/phase2_replay/trajectory.npz'))
    parser.add_argument('--output', type=Path, default=None,
                        help='Default: comparison.png beside the trajectory file.')
    parser.add_argument('--start-time', type=float, default=0.0,
                        help='Condition-frame time within the trajectory in seconds.')
    parser.add_argument('--phase2-ckpt', default=cfg.phase2_ckpt)
    parser.add_argument('--vae-pth', default=cfg.vae_pth)
    parser.add_argument('--device', default=cfg.device)
    parser.add_argument('--num-steps', type=int, default=6)
    parser.add_argument('--v-cmd', type=float, default=cfg.v_cmd)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def predict(runtime, depth, state_goal, v_cmd, num_steps):
    runtime.infer(depth, state_goal, v_cmd=v_cmd, num_steps=num_steps)
    decoded = runtime.vae.decode(runtime._video_latent_buf)
    return decoded[0, :, 0].float().cpu().numpy()


def colorize(depth):
    pixels = (np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)


def save_comparison(gt, pred, output):
    assert gt.shape == (9, 96, 160) and pred.shape == gt.shape
    label_w, time_h, gap = 190, 30, 8
    cell_h = time_h + 96
    canvas = np.full((2 * cell_h + gap + 16, label_w + 9 * 160 + 8 * gap, 3),
                     255, dtype=np.uint8)
    for row, (name, frames) in enumerate((('Ground truth (depth)', gt),
                                          ('Phase-2 prediction', pred))):
        y = 8 + row * (cell_h + gap)
        cv2.putText(canvas, name, (10, y + time_h + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        if row == 1:
            cv2.putText(canvas, '(t=0: condition)', (10, y + time_h + 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        for col, frame in enumerate(frames):
            x = label_w + col * (160 + gap)
            label = f'{col * 0.2:.1f} s'
            cv2.putText(canvas, label, (x + 48, y + 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            canvas[y + time_h:y + time_h + 96, x:x + 160] = colorize(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f'Failed to write {output}')


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    with np.load(args.trajectory, allow_pickle=False) as trajectory:
        depths = trajectory['depths']
        state_goals = trajectory['state_goals']
        dt = float(trajectory['dt'])
        stride = int(trajectory['frame_stride'])
        start = int(round(args.start_time / dt))
        indices = start + np.arange(9) * stride
        if start < 0 or indices[-1] >= len(depths):
            raise ValueError(
                f'Need frames {start}..{start + 8 * stride}, but trajectory has '
                f'{len(depths)} frames. Choose an earlier --start-time.'
            )
        gt = depths[indices].astype(np.float32)
        state_goal = state_goals[start].copy()

    cfg.device = args.device
    cfg.phase2_ckpt = args.phase2_ckpt
    cfg.vae_pth = args.vae_pth
    runtime = FlowPilotRuntime(cfg)
    try:
        pred = predict(runtime, gt[0], state_goal, args.v_cmd, args.num_steps)
    finally:
        runtime.shutdown()
    output = args.output or args.trajectory.with_name('comparison.png')
    save_comparison(gt, pred, output)
    print(f'[replay] condition frame={start}, times=0.0..1.6 s (9 columns)')
    print(f'[replay] saved {output}')


if __name__ == '__main__':
    main()
