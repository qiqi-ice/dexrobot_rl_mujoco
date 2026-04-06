import argparse
import os

import torch

from dexrobot_mujoco.rl import DexRobotGraspEnv, GraspEnvConfig, PPOConfig, PPOTrainer, SubprocVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO for integrated arm-dexterous-hand grasping in MuJoCo.")
    parser.add_argument("--output-dir", default="outputs/grasp_ppo")
    parser.add_argument("--total-steps", type=int, default=10_000_000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--episode-horizon", type=int, default=250)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--train-epochs", type=int, default=20)
    # Transformer architecture
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--ffn-dim-factor", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--no-compile", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def main():
    args = parse_args()
    resolved_device = resolve_device(args.device)
    if resolved_device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU for policy optimization.")

    total_batch = args.num_envs * args.rollout_steps
    print(f"Parallel envs: {args.num_envs}  |  Rollout steps/env: {args.rollout_steps}  →  batch: {total_batch}")
    print(f"Minibatch: {args.minibatch_size}  |  Epochs: {args.train_epochs}")
    print(f"Transformer: d_model={args.d_model}  heads={args.num_heads}  layers={args.num_layers}  "
          f"tokens={args.num_tokens}  ffn×{args.ffn_dim_factor}")

    def make_env(seed_offset):
        def _fn():
            return DexRobotGraspEnv(GraspEnvConfig(
                seed=args.seed + seed_offset,
                episode_horizon=args.episode_horizon,
            ))
        return _fn

    if args.num_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(args.num_envs)])
    else:
        env = DexRobotGraspEnv(GraspEnvConfig(seed=args.seed, episode_horizon=args.episode_horizon))

    trainer = PPOTrainer(
        env,
        PPOConfig(
            seed=args.seed,
            total_steps=args.total_steps,
            rollout_steps=args.rollout_steps,
            minibatch_size=args.minibatch_size,
            train_epochs=args.train_epochs,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            num_tokens=args.num_tokens,
            ffn_dim_factor=args.ffn_dim_factor,
            dropout=args.dropout,
            device=resolved_device,
        ),
    )

    if not args.no_compile and resolved_device == "cuda":
        print("Compiling policy with torch.compile...")
        trainer.policy = torch.compile(trainer.policy)

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.train(args.output_dir)

    if args.num_envs > 1:
        env.close()


if __name__ == "__main__":
    main()
