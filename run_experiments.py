import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_CONDITIONS = (
    "sparse", "potential_geo", "potential_bfs", "intention_naive",
)


def run_one(condition, seed, args):
    stdout_dir = Path(args.stdout_dir)
    stdout_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = stdout_dir / f"{condition}_seed{seed}.stdout.log"
    command = [
        sys.executable, "train.py", "--wrapper", condition,
        "--seed", str(seed), "--env-id", args.env_id,
        "--total-timesteps", str(args.total_timesteps),
        "--eval-freq", str(args.eval_freq),
        "--eval-episodes", str(args.eval_episodes),
        "--eval-seed-base", str(args.eval_seed_base),
        "--shaping-coeff", str(args.shaping_coeff),
        "--obs", args.obs_mode,
        "--log-dir", args.log_dir, "--save-dir", args.save_dir,
    ]
    with stdout_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"{condition} seed {seed} failed; see {stdout_path}")
    return condition, seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 50)))
    parser.add_argument("--env-id", default="MiniGrid-FourRooms-v0")
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--eval-freq", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-seed-base", type=int, default=30_000)
    parser.add_argument("--shaping-coeff", type=float, default=1.0)
    parser.add_argument(
        "--obs", dest="obs_mode", choices=("partial", "augmented"),
        default="partial",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--log-dir", default="task05_logs")
    parser.add_argument("--save-dir", default="task05_models")
    parser.add_argument("--stdout-dir", default="task05_stdout")
    args = parser.parse_args()
    if args.max_workers > 4:
        raise ValueError("task05 concurrency must not exceed 4")
    jobs = [(condition, seed) for condition in args.conditions for seed in args.seeds]
    print(f"Running {len(jobs)} experiments with max_workers={args.max_workers}")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_one, condition, seed, args): (condition, seed)
            for condition, seed in jobs
        }
        for future in as_completed(futures):
            condition, seed = futures[future]
            future.result()
            print(f"completed {condition} seed {seed}", flush=True)


if __name__ == "__main__":
    main()
