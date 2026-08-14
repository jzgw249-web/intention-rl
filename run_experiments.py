import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_CONDITIONS = ("sparse", "intention_pb", "dense", "intention_naive")


def run_one(condition, seed, args):
    output_dir = Path(args.stdout_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{condition}_seed{seed}.stdout.log"
    command = [
        sys.executable, "train.py",
        "--wrapper", condition,
        "--seed", str(seed),
        "--env-id", args.env_id,
        "--total-timesteps", str(args.total_timesteps),
        "--eval-freq", str(args.eval_freq),
        "--eval-episodes", str(args.eval_episodes),
        "--shaping-coeff", str(args.shaping_coeff),
        "--log-dir", args.log_dir,
        "--save-dir", args.save_dir,
    ]
    with output_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"{condition} seed {seed} failed; see {output_path}")
    return condition, seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 50)))
    parser.add_argument("--env-id", default="MiniGrid-Empty-8x8-v0")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--shaping-coeff", type=float, default=1.0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--log-dir", default="main_logs")
    parser.add_argument("--save-dir", default="main_models")
    parser.add_argument("--stdout-dir", default="batch_stdout")
    args = parser.parse_args()

    jobs = [(condition, seed) for condition in args.conditions for seed in args.seeds]
    print(f"Running {len(jobs)} experiments with max_workers={args.max_workers}")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_one, condition, seed, args): (condition, seed)
                   for condition, seed in jobs}
        for future in as_completed(futures):
            condition, seed = futures[future]
            future.result()
            print(f"completed {condition} seed {seed}", flush=True)


if __name__ == "__main__":
    main()
