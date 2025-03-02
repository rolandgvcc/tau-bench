# Copyright Sierra

import os
import json
import random
import argparse
from typing import List
from datetime import datetime

from tau_bench.envs import get_env
from tau_bench.agents.grok_agent import ToolCallingAgent
from tau_bench.types import EnvRunResult, RunConfig
from tau_bench.envs.user import UserStrategy
from concurrent.futures import ThreadPoolExecutor
from tau_bench.run import display_metrics


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Run script for Grok agent")
    parser.add_argument(
        "--num-trials", type=int, default=1, help="Number of trials to run"
    )
    parser.add_argument(
        "--env",
        type=str,
        choices=["retail", "airline"],
        default="retail",
        help="Environment to run",
    )
    parser.add_argument("--model", type=str, default="grok", help="Grok model name")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="Task split",
    )
    parser.add_argument(
        "--start-index", type=int, default=0, help="Starting task index"
    )
    parser.add_argument(
        "--end-index", type=int, default=-1, help="Ending task index (-1 for all tasks)"
    )
    parser.add_argument(
        "--task-ids", type=int, nargs="+", help="Specific task IDs to run"
    )
    parser.add_argument(
        "--log-dir", type=str, default="results", help="Directory for logs"
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=1, help="Number of parallel tasks"
    )
    parser.add_argument("--seed", type=int, default=10, help="Random seed")
    parser.add_argument("--shuffle", type=int, default=0, help="Shuffle tasks (0 or 1)")
    parser.add_argument(
        "--user-strategy",
        type=str,
        default="llm",
        choices=[item.value for item in UserStrategy],
        help="User strategy",
    )

    args = parser.parse_args()
    return RunConfig(
        model_provider="openai",
        user_model_provider="openai",
        model=args.model,
        user_model=args.model,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy="tool-calling",  # Fixed to use Grok's tool-calling
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=None,  # Not needed for Grok
    )


def run(config: RunConfig) -> List[EnvRunResult]:
    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    ckpt_path = f"{config.log_dir}/grok-{config.model}-{config.temperature}_range_{config.start_index}-{config.end_index}_{time_str}.json"
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )

    agent = ToolCallingAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model=config.model,
        provider=config.model_provider,
        temperature=config.temperature,
    )

    end_index = (
        len(env.tasks)
        if config.end_index == -1
        else min(config.end_index, len(env.tasks))
    )
    results: List[EnvRunResult] = []

    idxs = (
        config.task_ids
        if config.task_ids and len(config.task_ids) > 0
        else list(range(config.start_index, end_index))
    )
    if config.shuffle:
        random.shuffle(idxs)

    def _run(idx: int) -> EnvRunResult:
        isolated_env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            task_split=config.task_split,
            user_provider=config.user_model_provider,
            task_index=idx,
        )

        print(f"Running task {idx}")
        res = agent.solve(env=isolated_env, task_index=idx)
        result = EnvRunResult(
            task_id=idx,
            reward=res.reward,
            info=res.info,
            traj=res.messages,
            trial=0,
        )
        print("✅" if result.reward == 1 else "❌", f"task_id={idx}", result.info)
        return result

    for i in range(config.num_trials):
        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            trial_results = list(executor.map(_run, idxs))
            results.extend(trial_results)

    # Add this line to display metrics
    display_metrics(results)

    with open(ckpt_path, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")

    return results


def main():
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
