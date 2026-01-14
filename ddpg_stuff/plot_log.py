#!/usr/bin/env python3
import re
import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt

# Matches: "Episode 9,Reward = 187.126444"
PATTERN = re.compile(r"Episode\s+(\d+),\s*Reward\s*=\s*([-+]?(?:\d*\.\d+|\d+))")


def load_rewards(log_path: str) -> pd.DataFrame:
    episodes, rewards = [], []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                episodes.append(int(m.group(1)))
                rewards.append(float(m.group(2)))

    if not episodes:
        raise ValueError(
            f"No episode/reward lines found in {log_path}. "
            "Expected like: INFO:root:Episode 1,Reward = 261.880222"
        )

    df = pd.DataFrame({"episode": episodes, "reward": rewards})
    df = df.sort_values("episode").drop_duplicates("episode", keep="last").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="run.log", help="Path to log file (default: run.log)")
    ap.add_argument("--ma", type=int, nargs="+", default=[50, 100], help="Moving-average windows (default: 50 100)")
    ap.add_argument("--out", default="", help="Optional output image path, e.g. rewards.png")
    args = ap.parse_args()

    df = load_rewards(args.log)

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["reward"], label="Reward", alpha=0.30)

    for w in args.ma:
        df[f"ma{w}"] = df["reward"].rolling(window=w, min_periods=1).mean()
        plt.plot(df["episode"], df[f"ma{w}"], label=f"MA({w})", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Rewards vs Episode")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
