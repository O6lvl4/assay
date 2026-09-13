#!/usr/bin/env python3
"""Print the score of a harbor job directory.

Reads each trial's result.json rather than the job-level summary, because the two can
disagree in the case that matters: a trial that was cancelled or that errored records no
reward AND no cost, and the job summary happily averages it in as a zero. A zero earned by
failing the task and a zero from a trial that never finished are different measurements,
and the exception column is what tells them apart.

The cost total is labelled a floor for the same reason. golemide prints its spend on its
summary line; a trial killed before that line spent money this script cannot see. That is
how an earlier run of this harness reported "$2.2" for a night that had actually spent
more -- so the number here never claims to be the spend unless every trial priced itself.
"""

import json
import pathlib
import sys
from datetime import datetime


def seconds(a, b):
    if not (a and b):
        return None
    fmt = "%Y-%m-%dT%H:%M:%S.%f%z"
    return (
        datetime.strptime(b.replace("Z", "+0000"), fmt)
        - datetime.strptime(a.replace("Z", "+0000"), fmt)
    ).total_seconds()


def main(job_dir):
    job = pathlib.Path(job_dir)
    rows = []
    for f in sorted(job.glob("*/result.json")):
        d = json.load(open(f))
        verifier = d.get("verifier_result") or {}
        agent = d.get("agent_result") or {}
        execution = d.get("agent_execution") or {}
        rows.append(
            dict(
                task=d["task_id"]["name"],
                reward=(verifier.get("rewards") or {}).get("reward"),
                cost=agent.get("cost_usd"),
                exc=(d.get("exception_info") or {}).get("exception_type"),
                secs=seconds(execution.get("started_at"), execution.get("finished_at")),
            )
        )
    if not rows:
        print(f"no trials found in {job}")
        return 1

    fmt = "%-34s %7s %10s %7s  %s"
    print(fmt % ("task", "reward", "cost_usd", "secs", "exception"))
    for r in sorted(rows, key=lambda r: (-(r["reward"] or 0), r["task"])):
        print(
            fmt
            % (
                r["task"],
                "-" if r["reward"] is None else r["reward"],
                "-" if r["cost"] is None else r["cost"],
                "-" if r["secs"] is None else "%.0f" % r["secs"],
                r["exc"] or "",
            )
        )

    solved = [r for r in rows if r["reward"] == 1.0]
    finished = [r for r in rows if r["reward"] is not None]
    unfinished = [r for r in rows if r["reward"] is None]
    priced = [r["cost"] for r in rows if r["cost"] is not None]

    print()
    if not finished:
        # "0/0 = 0.0%" reads as a measured zero. Nothing was measured.
        print("score: none of the %d trial(s) produced a reward, so there is no score" % len(rows))
    else:
        print(
            "score: %d/%d = %.1f%%  (of %d trials that produced a reward)"
            % (len(solved), len(finished), 100.0 * len(solved) / len(finished), len(finished))
        )
    if unfinished:
        print(
            "  %d trial(s) produced no reward and are excluded, not counted as failures: %s"
            % (len(unfinished), ", ".join(r["task"] for r in unfinished))
        )
    print(
        "cost: $%.4f over %d/%d trials%s"
        % (
            sum(priced),
            len(priced),
            len(rows),
            ""
            if len(priced) == len(rows)
            else "  -- a FLOOR: the rest spent money with no recorded line",
        )
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: tb2-score.py <job-dir>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
