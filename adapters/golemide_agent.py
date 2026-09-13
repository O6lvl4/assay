"""Run golemide as a Harbor agent, so it can be measured against Claude Code directly.

Why this exists
---------------
Published leaderboard numbers cannot answer "is this system better than Claude Code":

* the scaffold behind each row is undisclosed, and the scaffold moves a score more than a
  model generation does (Claude Opus 4 on aider polyglot: 37.3% at one attempt, 72.0% at
  two -- larger than the whole 3.5-Sonnet-to-Opus-4 span on the same benchmark);
* the Claude models that ARE on the comparable coding leaderboards are from 2025-05, so
  beating them is not beating current Claude;
* the agentic composite cannot be decomposed at all -- the model holding the published
  80.2 appears on none of its own component leaderboards.

Harbor removes all three at once. `claude-code` is a built-in agent, so both systems run
on the same 89 tasks, in the same containers, on the same day, and the comparison needs no
published figure. See ../DESIGN.md for the measurements behind each claim.

How it works
------------
golemide is a native binary that takes a directory and a verify command and runs an
edit/verify loop until the command exits zero. Harbor runs agents inside the task
container, so the binary has to be a Linux one: built from the Almide release's
`almide-linux-aarch64` toolchain, uploaded in `setup`, and executed in `run`.

The verify command is the task's own test entrypoint, discovered in the container rather
than assumed, because a wrong verify command silently measures nothing -- the loop stops on
the first thing that exits zero.

Usage
-----
    hb run -a adapters.golemide_agent:GolemideAgent \
           -m cf:glm-5.3 \
           -d terminal-bench/terminal-bench-2 \
           -l 5 -n 2

`GOLEMIDE_BINARY` points at the Linux build. Cloudflare credentials are read from the
host environment and forwarded into the container, never written to disk in it.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import override

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# golemide prints a final line like: SOLVED in 2 attempt(s), $0.031, 47s
_COST = re.compile(r"\$([0-9]+\.[0-9]+)")
_TOKENS_IN = re.compile(r"\bin=([0-9]+)")
_TOKENS_OUT = re.compile(r"\bout=([0-9]+)")

# Where the binary lands in the container. /usr/local/bin is on PATH in every image the
# Terminal-Bench tasks use, but the path is explicit at the call site anyway so a task
# with an unusual PATH cannot change what runs.
_REMOTE_BIN = "/usr/local/bin/golemide"

# Candidate verify commands, most specific first. Discovered by probing the container, not
# assumed: golemide stops at the first command that exits zero, so handing it a command
# that trivially passes would make every task look solved.
_VERIFY_CANDIDATES: list[tuple[str, str]] = [
    ("run-tests.sh", "bash run-tests.sh"),
    ("tests/run-tests.sh", "bash tests/run-tests.sh"),
    ("Makefile", "make test"),
    ("pyproject.toml", "python -m pytest -q"),
    ("pytest.ini", "python -m pytest -q"),
    ("tests", "python -m pytest -q tests"),
    ("Cargo.toml", "cargo test -q"),
    ("package.json", "npm test"),
]


class GolemideAgent(BaseAgent):
    """golemide, driven inside a Harbor task container."""

    capabilities = AgentCapabilities()

    @staticmethod
    @override
    def name() -> str:
        return "golemide"

    @override
    def version(self) -> str | None:
        # The binary's own identity, so a result can be traced to a build.
        binary = os.environ.get("GOLEMIDE_BINARY", "")
        if not binary:
            return None
        try:
            return Path(binary).stat().st_mtime.__str__()
        except OSError:
            return None

    # --- setup -------------------------------------------------------------

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        binary = os.environ.get("GOLEMIDE_BINARY")
        if not binary:
            raise RuntimeError(
                "GOLEMIDE_BINARY is unset. It must point at a Linux build of golemide "
                "matching the container architecture; a macOS binary will upload fine and "
                "then fail to execute, which looks like the agent solving nothing."
            )
        src = Path(binary)
        if not src.is_file():
            raise RuntimeError(f"GOLEMIDE_BINARY does not exist: {src}")

        await environment.upload_file(src, _REMOTE_BIN)
        await environment.exec(f"chmod +x {shlex.quote(_REMOTE_BIN)}")

        # Fail here rather than mid-run: a binary built for the wrong architecture reports
        # "Exec format error", and finding that out per-task wastes the whole job.
        #
        # `observe` is the probe because it is the only subcommand that exits zero without
        # calling a model. `--help` exits 1 (golemide has no such flag and treats it as an
        # unknown command), which as a probe would have failed every task in the job while
        # looking like an architecture problem.
        probe = await environment.exec(f"{shlex.quote(_REMOTE_BIN)} observe --root /tmp")
        if probe.return_code != 0:
            raise RuntimeError(
                f"golemide will not execute in this container (exit {probe.return_code}). "
                f"Check the binary's architecture.\n{probe.stdout}\n"
            )

    # --- run ---------------------------------------------------------------

    async def _discover_verify(self, environment: BaseEnvironment, root: str) -> str | None:
        """The task's own test entrypoint, found by looking rather than guessing."""
        for marker, command in _VERIFY_CANDIDATES:
            probe = await environment.exec(
                f"test -e {shlex.quote(f'{root}/{marker}')}", timeout_sec=20
            )
            if probe.return_code == 0:
                return command
        return None

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        root = "/app"
        if not await environment.is_dir(root):
            probe = await environment.exec("pwd")
            root = (probe.stdout or "/").strip() or "/"

        verify = await self._discover_verify(environment, root)
        if verify is None:
            # Recorded, not silently substituted. A task whose tests this adapter cannot
            # find is a task this adapter cannot measure, and saying so is the only honest
            # outcome -- a fallback command that exits zero would score it as solved.
            self.logger.warning(
                "no verify command found under %s; golemide needs one to have a success "
                "signal, so this trial is left unattempted", root
            )
            return

        creds = {
            k: v
            for k, v in os.environ.items()
            if k in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")
        }
        if not creds:
            raise RuntimeError(
                "no Cloudflare credentials in the host environment; golemide cannot reach "
                "a model and every task would fail for a reason unrelated to the agent"
            )

        model = self.model_name or "cf:glm-5.3"
        command = (
            f"{shlex.quote(_REMOTE_BIN)} solve {shlex.quote(instruction)} "
            f"--root {shlex.quote(root)} --verify {shlex.quote(verify)} "
            f"--model {shlex.quote(model)} --attempts 6"
        )

        result = await environment.exec(
            command,
            cwd=root,
            env=creds,
            timeout_sec=None,
        )

        out = (result.stdout or "") + "\n" + (getattr(result, "stderr", "") or "")
        (self.logs_dir / "golemide.log").write_text(out)

        # Harbor grades the container, not golemide's own verdict, so nothing here decides
        # pass or fail. What it does record is the cost, which is the number this project
        # has been able to move: -38% to -41.5% per attempt across three measured A/Bs.
        costs = _COST.findall(out)
        if costs:
            context.cost_usd = float(costs[-1])
        tin = _TOKENS_IN.findall(out)
        if tin:
            context.n_input_tokens = int(tin[-1])
        tout = _TOKENS_OUT.findall(out)
        if tout:
            context.n_output_tokens = int(tout[-1])
