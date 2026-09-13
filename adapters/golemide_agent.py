"""Run golemide as a Harbor agent, so it can be measured against Claude Code directly.

Why this exists
---------------
Published leaderboard numbers cannot answer "is this system better than Claude Code":

* the scaffold behind each row is undisclosed, and the scaffold moves a score more than a
  model generation does (Claude Opus 4 on aider polyglot: 37.3% at one attempt, 72.0% at
  two -- larger than the whole 3.5-Sonnet-to-Opus-4 span on the same benchmark);
* the Claude models that ARE on the comparable coding leaderboards are from 2025-05, so
  beating them is not beating current Claude;
* the agentic composite cannot be decomposed -- the model holding the published 80.2
  appears on none of its own component leaderboards.

Harbor removes all three at once. `claude-code` is a built-in agent, so both systems run on
the same 89 tasks, in the same containers, on the same day, and the comparison needs no
published figure. See ../DESIGN.md for the measurements behind each claim.

How it works, and why it is not the obvious way
-----------------------------------------------
The obvious design is to upload a Linux golemide into the task container. That was built
and abandoned after three measured failures, each caught before any model call:

1. the aarch64 build hit "cannot execute: required file not found" -- Terminal-Bench images
   are x86_64 even on an arm64 host, which runs them under emulation;
2. the x86_64 build hit "GLIBC_2.39 not found" -- the task image is Debian 12 (glibc 2.36)
   while the Almide release toolchain links against 2.39;
3. building Almide from source against 2.36 exhausted the machine's memory under emulation,
   and `CARGO_BUILD_TARGET=...-musl` is ignored by `almide build`, so the static binary that
   would sidestep glibc entirely is not available.

So golemide runs on the HOST, where its native binary already works, and only the verify
command crosses into the container. That is a fit rather than a workaround: golemide's whole
interface is a directory plus a shell command whose exit status defines success, so the
container boundary lands exactly on the shell command.

The bridge: a container's hostname is its short id in Docker (verified), so the host can
reach it with `docker exec`. Each verify pushes the host's working copy in with `docker cp`
and then runs the task's own tests inside, which keeps the two trees in step without
golemide knowing anything about containers.

Usage
-----
    hb run -a adapters.golemide_agent:GolemideAgent \
           -m cf:glm-5.3 \
           -d terminal-bench/terminal-bench-2 \
           -l 5 -n 2

`GOLEMIDE_BINARY` points at the host build. Cloudflare credentials are read from the host
environment; they never enter the container, because golemide never runs there.

What this adapter found, and why it cannot be fixed here
--------------------------------------------------------
The bridge works. Two trials completed with no exceptions, and both scored zero for one
reason, logged verbatim:

    no verify command found under /app; leaving this trial unattempted

Not a discovery bug. The task container holds `gates.txt` and `sim.c` and **no tests at
all** -- Harbor injects and runs them after the agent exits, which is how the oracle agent
scores 1.0. Terminal-Bench withholds the success signal on purpose.

golemide's entire interface is `--verify CMD`, "the command whose exit status defines
success". On this benchmark no such command exists for the agent to have. So golemide
cannot compete here as designed, and no amount of adapter work changes that: the benchmark
measures the one thing golemide delegates to its caller -- deciding when the job is done,
without being told.

That is the same gap `emet`'s design document names as its entrance gate: "the entrance
asks whether a request can be turned into acceptance criteria". It was written down as the
missing piece before this run measured it.

So the honest boundary: a test-driven loop can be made cheaper, more careful and better
targeted -- all measured, all real -- and it still cannot enter a benchmark that refuses to
tell it what passing means. Closing that needs a component that manufactures acceptance
criteria from prose and abstains when it cannot, which is a different program from this one.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shlex
import shutil
import tempfile
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

# Candidate verify commands, most specific first. Discovered by probing the container, not
# assumed: golemide stops at the first command that exits zero, so a command that trivially
# passes would score every task as solved.
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
    """golemide on the host, with its verify command bridged into the task container."""

    capabilities = AgentCapabilities()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._container: str | None = None

    @staticmethod
    @override
    def name() -> str:
        return "golemide"

    @override
    def version(self) -> str | None:
        # A content hash, not a timestamp: two builds of one source must compare equal, and
        # a changed source must not be able to report an unchanged version.
        path = os.environ.get("GOLEMIDE_BINARY")
        if path and Path(path).is_file():
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
            return f"{Path(path).name}@{digest}"
        return None

    async def _host(self, *argv: str) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        return proc.returncode or 0, stdout.decode(errors="replace")

    # --- setup -------------------------------------------------------------

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        binary = os.environ.get("GOLEMIDE_BINARY")
        if not binary or not Path(binary).is_file():
            raise RuntimeError(
                "GOLEMIDE_BINARY must point at a golemide build for THIS host (golemide "
                f"runs on the host, not in the container). Got: {binary!r}"
            )

        # The container's hostname is its short id, which is how the host reaches it.
        probe = await environment.exec("hostname", timeout_sec=20)
        cid = (probe.stdout or "").strip()
        if not cid:
            raise RuntimeError("could not read the container's hostname; no way to bridge")
        self._container = cid

        # Prove the bridge before a task depends on it: a `docker exec` that fails later
        # would look like golemide being unable to solve anything.
        rc, out = await self._host("docker", "exec", cid, "true")
        if rc != 0:
            raise RuntimeError(
                f"cannot `docker exec` into {cid} from the host (exit {rc}): {out.strip()}"
            )

    # --- run ---------------------------------------------------------------

    async def _discover_verify(self, environment: BaseEnvironment, root: str) -> str | None:
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
        cid = self._container
        if cid is None:
            raise RuntimeError("setup did not run")

        if await environment.is_dir("/app"):
            root = "/app"
        else:
            root = ((await environment.exec("pwd")).stdout or "/").strip() or "/"

        verify = await self._discover_verify(environment, root)
        if verify is None:
            # Recorded, not substituted. A task whose tests cannot be found is a task this
            # adapter cannot measure; a fallback that exits zero would score it as solved.
            self.logger.warning(
                "no verify command found under %s; leaving this trial unattempted", root
            )
            return

        work = Path(tempfile.mkdtemp(prefix="golemide-harbor-"))
        try:
            local = work / "src"
            await environment.download_dir(root, local)

            # The verify command golemide runs on the host: push the working copy into the
            # container, then run the task's own tests there. The exit status passes through
            # untouched, so golemide's success signal is the container's, not the host's.
            script = work / "verify.sh"
            script.write_text(
                "#!/bin/sh\n"
                f"docker cp {shlex.quote(str(local))}/. "
                f"{cid}:{shlex.quote(root)}/ >/dev/null 2>&1 || exit 111\n"
                f"docker exec -w {shlex.quote(root)} {cid} sh -lc {shlex.quote(verify)}\n"
                "exit $?\n"
            )
            script.chmod(0o755)

            command = [
                os.environ["GOLEMIDE_BINARY"],
                "solve",
                instruction,
                "--root", str(local),
                "--verify", str(script),
                "--model", self.model_name or "cf:glm-5.3",
                "--attempts", "6",
            ]
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=dict(os.environ),
            )
            stdout, _ = await proc.communicate()
            out = stdout.decode(errors="replace")
            (self.logs_dir / "golemide.log").write_text(out)

            # The final state has to be in the container, because that is what Harbor
            # grades. The verify script syncs on every attempt, but a run that ends without
            # a passing verify would otherwise leave the last edit on the host only.
            await self._host("docker", "cp", f"{local}/.", f"{cid}:{root}/")

            # Harbor grades the container, so nothing here decides pass or fail. What this
            # records is cost -- the quantity this project has actually moved (-38% to
            # -41.5% per attempt across three measured A/Bs).
            costs = _COST.findall(out)
            if costs:
                context.cost_usd = float(costs[-1])
            tin = _TOKENS_IN.findall(out)
            if tin:
                context.n_input_tokens = int(tin[-1])
            tout = _TOKENS_OUT.findall(out)
            if tout:
                context.n_output_tokens = int(tout[-1])
        finally:
            shutil.rmtree(work, ignore_errors=True)
