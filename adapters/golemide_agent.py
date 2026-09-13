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

The success signal, which this benchmark refuses to provide
-----------------------------------------------------------
The bridge worked on the first attempt and both trials still scored zero, for one reason
logged verbatim:

    no verify command found under /app; leaving this trial unattempted

Not a discovery bug. The task container holds `gates.txt` and `sim.c` and **no tests at
all** -- Harbor injects and runs them after the agent exits, which is how the oracle agent
scores 1.0. Terminal-Bench withholds the success signal on purpose, because deciding when
the job is done is part of what it measures. golemide's entire interface is `--verify CMD`,
"the command whose exit status defines success", so without one it attempts nothing.

That is exactly the gap `emet`'s design document names as its entrance gate: "the entrance
asks whether a request can be turned into acceptance criteria". It was written down as the
missing piece before this run measured it.

So the verify command is sought in three tiers, weakest claim last:

1. `_discover_verify` -- the task's own test entrypoint, if it ships one;
2. `_derive_verify` -- build-and-run on the files present. Measured to fire on neither task
   tried: Terminal-Bench tasks do not share a shape ("write a MIPS interpreter that runs
   Doom" has no mechanical build check), so this is the floor and is left unextended;
3. `_stated_verify` -- `emet`'s entrance gate, at the smallest size that is still the real
   thing. Three independent readings of the request are sampled, and the criteria are used
   only if a majority of them test THE SAME THING (compared by which programs they invoke
   and which paths they touch, not by wording). Then the survivor must fail on the untouched
   task, and must not be an implementation in disguise.

Each of those three checks exists because a weaker version was measured and found wanting:

* a single answer can only be checked for whether it runs, and "it runs" is not "it is the
  right check";
* a criterion that already passes cannot guide an edit;
* asked for a check on an open-ended task, the model returns the SOLUTION as a heredoc
  (`cat > mips.c <<EOF ...`) -- measured, and the reason the shape test is there.

The sampling is the part that makes this a gate rather than a guess: it measures the
REQUEST's determinacy, not the answer's correctness. If three readings disagree about what
the task even checks, the request could not be turned into criteria and abstaining is the
honest outcome.

Nothing in tier 3 can inflate a score: Harbor grades the container independently afterwards,
so stated criteria can only guide or waste attempts, never award them. Abstaining is
therefore cheaper than guessing, and all three tiers abstain rather than substitute a
command that would trivially pass.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
import urllib.request
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

# How many independent readings of the request to take before trusting any of them.
# Three is the smallest number that can show a majority AND a split; two can only agree
# or tie, which cannot distinguish "determinate" from "ambiguous".
_CRITERIA_SAMPLES = 3

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

    async def _derive_verify(self, environment: BaseEnvironment, root: str) -> str | None:
        """A success signal built from what the task ships, when it ships no tests.

        Terminal-Bench withholds its tests -- they are injected after the agent exits -- so
        a loop that needs `--verify` has nothing to iterate against and does nothing at all.
        That is what the first run measured: two trials, zero attempted.

        Deriving one is legitimate rather than a way of scoring itself: Harbor grades the
        container independently afterwards, so a weak or wrong signal here cannot inflate
        the result. It can only fail to guide. The risk runs the other way -- a signal that
        passes too easily makes golemide stop early -- so these are build-and-run checks,
        the weakest claim that is still a claim: the code the task ships must still compile
        and execute after the edit.

        This is the honest floor, not the ideal. The ideal is a component that reads the
        instruction, states acceptance criteria, and abstains when it cannot -- `emet`'s
        entrance gate, which is designed and unbuilt.

        And the floor is not high enough, which was worth measuring rather than assuming.
        It fired on nothing across the two tasks tried, because Terminal-Bench tasks are not
        shaped like exercises:

            circuit-fibsqrt      /app = gates.txt, sim.c
            make-mips-interpreter  /app = doom.wad, doomgeneric, doomgeneric_mips

        The second one is "write a MIPS interpreter that can run Doom". No mechanical
        build-and-run check means anything there; `cc` on a top-level `.c` is not even
        applicable. Deriving a signal per task shape is a losing game on a benchmark whose
        whole point is that the tasks do not share a shape.

        So this function stays as the floor for simple tasks. What extends past it is
        `_stated_verify`, which asks a model to state the criteria and then checks that the
        criteria discriminate -- not another file-shape heuristic.
        """
        probe = await environment.exec(
            f"ls -1 {shlex.quote(root)} 2>/dev/null", timeout_sec=20
        )
        names = [n.strip() for n in (probe.stdout or "").splitlines() if n.strip()]
        if not names:
            return None

        def has(ext: str) -> str | None:
            return next((n for n in names if n.endswith(ext)), None)

        c_file = has(".c")
        if c_file:
            # Compile with warnings as information, then run it. A task whose program needs
            # arguments will exit non-zero and that is still a usable signal: it separates
            # "builds and runs" from "does not build".
            return (
                f"cc -O1 -o /tmp/_assay_build {shlex.quote(c_file)} 2>&1 && /tmp/_assay_build"
            )
        py_file = has(".py")
        if py_file:
            return f"python3 -c 'import py_compile,sys; py_compile.compile({py_file!r}, doraise=True)'"
        rs_file = has(".rs")
        if rs_file:
            return f"rustc --edition 2021 -o /tmp/_assay_build {shlex.quote(rs_file)} && /tmp/_assay_build"
        sh_file = has(".sh")
        if sh_file:
            return f"sh -n {shlex.quote(sh_file)}"
        return None

    @staticmethod
    def _criteria_signature(command: str) -> frozenset[str]:
        """What a candidate check actually looks at, as a comparable set.

        Two checks that invoke different programs against different files are two different
        interpretations of the request, however similar their prose. Comparing raw strings
        would call every rewording a disagreement and every coincidence an agreement; this
        compares the executables named and the paths touched, which is what the check does.
        """
        words = re.findall(r"[A-Za-z0-9_./-]+", command)
        interesting = set()
        for w in words:
            if w in {"-c", "-e", "&&", "||", ";", "set", "sh", "bash", "exit", "$?", "then", "fi"}:
                continue
            if w.startswith("-"):
                continue
            if "." in w or "/" in w or w.isalpha():
                interesting.add(w.lower().lstrip("./"))
        return frozenset(interesting)

    async def _sample_criteria(self, prompt: str, k: int) -> list[str]:
        """k independent statements of the criteria, for measuring their stability."""
        account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        token = os.environ.get("CLOUDFLARE_API_TOKEN")
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{account}"
            "/ai/run/@cf/zai-org/glm-5.3-flash"
        )
        out: list[str] = []
        for _ in range(k):
            body = json.dumps(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8192,
                    "reasoning_effort": "low",
                    # Sampling, not greedy: k identical answers from a deterministic decode
                    # would measure nothing about the request.
                    "temperature": 0.7,
                }
            ).encode()
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=240) as resp:
                    payload = json.loads(resp.read().decode())
            except Exception as exc:
                self.logger.warning("a criteria sample failed: %s", exc)
                continue
            result = payload.get("result") or {}
            text = result.get("response") or ""
            choices = result.get("choices") or []
            if not text and choices:
                text = ((choices[0] or {}).get("message") or {}).get("content") or ""
            if not text.strip():
                # The third silent abstention of this kind, so it is logged like the others:
                # an empty content field looked identical to "the model declined", and the
                # cause was upstream both times (response shape, then token budget).
                self.logger.warning(
                    "a criteria sample returned no content (finish_reason=%s)",
                    ((choices or [{}])[0] or {}).get("finish_reason"),
                )
                continue
            picked = ""
            for line in text.strip().splitlines():
                line = line.strip().strip("`")
                if line and not line.startswith("#"):
                    picked = line
                    break
            if picked:
                out.append(picked)
            else:
                self.logger.warning("a criteria sample had no usable line: %.80s", text)
        return out

    async def _stated_verify(
        self, instruction: str, environment: BaseEnvironment, root: str
    ) -> str | None:
        """Acceptance criteria stated from the instruction, kept only if they discriminate.

        This is `emet`'s entrance gate at its smallest: turn a request into a check, and
        abstain when you cannot. Terminal-Bench withholds its tests, so without this a
        test-driven loop has nothing to iterate against and attempts nothing at all.

        The gate is the second step, not the first. A model asked for a check will happily
        produce one that passes on anything, and a criterion that passes before any edit
        cannot guide an edit. So the proposed command is RUN ON THE UNTOUCHED CONTAINER and
        kept only if it fails there. That is a measurement, not a judgement of the model's
        answer -- the distinction `emet` is built on.

        Nothing here can inflate a score: Harbor grades the container independently
        afterwards. A bad criterion can only waste attempts, which is why it is cheaper to
        abstain than to guess.
        """
        account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        token = os.environ.get("CLOUDFLARE_API_TOKEN")
        if not account or not token:
            # Every abstention says why. A silent one is indistinguishable from a principled
            # one, and this branch was silent while a response-shape bug upstream was the
            # real cause -- which cost a run to find.
            self.logger.warning(
                "no Cloudflare credentials on the host, so criteria cannot be stated; "
                "abstaining"
            )
            return None

        listing = (
            await environment.exec(f"ls -RF {shlex.quote(root)} 2>/dev/null | head -60")
        ).stdout or ""

        prompt = (
            "You are given a task and the files present. Reply with ONE shell command, and "
            "nothing else, that exits 0 only when the task is COMPLETE and non-zero while "
            "it is incomplete. It runs in the project directory. It must actually test the "
            "work: a command that always succeeds is useless. No explanation, no markdown, "
            "no backticks.\n\n"
            # Trimmed hard. A 3000-character instruction plus a 2000-character listing made
            # the model reason long enough to blow both the token budget and the request
            # timeout; the same prompt at these sizes answers in about two seconds. What the
            # gate needs is the task's OBJECT, not its full prose.
            f"TASK:\n{instruction[:1200]}\n\nFILES:\n{listing[:800]}\n"
        )
        # `max_tokens` covers the reasoning as well as the answer, and glm-5.3-flash spends
        # it on `reasoning_content` first. At 300 tokens it returned finish_reason "length"
        # with an EMPTY content field; at 4096 it did the same. That is the worst kind of
        # failure here, because an empty answer is indistinguishable from a principled
        # abstention -- the adapter reported "could not state criteria" when the real cause
        # was its own token budget.
        #
        # `reasoning_effort: low` is the fix, measured against the alternative: at low effort
        # 4096 tokens returns a usable command with finish_reason "stop", while raising the
        # budget to 16384 without it timed out instead. Stating a check does not need deep
        # reasoning; the check's quality is decided by the discriminate test below, not by
        # how long the model thought about it.
        # k interpretations, then agreement. This is the gate: a single answer cannot be
        # checked for anything except whether it runs, and "it runs" is not "it is the right
        # check". Sampling measures the REQUEST -- if three independent readings of it test
        # different things, the request was not determinate enough to turn into criteria, and
        # the honest move is to abstain rather than pick one and call it the goal.
        samples = await self._sample_criteria(prompt, _CRITERIA_SAMPLES)
        if len(samples) < 2:
            self.logger.warning(
                "only %d criteria sample(s) came back; not enough to measure agreement, "
                "abstaining", len(samples)
            )
            return None

        groups: dict[frozenset[str], list[str]] = {}
        for s in samples:
            groups.setdefault(self._criteria_signature(s), []).append(s)
        best_sig, best = max(groups.items(), key=lambda kv: len(kv[1]))
        if len(best) * 2 <= len(samples):
            # No majority: the readings disagree about what the task even checks.
            self.logger.warning(
                "%d samples produced %d different interpretations with no majority; the "
                "request is not determinate enough to state criteria, abstaining: %s",
                len(samples), len(groups),
                " | ".join(sorted(",".join(sorted(g)) for g in groups)[:3]),
            )
            return None

        candidate = best[0]
        self.logger.info(
            "criteria agreed by %d/%d samples on %s", len(best), len(samples),
            ",".join(sorted(best_sig)) or "(nothing identifiable)",
        )

        # Refuse the shapes that cannot discriminate by construction.
        if re.fullmatch(r"(true|:|exit\s+0|/bin/true)\s*;?", candidate):
            self.logger.warning("stated criteria always pass; abstaining: %r", candidate)
            return None

        # The measurement: criteria that pass before any edit are worthless.
        probe = await environment.exec(candidate, cwd=root, timeout_sec=180)
        if probe.return_code == 0:
            self.logger.warning(
                "stated criteria already pass on the untouched task, so they cannot guide "
                "an edit; abstaining: %r", candidate
            )
            return None

        # What the discriminate test cannot catch, measured rather than assumed. Asked for a
        # check on an open-ended task ("write a MIPS interpreter that runs Doom"), the model
        # returns the SOLUTION disguised as a check -- a `cat > mips.c <<EOF ...` heredoc
        # that writes an implementation and then compiles it. Failing-before-the-edit rejects
        # criteria that always pass; it cannot reject criteria that contain their own answer.
        #
        # So this tier gets the loop moving and cannot be trusted to aim it. A gate that could
        # is `emet`'s: sample k interpretations of the request and abstain when they disagree,
        # which measures the request's determinacy instead of trusting one answer about it. A
        # single call has no way to tell a criterion from a solution.
        if len(candidate) > 400 or "<<" in candidate or "cat >" in candidate:
            self.logger.warning(
                "stated criteria look like an implementation rather than a check "
                "(%d chars); abstaining: %.120s", len(candidate), candidate
            )
            return None

        self.logger.info("stated criteria discriminate (exit %s): %s", probe.return_code, candidate)
        return candidate

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
        source = "the task's own"
        if verify is None:
            verify = await self._derive_verify(environment, root)
            source = "derived from the task's files"
            # The same discriminate test the stated tier gets, and it was missing here --
            # applied only where the idea happened to be under consideration, which is how a
            # principle becomes a special case.
            #
            # Measured: on circuit-fibsqrt the derived command was
            # `cc -O1 -o /tmp/_assay_build sim.c && /tmp/_assay_build`, and it PASSES on the
            # untouched task, because `sim.c` compiles and runs fine -- the task is about what
            # the simulation computes, not whether it builds. golemide then correctly refused
            # to work ("already passes -- nothing was changed") and the trial scored zero on a
            # criterion that could never have guided anything.
            if verify is not None:
                probe = await environment.exec(verify, cwd=root, timeout_sec=180)
                if probe.return_code == 0:
                    self.logger.warning(
                        "derived criteria already pass on the untouched task, so they cannot "
                        "guide an edit; abstaining: %s", verify
                    )
                    verify = None
        if verify is None:
            # Last resort, and the only one that reads the instruction: state the criteria,
            # then keep them only if they fail on the untouched task.
            verify = await self._stated_verify(instruction, environment, root)
            source = "stated from the instruction"
        attempts = "6"
        if verify is None:
            # Acting without a success signal, which is what this benchmark actually asks
            # for. The three tiers above all abstained, and abstaining means golemide does
            # nothing and scores zero honestly -- correct behaviour for a loop built around
            # having an oracle, and still zero.
            #
            # So the last resort inverts the premise: a verify that always reports "not
            # done" turns the loop into N best-effort passes. golemide reads the
            # instruction, edits, is told it is not finished, and edits again with its own
            # diff in the history. It never believes it is done; Harbor decides that
            # afterwards, which it was going to do regardless.
            #
            # Attempts are capped low here on purpose. With no signal there is nothing to
            # tell improvement from thrash, and a loop that cannot perceive progress should
            # not be given six chances to churn the same files.
            verify = "/bin/false"
            source = "NONE -- acting without a success signal, best effort only"
            attempts = "2"
            self.logger.warning(
                "no criteria could be discovered, derived or stated for %s; running %s "
                "best-effort passes with no success signal instead of not attempting",
                root, attempts,
            )
        self.logger.info("verify (%s): %s", source, verify)

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
                "--attempts", attempts,
                # Looking around before editing. On a task that ships no tests this is the
                # only way the agent learns anything beyond the file listing, and it is what
                # the no-signal tier is otherwise missing.
                "--explore", os.environ.get("GOLEMIDE_EXPLORE", "1"),
                # Many of these tasks have nothing to repair: the instruction says "create
                # /app/filter.py" and the container ships no source at all. Without this,
                # golemide refuses before calling the model -- which is how three of the ten
                # pilot tasks ended in under 45 seconds for $0.00 and a reward of zero.
                "--create",
            ]
            # Write straight to the log file rather than buffering through a pipe. Harbor
            # kills an agent at the timeout the TASK declares -- `[agent] timeout_sec` in its
            # task.toml, which across terminal-bench-2 runs from 600s to 12000s, so there is no
            # single deadline to design around -- and `communicate()` holds everything in memory
            # until the process exits, so a killed run lost its entire log, including the cost
            # line. That run's spend became unreportable, which is the one kind of missing
            # record that cannot be reconstructed afterwards.
            log_path = self.logs_dir / "golemide.log"
            try:
                with open(log_path, "wb") as sink:
                    proc = await asyncio.create_subprocess_exec(
                        *command,
                        stdout=sink,
                        stderr=asyncio.subprocess.STDOUT,
                        env=dict(os.environ),
                    )
                    try:
                        await proc.wait()
                    except asyncio.CancelledError:
                        # Killed from outside (Harbor's timeout). Reap the child first so it
                        # cannot outlive the trial and keep spending.
                        proc.kill()
                        await proc.wait()
                        raise
            finally:
                # Cost is recorded even when the run was killed, which is the whole point of
                # this restructuring: money was spent either way, and the timed-out case is
                # exactly the one where the old code lost the record.
                out = log_path.read_text(errors="replace") if log_path.exists() else ""
                costs = _COST.findall(out)
                if costs:
                    context.cost_usd = float(costs[-1])
                tin = _TOKENS_IN.findall(out)
                if tin:
                    context.n_input_tokens = int(tin[-1])
                tout = _TOKENS_OUT.findall(out)
                if tout:
                    context.n_output_tokens = int(tout[-1])
                if not costs:
                    self.logger.warning(
                        "no cost line in golemide's output (%d bytes); spend for this trial "
                        "is unrecorded", len(out)
                    )

            # The final state has to be in the container, because that is what Harbor
            # grades. The verify script syncs on every attempt, but a run that ends without
            # a passing verify would otherwise leave the last edit on the host only.
            await self._host("docker", "cp", f"{local}/.", f"{cid}:{root}/")
        finally:
            shutil.rmtree(work, ignore_errors=True)
