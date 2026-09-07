"""Generic allowlisted-script runner for a local (optionally LAN-exposed) web
console.

This is the whole reason such
a console can safely bind beyond loopback: the client can only name an action
id from a caller-supplied `ACTIONS` registry and provide values for that
action's declared arguments. The command line is assembled here, server-side,
and never accepted from the client -- a generic "run a command" endpoint
would otherwise be remote code execution for anyone who can reach the port.

Jobs run one at a time by design: a game's pipeline scripts commonly write
the same files, and two concurrent writers to the same jsonl would interleave
their output.

`build_cmd` interprets an action's `args` spec (a list of dicts): `fixed`
appends a constant value (optionally glob-expanded server-side, never via a
shell); `flag` appends a bare switch when truthy; `int`/`text`/`choice` pull a
value from the caller-supplied `values` dict, falling back to the spec's
`default`, and `choice` is validated against an explicit `choices` list.
"""

import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class Runner:
    """Runs one allowlisted action at a time, buffering its output."""

    def __init__(self, actions_by_id: dict, repo_root: Path):
        self.actions_by_id = actions_by_id
        self.repo_root = repo_root
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.order: list[str] = []
        self.active: str | None = None

    def build_cmd(self, action: dict, values: dict) -> list[str]:
        cmd = [sys.executable, action["cmd"]]
        for spec in action["args"]:
            name, kind = spec.get("name"), spec["type"]
            if kind == "fixed":
                if spec.get("glob"):
                    # Expand here rather than handing a pattern to a shell.
                    import glob as _g
                    hits = sorted(_g.glob(str(self.repo_root / spec["value"][0])))
                    if name:
                        cmd.append(name)
                    cmd += hits or spec["value"]
                else:
                    if name:
                        cmd.append(name)
                    cmd += list(spec["value"])
                continue

            raw = values.get(spec.get("key") or name or spec.get("label"))
            if kind == "flag":
                if raw if raw is not None else spec.get("default"):
                    cmd.append(name)
            elif kind == "int":
                v = int(raw) if str(raw or "").strip() else spec.get("default")
                cmd += ([name] if name else []) + [str(int(v))]
            elif kind in ("text", "choice"):
                v = raw if raw not in (None, "") else spec.get("default", "")
                if v == "":
                    continue
                if kind == "choice" and v not in spec.get("choices", []):
                    raise ValueError(f"{v!r} is not an allowed value")
                cmd += ([name] if name else []) + [str(v)]
        return cmd

    def start(self, action_id: str, values: dict) -> dict:
        action = self.actions_by_id.get(action_id)
        if action is None:
            return {"ok": False, "error": f"unknown action {action_id!r}"}
        with self.lock:
            if self.active and self.jobs[self.active]["status"] == "running":
                return {"ok": False, "error": "Another job is still running. "
                                              "These scripts share output files, so they run one at a time."}
            try:
                cmd = self.build_cmd(action, values)
            except (ValueError, TypeError) as e:
                return {"ok": False, "error": str(e)}

            jid = f"{action_id}-{int(time.time() * 1000)}"
            job = {
                "id": jid, "action": action_id, "label": action["label"],
                "cmd": " ".join(Path(c).name if i == 1 else c for i, c in enumerate(cmd)),
                "status": "running", "returncode": None, "lines": [],
                "started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "ended": None,
            }
            self.jobs[jid] = job
            self.order.append(jid)
            self.active = jid

        proc = subprocess.Popen(
            cmd, cwd=self.repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        job["_proc"] = proc
        threading.Thread(target=self._pump, args=(job, proc), daemon=True).start()
        return {"ok": True, "job": jid}

    def _pump(self, job: dict, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stdout:
                # Progress bars emit \r; keep only the last segment so the log
                # doesn't fill with partial redraws.
                job["lines"].append(line.rstrip("\n").split("\r")[-1])
                if len(job["lines"]) > 4000:
                    del job["lines"][:1000]
        finally:
            proc.wait()
            job["returncode"] = proc.returncode
            job["status"] = "done" if proc.returncode == 0 else (
                "cancelled" if job.get("_cancelled") else "failed")
            job["ended"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            job.pop("_proc", None)
            with self.lock:
                if self.active == job["id"]:
                    self.active = None

    def cancel(self, jid: str) -> dict:
        job = self.jobs.get(jid)
        if not job or job["status"] != "running":
            return {"ok": False, "error": "not running"}
        job["_cancelled"] = True
        proc = job.get("_proc")
        if proc:
            proc.terminate()
        return {"ok": True}

    def view(self, jid: str, since: int = 0) -> dict | None:
        job = self.jobs.get(jid)
        if job is None:
            return None
        return {k: v for k, v in job.items() if not k.startswith("_")} | {
            "lines": job["lines"][since:], "total_lines": len(job["lines"]),
        }

    def history(self, n: int = 12) -> list[dict]:
        return [{"id": j, "label": self.jobs[j]["label"], "status": self.jobs[j]["status"],
                 "started": self.jobs[j]["started"], "returncode": self.jobs[j]["returncode"]}
                for j in reversed(self.order[-n:])]


def lan_ip() -> str:
    """Best-guess LAN address. The UDP socket is never actually sent on."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()
