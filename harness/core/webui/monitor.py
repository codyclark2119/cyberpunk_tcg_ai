"""Game-agnostic monitoring console: run a repo's scripts and watch them.

    from harness.core.webui.monitor import serve
    serve(ACTIONS, repo_root=REPO_ROOT, title="one_piece_llm")

`runner.Runner` already generalized the *execution* half -- an allowlist of
actions, one job at a time, no shell. This is the server around it, and it is
the part every repo was otherwise going to write again: `magic-llm`'s
`scripts/webui.py` carries its own copy of `Runner` (pre-extraction, wired to
module globals instead of parameters) plus ~500 lines of console HTML, and
nothing of that was reachable from the other two repos.

Nothing here imports a game, a model, a card database or a gold set. The only
input is the caller's `ACTIONS` list, which is also the only thing that can
run. That is what makes one console serve three repos: the repo-specific part
is data, not code.

## The allowlist is the security model, not a convenience

This server can bind beyond loopback on request, so it is reachable by
anything on the network. A generic "run this command" endpoint would be remote
code execution for anyone who can reach the port. Instead the client names an
action **id** and supplies values for that action's *declared* arguments; the
command line is assembled server-side by `Runner.build_cmd` and never accepted
from the client. Free text is passed as separate argv entries, never through a
shell.

Off loopback a token is also required -- generated at startup, printed with
the URL, kept in a cookie after the first request. `--no-auth` opts out for a
trusted network, and binding to loopback needs no token because reaching it
already means local access.

## Jobs run one at a time

Not tidiness -- correctness. Pipeline scripts in these repos commonly write
the same `.jsonl`, and two concurrent writers interleave lines into a file
that still parses. `Runner` serializes, and this server surfaces the refusal
rather than queueing silently.

## Extending it without forking it

`extra_routes(app)` is called with the FastAPI app after the monitor's own
routes are registered, so a repo can add views that need its game modules --
position authoring, a gold-set browser -- while the runner half stays shared.
Anything mounted that way is the caller's responsibility, including whether it
is safe to expose off loopback.
"""

import secrets
from pathlib import Path

from harness.core.webui.runner import Runner, lan_ip

__all__ = ["build_app", "serve", "CONSOLE_HTML"]


def _public_action(a: dict) -> dict:
    """An action as the client may see it.

    `cmd` is deliberately withheld. The client never needs it -- it names an
    id -- and publishing the script path invites building a request around it
    rather than around the allowlist.
    """
    return {
        "id": a["id"], "label": a.get("label", a["id"]),
        "group": a.get("group", "Scripts"), "desc": a.get("desc", ""),
        "eta": a.get("eta", ""), "writes": bool(a.get("writes")),
        "args": [
            {k: v for k, v in spec.items() if k != "value"}
            for spec in a.get("args", []) if spec.get("type") != "fixed"
        ],
    }


def build_app(actions: list[dict], repo_root: Path, title: str = "monitor",
              token: str | None = None, runner: Runner | None = None,
              extra_routes=None):
    """The FastAPI app. Separated from `serve` so tests need no network."""
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

    by_id = {a["id"]: a for a in actions}
    if len(by_id) != len(actions):
        # Two actions sharing an id means one is unreachable -- and which one
        # wins depends on list order, so it would fail as a confusing "that
        # ran the wrong script" rather than as a startup error.
        dupes = sorted({a["id"] for a in actions if list(
            x["id"] for x in actions).count(a["id"]) > 1})
        raise ValueError(f"duplicate action id(s): {dupes}")

    runner = runner or Runner(by_id, repo_root)
    app = FastAPI(title=f"{title} monitor")

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if not token:
            return await call_next(request)
        supplied = (request.query_params.get("t")
                    or request.cookies.get("montoken")
                    or request.headers.get("x-token"))
        if supplied != token:
            return PlainTextResponse(
                "Token required. Use the URL printed by the server.", 401)
        response = await call_next(request)
        if request.query_params.get("t") == token:
            response.set_cookie("montoken", token, max_age=86400 * 7, samesite="lax")
        return response

    @app.get("/", response_class=HTMLResponse)
    def index():
        return CONSOLE_HTML.replace("__TITLE__", title)

    @app.get("/api/actions")
    def list_actions():
        groups: dict[str, list] = {}
        for a in actions:
            groups.setdefault(a.get("group", "Scripts"), []).append(_public_action(a))
        return {"title": title, "groups": groups,
                "active": runner.active, "repo": str(repo_root)}

    @app.post("/api/run/{action_id}")
    async def run(action_id: str, payload: dict | None = None):
        result = runner.start(action_id, (payload or {}).get("values") or {})
        if not result.get("ok"):
            # 409 for "something else is running", 400 for a bad request --
            # distinguishable by the client without parsing the message.
            code = 409 if "still running" in result.get("error", "") else 400
            return JSONResponse(result, status_code=code)
        return result

    @app.get("/api/job/{job_id}")
    def job(job_id: str, since: int = 0):
        view = runner.view(job_id, since)
        if view is None:
            return JSONResponse({"error": "no such job"}, status_code=404)
        return view

    @app.post("/api/cancel/{job_id}")
    def cancel(job_id: str):
        result = runner.cancel(job_id)
        return result if result.get("ok") else JSONResponse(result, status_code=400)

    @app.get("/api/history")
    def history(n: int = 12):
        return {"jobs": runner.history(n), "active": runner.active}

    if extra_routes is not None:
        extra_routes(app)
    app.state.runner = runner
    return app


def serve(actions: list[dict], repo_root: Path, title: str = "monitor",
          host: str = "127.0.0.1", port: int = 8765, lan: bool = False,
          no_auth: bool = False, extra_routes=None) -> None:
    """Run the console. Prints the URL, including the token when there is one."""
    import uvicorn

    bind = "0.0.0.0" if lan else host
    # A token only when reachable from elsewhere: on loopback, being able to
    # reach the port already means local access, so a token adds friction
    # without adding a boundary.
    token = None if (no_auth or not lan) else secrets.token_urlsafe(12)
    app = build_app(actions, repo_root, title=title, token=token,
                    extra_routes=extra_routes)

    shown = f"http://{lan_ip()}:{port}" if lan else f"http://{host}:{port}"
    print(f"{title} monitor -> {shown}" + (f"/?t={token}" if token else ""))
    if lan and not token:
        print("  WARNING: bound to all interfaces with --no-auth. Anything on "
              "this network can run the allowlisted scripts.")
    print(f"  {len(actions)} action(s); jobs run one at a time.")
    uvicorn.run(app, host=bind, port=port, log_level="warning")


CONSOLE_HTML = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ monitor</title>
<style>
 :root{--bg:#0f1115;--fg:#e6e6e6;--dim:#9aa0a6;--card:#171a21;--line:#272b33;
       --ok:#4ade80;--bad:#f87171;--run:#60a5fa;--accent:#818cf8}
 @media(prefers-color-scheme:light){:root{--bg:#f7f7f8;--fg:#16181d;--dim:#5f6368;
       --card:#fff;--line:#e3e5e9;--ok:#15803d;--bad:#b91c1c;--run:#1d4ed8;--accent:#4f46e5}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
 header{padding:14px 18px;border-bottom:1px solid var(--line);display:flex;
        gap:12px;align-items:baseline;flex-wrap:wrap}
 h1{font-size:15px;margin:0;font-weight:600}
 .repo{color:var(--dim);font-size:12px;font-family:ui-monospace,monospace}
 main{display:grid;grid-template-columns:minmax(260px,1fr) minmax(0,2fr);
      gap:16px;padding:16px;align-items:start}
 @media(max-width:820px){main{grid-template-columns:1fr}}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
 .group{font-size:11px;text-transform:uppercase;letter-spacing:.08em;
        color:var(--dim);margin:14px 0 6px}
 .group:first-child{margin-top:0}
 .act{border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:8px}
 .act b{font-weight:600}
 .act .desc{color:var(--dim);font-size:12.5px;margin:4px 0 0}
 .meta{color:var(--dim);font-size:11.5px;margin-top:4px}
 .writes{color:var(--bad)}
 .args{margin-top:8px;display:grid;gap:6px}
 label{display:flex;gap:8px;align-items:center;font-size:12.5px;color:var(--dim)}
 input[type=text],input[type=number],select{background:var(--bg);color:var(--fg);
        border:1px solid var(--line);border-radius:6px;padding:4px 7px;font:inherit;
        font-size:12.5px;flex:1;min-width:0}
 button{background:var(--accent);color:#fff;border:0;border-radius:7px;
        padding:6px 12px;font:inherit;font-weight:600;cursor:pointer}
 button:disabled{opacity:.45;cursor:not-allowed}
 button.ghost{background:transparent;color:var(--dim);border:1px solid var(--line)}
 pre{background:#0b0d11;color:#dfe3e8;border-radius:8px;padding:10px;margin:0;
     overflow:auto;max-height:62vh;font:12px/1.45 ui-monospace,monospace;white-space:pre-wrap}
 @media(prefers-color-scheme:light){pre{background:#12141a}}
 .status{font-weight:600}
 .running{color:var(--run)} .done{color:var(--ok)} .failed,.cancelled{color:var(--bad)}
 .hist{font-size:12px;color:var(--dim);margin-top:10px;display:grid;gap:3px}
 .hist a{color:inherit;cursor:pointer;text-decoration:none}
 .hist a:hover{color:var(--fg)}
 .err{color:var(--bad);font-size:12.5px;margin-top:8px;min-height:1em}
</style>
<header>
  <h1>__TITLE__</h1>
  <span class="repo" id="repo"></span>
</header>
<main>
  <div class="card" id="actions">loading…</div>
  <div class="card">
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      <span class="status" id="status">idle</span>
      <span class="repo" id="jobname"></span>
      <span style="flex:1"></span>
      <button class="ghost" id="cancel" disabled>Cancel</button>
    </div>
    <div class="err" id="err"></div>
    <pre id="log">No job selected.</pre>
    <div class="hist" id="hist"></div>
  </div>
</main>
<script>
const $ = s => document.querySelector(s);
let CUR = null, SEEN = 0, TIMER = null;

function tok(){ const u = new URL(location.href); const t = u.searchParams.get("t");
  return t ? ("?t=" + encodeURIComponent(t)) : ""; }
async function api(path, opts){ const r = await fetch(path + tok(), opts); return r; }

async function loadActions(){
  const r = await api("/api/actions"); const d = await r.json();
  $("#repo").textContent = d.repo;
  const host = $("#actions"); host.innerHTML = "";
  for (const [group, acts] of Object.entries(d.groups)){
    const g = document.createElement("div"); g.className = "group"; g.textContent = group;
    host.appendChild(g);
    for (const a of acts){
      const el = document.createElement("div"); el.className = "act";
      const bits = [];
      if (a.eta) bits.push(a.eta);
      if (a.writes) bits.push('<span class="writes">writes files</span>');
      el.innerHTML = `<b>${a.label}</b>`
        + (a.desc ? `<div class="desc">${a.desc}</div>` : "")
        + (bits.length ? `<div class="meta">${bits.join(" · ")}</div>` : "");
      const args = document.createElement("div"); args.className = "args";
      for (const s of a.args){
        const lab = document.createElement("label");
        const name = s.label || s.name || s.key || "";
        if (s.type === "flag"){
          lab.innerHTML = `<input type=checkbox ${s.default ? "checked" : ""}
            data-k="${s.name||s.key}"> ${name}`;
        } else if (s.type === "choice"){
          lab.innerHTML = `${name} <select data-k="${s.name||s.key}">` +
            (s.choices||[]).map(c => `<option ${c===s.default?"selected":""}>${c}</option>`).join("")
            + `</select>`;
        } else {
          const t = s.type === "int" ? "number" : "text";
          lab.innerHTML = `${name} <input type=${t} data-k="${s.name||s.key}"
            value="${s.default ?? ""}">`;
        }
        args.appendChild(lab);
      }
      el.appendChild(args);
      const b = document.createElement("button"); b.textContent = "Run";
      b.onclick = () => run(a.id, args, b);
      el.appendChild(b);
      host.appendChild(el);
    }
  }
}

async function run(id, argsEl, btn){
  const values = {};
  argsEl.querySelectorAll("[data-k]").forEach(i => {
    values[i.dataset.k] = i.type === "checkbox" ? i.checked : i.value;
  });
  btn.disabled = true; $("#err").textContent = "";
  const r = await api("/api/run/" + id, {method:"POST",
    headers:{"content-type":"application/json"}, body: JSON.stringify({values})});
  const d = await r.json(); btn.disabled = false;
  if (!r.ok){ $("#err").textContent = d.error || "failed to start"; return; }
  watch(d.job);
}

function watch(jid){ CUR = jid; SEEN = 0; $("#log").textContent = "";
  clearInterval(TIMER); TIMER = setInterval(poll, 700); poll(); }

async function poll(){
  if (!CUR) return;
  const r = await api(`/api/job/${CUR}&since=${SEEN}`.replace("&", tok()?"&":"?"));
  if (!r.ok) { clearInterval(TIMER); return; }
  const d = await r.json();
  if (d.lines && d.lines.length){
    $("#log").textContent += d.lines.join("\n") + "\n";
    $("#log").scrollTop = $("#log").scrollHeight;
    SEEN = d.total_lines;
  }
  $("#status").textContent = d.status; $("#status").className = "status " + d.status;
  $("#jobname").textContent = d.cmd || "";
  $("#cancel").disabled = d.status !== "running";
  if (d.status !== "running"){ clearInterval(TIMER); loadHistory(); }
}

$("#cancel").onclick = async () => { if (CUR) await api("/api/cancel/" + CUR, {method:"POST"}); };

async function loadHistory(){
  const r = await api("/api/history"); const d = await r.json();
  $("#hist").innerHTML = d.jobs.map(j =>
    `<a data-j="${j.id}">${j.status === "done" ? "✓" : j.status === "running" ? "…" : "✗"} `
    + `${j.label} <span style="opacity:.6">${j.started}</span></a>`).join("");
  $("#hist").querySelectorAll("a").forEach(a =>
    a.onclick = () => watch(a.dataset.j));
}

loadActions(); loadHistory();
</script>
"""
