#!/usr/bin/env python3
"""Stage a .weave file into the Oriverse editor through the local Oriverse MCP server.

    python ori_push.py fox.weave --agent <uuid>            # stage_new_world (compile + validate)
    python ori_push.py fox.weave --agent <uuid> --submit   # ...then submit_pending_world ([Load] button)

Without --agent a new editing session is created via list_editors and its id is printed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://127.0.0.1:39837/mcp"


class Mcp:
    def __init__(self, url: str):
        self.url = url
        self.session: str | None = None
        self.next_id = 1
        self.rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                "clientInfo": {"name": "glb2ori-push", "version": "1.0"}})
        self.notify("notifications/initialized")

    def _post(self, payload: dict) -> dict | None:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(self.url, json.dumps(payload).encode(), headers)
        with urllib.request.urlopen(req, timeout=600) as r:
            self.session = r.headers.get("Mcp-Session-Id", self.session)
            body = r.read().decode()
            if not body.strip():
                return None
            if "text/event-stream" in r.headers.get("Content-Type", ""):
                msgs = [json.loads(l[5:]) for l in body.splitlines() if l.startswith("data:") and l[5:].strip()]
                return next((m for m in reversed(msgs) if "id" in m), msgs[-1] if msgs else None)
            return json.loads(body)

    def notify(self, method: str, params: dict | None = None) -> None:
        self._post({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def rpc(self, method: str, params: dict) -> dict:
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params}
        self.next_id += 1
        res = self._post(msg)
        if res is None or "error" in res:
            raise RuntimeError(f"{method} failed: {res and res.get('error')}")
        return res["result"]

    def call(self, tool: str, args: dict) -> tuple[bool, str]:
        res = self.rpc("tools/call", {"name": tool, "arguments": args})
        text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        return not res.get("isError", False), text


def push(weave: str | Path, agent: str | None = None, submit: bool = False, url: str = DEFAULT_URL) -> dict:
    """Stage (and optionally submit) a .weave file; returns {ok, agent, stage, submit}."""
    mcp = Mcp(url)
    if not agent:
        _, text = mcp.call("list_editors", {})
        m = re.search(r"Editing session agent: ([0-9a-f-]{36})", text)
        if not m:
            return {"ok": False, "agent": None, "stage": text, "submit": None}
        agent = m.group(1)
    ok, stage = mcp.call("stage_new_world", {"agent": agent, "world_description": Path(weave).read_text()})
    ok = ok and "run_result: ok" in stage
    sub = None
    if ok and submit:
        ok, sub = mcp.call("submit_pending_world", {"agent": agent, "summary": f"Import {Path(weave).stem} (glb2ori)"})
    return {"ok": ok, "agent": agent, "stage": stage, "submit": sub}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weave", type=Path)
    ap.add_argument("--agent", help="Oriverse editing session id (from list_editors)")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--submit", action="store_true", help="also submit_pending_world so the editor shows [Load]")
    a = ap.parse_args()
    res = push(a.weave, a.agent, a.submit, a.url)
    print(f"agent: {res['agent']}", file=sys.stderr)
    print(res["stage"])
    if res["submit"]:
        print(res["submit"])
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
