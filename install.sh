#!/usr/bin/env bash
# One-shot setup: venv + deps + self-test, then prints the MCP server config to paste.
set -euo pipefail
cd "$(dirname "$0")"
DIR="$(pwd)"
PY="${PYTHON:-python3}"

"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -m unittest discover -s tests -q

cat <<EOF

glb2ori installed.

CLI:
  $DIR/.venv/bin/python $DIR/glb2ori.py model.glb --inspect
  $DIR/.venv/bin/python $DIR/glb2ori.py model.glb -o model.weave --tris 8000 --height 150 --json

MCP server config (Claude Desktop/Code, Cursor, Windsurf, Devin, ...):
{
  "mcpServers": {
    "glb2ori": {
      "command": "$DIR/.venv/bin/python",
      "args": ["$DIR/glb2ori_mcp.py"]
    }
  }
}

Claude Code:  claude mcp add glb2ori -- $DIR/.venv/bin/python $DIR/glb2ori_mcp.py
EOF
