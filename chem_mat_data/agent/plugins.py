"""
Plugin installation logic for the agent system.

This module handles the installation of external MCP servers and tools that
enhance the agent's capabilities. The plugins are installed into the
``.mcp-servers/`` directory inside the project root.

Each backend has its own set of plugins:

- **Claude**: ``crawl4ai-mcp-server`` (pip) for advanced web crawling, and
  ``mcp-file-downloader`` (Node.js) for robust file downloads including
  browser-mode support.
- **OpenCode**: ``mcp-file-downloader`` (Node.js) for file downloads. The
  other tools (crawl4ai, websearch) are configured as OpenCode plugins
  separately.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from chem_mat_data.utils import PROJECT_PATH


#: Directory where MCP servers are installed.
MCP_SERVERS_DIR: Path = Path(PROJECT_PATH) / ".mcp-servers"

#: URL of the mcp-file-downloader Git repository.
FILE_DOWNLOADER_REPO: str = "https://github.com/abs222222/mcp-file-downloader.git"

#: Subdirectory name for the file downloader inside .mcp-servers/.
FILE_DOWNLOADER_DIR_NAME: str = "file-downloader"


class PluginInstallError(Exception):
    """Raised when a plugin installation step fails."""
    pass


def _run(
    cmd: List[str],
    cwd: str | Path | None = None,
    description: str = "",
) -> subprocess.CompletedProcess:
    """
    Run a subprocess command and raise :class:`PluginInstallError` on failure.

    :param cmd: Command and arguments.
    :param cwd: Working directory.
    :param description: Human-readable description for error messages.

    :returns: The completed process.
    :raises PluginInstallError: If the command exits with a non-zero code.
    """
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PluginInstallError(
            f"{description or ' '.join(cmd)} failed (exit {result.returncode}):\n{detail}"
        )
    return result


def _check_command(name: str) -> bool:
    """Return ``True`` if ``name`` is available on PATH."""
    return shutil.which(name) is not None


# =========================================================================
# File downloader (shared between backends)
# =========================================================================

def install_file_downloader() -> Path:
    """
    Install the mcp-file-downloader MCP server into ``.mcp-servers/``.

    Clones the repository from GitHub and runs ``npm install``. If the
    directory already exists, it is removed and re-cloned to ensure a
    clean state.

    :returns: Absolute path to the ``download-server.js`` entry point.
    :raises PluginInstallError: If git, npm, or npx commands fail.
    """
    if not _check_command("git"):
        raise PluginInstallError("git is not installed. Please install git first.")
    if not _check_command("npm"):
        raise PluginInstallError("npm is not installed. Please install Node.js (>= 18) first.")

    MCP_SERVERS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MCP_SERVERS_DIR / FILE_DOWNLOADER_DIR_NAME

    # Clean previous installation
    if dest.exists():
        shutil.rmtree(dest)

    _run(
        ["git", "clone", FILE_DOWNLOADER_REPO, str(dest)],
        description="git clone mcp-file-downloader",
    )
    _run(
        ["npm", "install"],
        cwd=dest,
        description="npm install (file-downloader)",
    )
    _run(
        ["npx", "playwright", "install", "chromium"],
        cwd=dest,
        description="npx playwright install chromium",
    )

    server_js = dest / "download-server.js"
    if not server_js.exists():
        raise PluginInstallError(
            f"Expected server file not found: {server_js}"
        )
    return server_js


# =========================================================================
# crawl4ai MCP server (Claude backend)
# =========================================================================

def install_crawl4ai() -> str:
    """
    Install the crawl4ai MCP server via pip.

    Installs the ``crawl4ai-mcp-server`` package which provides the
    ``crawl4ai-mcp`` command, then runs ``crawl4ai-setup`` to configure
    the browser backend.

    :returns: The command name to start the server (``"crawl4ai-mcp"``).
    :raises PluginInstallError: If pip install or setup fails.
    """
    if not _check_command("pip"):
        raise PluginInstallError("pip is not installed.")

    _run(
        ["pip", "install", "crawl4ai-mcp-server"],
        description="pip install crawl4ai-mcp-server",
    )
    _run(
        ["crawl4ai-setup"],
        description="crawl4ai-setup (browser configuration)",
    )

    if not _check_command("crawl4ai-mcp"):
        raise PluginInstallError(
            "crawl4ai-mcp command not found after installation. "
            "Check that the pip install target is on your PATH."
        )
    return "crawl4ai-mcp"


# =========================================================================
# Backend-specific installation orchestration
# =========================================================================

def install_claude_plugins(config) -> Dict[str, Any]:
    """
    Install all plugins required by the Claude backend.

    Installs ``crawl4ai-mcp-server`` (pip) and ``mcp-file-downloader``
    (Node.js), then updates the config file with the MCP server entries.

    :param config: The :class:`~chem_mat_data.config.Config` singleton.

    :returns: Dict of installed MCP server configurations (as written to
        the config file).
    :raises PluginInstallError: If any installation step fails.
    """
    mcp_servers: Dict[str, Any] = {}

    # 1. crawl4ai
    crawl4ai_cmd = install_crawl4ai()
    mcp_servers["crawl4ai"] = {
        "command": crawl4ai_cmd,
        "args": [],
    }

    # 2. file downloader
    server_js = install_file_downloader()
    mcp_servers["file_downloader"] = {
        "command": "node",
        "args": [str(server_js)],
    }

    # 3. Update config
    _update_config_mcp_servers(config, "claude", mcp_servers)

    return mcp_servers


def install_opencode_plugins(config) -> Path:
    """
    Install plugins required by the OpenCode backend.

    Installs the ``mcp-file-downloader`` Node.js MCP server. The other
    OpenCode plugins (crawl4ai, websearch_cited) must be configured in
    OpenCode's own configuration — a config snippet is returned for the
    user to copy.

    :param config: The :class:`~chem_mat_data.config.Config` singleton.

    :returns: Path to the installed ``download-server.js``.
    :raises PluginInstallError: If any installation step fails.
    """
    server_js = install_file_downloader()
    return server_js


def _update_config_mcp_servers(
    config,
    backend_name: str,
    mcp_servers: Dict[str, Any],
) -> None:
    """
    Write MCP server entries into the ``[agent.<backend>.mcp_servers]``
    section of the config file.

    Merges with any existing entries rather than replacing them.

    :param config: The Config singleton.
    :param backend_name: Backend name (e.g. ``"claude"``).
    :param mcp_servers: Dict mapping server name to config dict.
    """
    import tomlkit

    if "agent" not in config.config_data:
        config.config_data["agent"] = tomlkit.table()

    agent_section = config.config_data["agent"]

    if backend_name not in agent_section:
        agent_section[backend_name] = tomlkit.table()

    backend_section = agent_section[backend_name]

    if "mcp_servers" not in backend_section:
        backend_section["mcp_servers"] = tomlkit.table()

    for server_name, server_config in mcp_servers.items():
        backend_section["mcp_servers"][server_name] = server_config

    config.save()


def get_opencode_config_snippet(server_js_path: Path) -> str:
    """
    Generate an OpenCode configuration snippet for the user to add to
    their OpenCode config file.

    :param server_js_path: Absolute path to the download-server.js file.

    :returns: A formatted JSON string with the OpenCode config snippet.
    """
    return f'''\
Add the following to your OpenCode configuration:

"mcp": {{
    "file_downloader": {{
        "type": "local",
        "command": ["node", "{server_js_path}"],
        "enabled": true,
        "timeout": 15000
    }}
}},
"plugin": [
    "opencode-websearch-cited@1.2.0"
],
"agent": {{
    "dataset-links": {{
        "description": "Download the dataset files from the provided publication.",
        "permission": {{
            "websearch_*": "allow",
            "crawlfetch": "allow",
            "webfetch": "deny",
            "file_downloader_*": "allow",
            "bash": "deny",
            "read": "deny",
            "glob": "deny",
            "grep": "deny",
            "list": "deny",
            "edit": "deny",
            "patch": "deny",
            "codesearch": "deny"
        }}
    }},
    "script-generation": {{
        "description": "Generate a processing script for the downloaded dataset.",
        "permission": {{
            "websearch_*": "deny",
            "crawlfetch": "deny",
            "webfetch": "deny",
            "file_downloader_*": "deny",
            "bash": "allow",
            "read": "allow",
            "glob": "allow",
            "grep": "allow",
            "list": "allow",
            "edit": "deny",
            "patch": "deny",
            "codesearch": "allow"
        }}
    }}
}}'''


# =========================================================================
# Virtual environment management for script execution
# =========================================================================

#: Default venv location inside the artifacts directory.
VENV_DIR: Path = Path(__file__).resolve().parent / "artifacts" / ".venv"


def create_agent_venv(venv_path: Path | None = None) -> Path:
    """
    Create a fresh virtual environment for running generated scripts.

    Uses ``uv`` to create the venv and install the project in editable
    mode with dev dependencies. If the venv directory already exists it
    is removed first to ensure a clean state.

    :param venv_path: Where to create the venv. Defaults to
        ``agent/artifacts/.venv/``.

    :returns: Absolute path to the Python interpreter inside the venv.
    :raises PluginInstallError: If uv is not installed or a step fails.
    """
    import sys

    venv_path = venv_path or VENV_DIR

    # Locate uv binary
    uv_bin = _find_uv()

    # Clean previous venv
    if venv_path.exists():
        shutil.rmtree(venv_path)

    # Create venv
    _run(
        [uv_bin, "venv", str(venv_path)],
        description="uv venv (create virtual environment)",
    )

    # Determine the python interpreter path inside the venv
    if sys.platform == "win32":
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        python_path = venv_path / "bin" / "python"

    if not python_path.exists():
        raise PluginInstallError(
            f"Python interpreter not found at {python_path} after venv creation"
        )

    # Install the project in editable mode with dev deps
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_path)

    _run(
        [uv_bin, "pip", "install", "-e", f"{PROJECT_PATH}[dev]"],
        cwd=PROJECT_PATH,
        description="uv pip install -e .[dev]",
    )

    return python_path


def _find_uv() -> str:
    """
    Locate the ``uv`` binary.

    Tries the ``uv`` Python package first (``find_uv_bin()``), then
    falls back to looking for ``uv`` on the system PATH.

    :returns: Absolute path to the uv binary.
    :raises PluginInstallError: If uv cannot be found.
    """
    # Try the uv Python package
    try:
        from uv import find_uv_bin
        return str(find_uv_bin())
    except (ImportError, FileNotFoundError):
        pass

    # Fall back to PATH
    uv_path = shutil.which("uv")
    if uv_path:
        return uv_path

    raise PluginInstallError(
        "uv is not installed. Install it with: pip install uv  "
        "or see https://docs.astral.sh/uv/getting-started/installation/"
    )
