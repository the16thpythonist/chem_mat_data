"""
Isolated environment management for running generated dataset processing scripts.

A ``create_graph_datasets__<name>.py`` script produced by the ``dataset-extraction``
skill has to be executed before it can be trusted, and executing freshly generated
code in the active development environment is a poor idea: a stray dependency or a
partially applied install would silently change the environment the rest of the
project runs in.

:func:`create_agent_venv` therefore builds a throwaway virtual environment at
``.dataset_work/.venv`` with the project installed in editable mode, recreated from
scratch on every call so a broken environment is never something to debug.

Exposed through ``cmmanage agent venv``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from chem_mat_data.utils import PROJECT_PATH


#: Default venv location, inside the gitignored dataset working directory at the
#: repository root. Deliberately not inside the package itself: a venv written into
#: the source tree ends up in the installed distribution.
VENV_DIR: Path = Path(PROJECT_PATH) / '.dataset_work' / '.venv'


class EnvironmentSetupError(Exception):
    """Raised when a step of the environment setup fails."""

    pass


def _run(
    cmd: List[str],
    cwd: str | Path | None = None,
    description: str = '',
    env: Dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """
    Run a subprocess command and raise :class:`EnvironmentSetupError` on failure.

    :param cmd: Command and arguments.
    :param cwd: Working directory.
    :param description: Human-readable description used in the error message.
    :param env: Environment variables for the subprocess. Inherits the current
        environment when ``None``.

    :returns: The completed process.
    :raises EnvironmentSetupError: If the command exits with a non-zero code.
    """
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise EnvironmentSetupError(
            f'{description or " ".join(cmd)} failed (exit {result.returncode}):\n{detail}'
        )
    return result


def find_uv() -> str:
    """
    Locate the ``uv`` binary.

    Prefers the ``uv`` Python package, which pins a known-good binary, and falls back
    to whatever is on ``PATH``.

    :returns: Absolute path to the uv binary.
    :raises EnvironmentSetupError: If uv cannot be found.
    """
    try:
        from uv import find_uv_bin

        return str(find_uv_bin())
    except (ImportError, FileNotFoundError):
        pass

    uv_path = shutil.which('uv')
    if uv_path:
        return uv_path

    raise EnvironmentSetupError(
        'uv is not installed. Install it with `pip install uv` or see '
        'https://docs.astral.sh/uv/getting-started/installation/'
    )


def create_agent_venv(venv_path: Path | None = None) -> Path:
    """
    Create a fresh virtual environment for running generated scripts.

    Any existing environment at the target path is removed first, so the result is
    always a clean state rather than an accumulation of previous runs. The project is
    installed in editable mode with its development extras.

    Example:

    .. code-block:: python

        python_path = create_agent_venv()
        subprocess.run([str(python_path), 'chem_mat_data/scripts/create_graph_datasets__esol.py'])

    :param venv_path: Where to create the environment. Defaults to :data:`VENV_DIR`.

    :returns: Absolute path to the Python interpreter inside the environment.
    :raises EnvironmentSetupError: If uv is unavailable or any step fails.
    """
    import sys

    venv_path = Path(venv_path) if venv_path else VENV_DIR
    uv_bin = find_uv()

    if venv_path.exists():
        shutil.rmtree(venv_path)

    venv_path.parent.mkdir(parents=True, exist_ok=True)

    _run(
        [uv_bin, 'venv', str(venv_path)],
        description='uv venv (create virtual environment)',
    )

    if sys.platform == 'win32':
        python_path = venv_path / 'Scripts' / 'python.exe'
    else:
        python_path = venv_path / 'bin' / 'python'

    if not python_path.exists():
        raise EnvironmentSetupError(
            f'python interpreter not found at {python_path} after venv creation'
        )

    # VIRTUAL_ENV has to reach the subprocess for `uv pip install` to target the
    # environment just created rather than whatever happens to be active.
    env = os.environ.copy()
    env['VIRTUAL_ENV'] = str(venv_path)
    env.pop('CONDA_PREFIX', None)

    _run(
        [uv_bin, 'pip', 'install', '-e', f'{PROJECT_PATH}[dev]'],
        cwd=PROJECT_PATH,
        description='uv pip install -e .[dev]',
        env=env,
    )

    return python_path
