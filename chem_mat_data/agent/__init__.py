"""
Deterministic helpers supporting the ``dataset-extraction`` skill.

The instructions for turning a scientific publication into a ChemMatData processing
script - how to find the data, which files to take, how the script has to look, what
the metadata must contain - live in ``.claude/skills/dataset-extraction/`` and are
executed by an agent in Claude Code, not by this package.

What remains here are the two pieces that should not depend on a model's judgement:

- :mod:`chem_mat_data.agent.environment` builds the clean virtual environment in
  which a generated script is executed.
- :mod:`chem_mat_data.agent.manifest` owns the per-run state that survives between
  stages and between sessions.
- :mod:`chem_mat_data.agent.gates` implements the post-conditions of each of the six
  stages.
- :mod:`chem_mat_data.agent.verification` checks a finished results folder against the
  hard post-conditions of a processing run.

Both are exposed through the ``cmmanage agent`` command group.
"""

from chem_mat_data.agent.environment import create_agent_venv
from chem_mat_data.agent.environment import EnvironmentSetupError
from chem_mat_data.agent.gates import GATES
from chem_mat_data.agent.gates import is_permissive
from chem_mat_data.agent.gates import run_gate
from chem_mat_data.agent.manifest import Manifest
from chem_mat_data.agent.manifest import StageState
from chem_mat_data.agent.manifest import STAGES
from chem_mat_data.agent.verification import ERROR
from chem_mat_data.agent.verification import WARNING
from chem_mat_data.agent.verification import VerificationIssue
from chem_mat_data.agent.verification import VerificationReport
from chem_mat_data.agent.verification import verify_results

__all__ = [
    'GATES',
    'Manifest',
    'STAGES',
    'StageState',
    'is_permissive',
    'run_gate',
    'create_agent_venv',
    'EnvironmentSetupError',
    'ERROR',
    'WARNING',
    'VerificationIssue',
    'VerificationReport',
    'verify_results',
]
