"""
Dataset discovery pipeline for chemistry and materials-science papers.

This subpackage bundles the lightweight first stage of the dataset-discovery
pipeline originally released at https://github.com/aimat-lab/Dataset-Discovery-Pipeline.
It exposes three composable steps:

1. ``openalex`` — iterate chemistry/materials papers from the OpenAlex API with
   resumable cursor state and an on-disk de-duplication set.
2. ``classifier`` — score the ``title + abstract`` of each paper with a
   fine-tuned TinyLlama-1.1B classifier (QLoRA adapter) and flag those above a
   probability threshold as likely dataset releases.
3. ``ledger`` — screen a flagged paper against everything already answered: the
   shipped catalog, the human-decided rejection list, and the verdict ledger of
   previous sweeps.

The pipeline is intentionally kept separate from the existing ``agent``
subsystem: ``discovery`` is the cheap upstream filter that narrows the whole of
OpenAlex down to a few hundred plausible hits. The expensive downstream verifier
that confirms each hit by actually browsing the paper is the ``/dataset-search``
Claude Code skill, which drives these commands; papers it confirms are then
processed by the ``/dataset-extraction`` skill.

The command-line surface lives in ``chem_mat_data.manage`` under
``cmmanage discover {model-info,model-download,fetch,score,stream,check,verdict}``.
"""

from chem_mat_data.discovery.io import (
    JsonlRecord,
    iter_jsonl,
    write_jsonl_line,
    FetchState,
    build_csv_row,
    STREAM_CSV_BASE_COLUMNS,
)
from chem_mat_data.discovery.ledger import (
    ScreenResult,
    Verdict,
    VerdictLedger,
    screen,
    screen_many,
)

__all__ = [
    'JsonlRecord',
    'iter_jsonl',
    'write_jsonl_line',
    'FetchState',
    'build_csv_row',
    'STREAM_CSV_BASE_COLUMNS',
    'ScreenResult',
    'Verdict',
    'VerdictLedger',
    'screen',
    'screen_many',
]
