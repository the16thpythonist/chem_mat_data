"""
Dataset discovery pipeline for chemistry and materials-science papers.

This subpackage bundles the lightweight first stage of the dataset-discovery
pipeline originally released at https://github.com/aimat-lab/Dataset-Discovery-Pipeline.
It exposes two composable steps:

1. ``openalex`` — iterate chemistry/materials papers from the OpenAlex API with
   resumable cursor state and an on-disk de-duplication set.
2. ``classifier`` — score the ``title + abstract`` of each paper with a
   fine-tuned TinyLlama-1.1B classifier (QLoRA adapter) and flag those above a
   probability threshold as likely dataset releases.

The pipeline is intentionally kept separate from the existing ``agent``
subsystem: ``discovery`` is the cheap upstream filter that narrows the whole
of OpenAlex down to a few hundred plausible hits, while ``agent identify``
is the expensive downstream verifier that confirms each hit by actually
browsing the paper.

The command-line surface lives in ``chem_mat_data.manage`` under
``cmmanage discover {model-info,model-download,fetch,score,stream}``.
"""

from chem_mat_data.discovery.io import (
    JsonlRecord,
    iter_jsonl,
    write_jsonl_line,
    FetchState,
    build_csv_row,
    STREAM_CSV_BASE_COLUMNS,
)

__all__ = [
    'JsonlRecord',
    'iter_jsonl',
    'write_jsonl_line',
    'FetchState',
    'build_csv_row',
    'STREAM_CSV_BASE_COLUMNS',
]
