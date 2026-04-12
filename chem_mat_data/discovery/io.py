"""
Filesystem helpers for the dataset-discovery pipeline.

This module contains three small utilities used by both the OpenAlex fetcher
and the classifier scorer:

- :func:`iter_jsonl` — stream JSON objects from a ``.jsonl`` file while
  tolerating blank lines and malformed records.
- :func:`write_jsonl_line` — append a single JSON object with a trailing
  newline and no ASCII-only escaping so that Unicode titles survive round-trips.
- :class:`FetchState` — persistent pagination state consisting of a single
  cursor token and an on-disk set of already-seen OpenAlex IDs. The state
  files live next to the target JSONL so that interrupted fetches can be
  resumed from exactly where they left off.

The design choice to keep the cursor in one tiny file and the seen-IDs in a
newline-delimited text file (rather than e.g. SQLite) is deliberate: it makes
the resumable state trivially inspectable and lets users edit it by hand when
a run goes wrong.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import IO, Any, Dict, Iterator, Optional, Set, Union

JsonlRecord = Dict[str, Any]
PathLike = Union[str, os.PathLike]


#: Column order used by :func:`build_csv_row` and the ``discover stream``
#: command. Extended with ``title`` only when ``include_title=True``.
STREAM_CSV_BASE_COLUMNS = (
    'openalex_id',
    'url',
    'score',
    'flagged',
    'publication_year',
)


def _short_openalex_id(full_id: Optional[str]) -> str:
    """Return ``W123`` from a full ``https://openalex.org/W123`` URL."""
    if not full_id:
        return ''
    return full_id.rsplit('/', 1)[-1]


def _normalize_doi_url(doi: Optional[str]) -> Optional[str]:
    """
    Normalize a DOI value into a ``https://doi.org/...`` URL.

    OpenAlex returns DOIs either as a bare identifier (``10.1234/foo``) or as
    a full URL, depending on the endpoint. We coerce both forms to the
    canonical ``https://doi.org/`` variant so the CSV's ``url`` column is
    directly clickable from a spreadsheet viewer.
    """
    if not doi:
        return None
    if doi.startswith('http://') or doi.startswith('https://'):
        return doi
    return f'https://doi.org/{doi.lstrip("/")}'


def build_csv_row(
    record: JsonlRecord,
    score: float,
    threshold: float,
    include_title: bool = False,
) -> Dict[str, Any]:
    """
    Project a scored OpenAlex record into the CSV-row dict used by ``stream``.

    The column order is stable and matches :data:`STREAM_CSV_BASE_COLUMNS`
    (plus an optional ``title`` column when ``include_title`` is True) so that
    ``csv.DictWriter`` produces a deterministic header.

    :param record: A record as produced by
        :func:`chem_mat_data.discovery.openalex.to_record`.
    :param score: Classifier positive-class probability.
    :param threshold: Flagging threshold — rows with ``score > threshold``
        get ``flagged=1``.
    :param include_title: When True, append a trailing ``title`` column.

    :returns: An :class:`~collections.OrderedDict` ready for
        :meth:`csv.DictWriter.writerow`.
    """
    short_id = _short_openalex_id(record.get('openalex_id'))
    url = _normalize_doi_url(record.get('doi'))
    if not url:
        # Fall back to the OpenAlex landing page — either the full id if
        # available, or a reconstructed URL from the short identifier.
        full_id = record.get('openalex_id')
        url = full_id or (f'https://openalex.org/{short_id}' if short_id else '')

    row: 'OrderedDict[str, Any]' = OrderedDict()
    row['openalex_id'] = short_id
    row['url'] = url
    row['score'] = f'{float(score):.4f}'
    row['flagged'] = 1 if float(score) > threshold else 0
    row['publication_year'] = record.get('publication_year') or ''
    if include_title:
        row['title'] = record.get('title') or ''
    return row


def iter_jsonl(path: PathLike) -> Iterator[JsonlRecord]:
    """
    Yield JSON records from a ``.jsonl`` file one at a time.

    Blank lines and lines that fail to parse as JSON are silently skipped so
    that a corrupted tail (common when a previous run was interrupted) does
    not abort the whole pipeline.

    :param path: Path to the ``.jsonl`` file to read.

    :returns: Generator of dict records.

    :raises FileNotFoundError: If the input file does not exist.
    """
    p = Path(path)
    with p.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl_line(fp: IO[str], record: JsonlRecord) -> None:
    """
    Append one JSON record to an open text file with a trailing newline.

    Uses ``ensure_ascii=False`` so that Unicode characters (Greek letters,
    umlauts, Chinese names, …) are preserved verbatim rather than escaped to
    ``\\uXXXX`` sequences — this keeps the JSONL files human-readable.

    :param fp: An open text-mode file handle positioned for appending.
    :param record: The dict to serialize.
    """
    fp.write(json.dumps(record, ensure_ascii=False) + '\n')


class FetchState:
    """
    Persistent pagination bookkeeping for the OpenAlex fetcher.

    Two files live next to the target JSONL, sharing its stem:

    - ``<target>.cursor`` — single-line file storing the OpenAlex ``next_cursor``
      token. Re-read on load, rewritten after every batch, and used to resume
      traversal. Starts at ``"*"`` when absent (the OpenAlex sentinel for
      "beginning of cursor pagination").
    - ``<target>.ids`` — newline-delimited set of OpenAlex IDs already written
      to the JSONL, used to de-duplicate across resumed runs.

    The class supports both one-shot usage (``load`` + ``update_cursor``) and
    streaming usage (``open`` + ``mark_seen`` + ``close``) where new IDs are
    flushed to disk immediately so that a crash mid-fetch never loses seen-ID
    entries.

    :param target_path: The path of the output JSONL file whose state this
        instance manages. The ``.cursor`` and ``.ids`` sibling files are derived
        from it.
    """

    def __init__(self, target_path: PathLike):
        self.target_path = Path(target_path)
        self.cursor_path = Path(str(self.target_path) + '.cursor')
        self.ids_path = Path(str(self.target_path) + '.ids')
        self.cursor: str = '*'
        self.seen_ids: Set[str] = set()
        self._ids_fp: Optional[IO[str]] = None

    def load(self) -> None:
        """
        Populate ``cursor`` and ``seen_ids`` from any existing sibling files.

        Missing files are treated as empty state, so calling ``load`` on a
        fresh target is always safe.
        """
        if self.cursor_path.exists():
            self.cursor = self.cursor_path.read_text(encoding='utf-8').strip() or '*'

        if self.ids_path.exists():
            with self.ids_path.open(encoding='utf-8') as f:
                self.seen_ids = {line.strip() for line in f if line.strip()}

    def open(self) -> None:
        """
        Open the seen-IDs file for appending and load any existing state.

        Must be paired with :meth:`close`. Using ``FetchState`` as a context
        manager (via ``with``) is the recommended pattern.
        """
        self.load()
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        self._ids_fp = self.ids_path.open('a', encoding='utf-8')

    def close(self) -> None:
        """Close the underlying seen-IDs file handle if it is open."""
        if self._ids_fp is not None:
            self._ids_fp.close()
            self._ids_fp = None

    def __enter__(self) -> 'FetchState':
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def mark_seen(self, oid: str) -> None:
        """
        Record an OpenAlex ID as already fetched and flush it to disk.

        Only valid between :meth:`open` and :meth:`close`. The immediate flush
        is intentional: if the process is killed mid-batch, the next run will
        correctly skip the IDs we already wrote to the JSONL.

        :param oid: The OpenAlex work identifier (full URL form).
        """
        if self._ids_fp is None:
            raise RuntimeError(
                'FetchState.mark_seen called before open(); '
                'use the FetchState as a context manager or call .open() first.'
            )
        if oid in self.seen_ids:
            return
        self.seen_ids.add(oid)
        self._ids_fp.write(oid + '\n')
        self._ids_fp.flush()

    def update_cursor(self, cursor: Optional[str]) -> None:
        """
        Persist a new cursor value. ``None`` is a no-op so callers do not have
        to special-case the end of pagination.

        :param cursor: The next-page cursor token returned by OpenAlex.
        """
        if cursor is None:
            return
        self.cursor = cursor
        self.cursor_path.parent.mkdir(parents=True, exist_ok=True)
        self.cursor_path.write_text(cursor, encoding='utf-8')
