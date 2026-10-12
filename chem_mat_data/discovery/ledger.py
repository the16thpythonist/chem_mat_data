"""
Screening ledgers for the dataset-search workflow.

The discovery classifier flags papers that *look* like dataset releases, but a
flagged paper is only worth an expensive per-paper verification if nobody has
answered the question already. Three files in this repository hold prior
answers, and this module reads all three behind one function:

- ``chem_mat_data/metadata.yml`` — the shipped catalog. A paper whose data is
  already distributed by ChemMatData does not need finding again.
- ``docs/rejected_datasets.md`` — datasets investigated and refused **by a
  person**, with the evidence. A hit here is authoritative and only a person
  may overturn it.
- ``docs/search_verdicts.md`` — the verdict ledger written by this workflow.
  It exists so that the second sweep does not re-verify what the first one
  already decided; the resumable ``.ids`` sidecar prevents re-*fetching* a
  paper, but says nothing about whether it was ever *judged*.

The public surface is deliberately small:

- :func:`screen` — answer "has this paper been dealt with?" for one URL.
- :func:`screen_many` — the same for a batch, loading each ledger once.
- :class:`VerdictLedger` — read and append the verdict ledger.

Everything is keyed on the **bare DOI** (``10.1021/acs.jcim.0c01041``) rather
than on the URL as written, because the same paper appears across these files
as a ``https://doi.org/`` link, a publisher URL and a bare identifier. URLs
that carry no DOI at all — a GitHub repository, an OpenAlex landing page — are
matched on their normalized full form instead.

Example:

.. code-block:: python

    from chem_mat_data.discovery.ledger import screen

    result = screen('https://doi.org/10.1021/acs.jcim.0c01041')
    print(result.status)   # 'in_catalog'
    print(result.detail)   # 'tmqm'
    print(result.skip)     # True
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import yaml

from chem_mat_data.utils import METADATA_PATH, PROJECT_PATH

PathLike = Union[str, os.PathLike]

#: Default location of the shipped dataset catalog.
CATALOG_PATH: Path = Path(METADATA_PATH)

#: Default location of the human-decided rejection list.
REJECTED_PATH: Path = Path(PROJECT_PATH) / 'docs' / 'rejected_datasets.md'

#: Default location of the verdict ledger written by ``discover verdict``.
VERDICTS_PATH: Path = Path(PROJECT_PATH) / 'docs' / 'search_verdicts.md'

#: Default location of the standing list of leads this workflow promotes into.
LEADS_PATH: Path = Path(PROJECT_PATH) / 'MISSING_DATASETS.md'

#: Screening outcomes. ``new`` is the only one that warrants spending a
#: verification subagent on the paper.
STATUS_NEW: str = 'new'
STATUS_IN_CATALOG: str = 'in_catalog'
STATUS_REJECTED: str = 'rejected'
STATUS_VERIFIED: str = 'verified'

#: The paper is already a lead in ``MISSING_DATASETS.md``. Deliberately **not** a
#: skip status: those rows were confirmed from the title alone by an LLM, and 26%
#: of that population was later measured as mislabelled
#: (``POSITIVES_VALIDATION.md``: 37 of 142 rejected on re-examination). Skipping
#: them would entrench a title-only guess and deny it the record URL and entry
#: counts that only a verification subagent produces. The status exists so the
#: agent is *told*, and can enrich the existing row instead of appending a
#: duplicate one.
STATUS_LISTED: str = 'listed'

#: Outcomes that mean "do not spend a subagent on this paper".
SKIP_STATUSES: Tuple[str, ...] = (STATUS_IN_CATALOG, STATUS_REJECTED, STATUS_VERIFIED)

#: Verdict vocabulary accepted by :meth:`VerdictLedger.append`. Kept small and
#: fixed for the same reason the license gate fixes its own vocabulary: a free
#: text verdict cannot be matched against later.
DECISIONS: Tuple[str, ...] = ('confirmed', 'rejected', 'unclear')

#: Matches a DOI anywhere in a string. Deliberately permissive on the suffix —
#: DOIs legitimately contain slashes, dots, parentheses and hyphens — but
#: trailing punctuation that belongs to the surrounding prose is trimmed by
#: :func:`normalize_doi`.
_DOI_RE = re.compile(r'10\.\d{4,9}/[^\s"<>|\)\]]+', re.IGNORECASE)

#: Trailing characters that are almost always markdown or prose punctuation
#: rather than part of the DOI itself.
_DOI_TRAILING = ').,;:]}>*_'


def normalize_doi(value: Optional[str]) -> Optional[str]:
    """
    Extract and normalise the bare DOI contained in ``value``.

    Accepts any of the forms these files actually use — a full
    ``https://doi.org/10.x/y`` URL, a ``doi:10.x/y`` prefix, a markdown link
    body, or the bare identifier — and returns the lowercase bare DOI. Returns
    ``None`` when the value contains no DOI at all, which is the normal case
    for a Zenodo record URL or a GitHub repository.

    :param value: Any string that may contain a DOI, or ``None``.

    :returns: The lowercase bare DOI, or ``None``.
    """
    if not value:
        return None
    match = _DOI_RE.search(value)
    if not match:
        return None
    return match.group(0).rstrip(_DOI_TRAILING).lower()


def normalize_url(value: Optional[str]) -> Optional[str]:
    """
    Normalise a URL for equality comparison.

    Strips the scheme, a leading ``www.``, any trailing slash and lowercases
    the result, so that ``https://Zenodo.org/records/123/`` and
    ``http://zenodo.org/records/123`` compare equal. This is the fallback key
    for entries that carry no DOI.

    :param value: A URL, or ``None``.

    :returns: The normalised form, or ``None`` when the input is empty.
    """
    if not value:
        return None
    text = value.strip()
    for scheme in ('https://', 'http://'):
        if text.lower().startswith(scheme):
            text = text[len(scheme):]
            break
    if text.lower().startswith('www.'):
        text = text[4:]
    return text.rstrip('/').lower() or None


def _keys_for(value: Optional[str]) -> Set[str]:
    """
    Return every lookup key a single URL should be indexed and searched under.

    A URL contributes its bare DOI when it has one and its normalised URL form
    always, so that a paper recorded as a publisher link in one file and as a
    ``doi.org`` link in another still matches.
    """
    keys: Set[str] = set()
    doi = normalize_doi(value)
    if doi:
        keys.add(f'doi:{doi}')
    url = normalize_url(value)
    if url:
        keys.add(f'url:{url}')
    return keys


# == the shipped catalog ==

@dataclass
class CatalogIndex:
    """
    Lookup index over the ``sources`` URLs of the shipped dataset catalog.

    :ivar by_key: Maps a lookup key (see :func:`_keys_for`) to the dataset name.
    :ivar unindexed: Names of catalog datasets that declare no ``sources`` at
        all. These cannot be matched by URL, which is a real blind spot rather
        than a proof of absence — callers should surface it.
    :ivar total: Total number of datasets in the catalog.
    """

    by_key: Dict[str, str] = field(default_factory=dict)
    unindexed: List[str] = field(default_factory=list)
    total: int = 0

    def lookup(self, value: Optional[str]) -> Optional[str]:
        """Return the dataset name whose sources contain ``value``, or ``None``."""
        for key in _keys_for(value):
            name = self.by_key.get(key)
            if name:
                return name
        return None


def load_catalog(path: Optional[PathLike] = None) -> CatalogIndex:
    """
    Build a :class:`CatalogIndex` from ``metadata.yml``.

    A missing or unreadable catalog yields an empty index rather than raising:
    the screening step is an optimisation, and failing it should degrade the
    run to "verify this paper anyway" rather than stop it.

    :param path: Override for the catalog location.

    :returns: The populated index.
    """
    catalog_path = Path(path) if path is not None else CATALOG_PATH
    index = CatalogIndex()
    try:
        with catalog_path.open(encoding='utf-8') as file:
            data = yaml.safe_load(file) or {}
    except (OSError, yaml.YAMLError):
        return index

    datasets = data.get('datasets') or {}
    if not isinstance(datasets, dict):
        return index

    index.total = len(datasets)
    for name, entry in datasets.items():
        if not isinstance(entry, dict):
            continue
        sources = entry.get('sources') or []
        if isinstance(sources, str):
            sources = [sources]
        found = False
        for source in sources:
            for key in _keys_for(source):
                index.by_key.setdefault(key, name)
                found = True
        if not found:
            index.unindexed.append(name)
    index.unindexed.sort()
    return index


# == the human-decided rejection list ==

@dataclass
class RejectedEntry:
    """One ``### `` section of ``docs/rejected_datasets.md``."""

    heading: str
    keys: Set[str] = field(default_factory=set)


def load_rejected(path: Optional[PathLike] = None) -> List[RejectedEntry]:
    """
    Parse the rejection list into one entry per ``### `` heading.

    Every DOI and URL appearing anywhere inside a section becomes a lookup key
    for that section, which intentionally over-matches: a paper that shares a
    data record or an upstream database with a rejected entry is exactly the
    case worth stopping on, even when the publication itself is different.

    :param path: Override for the rejection list location.

    :returns: The parsed entries, empty when the file is absent.
    """
    rejected_path = Path(path) if path is not None else REJECTED_PATH
    try:
        text = rejected_path.read_text(encoding='utf-8')
    except OSError:
        return []

    entries: List[RejectedEntry] = []
    current: Optional[RejectedEntry] = None
    for line in text.splitlines():
        if line.startswith('### '):
            current = RejectedEntry(heading=line[4:].strip())
            entries.append(current)
            continue
        if current is None:
            continue
        for doi in _DOI_RE.findall(line):
            current.keys.add(f'doi:{doi.rstrip(_DOI_TRAILING).lower()}')
        for url in re.findall(r'https?://[^\s\)\]|]+', line):
            normalized = normalize_url(url)
            if normalized:
                current.keys.add(f'url:{normalized}')
    return entries


def lookup_rejected(value: Optional[str], entries: Sequence[RejectedEntry]) -> Optional[str]:
    """Return the heading of the rejection entry matching ``value``, or ``None``."""
    keys = _keys_for(value)
    if not keys:
        return None
    for entry in entries:
        if keys & entry.keys:
            return entry.heading
    return None


# == the standing list of leads ==

def load_leads(path: Optional[PathLike] = None) -> Dict[str, str]:
    """
    Index ``MISSING_DATASETS.md`` by every identifier its rows carry.

    The file is a set of markdown tables of candidate datasets, one per row, with
    a URL column. Every DOI and URL found in a table row becomes a lookup key for
    that row, and the value is the row's second cell — the dataset/paper name — so
    that a hit can be reported usefully rather than as a bare "seen before".

    Unlike the catalog and the rejection list, a hit here is **not** a reason to
    skip the paper; see :data:`STATUS_LISTED`.

    :param path: Override for the leads file location.

    :returns: Mapping of lookup key to the lead's name. Empty when the file is
        absent, since screening must degrade rather than fail.
    """
    leads_path = Path(path) if path is not None else LEADS_PATH
    try:
        text = leads_path.read_text(encoding='utf-8')
    except OSError:
        return {}

    index: Dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith('|'):
            continue
        cells = [cell.strip() for cell in stripped.strip('|').split('|')]
        if len(cells) < 2 or set(cells[0]) <= {'-', ':'}:
            continue
        name = cells[1]
        for cell in cells:
            for key in _keys_for(cell):
                index.setdefault(key, name)
    return index


def lookup_lead(value: Optional[str], leads: Dict[str, str]) -> Optional[str]:
    """Return the name of the lead matching ``value``, or ``None``."""
    for key in _keys_for(value):
        name = leads.get(key)
        if name:
            return name
    return None


# == the verdict ledger ==

#: Column order of the markdown table in ``docs/search_verdicts.md``.
VERDICT_COLUMNS: Tuple[str, ...] = ('url', 'decision', 'name', 'record_url', 'why', 'date')

#: Preamble written when the ledger file is created. It explains the file to a
#: human reader, since the file is committed and will be read long after the
#: run that produced it.
VERDICT_HEADER: str = """# Dataset search verdicts

Papers judged by the `/dataset-search` skill, one row per paper, so that a later
sweep does not spend a verification subagent re-answering a settled question.

`decision` is one of `confirmed` (the paper releases a dataset), `rejected` (it
does not), or `unclear` (the verification could not tell). **These are machine
verdicts about whether a dataset exists — they say nothing about licensing.**
Whether a confirmed dataset may actually be redistributed is decided by the
`/dataset-extraction` skill's license stage, and rejections made *by a person*
live in [`rejected_datasets.md`](rejected_datasets.md), not here.

| url | decision | name | record_url | why | date |
|---|---|---|---|---|---|
"""


def _escape_cell(value: Optional[str]) -> str:
    """
    Render a value safe to place inside a markdown table cell.

    Newlines are collapsed so one verdict stays one row, and pipes are
    backslash-escaped so a ``why`` containing one cannot invent a column.
    :func:`_unescape_cell` is the exact inverse.
    """
    if value is None:
        return ''
    return ' '.join(str(value).split()).replace('|', '\\|')


def _unescape_cell(value: str) -> str:
    """Inverse of :func:`_escape_cell` for a single already-split cell."""
    return value.strip().replace('\\|', '|')


def _split_row(line: str) -> List[str]:
    """
    Split a markdown table row into unescaped cells.

    Splits on pipes that are *not* backslash-escaped, so that a cell whose text
    legitimately contains a pipe survives the round-trip instead of silently
    truncating the row.
    """
    return [_unescape_cell(cell) for cell in re.split(r'(?<!\\)\|', line)]


@dataclass
class Verdict:
    """One row of the verdict ledger."""

    url: str
    decision: str
    name: str = ''
    record_url: str = ''
    why: str = ''
    date: str = ''


class VerdictLedger:
    """
    Read/append interface to ``docs/search_verdicts.md``.

    The ledger is a markdown table rather than JSONL because it is committed
    and read by people — the same reasoning that makes ``rejected_datasets.md``
    and ``MISSING_DATASETS.md`` markdown. Parsing is tolerant: any line that is
    not a well-formed row is skipped rather than raising, so hand-edits and
    added prose cannot break a sweep.

    Example:

    .. code-block:: python

        ledger = VerdictLedger.load()
        if not ledger.lookup('https://doi.org/10.1021/foo'):
            ledger.append(Verdict(url='https://doi.org/10.1021/foo',
                                  decision='confirmed', name='foo_db',
                                  why='Zenodo record with 12k SMILES + logP'))
    """

    def __init__(self, path: Optional[PathLike] = None, verdicts: Optional[List[Verdict]] = None):
        self.path: Path = Path(path) if path is not None else VERDICTS_PATH
        self.verdicts: List[Verdict] = verdicts if verdicts is not None else []
        self._by_key: Dict[str, Verdict] = {}
        for verdict in self.verdicts:
            self._index(verdict)

    def _index(self, verdict: Verdict) -> None:
        """
        Register a verdict under every key it should be findable by.

        Later verdicts **overwrite** earlier ones for the same key. That is what
        makes ``discover verdict --force`` mean something: the ledger is
        append-only, so a correction is a second row, and if the first row kept
        the key the correction would be written to disk and never read back —
        the tool would report success for a change that did nothing.
        """
        for key in _keys_for(verdict.url) | _keys_for(verdict.record_url):
            self._by_key[key] = verdict

    @classmethod
    def load(cls, path: Optional[PathLike] = None) -> 'VerdictLedger':
        """
        Read the ledger from disk, returning an empty one when it does not exist.

        :param path: Override for the ledger location.

        :returns: The loaded ledger.
        """
        ledger_path = Path(path) if path is not None else VERDICTS_PATH
        try:
            text = ledger_path.read_text(encoding='utf-8')
        except OSError:
            return cls(path=ledger_path)

        verdicts: List[Verdict] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith('|') or not stripped.endswith('|'):
                continue
            cells = _split_row(stripped[1:-1])
            if len(cells) < 2:
                continue
            # Skip the header row and the ``|---|---|`` separator beneath it.
            if cells[0] in ('url', '') or set(cells[0]) <= {'-', ':'}:
                continue
            if cells[1] not in DECISIONS:
                continue
            padded = (cells + [''] * len(VERDICT_COLUMNS))[:len(VERDICT_COLUMNS)]
            verdicts.append(Verdict(**dict(zip(VERDICT_COLUMNS, padded))))
        return cls(path=ledger_path, verdicts=verdicts)

    def lookup(self, value: Optional[str]) -> Optional[Verdict]:
        """Return the recorded verdict matching ``value``, or ``None``."""
        for key in _keys_for(value):
            verdict = self._by_key.get(key)
            if verdict:
                return verdict
        return None

    def append(self, verdict: Verdict) -> None:
        """
        Append one verdict to the ledger, creating the file if needed.

        The write is an append rather than a rewrite so that a crashed sweep
        loses at most the row it was mid-way through, and so that concurrent
        verification batches cannot clobber each other's rows.

        :param verdict: The verdict to record. ``decision`` must be one of
            :data:`DECISIONS`.

        :raises ValueError: If ``decision`` is outside the fixed vocabulary, or
            ``url`` is empty.
        """
        if verdict.decision not in DECISIONS:
            raise ValueError(
                f'decision must be one of {", ".join(DECISIONS)}, got {verdict.decision!r}'
            )
        if not (verdict.url or '').strip():
            raise ValueError('a verdict needs a url')

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(VERDICT_HEADER, encoding='utf-8')

        row = '| ' + ' | '.join(
            _escape_cell(getattr(verdict, column)) for column in VERDICT_COLUMNS
        ) + ' |\n'
        with self.path.open('a', encoding='utf-8') as file:
            file.write(row)
        self.verdicts.append(verdict)
        self._index(verdict)


# == screening ==

@dataclass
class ScreenResult:
    """
    The outcome of screening one paper against all three ledgers.

    :ivar url: The URL as supplied by the caller.
    :ivar status: One of :data:`STATUS_NEW`, :data:`STATUS_IN_CATALOG`,
        :data:`STATUS_REJECTED`, :data:`STATUS_VERIFIED`, :data:`STATUS_LISTED`.
    :ivar detail: What was matched — a dataset name, a rejection heading, a lead
        name, or a prior decision. Empty for :data:`STATUS_NEW`.
    :ivar source: Which ledger produced the match, for the human reading the
        run's notes. Empty for :data:`STATUS_NEW`.
    :ivar title: The paper title, when the caller had one. Screening never looks
        it up — it is carried through so that the briefs written from this output
        can name the paper instead of a bare DOI.
    :ivar openalex_id: The short OpenAlex identifier (``W123…``), when the caller
        had one. Carried for the same reason as ``title``: it is the only stable,
        filesystem-safe name for a paper, and a DOI cannot be used as a filename.
    """

    url: str
    status: str = STATUS_NEW
    detail: str = ''
    source: str = ''
    title: str = ''
    openalex_id: str = ''

    @property
    def skip(self) -> bool:
        """True when this paper should not be sent to a verification subagent."""
        return self.status in SKIP_STATUSES

    def to_dict(self) -> Dict[str, object]:
        """Render as a plain dict, for the ``--json`` output of ``discover check``."""
        return {
            'openalex_id': self.openalex_id,
            'url': self.url,
            'title': self.title,
            'status': self.status,
            'detail': self.detail,
            'source': self.source,
            'skip': self.skip,
        }


def screen_many(
    urls: Sequence[str],
    catalog: Optional[CatalogIndex] = None,
    rejected: Optional[Sequence[RejectedEntry]] = None,
    ledger: Optional[VerdictLedger] = None,
    leads: Optional[Dict[str, str]] = None,
) -> List[ScreenResult]:
    """
    Screen a batch of papers, loading each source at most once.

    Checked in order of authority. The human-decided rejection list comes first,
    because a hit there ends the matter and must be reported even when the paper
    is also in the catalog; then the shipped catalog; then this workflow's own
    prior verdicts. All three mean *skip*.

    The standing leads list is checked last and means something weaker — see
    :data:`STATUS_LISTED`. It is reported so the caller can enrich the existing
    row rather than append a duplicate one, but the paper is still worth
    verifying, so :attr:`ScreenResult.skip` stays False.

    :param urls: The paper URLs (or DOIs) to screen.
    :param catalog: Pre-loaded catalog index; loaded from disk when omitted.
    :param rejected: Pre-loaded rejection entries; loaded from disk when omitted.
    :param ledger: Pre-loaded verdict ledger; loaded from disk when omitted.
    :param leads: Pre-loaded leads index; loaded from disk when omitted.

    :returns: One :class:`ScreenResult` per input URL, in input order.
    """
    catalog = catalog if catalog is not None else load_catalog()
    rejected = rejected if rejected is not None else load_rejected()
    ledger = ledger if ledger is not None else VerdictLedger.load()
    leads = leads if leads is not None else load_leads()

    results: List[ScreenResult] = []
    for url in urls:
        heading = lookup_rejected(url, rejected)
        if heading:
            results.append(ScreenResult(url, STATUS_REJECTED, heading, 'docs/rejected_datasets.md'))
            continue

        name = catalog.lookup(url)
        if name:
            results.append(ScreenResult(url, STATUS_IN_CATALOG, name, 'chem_mat_data/metadata.yml'))
            continue

        verdict = ledger.lookup(url)
        if verdict:
            detail = verdict.decision
            if verdict.name:
                detail = f'{verdict.decision} ({verdict.name})'
            results.append(ScreenResult(url, STATUS_VERIFIED, detail, 'docs/search_verdicts.md'))
            continue

        lead = lookup_lead(url, leads)
        if lead:
            results.append(ScreenResult(url, STATUS_LISTED, lead, 'MISSING_DATASETS.md'))
            continue

        results.append(ScreenResult(url))
    return results


def screen(url: str, **kwargs) -> ScreenResult:
    """
    Screen a single paper. Convenience wrapper around :func:`screen_many`.

    :param url: The paper URL or DOI.

    :returns: The screening result.
    """
    return screen_many([url], **kwargs)[0]
