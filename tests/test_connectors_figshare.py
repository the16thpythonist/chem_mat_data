"""
Tests for :class:`~chem_mat_data.connectors.FigshareSource`.

The class existed for months with no tests at all, and with a defect that only shows up
in the case it is most likely to be used for: a paper's data availability section cites
the Figshare **article** - a DOI or a landing page - while only per-file ``ndownloader``
URLs can actually be downloaded. Passing the article link used to fall through to a
branch commented "accept the URL as-is but warn the user", which emitted no warning and
downloaded the HTML landing page. With ``relative_path`` set that surfaces as a confusing
zip error; without it, an HTML file is returned as though it were the data.

The mocked tests below cover every path offline. One live test at the end is marked
``network`` - deselected by default - because everything else here asserts what I believe
Figshare's API does, and only that one asserts what it actually does.
"""

import io
import json
import os
import zipfile
from unittest.mock import Mock, patch

import pytest

from chem_mat_data.connectors import USER_AGENT, FigshareSource


#: A stand-in for the article endpoint's payload, shaped like QM40's: several files with
#: the same extension, where taking the wrong one is silent.
ARTICLE_FILES = [
    {'id': 111, 'name': 'QM40_main.csv', 'size': 48_600_000,
     'download_url': 'https://ndownloader.figshare.com/files/111'},
    {'id': 222, 'name': 'QM40_xyz.csv', 'size': 432_100_000,
     'download_url': 'https://ndownloader.figshare.com/files/222'},
    {'id': 333, 'name': 'QM40_bond.csv', 'size': 245_000_000,
     'download_url': 'https://ndownloader.figshare.com/files/333'},
]


def api_response(payload):
    """Build a mock response that returns *payload* from ``.json()``."""
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    response.status_code = 200
    return response


def download_response(content: bytes):
    """Build a mock streaming response carrying *content*."""
    response = Mock()
    response.status_code = 200
    response.headers = {'content-length': str(len(content))}
    response.iter_content.return_value = [content]
    response.raise_for_status.return_value = None
    return response


def zip_bytes(members: dict) -> bytes:
    """Build an in-memory ZIP archive from ``{path: text}``."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for path, text in members.items():
            archive.writestr(path, text)
    return buffer.getvalue()


# -- what a link is understood to mean ------------------------------------------------

@pytest.mark.parametrize('url,expected', [
    # per-file, which is the only directly fetchable form
    ('https://ndownloader.figshare.com/files/25766582', (25766582, None, None)),
    ('https://springernature.figshare.com/ndownloader/files/25766582',
     (25766582, None, None)),
    # article landing pages
    ('https://figshare.com/articles/dataset/QM40/25993060', (None, 25993060, None)),
    ('https://figshare.com/articles/dataset/QM40/25993060/2', (None, 25993060, 2)),
    ('https://figshare.com/articles/25993060', (None, 25993060, None)),
    # institutional portals, where publisher-hosted supplementary data lives
    ('https://springernature.figshare.com/articles/dataset/Some_Data/12345',
     (None, 12345, None)),
    # DOIs, bare and resolved
    ('10.6084/m9.figshare.25993060', (None, 25993060, None)),
    ('10.6084/m9.figshare.25993060.v3', (None, 25993060, 3)),
    ('https://doi.org/10.6084/m9.figshare.25993060.v1', (None, 25993060, 1)),
])
def test_every_link_form_is_understood(url, expected):
    """
    All four shapes in the wild resolve to the same three facts.

    A caller has whichever one the paper happened to print; they should not have to
    know that only one of them can be downloaded.
    """
    assert FigshareSource._parse_link(url) == expected


def test_an_unrecognised_link_raises_instead_of_downloading_html():
    """
    The regression this file exists for.

    Silently downloading a landing page is worse than failing, because the failure
    surfaces much later - as a zip error, or as an HTML file treated as data.
    """
    with pytest.raises(ValueError) as excinfo:
        FigshareSource(url='https://figshare.com/collections/Some_Collection/98765')

    message = str(excinfo.value)
    assert 'not a recognised Figshare link' in message
    assert 'HTML landing page' in message, 'the error should say why this is refused'


def test_an_explicit_version_beats_the_one_in_the_link():
    """Asking for a version explicitly should not be silently overridden by the URL."""
    assert FigshareSource._parse_link(
        'https://doi.org/10.6084/m9.figshare.25993060.v1', version=4
    ) == (None, 25993060, 4)


def test_constructor_rejects_contradictory_arguments():
    """Two sources of truth for what to download is a caller error, not a preference."""
    with pytest.raises(ValueError):
        FigshareSource()
    with pytest.raises(ValueError):
        FigshareSource(file_id=1, url='https://ndownloader.figshare.com/files/2')
    with pytest.raises(ValueError):
        FigshareSource(article_id=1, url='https://figshare.com/articles/dataset/x/2')


# -- resolving an article to one file -------------------------------------------------

@patch('chem_mat_data.connectors.requests.get')
def test_a_multi_file_article_refuses_to_guess(mock_get):
    """
    The expensive mistake this guards against.

    QM40 ships three CSVs with overlapping columns. Picking the wrong one yields a
    valid dataset of the wrong thing, and nothing downstream would notice - so the
    error has to carry the listing rather than a suggestion to go and look.
    """
    mock_get.return_value = api_response({'files': ARTICLE_FILES})

    with pytest.raises(ValueError) as excinfo:
        FigshareSource(article_id=25993060).resolve()

    message = str(excinfo.value)
    assert 'holds 3 files' in message
    assert 'name=' in message
    for entry in ARTICLE_FILES:
        assert entry['name'] in message, 'the listing must name every candidate'
    assert 'MB' in message, 'sizes make the choice obvious; raw bytes do not'


@patch('chem_mat_data.connectors.requests.get')
def test_a_single_file_article_needs_no_name(mock_get):
    """With no choice to make, refusing to choose would be pedantry."""
    mock_get.return_value = api_response({'files': [ARTICLE_FILES[0]]})

    source = FigshareSource(article_id=25993060)

    assert source.resolve() == 'https://ndownloader.figshare.com/files/111'
    assert source.file_id == 111
    assert source.filename == 'QM40_main.csv'


@pytest.mark.parametrize('name,expected_id', [
    ('QM40_main.csv', 111),      # exact
    ('qm40_MAIN.csv', 111),      # case-insensitive
    ('*xyz*', 222),              # glob
    ('bond', 333),               # substring
])
@patch('chem_mat_data.connectors.requests.get')
def test_a_file_can_be_named_loosely(mock_get, name, expected_id):
    """
    Four ways of naming the same file, because a caller reading a paper has the name
    approximately right far more often than exactly right.
    """
    mock_get.return_value = api_response({'files': ARTICLE_FILES})

    source = FigshareSource(article_id=25993060, name=name)
    source.resolve()

    assert source.file_id == expected_id


@patch('chem_mat_data.connectors.requests.get')
def test_an_ambiguous_name_raises_rather_than_taking_the_first(mock_get):
    """A pattern matching two files is the same problem as naming none."""
    mock_get.return_value = api_response({'files': ARTICLE_FILES})

    with pytest.raises(ValueError) as excinfo:
        FigshareSource(article_id=25993060, name='QM40*').resolve()

    assert 'matches 3 files' in str(excinfo.value)


@patch('chem_mat_data.connectors.requests.get')
def test_a_name_that_matches_nothing_shows_what_is_there(mock_get):
    """A typo should be answerable from the error alone."""
    mock_get.return_value = api_response({'files': ARTICLE_FILES})

    with pytest.raises(ValueError) as excinfo:
        FigshareSource(article_id=25993060, name='QM40_mian.csv').resolve()

    message = str(excinfo.value)
    assert "no file named 'QM40_mian.csv'" in message
    assert 'QM40_main.csv' in message


@patch('chem_mat_data.connectors.requests.get')
def test_the_exact_match_wins_over_the_substring(mock_get):
    """
    A name that is both an exact match and a substring of another must not be
    ambiguous - otherwise 'data.csv' could never be requested alongside 'data.csv.gz'.
    """
    files = [
        {'id': 1, 'name': 'data.csv', 'size': 10, 'download_url': 'u1'},
        {'id': 2, 'name': 'data.csv.gz', 'size': 5, 'download_url': 'u2'},
    ]
    mock_get.return_value = api_response({'files': files})

    source = FigshareSource(article_id=1, name='data.csv')
    source.resolve()

    assert source.file_id == 1


@patch('chem_mat_data.connectors.requests.get')
def test_a_version_is_requested_from_the_versioned_endpoint(mock_get):
    """Pinning a version is the whole point of citing one; it must reach the API."""
    mock_get.return_value = api_response({'files': [ARTICLE_FILES[0]]})

    FigshareSource(url='https://doi.org/10.6084/m9.figshare.25993060.v2').resolve()

    assert mock_get.call_args[0][0].endswith('/articles/25993060/versions/2')


@patch('chem_mat_data.connectors.requests.get')
def test_the_files_endpoint_is_used_when_the_article_omits_them(mock_get):
    """Tolerate an API response that does not embed the file list."""
    mock_get.side_effect = [
        api_response({'title': 'QM40'}),
        api_response([ARTICLE_FILES[0]]),
    ]

    source = FigshareSource(article_id=25993060)
    source.resolve()

    assert source.file_id == 111
    assert mock_get.call_args[0][0].endswith('/files')


@patch('chem_mat_data.connectors.requests.get')
def test_resolving_twice_hits_the_api_once(mock_get):
    """`fetch` calls `resolve`, so a caller that resolved first must not pay twice."""
    mock_get.return_value = api_response({'files': [ARTICLE_FILES[0]]})

    source = FigshareSource(article_id=25993060)
    source.resolve()
    source.resolve()

    assert mock_get.call_count == 1


def test_a_file_source_has_no_article_to_list():
    """Asking a per-file source what else is in the article is a category error."""
    with pytest.raises(ValueError):
        FigshareSource(file_id=111).list_files()


# -- downloading ----------------------------------------------------------------------

@patch('chem_mat_data.connectors.requests.get')
def test_an_article_link_downloads_the_named_file(mock_get):
    """End to end from a DOI, which is the case the class exists for."""
    mock_get.side_effect = [
        api_response({'files': ARTICLE_FILES}),
        download_response(b'smiles,homo\nCCO,-0.25\n'),
    ]

    with FigshareSource(
        url='https://doi.org/10.6084/m9.figshare.25993060', name='QM40_main.csv'
    ) as source:
        path = source.fetch()
        # Read inside the block: leaving it removes the temporary directory.
        assert open(path).read().startswith('smiles,homo')
        assert path.endswith('QM40_main.csv'), (
            'the file should land on disk under its real name, not the numeric id '
            'that the ndownloader URL ends in'
        )


@patch('chem_mat_data.connectors.requests.get')
def test_a_file_id_still_works_unchanged(mock_get):
    """
    The two scripts already using this class pass `file_id` and `relative_path`.

    They must keep working without touching the API at all.
    """
    mock_get.return_value = download_response(
        zip_bytes({'PFAS-Map/test_data/PFAS_TTR_Binding.csv': 'smiles,y\nCF,1\n'})
    )

    with FigshareSource(
        file_id=25766582, relative_path='PFAS-Map/test_data/PFAS_TTR_Binding.csv'
    ) as source:
        path = source.fetch()
        assert open(path).read().startswith('smiles,y')

    assert mock_get.call_count == 1, 'a file id needs no API lookup'


@patch('chem_mat_data.connectors.requests.get')
def test_every_request_identifies_as_chem_mat_data(mock_get):
    """
    Figshare holds a client posing as a browser at HTTP 202 forever.

    Both the API lookup and the download must send the honest ``USER_AGENT``, or the
    retry loop has nothing to wait for.
    """
    mock_get.side_effect = [
        api_response({'files': [ARTICLE_FILES[0]]}),
        download_response(b'smiles,homo\nCCO,-0.25\n'),
    ]

    with FigshareSource(article_id=25993060) as source:
        source.fetch()

    assert mock_get.call_count == 2
    for call in mock_get.call_args_list:
        assert call.kwargs['headers']['User-Agent'] == USER_AGENT


@patch('chem_mat_data.connectors.time.sleep')
@patch('chem_mat_data.connectors.requests.get')
def test_a_202_is_retried(mock_get, mock_sleep):
    """
    Publisher subdomains answer 202 while the file is prepared.

    Sleep is patched out so the retry logic is tested rather than waited on.
    """
    accepted = Mock(status_code=202)
    mock_get.side_effect = [accepted, accepted, download_response(b'data')]

    with FigshareSource(file_id=1) as source:
        path = source.fetch()
        assert open(path, 'rb').read() == b'data'

    assert mock_get.call_count == 3


@patch('chem_mat_data.connectors.time.sleep')
@patch('chem_mat_data.connectors.requests.get')
def test_a_permanent_202_gives_up_loudly(mock_get, mock_sleep):
    """An endless 202 must not be mistaken for an empty file."""
    mock_get.return_value = Mock(status_code=202)

    with pytest.raises(RuntimeError) as excinfo:
        with FigshareSource(file_id=1, max_retries=2) as source:
            source.fetch()

    assert '202' in str(excinfo.value)


# -- the one test that checks Figshare rather than my beliefs about it ----------------

@pytest.mark.network
def test_a_real_article_resolves():
    """
    Everything above mocks the API, so it verifies my assumptions and nothing else.

    This one asks Figshare. It is deselected by default (the project's addopts exclude
    the `network` marker) and exists so the contract is checked against reality when
    someone chooses to, rather than never.

    Uses the QM40 record, which at the time of writing holds a single 332 MB archive.
    The assertions are deliberately about the *contract* - the article endpoint embeds
    a file list, and each entry carries id, name, size and a download URL - rather than
    about that record's current contents, which upstream may change without warning.
    """
    source = FigshareSource(url='https://doi.org/10.6084/m9.figshare.25993060')
    files = source.list_files()

    assert files, 'the QM40 article should list files'
    for entry in files:
        assert {'id', 'name', 'size', 'download_url'} <= set(entry)
        assert 'ndownloader' in entry['download_url']

    if len(files) == 1:
        # A single file needs no name, and resolving must yield a fetchable URL.
        assert 'figshare.com' in source.resolve()
    else:
        # With a choice to make, resolving without a name must still refuse to guess.
        with pytest.raises(ValueError):
            source.resolve()


def test_directory_decides_where_the_download_lands(tmp_path):
    # A multi-gigabyte file must not be forced onto the system temp disk, which is
    # often the small one.
    source = FigshareSource(file_id=1, directory=str(tmp_path))
    source.prepare()
    try:
        assert os.path.dirname(source.temp_dir) == str(tmp_path)
    finally:
        source.cleanup()
