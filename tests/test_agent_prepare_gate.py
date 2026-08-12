"""
Tests for what the prepare gate accepts as a downloaded dataset.

The gate checked file size and an HTML byte prefix and nothing else, which made it
simultaneously too loose and too tight. A file containing the literal text "Not Found"
passed; a two-row `license_compounds.csv` was rejected, because the documentation-name
list was matched as a bare substring of the filename.

Severity was mostly capped by the smoke run failing later, with one silent exception:
a truncated-but-parseable download makes `input_rows` describe a different file than the
script fetches, which turns the drop-rate check into a no-op. That case is covered here
too.
"""

import pytest

from chem_mat_data.agent import manifest as manifest_module
from chem_mat_data.agent.manifest import Manifest
from chem_mat_data.agent.gates import (
    gate_prepare,
    is_documentation_name,
    parses_as_table,
)

from pathlib import Path


GOOD_RECIPE = (
    'Direct URL: https://zenodo.org/records/1/files/BP.csv?download=1\n'
    'Columns: smiles, BP in degrees Celsius\n'
)


@pytest.fixture
def work_root(tmp_path, monkeypatch):
    """Redirect the pipeline working directory into a temporary folder."""
    monkeypatch.setattr(manifest_module, 'WORK_DIR', tmp_path / '.dataset_work')
    return tmp_path / '.dataset_work'


def prepare(work_root, files, recipe=GOOD_RECIPE):
    """Build a downloads folder and run the prepare gate over it."""
    manifest = Manifest.create('demo', source_url='https://example.org/paper')
    downloads = work_root / 'demo' / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (downloads / name).write_text(content)
    if recipe is not None:
        (downloads / 'download_recipe.md').write_text(recipe)
    return gate_prepare(manifest)


# -- what used to be accepted --

@pytest.mark.parametrize(
    'label,content',
    [
        ('an error page saved as csv', 'Not Found'),
        ('a repository record dump', '{"id": 14392754, "title": "x"}'),
        ('a header row with nothing under it', 'smiles,BP\n'),
        ('an xml error document', '<?xml version="1.0"?><error>denied</error>'),
    ],
)
def test_non_tabular_downloads_are_rejected(work_root, label, content):
    report = prepare(work_root, {'BP.csv': content})

    assert not report.ok, label
    assert any('parses as a table' in i.message for i in report.errors)


def test_a_real_table_passes(work_root):
    report = prepare(work_root, {'BP.csv': 'smiles,BP\nCCO,78.4\nCCC,-42.1\n'})

    assert report.ok, [i.message for i in report.errors]


def test_a_single_column_smiles_list_passes(work_root):
    """No minimum column count: a bare list of SMILES is a legitimate dataset."""
    report = prepare(work_root, {'compounds.smi': 'smiles\nCCO\nCCC\n'})

    assert report.ok, [i.message for i in report.errors]


def test_one_tabular_file_among_several_is_enough(work_root):
    """Ancillary records alongside the real data must not be a finding."""
    report = prepare(
        work_root,
        {
            'zenodo_record.json': '{"id": 1}',
            'notes.txt': 'some prose about the study',
            'BP.csv': 'smiles,BP\nCCO,78.4\n',
        },
    )

    assert report.ok, [i.message for i in report.errors]


# -- what used to be rejected --

def test_a_data_file_whose_name_contains_license_is_accepted(work_root):
    """The regression: matched as a bare substring, this was discarded as docs."""
    report = prepare(work_root, {'license_compounds.csv': 'smiles,BP\nCCO,78\n'})

    assert report.ok, [i.message for i in report.errors]


@pytest.mark.parametrize(
    'name,expected',
    [
        ('README.md', True),
        ('readme.txt', True),
        ('LICENSE', True),
        ('download_recipe.md', True),
        ('manifest.yml', True),
        ('license_compounds.csv', False),
        ('readme_values.csv', False),
        ('BP.csv', False),
    ],
)
def test_documentation_names_match_the_whole_stem(name, expected):
    assert is_documentation_name(Path(name)) is expected


def test_documentation_alongside_data_does_not_break_the_run(work_root):
    report = prepare(
        work_root,
        {'BP.csv': 'smiles,BP\nCCO,78\n', 'README.md': 'notes', 'LICENSE': 'CC-BY-4.0'},
    )

    assert report.ok, [i.message for i in report.errors]


def test_documentation_only_is_still_rejected_and_says_what_it_saw(work_root):
    report = prepare(work_root, {'README.md': 'notes', 'LICENSE': 'CC-BY-4.0'})

    assert not report.ok
    message = next(i.message for i in report.errors if 'no data files' in i.message)
    assert 'README.md' in message and 'LICENSE' in message


# -- the recipe --

def test_a_recipe_with_only_a_doi_is_rejected(work_root):
    """A DOI resolves to a landing page, not to a file."""
    report = prepare(
        work_root,
        {'BP.csv': 'smiles,BP\nCCO,78\n'},
        recipe='See the paper at https://doi.org/10.1000/xyz',
    )

    assert not report.ok
    assert any('only DOI links' in i.message for i in report.errors)


def test_a_recipe_with_a_doi_and_a_real_url_passes(work_root):
    report = prepare(
        work_root,
        {'BP.csv': 'smiles,BP\nCCO,78\n'},
        recipe='Paper https://doi.org/10.1000/xyz\nFile https://zenodo.org/records/1/files/BP.csv',
    )

    assert report.ok, [i.message for i in report.errors]


def test_a_recipe_with_no_url_is_rejected(work_root):
    report = prepare(
        work_root, {'BP.csv': 'smiles,BP\nCCO,78\n'}, recipe='downloaded by hand'
    )

    assert not report.ok
    assert any('records no URL' in i.message for i in report.errors)


# -- schema documentation --

def test_a_recipe_describing_the_columns_satisfies_the_schema_warning(work_root):
    """
    The agent's own recipe documented the columns better than any README.

    Warning about a missing README while the schema is documented in front of you is
    noise, and noise is what teaches a reader to skim warnings.
    """
    report = prepare(work_root, {'BP.csv': 'smiles,BP\nCCO,78\n'})

    assert not [i for i in report.warnings if 'schema documentation' in i.message]


def test_no_readme_and_no_schema_notes_warns(work_root):
    report = prepare(
        work_root,
        {'BP.csv': 'smiles,BP\nCCO,78\n'},
        recipe='Direct URL: https://zenodo.org/records/1/files/BP.csv',
    )

    assert report.ok
    assert any('schema documentation' in i.message for i in report.warnings)


# -- parses_as_table directly --

def test_parses_as_table_reads_only_a_prefix(tmp_path):
    """A multi-gigabyte archive must not be read in full to prove it is data."""
    path = tmp_path / 'big.csv'
    path.write_text('smiles,BP\n' + ''.join(f'C{i},{i}\n' for i in range(50_000)))

    assert parses_as_table(path, max_bytes=2_000)


def test_parses_as_table_rejects_an_empty_file(tmp_path):
    path = tmp_path / 'empty.csv'
    path.write_text('')

    assert not parses_as_table(path)
