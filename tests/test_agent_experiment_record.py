"""
Tests for checking a results folder against the record pycomex wrote about the run.

Every other gate check compares something the agent declared against something else the
same agent declared. ``experiment_meta.json`` is written by the experiment framework, so
it is the one witness that can contradict the agent - and until now nothing read it.

The failure it exists for: a run that raises partway through still leaves a folder with
a message pack, a metadata file and plausible counts, because molecules that fail
conversion are logged and skipped rather than raising. Such a folder passed all six
gates.

Field names and the status vocabulary are taken from pycomex itself
(``functional/experiment.py``: ``running`` -> ``failed`` / ``done``, with ``has_error``
and ``error_type`` set on failure), not from documentation.
"""

import json
from pathlib import Path

import pytest

from chem_mat_data.agent.verification import VerificationReport
from chem_mat_data.agent.gates import check_experiment_record


def write_meta(directory: Path, **overrides) -> Path:
    """
    Write an ``experiment_meta.json`` describing a successful run.

    :param directory: Folder to write into.
    :param overrides: Fields to replace.

    :returns: The folder.
    """
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        'name': 'create_graph_datasets__demo',
        'status': 'done',
        'has_error': False,
        'error_type': None,
        'start_time': 1786529051.28,
        'end_time': 1786529062.83,
        'duration': 11.5,
    }
    meta.update(overrides)
    (directory / 'experiment_meta.json').write_text(json.dumps(meta))
    return directory


def run(directory, script_path=None, label='') -> VerificationReport:
    """Run the check and return the report."""
    report = VerificationReport(results_dir=Path(directory))
    check_experiment_record(report, str(directory), script_path=script_path, label=label)
    return report


# -- the happy path --

def test_a_finished_run_passes(tmp_path):
    report = run(write_meta(tmp_path / 'debug'))

    assert report.ok
    assert not report.warnings


def test_a_finished_run_matching_its_script_passes(tmp_path):
    report = run(
        write_meta(tmp_path / 'debug'),
        script_path='chem_mat_data/scripts/create_graph_datasets__demo.py',
    )

    assert report.ok


# -- the failure this exists to catch --

def test_a_failed_run_is_an_error(tmp_path):
    """The regression: this folder used to pass all six gates."""
    report = run(write_meta(tmp_path / 'debug', status='failed', has_error=True))

    assert not report.ok
    assert any('did not' in i.message or 'raised' in i.message for i in report.errors)


def test_has_error_alone_is_an_error(tmp_path):
    report = run(write_meta(tmp_path / 'debug', has_error=True, error_type='ValueError'))

    assert not report.ok
    assert any('ValueError' in i.message for i in report.errors)


def test_a_run_still_marked_running_is_an_error(tmp_path):
    """
    Asserted positively against 'done', so a killed run is caught too.

    pycomex sets 'running' at start and only moves it at the end, so a process killed
    mid-flight leaves 'running' behind - which a check that merely excluded 'failed'
    would wave through.
    """
    report = run(write_meta(tmp_path / 'debug', status='running'))

    assert not report.ok
    assert any('running' in i.message for i in report.errors)


def test_the_error_type_is_reported(tmp_path):
    report = run(
        write_meta(
            tmp_path / 'debug', status='failed', has_error=True, error_type='KeyError'
        )
    )

    assert any('KeyError' in i.message for i in report.errors)


# -- results belonging to a different run --

def test_results_from_another_script_are_an_error(tmp_path):
    report = run(
        write_meta(tmp_path / 'debug', name='create_graph_datasets__something_else'),
        script_path='chem_mat_data/scripts/create_graph_datasets__demo.py',
    )

    assert not report.ok
    assert any('different run' in i.message for i in report.errors)


def test_name_is_not_checked_when_no_script_is_recorded(tmp_path):
    report = run(write_meta(tmp_path / 'debug', name='whatever'))

    assert report.ok


# -- degrading rather than crashing --

def test_a_missing_record_warns_rather_than_failing(tmp_path):
    """
    Older runs predate the record, so its absence cannot be fatal.

    It is still worth saying, because without it a crashed run is indistinguishable
    from a successful one.
    """
    directory = tmp_path / 'debug'
    directory.mkdir(parents=True)

    report = run(directory)

    assert report.ok
    assert any('experiment_meta.json' in i.message for i in report.warnings)


def test_unreadable_record_warns(tmp_path):
    directory = tmp_path / 'debug'
    directory.mkdir(parents=True)
    (directory / 'experiment_meta.json').write_text('{not json')

    report = run(directory)

    assert report.ok
    assert any('could not be read' in i.message for i in report.warnings)


def test_a_record_that_is_not_a_mapping_warns(tmp_path):
    directory = tmp_path / 'debug'
    directory.mkdir(parents=True)
    (directory / 'experiment_meta.json').write_text('["a", "list"]')

    report = run(directory)

    assert report.ok
    assert any('not a mapping' in i.message for i in report.warnings)


def test_no_results_dir_is_silent(tmp_path):
    """The caller reports a missing results dir; this check should not double up."""
    report = VerificationReport(results_dir=tmp_path)
    check_experiment_record(report, None)

    assert report.ok
    assert not report.warnings


# -- labelling --

def test_the_label_prefixes_messages(tmp_path):
    report = run(write_meta(tmp_path / 'debug', status='failed'), label='smoke run')

    assert all(i.message.startswith('smoke run: ') for i in report.errors)


# -- the script identity check: an independent witness the agent cannot forge --

def write_script(directory, body: str = 'print("hello")\n'):
    """Write a script and the framework's archived copy of it."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'experiment_code.py').write_text(body)
    return body


def test_an_edited_script_is_caught(tmp_path, monkeypatch):
    """
    `experiment_code.py` is written by the framework, so it can contradict the agent.

    If the script is edited after the run, the results describe code that no longer
    exists - and every other check becomes a claim about the wrong thing.
    """
    import chem_mat_data.agent.gates as gates_module

    results = tmp_path / 'debug'
    write_meta(results, name='create_graph_datasets__demo')
    write_script(results, 'print("what actually ran")\n')

    scripts = tmp_path / 'chem_mat_data' / 'scripts'
    scripts.mkdir(parents=True)
    (scripts / 'create_graph_datasets__demo.py').write_text('print("edited afterwards")\n')
    monkeypatch.setattr(gates_module, 'PROJECT_PATH', str(tmp_path))

    report = run(results, script_path='chem_mat_data/scripts/create_graph_datasets__demo.py')

    assert not report.ok
    assert any('has changed since these results' in i.message for i in report.errors)


def test_an_unchanged_script_passes(tmp_path, monkeypatch):
    import chem_mat_data.agent.gates as gates_module

    results = tmp_path / 'debug'
    write_meta(results, name='create_graph_datasets__demo')
    body = write_script(results, 'print("the real script")\n')

    scripts = tmp_path / 'chem_mat_data' / 'scripts'
    scripts.mkdir(parents=True)
    (scripts / 'create_graph_datasets__demo.py').write_text(body)
    monkeypatch.setattr(gates_module, 'PROJECT_PATH', str(tmp_path))

    report = run(results, script_path='chem_mat_data/scripts/create_graph_datasets__demo.py')

    assert report.ok, [i.message for i in report.errors]


def test_a_missing_archived_copy_is_not_an_error(tmp_path, monkeypatch):
    """Results predating the archived copy are legitimate; absence must not block."""
    import chem_mat_data.agent.gates as gates_module

    results = tmp_path / 'debug'
    write_meta(results, name='create_graph_datasets__demo')

    scripts = tmp_path / 'chem_mat_data' / 'scripts'
    scripts.mkdir(parents=True)
    (scripts / 'create_graph_datasets__demo.py').write_text('print("x")\n')
    monkeypatch.setattr(gates_module, 'PROJECT_PATH', str(tmp_path))

    assert run(results, script_path='chem_mat_data/scripts/create_graph_datasets__demo.py').ok
