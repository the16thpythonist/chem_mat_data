import tempfile
from pathlib import Path

import yaml
import pytest

from chem_mat_data.agent.verification import (
    ERROR,
    WARNING,
    VerificationReport,
    normalize_target_description_keys,
    verify_results,
)


def make_results_dir(
    directory: str,
    dataset_name: str = 'example',
    metadata: dict | None = None,
    write_mpack: bool = True,
    write_gz: bool = True,
    write_csv: bool = True,
    write_metadata: bool = True,
) -> Path:
    """
    Build a synthetic experiment results folder for testing.

    :param directory: The folder to populate.
    :param dataset_name: Base name for the dataset files.
    :param metadata: Metadata dict to write. A valid two-target default is used
        when ``None``.
    :param write_mpack: Whether to write the ``.mpack`` dataset file.
    :param write_gz: Whether to write the compressed dataset file.
    :param write_csv: Whether to write the raw CSV export.
    :param write_metadata: Whether to write ``metadata.yml``.

    :returns: The populated folder path.
    """
    path = Path(directory)

    if metadata is None:
        metadata = {
            'compounds': 1234,
            'targets': 2,
            'target_type': ['regression'],
            'description': 'A synthetic dataset used for testing.',
            'verbose': 'Example Dataset',
            'raw': ['csv'],
            'tags': ['Molecules', 'SMILES'],
            'sources': ['https://doi.org/10.0000/example'],
            'target_descriptions': {
                '0': 'alpha - the first target (eV)',
                '1': 'beta - the second target (eV)',
            },
        }

    if write_mpack:
        (path / f'{dataset_name}.mpack').write_bytes(b'\x00' * 32)
    if write_gz:
        (path / f'{dataset_name}.mpack.gz').write_bytes(b'\x00' * 16)
    if write_csv:
        (path / f'{dataset_name}.csv').write_text('smiles,alpha,beta\nCCO,1.0,2.0\n')
    if write_metadata:
        with (path / 'metadata.yml').open('w') as file:
            yaml.dump(metadata, file)

    return path


class TestNormalizeTargetDescriptionKeys:
    """
    Unit tests for the key normalisation helper.
    """

    def test_string_and_int_keys_both_normalize(self):
        """
        Both key conventions occur in the existing metadata, so both have to be
        accepted and reduced to integers.
        """
        assert normalize_target_description_keys({'0': 'a', '1': 'b'}) == {
            0: 'a',
            1: 'b',
        }
        assert normalize_target_description_keys({0: 'a', 1: 'b'}) == {0: 'a', 1: 'b'}

    def test_unparseable_keys_are_dropped(self):
        """
        A key that is neither an int nor an int-like string cannot identify a target
        index and is discarded, which then surfaces as a missing index.
        """
        result = normalize_target_description_keys({'0': 'a', 'not_a_number': 'b'})
        assert result == {0: 'a'}


class TestVerifyResults:
    """
    Unit tests for :func:`verify_results`.
    """

    def test_valid_results_pass(self):
        """
        A complete results folder should verify without errors or warnings and report
        the dataset name, compound count and target count.
        """
        with tempfile.TemporaryDirectory() as directory:
            make_results_dir(directory)
            report = verify_results(directory)

            assert isinstance(report, VerificationReport)
            assert report.ok
            assert report.errors == []
            assert report.warnings == []
            assert report.dataset_name == 'example'
            assert report.compounds == 1234
            assert report.targets == 2

    def test_missing_directory_is_an_error(self):
        """
        Pointing the check at a path that does not exist should fail rather than raise.
        """
        report = verify_results('/nonexistent/path/that/does/not/exist')
        assert not report.ok
        assert any('does not exist' in issue.message for issue in report.errors)

    def test_missing_mpack_is_an_error(self):
        """
        A missing message pack file means the run never reached the save step, which is
        the single most important thing this check catches.
        """
        with tempfile.TemporaryDirectory() as directory:
            make_results_dir(directory, write_mpack=True)
            (Path(directory) / 'example.mpack').unlink()

            report = verify_results(directory)
            assert not report.ok
            assert any('.mpack' in issue.message for issue in report.errors)

    def test_empty_mpack_is_an_error(self):
        """
        A zero-byte dataset file is as unusable as a missing one.
        """
        with tempfile.TemporaryDirectory() as directory:
            make_results_dir(directory)
            (Path(directory) / 'example.mpack').write_bytes(b'')

            report = verify_results(directory)
            assert not report.ok
            assert any('empty' in issue.message for issue in report.errors)

    def test_missing_metadata_is_an_error(self):
        """
        Without metadata.yml the dataset cannot be described to a user.
        """
        with tempfile.TemporaryDirectory() as directory:
            make_results_dir(directory, write_metadata=False)

            report = verify_results(directory)
            assert not report.ok
            assert any('metadata.yml' in issue.message for issue in report.errors)

    def test_malformed_metadata_is_an_error(self):
        """
        Metadata that is not valid YAML should be reported cleanly rather than raising
        out of the verification call.
        """
        with tempfile.TemporaryDirectory() as directory:
            make_results_dir(directory)
            (Path(directory) / 'metadata.yml').write_text('{[not: valid yaml')

            report = verify_results(directory)
            assert not report.ok

    def test_zero_compounds_is_an_error(self):
        """
        A run where every molecule failed conversion still writes files and exits
        cleanly, so the compound count is the only thing that catches it.
        """
        with tempfile.TemporaryDirectory() as directory:
            make_results_dir(
                directory,
                metadata={
                    'compounds': 0,
                    'targets': 1,
                    'description': 'x',
                    'tags': ['Molecules'],
                    'sources': ['https://example.org'],
                    'verbose': 'X',
                    'target_type': ['regression'],
                    'target_descriptions': {'0': 'a - the target'},
                },
            )

            report = verify_results(directory)
            assert not report.ok
            assert any('0 compounds' in issue.message for issue in report.errors)

    @pytest.mark.parametrize(
        'descriptions,targets,expect_ok',
        [
            ({'0': 'a', '1': 'b'}, 2, True),
            ({0: 'a', 1: 'b'}, 2, True),  # int keys are equally acceptable
            ({'0': 'a'}, 2, False),  # index 1 missing
            ({'1': 'b'}, 2, False),  # index 0 missing
            ({'0': 'a', '2': 'c'}, 2, False),  # index 1 missing, 2 out of range
            ({'0': 'a', '1': ''}, 2, False),  # empty description
        ],
    )
    def test_target_description_coverage(self, descriptions, targets, expect_ok):
        """
        Every target index from 0 to targets-1 needs a non-empty description. This is
        the check that most often catches an otherwise plausible-looking run.
        """
        with tempfile.TemporaryDirectory() as directory:
            make_results_dir(
                directory,
                metadata={
                    'compounds': 100,
                    'targets': targets,
                    'description': 'x',
                    'tags': ['Molecules'],
                    'sources': ['https://example.org'],
                    'verbose': 'X',
                    'target_type': ['regression'],
                    'target_descriptions': descriptions,
                },
            )

            report = verify_results(directory)
            assert report.ok is expect_ok

    def test_missing_descriptions_for_single_target_is_only_a_warning(self):
        """
        A single-target dataset is still interpretable without target_descriptions, so
        their absence should not block the workflow.
        """
        with tempfile.TemporaryDirectory() as directory:
            make_results_dir(
                directory,
                metadata={
                    'compounds': 100,
                    'targets': 1,
                    'description': 'x',
                    'tags': ['Molecules'],
                    'sources': ['https://example.org'],
                    'verbose': 'X',
                    'target_type': ['regression'],
                },
            )

            report = verify_results(directory)
            assert report.ok
            assert any(issue.level == WARNING for issue in report.warnings)

    def test_low_compound_count_warns_about_testing_flag(self):
        """
        A compound count at or below the 50-element testing cap is the signature of a
        run left in testing mode, which is easy to mistake for a finished dataset.
        """
        with tempfile.TemporaryDirectory() as directory:
            make_results_dir(
                directory,
                metadata={
                    'compounds': 50,
                    'targets': 1,
                    'description': 'x',
                    'tags': ['Molecules'],
                    'sources': ['https://example.org'],
                    'verbose': 'X',
                    'target_type': ['regression'],
                    'target_descriptions': {'0': 'a - the target'},
                },
            )

            report = verify_results(directory)
            assert report.ok
            assert any('__TESTING__' in issue.message for issue in report.warnings)

    def test_missing_descriptive_fields_are_warnings_only(self):
        """
        A dataset without tags or sources is worse but still usable, so these must not
        block the workflow the way a missing dataset file does.
        """
        with tempfile.TemporaryDirectory() as directory:
            make_results_dir(
                directory,
                metadata={
                    'compounds': 100,
                    'targets': 1,
                    'target_descriptions': {'0': 'a - the target'},
                },
            )

            report = verify_results(directory)
            assert report.ok
            messages = ' '.join(issue.message for issue in report.warnings)
            assert 'tags' in messages
            assert 'sources' in messages
            assert 'description' in messages

    def test_missing_compressed_file_is_a_warning(self):
        """
        A missing .mpack.gz usually just means COMPRESS was disabled.
        """
        with tempfile.TemporaryDirectory() as directory:
            make_results_dir(directory, write_gz=False)

            report = verify_results(directory)
            assert report.ok
            assert any('.mpack.gz' in issue.message for issue in report.warnings)

    def test_report_partitions_issues_by_level(self):
        """
        The report should separate blocking from non-blocking issues so that a caller
        can act on errors while still surfacing warnings.
        """
        with tempfile.TemporaryDirectory() as directory:
            make_results_dir(
                directory,
                write_gz=False,
                metadata={'compounds': 0, 'targets': 1},
            )

            report = verify_results(directory)
            assert not report.ok
            assert all(issue.level == ERROR for issue in report.errors)
            assert all(issue.level == WARNING for issue in report.warnings)
            assert len(report.issues) == len(report.errors) + len(report.warnings)
