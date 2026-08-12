"""
Tests that the two artifacts of one run are checked against each other.

The message pack and the CSV are produced from the same dataset by different code paths
and used to disagree completely: graphs came off a multiprocessing queue in completion
order while the CSV was written in source order. On a real run all 5432 rows were
mismatched, with identical contents. Nothing detected it, because both files existed and
both had the right length - the only checks there were.
"""

import csv as csv_module

import msgpack
import pytest

from chem_mat_data.agent.verification import (
    ARTIFACT_SAMPLE,
    VerificationReport,
    check_artifacts_agree,
)


def write_pair(tmp_path, mpack_smiles, csv_smiles):
    """
    Write a message pack and a CSV holding the given SMILES.

    :param tmp_path: Folder to write into.
    :param mpack_smiles: SMILES in message pack order.
    :param csv_smiles: SMILES in CSV order.

    :returns: Tuple of the two paths.
    """
    mpack_path = tmp_path / 'demo.mpack'
    graphs = [{'graph_repr': s, 'graph_labels': [1.0]} for s in mpack_smiles]
    mpack_path.write_bytes(msgpack.packb(graphs))

    csv_path = tmp_path / 'demo.csv'
    with open(csv_path, 'w', newline='') as file:
        writer = csv_module.writer(file)
        writer.writerow(['smiles', 'value'])
        for s in csv_smiles:
            writer.writerow([s, 1.0])

    return mpack_path, csv_path


def check(tmp_path, mpack_smiles, csv_smiles) -> VerificationReport:
    """Run the cross-artifact check and return the report."""
    mpack_path, csv_path = write_pair(tmp_path, mpack_smiles, csv_smiles)
    report = VerificationReport(results_dir=tmp_path)
    check_artifacts_agree(report, mpack_path, csv_path)
    return report


def test_matching_artifacts_pass(tmp_path):
    smiles = ['CCO', 'c1ccccc1', 'CC(=O)O']

    assert check(tmp_path, smiles, smiles).ok


def test_a_permutation_is_an_error(tmp_path):
    """The real defect: same contents, different order."""
    report = check(
        tmp_path,
        ['CC(C)CCC(CC)CC', 'CC(CC)(CC)C(O)=O', 'C[C@@H](N)C(C)(C)C'],
        ['C=O', 'OC(=O)c1cc(Cl)ccc1Cl', 'CN(C)CN(C)C'],
    )

    assert not report.ok
    assert any('not in the same order' in i.message for i in report.errors)


def test_the_error_names_the_first_disagreeing_row(tmp_path):
    report = check(tmp_path, ['CCO', 'CCC', 'CCCC'], ['CCO', 'CCCC', 'CCC'])

    assert not report.ok
    message = report.errors[0].message
    assert 'row 1' in message
    assert "'CCC'" in message


def test_differing_lengths_are_an_error(tmp_path):
    """A smoke run used to write a full-length CSV beside a 50-graph mpack."""
    report = check(tmp_path, ['CCO', 'CCC'], ['CCO', 'CCC', 'CCCC', 'CCCCC'])

    assert not report.ok
    assert any('disagree about how large' in i.message for i in report.errors)


def test_only_a_prefix_is_compared(tmp_path):
    """
    Beyond the sample the rows are not read.

    The failure this catches is a whole-file permutation, and reading every row of a
    multi-gigabyte dataset to confirm it would be its own problem.
    """
    size = ARTIFACT_SAMPLE * 3
    mpack_smiles = [f'C{i}' for i in range(size)]
    csv_smiles = list(mpack_smiles)
    csv_smiles[-1] = 'DIFFERENT'

    assert check(tmp_path, mpack_smiles, csv_smiles).ok


def test_a_difference_inside_the_sample_is_caught(tmp_path):
    size = ARTIFACT_SAMPLE * 3
    mpack_smiles = [f'C{i}' for i in range(size)]
    csv_smiles = list(mpack_smiles)
    csv_smiles[0] = 'DIFFERENT'

    assert not check(tmp_path, mpack_smiles, csv_smiles).ok


# -- degrading rather than crashing --

def test_an_unreadable_mpack_is_silent(tmp_path):
    """verify_results reports a broken mpack; this check must not double up."""
    mpack_path = tmp_path / 'demo.mpack'
    mpack_path.write_bytes(b'not a message pack')
    csv_path = tmp_path / 'demo.csv'
    csv_path.write_text('smiles,value\nCCO,1.0\n')

    report = VerificationReport(results_dir=tmp_path)
    check_artifacts_agree(report, mpack_path, csv_path)

    assert report.ok


def test_a_csv_without_a_smiles_column_is_silent(tmp_path):
    mpack_path, _ = write_pair(tmp_path, ['CCO'], ['CCO'])
    csv_path = tmp_path / 'other.csv'
    csv_path.write_text('name,value\nethanol,1.0\n')

    report = VerificationReport(results_dir=tmp_path)
    check_artifacts_agree(report, mpack_path, csv_path)

    assert report.ok


def test_an_empty_csv_is_silent(tmp_path):
    mpack_path, _ = write_pair(tmp_path, ['CCO'], ['CCO'])
    csv_path = tmp_path / 'empty.csv'
    csv_path.write_text('')

    report = VerificationReport(results_dir=tmp_path)
    check_artifacts_agree(report, mpack_path, csv_path)

    assert report.ok
