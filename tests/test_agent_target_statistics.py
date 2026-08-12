"""
Tests for observed target statistics and the plausibility checks built on them.

Units are the highest-consequence claim the pipeline makes and had no check anywhere:
`elanos_bp` shipped for months describing a boiling point "in Kelvin" whose minimum is
-88.6 degrees Celsius.

Comparing a declared range against an observed one would not have caught that. An agent
confidently wrong about units writes the same wrong string on both sides, so the two
agree. What catches it is physics: an absolute temperature scale cannot go negative.
That asymmetry is why the unit check runs unconditionally while the range check is
optional.
"""

import math

import msgpack
import numpy as np
import pytest

from chem_mat_data.agent.verification import (
    VerificationReport,
    stream_target_statistics,
)
from chem_mat_data.agent.gates import check_target_plausibility


def write_mpack(path, label_rows):
    """
    Write a minimal dataset message pack carrying the given label vectors.

    :param path: File to write.
    :param label_rows: One list of target values per graph.

    :returns: The path written to.
    """
    graphs = [{'graph_labels': list(row)} for row in label_rows]
    path.write_bytes(msgpack.packb(graphs))
    return path


def stats_for(tmp_path, label_rows):
    """Stream statistics for the given label vectors."""
    return stream_target_statistics(write_mpack(tmp_path / 'd.mpack', label_rows))


def check(statistics, draft) -> VerificationReport:
    """Run the plausibility checks and return the report."""
    report = VerificationReport(results_dir='.')
    check_target_plausibility(report, statistics, draft)
    return report


# -- streaming --

def test_statistics_match_the_values(tmp_path):
    statistics = stats_for(tmp_path, [[-88.6], [0.0], [548.0], [100.0]])

    assert len(statistics) == 1
    stat = statistics[0]
    assert stat.count == 4
    assert stat.minimum == pytest.approx(-88.6)
    assert stat.maximum == pytest.approx(548.0)
    assert stat.mean == pytest.approx((-88.6 + 0.0 + 548.0 + 100.0) / 4)
    assert stat.unique == 4


def test_multiple_targets_are_tracked_separately(tmp_path):
    statistics = stats_for(tmp_path, [[1.0, 10.0], [3.0, 30.0]])

    assert [s.index for s in statistics] == [0, 1]
    assert statistics[0].maximum == pytest.approx(3.0)
    assert statistics[1].maximum == pytest.approx(30.0)


def test_non_finite_values_are_counted_not_silently_dropped(tmp_path):
    statistics = stats_for(tmp_path, [[1.0], [float('nan')], [3.0], [None]])

    assert statistics[0].count == 2
    assert statistics[0].missing == 2
    assert statistics[0].maximum == pytest.approx(3.0)


def test_distinct_counting_stops_at_the_cap_and_says_so(tmp_path):
    """
    A capped count reported as an exact one is a wrong number.

    The count exists only to separate a continuous target from a constant or a few
    classes, so it is bounded - but the bound has to be visible, or the statistics
    stop being trustworthy at a glance.
    """
    from chem_mat_data.agent.verification import UNIQUE_CAP

    statistics = stats_for(tmp_path, [[float(i)] for i in range(UNIQUE_CAP * 3)])

    assert statistics[0].unique == UNIQUE_CAP
    assert statistics[0].unique_capped is True
    assert statistics[0].count == UNIQUE_CAP * 3


def test_a_small_number_of_distinct_values_is_not_capped(tmp_path):
    statistics = stats_for(tmp_path, [[1.0], [2.0], [1.0], [3.0]])

    assert statistics[0].unique == 3
    assert statistics[0].unique_capped is False


def test_an_unreadable_file_yields_no_statistics(tmp_path):
    path = tmp_path / 'broken.mpack'
    path.write_bytes(b'not a message pack')

    assert stream_target_statistics(path) == []


def test_a_missing_file_yields_no_statistics(tmp_path):
    assert stream_target_statistics(tmp_path / 'absent.mpack') == []


def test_numpy_extension_payloads_decode(tmp_path):
    """The datasets store arrays as a msgpack extension type; the local hook must decode it."""
    array = np.arange(4, dtype='float32')
    packed = msgpack.packb(
        {'data': array.tobytes(), 'dtype': 'float32', 'shape': [4]}
    )
    graphs = [{'graph_labels': [1.5], 'node_attributes': msgpack.ExtType(1, packed)}]
    path = tmp_path / 'ext.mpack'
    path.write_bytes(msgpack.packb(graphs))

    statistics = stream_target_statistics(path)

    assert statistics[0].minimum == pytest.approx(1.5)


# -- the unit check: the one that catches a mislabelled unit --

def test_kelvin_with_negative_values_is_an_error(tmp_path):
    """The real elanos_bp defect."""
    statistics = stats_for(tmp_path, [[-88.6], [548.0]])
    draft = {'target_descriptions': {'0': 'BP - Boiling Point (BP) in Kelvin'}}

    report = check(statistics, draft)

    assert not report.ok
    assert any('kelvin' in i.message.lower() for i in report.errors)


def test_celsius_with_negative_values_is_fine(tmp_path):
    """The corrected description must pass, or the check is useless."""
    statistics = stats_for(tmp_path, [[-88.6], [548.0]])
    draft = {
        'target_descriptions': {
            '0': 'BP - Experimental boiling point in degrees Celsius (°C)'
        }
    }

    assert check(statistics, draft).ok


def test_kelvin_with_positive_values_is_fine(tmp_path):
    statistics = stats_for(tmp_path, [[184.5], [821.0]])
    draft = {'target_descriptions': {'0': 'BP - boiling point in Kelvin'}}

    assert check(statistics, draft).ok


def test_a_log_transformed_target_is_out_of_scope(tmp_path):
    """
    Log-transformed values are routinely negative whatever the underlying unit is.

    Without this guard the check would fire on half the collection.
    """
    statistics = stats_for(tmp_path, [[-3.2], [1.1]])
    draft = {'target_descriptions': {'0': 'logK - log of the equilibrium constant (K)'}}

    assert check(statistics, draft).ok


def test_negative_molecular_weight_is_an_error(tmp_path):
    statistics = stats_for(tmp_path, [[-5.0], [180.0]])
    draft = {'target_descriptions': {'0': 'MW - molecular weight in g/mol'}}

    report = check(statistics, draft)

    assert not report.ok
    assert any('molecular mass' in i.message for i in report.errors)


def test_no_description_means_no_unit_check(tmp_path):
    statistics = stats_for(tmp_path, [[-10.0], [10.0]])

    assert check(statistics, {}).ok


# -- constant and missing columns --

def test_a_constant_target_is_an_error(tmp_path):
    statistics = stats_for(tmp_path, [[7.0], [7.0], [7.0]])

    report = check(statistics, {})

    assert not report.ok
    assert any('every row' in i.message for i in report.errors)


def test_missing_values_warn(tmp_path):
    statistics = stats_for(tmp_path, [[1.0], [float('nan')], [3.0]])

    report = check(statistics, {})

    assert report.ok
    assert any('missing' in i.message for i in report.warnings)


# -- the optional declared range --

def test_values_outside_the_declared_range_are_an_error(tmp_path):
    """
    The unit-conversion case: the paper states Celsius, the data is in Kelvin.

    Note what this does *not* catch: a descriptor column whose 0..1 range sits wholly
    inside the published -88.6..548 is within range, and is caught by the
    narrow-subset warning below rather than here.
    """
    statistics = stats_for(tmp_path, [[184.55], [821.15]])
    draft = {'target_ranges': {'0': {'min': -88.6, 'max': 548.0}}}

    report = check(statistics, draft)

    assert not report.ok
    assert any('states -88.6..548' in i.message for i in report.errors)


def test_a_descriptor_column_inside_the_published_range_only_warns(tmp_path):
    """
    Documents a real limit of the range check.

    D001 spans 0..1 and BP spans -88.6..548, so picking the descriptor column produces
    values that are technically within the published range. Only the narrow-subset
    warning fires, not an error.
    """
    statistics = stats_for(tmp_path, [[0.0], [1.0]])
    draft = {'target_ranges': {'0': {'min': -88.6, 'max': 548.0}}}

    report = check(statistics, draft)

    assert report.ok
    assert any('under a tenth' in i.message for i in report.warnings)


def test_values_inside_the_declared_range_pass(tmp_path):
    statistics = stats_for(tmp_path, [[-88.6], [548.0]])
    draft = {'target_ranges': {'0': {'min': -88.6, 'max': 548.0}}}

    assert check(statistics, draft).ok


def test_a_narrow_subset_of_the_declared_range_warns(tmp_path):
    """Suggests one split rather than the whole dataset."""
    statistics = stats_for(tmp_path, [[10.0], [20.0]])
    draft = {'target_ranges': {'0': {'min': 0.0, 'max': 500.0}}}

    report = check(statistics, draft)

    assert report.ok
    assert any('under a tenth' in i.message for i in report.warnings)


def test_ranges_are_optional(tmp_path):
    """No declared range must not be a finding - papers often state none."""
    statistics = stats_for(tmp_path, [[-88.6], [548.0]])

    assert check(statistics, {'target_descriptions': {'0': 'BP - in Celsius'}}).ok


def test_a_malformed_range_is_ignored_rather_than_crashing(tmp_path):
    statistics = stats_for(tmp_path, [[1.0], [2.0]])
    draft = {'target_ranges': {'0': {'min': 'low', 'max': None}}}

    assert check(statistics, draft).ok
