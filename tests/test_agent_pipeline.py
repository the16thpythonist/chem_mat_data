from pathlib import Path

import yaml
import pytest

from chem_mat_data.agent import manifest as manifest_module
from chem_mat_data.agent.manifest import (
    BLOCKED,
    PASSED,
    PENDING,
    STAGES,
    STALE,
    Manifest,
)
from chem_mat_data.agent import gates as gates_module
from chem_mat_data.agent.gates import (
    gate_execute,
    gate_license,
    gate_metadata,
    gate_prepare,
    gate_script,
    gate_verification,
    is_permissive,
    normalize_license,
    run_gate,
)


@pytest.fixture
def work_root(tmp_path, monkeypatch):
    """
    Redirect the pipeline working directory into a temporary folder.

    :param tmp_path: pytest temporary directory.
    :param monkeypatch: pytest monkeypatch fixture.

    :returns: The temporary working root.
    """
    monkeypatch.setattr(manifest_module, 'WORK_DIR', tmp_path / '.dataset_work')
    return tmp_path / '.dataset_work'


def valid_draft() -> dict:
    """Return a metadata draft that passes the metadata gate."""
    return {
        'dataset_name': 'demo_ds',
        'dataset_type': 'regression',
        'verbose': 'Demo Dataset',
        'description': (
            'A synthetic demonstration dataset of organic compounds with '
            'experimentally measured melting points in degrees Celsius, used to '
            'exercise the extraction pipeline from end to end.'
        ),
        'tags': ['Molecules', 'SMILES', 'Melting Point'],
        'sources': ['https://doi.org/10.0000/demo'],
        'target_count': 1,
        'target_descriptions': {'0': 'mpC - Experimental melting point (Celsius)'},
    }


class TestManifest:
    """
    Unit tests for the pipeline manifest.
    """

    def test_create_starts_every_stage_pending(self, work_root):
        manifest = Manifest.create('demo', source_url='https://example.org')
        assert all(manifest.state(stage).status == PENDING for stage in STAGES)
        assert manifest.next_stage() == 'prepare'
        assert not manifest.complete

    def test_round_trip_through_disk(self, work_root):
        manifest = Manifest.create('demo', source_url='https://example.org')
        manifest.metadata_draft = valid_draft()
        manifest.input_rows = 5000
        manifest.mark('prepare', PASSED)
        manifest.save()

        loaded = Manifest.load('demo')
        assert loaded.slug == 'demo'
        assert loaded.source_url == 'https://example.org'
        assert loaded.input_rows == 5000
        assert loaded.metadata_draft == valid_draft()
        assert loaded.state('prepare').status == PASSED
        assert loaded.next_stage() == 'metadata'

    def test_load_missing_manifest_raises(self, work_root):
        with pytest.raises(FileNotFoundError):
            Manifest.load('never-created')

    def test_next_stage_is_none_when_complete(self, work_root):
        manifest = Manifest.create('demo')
        for stage in STAGES:
            manifest.mark(stage, PASSED)
        assert manifest.next_stage() is None
        assert manifest.complete

    def test_rerunning_a_stage_invalidates_later_passed_stages(self, work_root):
        """
        Re-deciding the metadata after the script was written means the script no
        longer reflects what was approved, and that has to become visible rather than
        being silently accepted.
        """
        manifest = Manifest.create('demo')
        for stage in STAGES:
            manifest.mark(stage, PASSED)

        manifest.mark('metadata', PASSED)

        assert manifest.state('prepare').status == PASSED
        assert manifest.state('metadata').status == PASSED
        for stage in ('license', 'script', 'execute', 'verification'):
            assert manifest.state(stage).status == STALE
            assert 'metadata' in manifest.state(stage).notes

    def test_blocking_a_stage_does_not_invalidate_later_stages(self, work_root):
        manifest = Manifest.create('demo')
        manifest.mark('script', PASSED)
        manifest.mark('prepare', BLOCKED, blocker='paywall')

        assert manifest.state('script').status == PASSED
        assert manifest.blockers() == {'prepare': 'paywall'}

    def test_unmet_prerequisites(self, work_root):
        manifest = Manifest.create('demo')
        assert manifest.unmet_prerequisites('prepare') == []
        assert manifest.unmet_prerequisites('script') == [
            'prepare',
            'metadata',
            'license',
        ]

        manifest.mark('prepare', PASSED)
        manifest.mark('metadata', PASSED)
        assert manifest.unmet_prerequisites('script') == ['license']

    def test_unknown_stage_rejected(self, work_root):
        manifest = Manifest.create('demo')
        with pytest.raises(ValueError):
            manifest.mark('nonsense', PASSED)
        with pytest.raises(ValueError):
            manifest.state('nonsense')

    def test_run_gate_refuses_when_prerequisites_unmet(self, work_root):
        """
        Checking a script against metadata that was never agreed produces a
        meaningless verdict, so the gate must refuse rather than evaluate.
        """
        manifest = Manifest.create('demo')
        report = run_gate('script', manifest)
        assert not report.ok
        assert 'have not passed' in report.errors[0].message

    def test_run_gate_rejects_unknown_stage(self, work_root):
        manifest = Manifest.create('demo')
        with pytest.raises(ValueError):
            run_gate('nonsense', manifest)


class TestPrepareGate:
    """
    Unit tests for the prepare gate.
    """

    def build(
        self,
        work_root,
        csv: bytes = b'smiles,mpC\nCCO,-114\n',
        recipe=True,
        readme=True,
    ) -> Manifest:
        manifest = Manifest.create('demo', source_url='https://example.org')
        downloads = manifest.downloads_dir
        downloads.mkdir(parents=True, exist_ok=True)
        if csv is not None:
            (downloads / 'data.csv').write_bytes(csv)
        if recipe:
            (downloads / 'download_recipe.md').write_text(
                '# Recipe\n\nhttps://example.org/data.csv - plain CSV, 2 MB\n'
            )
        if readme:
            (downloads / 'README.md').write_text('mpC is melting point in Celsius.\n')
        return manifest

    def test_complete_prepare_passes(self, work_root):
        report = gate_prepare(self.build(work_root))
        assert report.ok
        assert report.warnings == []

    def test_html_saved_as_data_is_an_error(self, work_root):
        """
        Saving a repository landing page instead of the file behind it is the most
        common way prepare fails while appearing to succeed.
        """
        manifest = self.build(
            work_root, csv=b'<!DOCTYPE html>\n<html><body>Zenodo</body></html>'
        )
        report = gate_prepare(manifest)
        assert not report.ok
        assert 'HTML page' in report.errors[0].message

    def test_html_with_leading_whitespace_still_detected(self, work_root):
        manifest = self.build(work_root, csv=b'\n\n   <html><body>x</body></html>')
        report = gate_prepare(manifest)
        assert not report.ok

    def test_empty_data_file_is_an_error(self, work_root):
        report = gate_prepare(self.build(work_root, csv=b''))
        assert not report.ok
        assert any('empty' in issue.message for issue in report.errors)

    def test_missing_recipe_is_an_error(self, work_root):
        report = gate_prepare(self.build(work_root, recipe=False))
        assert not report.ok
        assert any('download_recipe' in issue.message for issue in report.errors)

    def test_recipe_without_url_is_an_error(self, work_root):
        manifest = self.build(work_root)
        (manifest.downloads_dir / 'download_recipe.md').write_text('no links here')
        report = gate_prepare(manifest)
        assert not report.ok
        assert any('no URL' in issue.message for issue in report.errors)

    def test_only_documentation_files_is_an_error(self, work_root):
        report = gate_prepare(self.build(work_root, csv=None))
        assert not report.ok
        assert any('no data files' in issue.message for issue in report.errors)

    def test_missing_readme_is_only_a_warning(self, work_root):
        report = gate_prepare(self.build(work_root, readme=False))
        assert report.ok
        assert any('README' in issue.message for issue in report.warnings)


class TestMetadataGate:
    """
    Unit tests for the metadata gate.
    """

    def test_valid_draft_passes(self, work_root):
        manifest = Manifest.create('demo')
        manifest.metadata_draft = valid_draft()
        report = gate_metadata(manifest)
        assert report.ok
        assert report.warnings == []

    def test_empty_draft_is_an_error(self, work_root):
        manifest = Manifest.create('demo')
        report = gate_metadata(manifest)
        assert not report.ok

    @pytest.mark.parametrize(
        'field', ['dataset_name', 'verbose', 'tags', 'sources', 'target_descriptions']
    )
    def test_missing_required_field_is_an_error(self, work_root, field):
        manifest = Manifest.create('demo')
        draft = valid_draft()
        del draft[field]
        manifest.metadata_draft = draft
        report = gate_metadata(manifest)
        assert not report.ok

    @pytest.mark.parametrize(
        'name,ok',
        [
            ('demo_ds', True),
            ('esol', True),
            ('Demo-DS', False),
            ('demo ds', False),
            ('DemoDS', False),
            ('1demo', True),
        ],
    )
    def test_dataset_name_must_be_snake_case(self, work_root, name, ok):
        manifest = Manifest.create('demo')
        draft = valid_draft()
        draft['dataset_name'] = name
        manifest.metadata_draft = draft
        report = gate_metadata(manifest)
        assert report.ok is ok

    def test_invalid_dataset_type_is_an_error(self, work_root):
        manifest = Manifest.create('demo')
        draft = valid_draft()
        draft['dataset_type'] = 'regr'
        manifest.metadata_draft = draft
        assert not gate_metadata(manifest).ok

    def test_target_descriptions_must_cover_target_count(self, work_root):
        manifest = Manifest.create('demo')
        draft = valid_draft()
        draft['target_count'] = 3
        draft['target_descriptions'] = {'0': 'a - x', '2': 'c - z'}
        manifest.metadata_draft = draft
        report = gate_metadata(manifest)
        assert not report.ok
        assert any('missing index' in issue.message for issue in report.errors)

    def test_short_description_is_a_warning(self, work_root):
        manifest = Manifest.create('demo')
        draft = valid_draft()
        draft['description'] = 'Too short.'
        manifest.metadata_draft = draft
        report = gate_metadata(manifest)
        assert report.ok
        assert any('description' in issue.message for issue in report.warnings)

    def test_description_without_separator_is_a_warning(self, work_root):
        manifest = Manifest.create('demo')
        draft = valid_draft()
        draft['target_descriptions'] = {'0': 'mpC'}
        manifest.metadata_draft = draft
        report = gate_metadata(manifest)
        assert report.ok
        assert any('<column>' in issue.message for issue in report.warnings)


class TestLicenseGate:
    """
    Unit tests for the license gate and its policy.
    """

    @pytest.mark.parametrize(
        'identifier,expected',
        [
            ('CC-BY-4.0', True),
            ('CC BY 4.0', True),
            ('cc_by_4.0', True),
            ('CC0', True),
            ('CC0-1.0', True),
            ('MIT', True),
            ('Apache-2.0', True),
            ('BSD-3-Clause', True),
            ('public domain', True),
            ('Unlicense', True),
            ('CC-BY-NC-4.0', False),
            ('CC-BY-SA-4.0', False),
            ('CC-BY-NC-SA-4.0', False),
            ('GPL-3.0', False),
            ('ODbL-1.0', False),
            ('proprietary', False),
            ('', False),
            (None, False),
        ],
    )
    def test_permissive_policy(self, identifier, expected):
        """
        ShareAlike and NonCommercial variants must not be matched by the CC-BY prefix.
        """
        assert is_permissive(identifier) is expected

    def test_normalize_license(self):
        assert normalize_license('CC BY 4.0') == 'cc-by-4.0'
        assert normalize_license('  CC_BY_4.0  ') == 'cc-by-4.0'

    def test_permissive_with_evidence_passes(self, work_root):
        manifest = Manifest.create('demo')
        manifest.license = {
            'identifier': 'CC-BY-4.0',
            'evidence_url': 'https://example.org/license',
            'redistribution': 'allowed',
            'attribution': 'Cite the original paper.',
            # A complete finding also states provenance: see TestLicenseProvenance.
            'upstream_database': None,
        }
        assert gate_license(manifest).ok

    def test_missing_evidence_is_an_error(self, work_root):
        manifest = Manifest.create('demo')
        manifest.license = {'identifier': 'MIT'}
        report = gate_license(manifest)
        assert not report.ok
        assert any('evidence_url' in issue.message for issue in report.errors)

    def test_no_license_found_escalates(self, work_root):
        manifest = Manifest.create('demo')
        manifest.license = {'evidence_url': 'https://example.org'}
        report = gate_license(manifest)
        assert not report.ok
        assert any('no license identified' in issue.message for issue in report.errors)

    def test_noncommercial_escalates(self, work_root):
        manifest = Manifest.create('demo')
        manifest.license = {
            'identifier': 'CC-BY-NC-4.0',
            'evidence_url': 'https://example.org',
        }
        assert not gate_license(manifest).ok

    def test_human_override_can_approve(self, work_root):
        manifest = Manifest.create('demo')
        manifest.license = {
            'identifier': 'CC-BY-NC-4.0',
            'evidence_url': 'https://example.org',
            'human_override': {
                'decision': 'approved',
                'reason': 'author granted written permission on 2026-08-01',
            },
        }
        report = gate_license(manifest)
        assert report.ok
        assert any('override' in issue.message for issue in report.warnings)

    def test_human_override_can_reject(self, work_root):
        manifest = Manifest.create('demo')
        manifest.license = {
            'identifier': 'CC-BY-4.0',
            'evidence_url': 'https://example.org',
            'human_override': {'decision': 'rejected', 'reason': 'not our data'},
        }
        assert not gate_license(manifest).ok

    def test_approval_requires_a_reason(self, work_root):
        manifest = Manifest.create('demo')
        manifest.license = {
            'identifier': 'CC-BY-NC-4.0',
            'evidence_url': 'https://example.org',
            'human_override': {'decision': 'approved'},
        }
        assert not gate_license(manifest).ok

    def test_redistribution_denied_is_an_error(self, work_root):
        manifest = Manifest.create('demo')
        manifest.license = {
            'identifier': 'CC-BY-4.0',
            'evidence_url': 'https://example.org',
            'redistribution': 'prohibited',
        }
        assert not gate_license(manifest).ok


class TestLicenseProvenance:
    """
    Unit tests for the upstream-provenance half of the license gate.

    A permissive license on the publication says nothing about the terms of a database
    the publication merely aggregated, and that is where the binding restriction
    usually lives. These cases are drawn from a real miss: a CC-BY-4.0 paper
    republishing data from a database that forbids systematic redistribution.
    """

    def build(self, source_url='https://example.org/paper', **extra) -> Manifest:
        """Build a manifest whose license is permissive apart from the given fields."""
        manifest = Manifest.create('demo', source_url=source_url)
        manifest.license = {
            'identifier': 'CC-BY-4.0',
            'evidence_url': 'https://example.org/record',
            'redistribution': 'allowed',
            'attribution': 'cite the paper',
            **extra,
        }
        return manifest

    def test_omitting_upstream_database_is_an_error(self, work_root):
        """
        The question must be answered explicitly. Silence is the failure mode that let
        a restrictive upstream database through unnoticed.
        """
        report = gate_license(self.build())
        assert not report.ok
        assert any('upstream_database' in issue.message for issue in report.errors)

    @pytest.mark.parametrize('value', [None, 'none', 'null', 'N/A'])
    def test_explicit_no_upstream_passes(self, work_root, value):
        assert gate_license(self.build(upstream_database=value)).ok

    def test_named_upstream_without_license_is_an_error(self, work_root):
        report = gate_license(
            self.build(
                upstream_database='enviPath',
                upstream_terms_url='https://envipath.org/terms',
            )
        )
        assert not report.ok
        assert any('was not' in issue.message for issue in report.errors)

    def test_named_upstream_with_restrictive_license_is_an_error(self, work_root):
        report = gate_license(
            self.build(
                upstream_database='enviPath',
                upstream_terms_url='https://envipath.org/terms',
                upstream_license='academic and non-commercial use only',
            )
        )
        assert not report.ok
        assert any('not permissive' in issue.message for issue in report.errors)

    def test_named_upstream_without_terms_url_is_an_error(self, work_root):
        report = gate_license(
            self.build(upstream_database='enviPath', upstream_license='CC-BY-4.0')
        )
        assert not report.ok
        assert any('upstream_terms_url' in issue.message for issue in report.errors)

    def test_named_upstream_with_permissive_license_passes(self, work_root):
        assert gate_license(
            self.build(
                upstream_database='PubChem',
                upstream_terms_url='https://pubchem.ncbi.nlm.nih.gov/docs/downloads',
                upstream_license='public domain',
            )
        ).ok

    def test_human_override_still_wins_over_upstream(self, work_root):
        assert gate_license(
            self.build(
                upstream_database='enviPath',
                upstream_license='non-commercial',
                human_override={
                    'decision': 'approved',
                    'reason': 'the aggregation was released CC-BY by the database maintainers',
                },
            )
        ).ok

    def test_preprint_versus_published_host_mismatch_warns(self, work_root):
        """
        The same work as a preprint and as a published article often carries different
        licenses; establishing the license on one while downloading the other is silent
        and easy.
        """
        report = gate_license(
            self.build(
                source_url='https://chemrxiv.org/engage/chemrxiv/article-details/abc',
                evidence_url='https://pmc.ncbi.nlm.nih.gov/articles/PMC10569042/',
                upstream_database=None,
            )
        )
        assert report.ok
        assert any('different licenses' in issue.message for issue in report.warnings)

    def test_matching_host_does_not_warn(self, work_root):
        report = gate_license(
            self.build(
                source_url='https://zenodo.org/records/123',
                evidence_url='https://www.zenodo.org/records/123',
                upstream_database=None,
            )
        )
        assert not any('different licenses' in i.message for i in report.warnings)

    def test_regression_the_case_the_gate_originally_missed(self, work_root):
        """
        End-to-end regression: a CC-BY-4.0 identifier evidenced on the published
        version, while the run started from a CC-BY-NC-ND preprint and the data
        actually derives from a non-commercial-only database. The gate used to pass
        this outright.
        """
        manifest = Manifest.create(
            'half_life',
            source_url='https://chemrxiv.org/engage/chemrxiv/article-details/64be471cb605c6803b425da6',
        )
        manifest.license = {
            'identifier': 'CC-BY-4.0',
            'evidence_url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10569042/',
            'redistribution': 'allowed',
            'attribution': 'Cite Hafner, Fenner & Scheidegger 2023.',
        }
        report = gate_license(manifest)
        assert not report.ok
        assert any('upstream_database' in issue.message for issue in report.errors)
        assert any('different licenses' in issue.message for issue in report.warnings)


def write_results(
    path: Path,
    name: str = 'demo_ds',
    compounds: int = 1000,
    targets: int = 1,
    extra: dict | None = None,
) -> Path:
    """
    Write a synthetic experiment results folder.

    :param path: Folder to populate.
    :param name: Dataset base name.
    :param compounds: Compound count to record.
    :param targets: Target count to record.
    :param extra: Additional metadata fields to merge in.

    :returns: The populated folder.
    """
    path.mkdir(parents=True, exist_ok=True)
    (path / f'{name}.mpack').write_bytes(b'\x00' * 32)
    (path / f'{name}.mpack.gz').write_bytes(b'\x00' * 16)
    (path / f'{name}.csv').write_text('smiles,mpC\nCCO,-114\n')

    metadata = {
        'compounds': compounds,
        'targets': targets,
        'target_type': ['regression'],
        'description': valid_draft()['description'],
        'verbose': valid_draft()['verbose'],
        'raw': ['csv'],
        'tags': valid_draft()['tags'],
        'sources': valid_draft()['sources'],
        'target_descriptions': valid_draft()['target_descriptions'],
    }
    metadata.update(extra or {})
    with (path / 'metadata.yml').open('w') as file:
        yaml.dump(metadata, file)
    return path


SCRIPT_TEMPLATE = """\
import os
import requests
import pandas as pd
from typing import Dict
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

DATASET_NAME: str = 'demo_ds'
DOWNLOAD_URL: str = 'https://example.org/data.csv'
__TESTING__ = {testing}

experiment = Experiment.extend(
    {template!r},
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals(),
)


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    path = os.path.join(e.path, 'raw.csv')
    if not os.path.exists(path):
        response = requests.get(DOWNLOAD_URL, timeout=300)
        with open(path, 'wb') as file:
            file.write(response.content)
    {extra}
    return {{}}


experiment.run_if_main()
"""


class TestScriptGate:
    """
    Unit tests for the script gate.
    """

    @pytest.fixture
    def scripts_dir(self, tmp_path, monkeypatch):
        """Redirect PROJECT_PATH so scripts resolve inside the temp tree."""
        scripts = tmp_path / 'chem_mat_data' / 'scripts'
        scripts.mkdir(parents=True)
        monkeypatch.setattr(gates_module, 'PROJECT_PATH', str(tmp_path))
        return scripts

    def build(
        self,
        work_root,
        scripts_dir,
        testing='False',
        template='create_graph_datasets.py',
        extra='',
        compounds=50,
        targets=1,
    ) -> Manifest:
        manifest = Manifest.create('demo')
        manifest.metadata_draft = valid_draft()
        for stage in ('prepare', 'metadata', 'license'):
            manifest.mark(stage, PASSED)

        script = scripts_dir / 'create_graph_datasets__demo_ds.py'
        script.write_text(
            SCRIPT_TEMPLATE.format(testing=testing, template=template, extra=extra)
        )
        manifest.script_path = 'chem_mat_data/scripts/create_graph_datasets__demo_ds.py'

        smoke = Path(scripts_dir) / 'results' / 'demo' / 'debug'
        write_results(smoke, compounds=compounds, targets=targets)
        manifest.smoke_results_dir = str(smoke)
        return manifest

    def test_valid_script_and_smoke_run_passes(self, work_root, scripts_dir):
        report = gate_script(self.build(work_root, scripts_dir))
        assert report.ok, [issue.message for issue in report.errors]

    def test_absolute_extend_path_is_an_error(self, work_root, scripts_dir):
        """
        An absolute base-template path works on the machine that generated it and
        breaks for everyone else.
        """
        manifest = self.build(
            work_root, scripts_dir, template='/abs/path/create_graph_datasets.py'
        )
        report = gate_script(manifest)
        assert not report.ok
        assert any('absolute path' in issue.message for issue in report.errors)

    def test_reaching_into_scratch_directory_is_an_error(self, work_root, scripts_dir):
        """
        The script must download its own data; reading .dataset_work makes it
        unreproducible for anyone else.
        """
        manifest = self.build(
            work_root,
            scripts_dir,
            extra="df = pd.read_csv('.dataset_work/demo/downloads/data.csv')",
        )
        report = gate_script(manifest)
        assert not report.ok
        assert any('.dataset_work' in issue.message for issue in report.errors)

    def test_target_count_mismatch_is_an_error(self, work_root, scripts_dir):
        manifest = self.build(work_root, scripts_dir, targets=4)
        report = gate_script(manifest)
        assert not report.ok
        assert any(
            'targets but the approved' in issue.message for issue in report.errors
        )

    def test_missing_smoke_run_is_an_error(self, work_root, scripts_dir):
        manifest = self.build(work_root, scripts_dir)
        manifest.smoke_results_dir = None
        report = gate_script(manifest)
        assert not report.ok

    def test_missing_script_is_an_error(self, work_root, scripts_dir):
        manifest = self.build(work_root, scripts_dir)
        manifest.script_path = 'chem_mat_data/scripts/does_not_exist.py'
        report = gate_script(manifest)
        assert not report.ok


class TestExecuteGate:
    """
    Unit tests for the execute gate, including the drop-rate policy.
    """

    @pytest.fixture
    def scripts_dir(self, tmp_path, monkeypatch):
        scripts = tmp_path / 'chem_mat_data' / 'scripts'
        scripts.mkdir(parents=True)
        monkeypatch.setattr(gates_module, 'PROJECT_PATH', str(tmp_path))
        return scripts

    def build(
        self,
        work_root,
        scripts_dir,
        tmp_path,
        testing='False',
        input_rows=1000,
        compounds=1000,
        max_drop=None,
    ) -> Manifest:
        manifest = Manifest.create('demo')
        manifest.metadata_draft = valid_draft()
        for stage in ('prepare', 'metadata', 'license', 'script'):
            manifest.mark(stage, PASSED)

        script = scripts_dir / 'create_graph_datasets__demo_ds.py'
        script.write_text(
            SCRIPT_TEMPLATE.format(
                testing=testing, template='create_graph_datasets.py', extra=''
            )
        )
        manifest.script_path = 'chem_mat_data/scripts/create_graph_datasets__demo_ds.py'
        manifest.input_rows = input_rows
        manifest.max_drop_rate = max_drop

        results = tmp_path / 'results' / 'debug'
        write_results(results, compounds=compounds)
        manifest.results_dir = str(results)
        return manifest

    def test_clean_full_run_passes(self, work_root, scripts_dir, tmp_path):
        report = gate_execute(self.build(work_root, scripts_dir, tmp_path))
        assert report.ok, [issue.message for issue in report.errors]

    def test_testing_flag_left_on_is_an_error(self, work_root, scripts_dir, tmp_path):
        manifest = self.build(work_root, scripts_dir, tmp_path, testing='True')
        report = gate_execute(manifest)
        assert not report.ok
        assert any('__TESTING__' in issue.message for issue in report.errors)

    @pytest.mark.parametrize(
        'input_rows,compounds,expect_ok,expect_warning',
        [
            (1000, 1000, True, False),  # no loss
            (1000, 985, True, False),  # 1.5% - below the 2% warn threshold
            (1000, 930, True, True),  # 7% - warns, matches tmQM's known loss
            (1000, 899, False, False),  # 10.1% - escalates
            (1000, 500, False, False),  # 50% - escalates
        ],
    )
    def test_drop_rate_policy(
        self,
        work_root,
        scripts_dir,
        tmp_path,
        input_rows,
        compounds,
        expect_ok,
        expect_warning,
    ):
        """
        Warn above 2%, escalate above 10%. Molecules that fail conversion are logged
        and skipped rather than raising, so this is the only thing that catches a run
        that quietly discarded most of its input.
        """
        manifest = self.build(
            work_root,
            scripts_dir,
            tmp_path,
            input_rows=input_rows,
            compounds=compounds,
        )
        report = gate_execute(manifest)
        assert report.ok is expect_ok
        if expect_warning and expect_ok:
            assert any('dropped' in issue.message for issue in report.warnings)

    def test_per_dataset_threshold_override(self, work_root, scripts_dir, tmp_path):
        """
        A known and explained loss should be allowed through with an override.
        """
        manifest = self.build(
            work_root,
            scripts_dir,
            tmp_path,
            input_rows=1000,
            compounds=700,
            max_drop=0.35,
        )
        manifest.drop_rate_reason = '7,700 rows have no SMILES in the source'
        report = gate_execute(manifest)
        assert report.ok

    def test_missing_input_rows_warns(self, work_root, scripts_dir, tmp_path):
        manifest = self.build(work_root, scripts_dir, tmp_path, input_rows=None)
        report = gate_execute(manifest)
        assert report.ok
        assert any('drop rate' in issue.message for issue in report.warnings)


class TestVerificationGate:
    """
    Unit tests for the final verification gate.
    """

    def build(self, tmp_path, written_extra=None) -> Manifest:
        manifest = Manifest.create('demo')
        manifest.metadata_draft = valid_draft()
        manifest.license = {
            'identifier': 'CC-BY-4.0',
            'evidence_url': 'https://example.org',
            'attribution': 'cite',
        }
        for stage in ('prepare', 'metadata', 'license', 'script', 'execute'):
            manifest.mark(stage, PASSED)

        results = tmp_path / 'results' / 'debug'
        write_results(results, extra=written_extra)
        manifest.results_dir = str(results)
        return manifest

    def test_complete_run_passes(self, work_root, tmp_path):
        report = gate_verification(self.build(tmp_path))
        assert report.ok, [issue.message for issue in report.errors]

    def test_incomplete_earlier_stage_is_an_error(self, work_root, tmp_path):
        manifest = self.build(tmp_path)
        manifest.mark('script', BLOCKED, blocker='smoke run failed')
        report = gate_verification(manifest)
        assert not report.ok
        assert any('script' in issue.message for issue in report.errors)

    def test_written_metadata_missing_approved_tags_is_an_error(
        self, work_root, tmp_path
    ):
        """
        The script carries its own METADATA literal, so what lands in metadata.yml can
        silently diverge from what the metadata stage approved.
        """
        manifest = self.build(tmp_path, written_extra={'tags': ['Molecules']})
        report = gate_verification(manifest)
        assert not report.ok
        assert any('absent from' in issue.message for issue in report.errors)

    def test_non_permissive_license_blocks_final_gate(self, work_root, tmp_path):
        manifest = self.build(tmp_path)
        manifest.license = {'identifier': 'CC-BY-NC-4.0', 'evidence_url': 'https://x'}
        report = gate_verification(manifest)
        assert not report.ok
        assert any('permissive' in issue.message for issue in report.errors)
