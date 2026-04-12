"""
Unit tests for :mod:`chem_mat_data.discovery.classifier`.

These tests monkey-patch the heavy torch / transformers / peft imports so
they can run in any environment, including CI boxes without a GPU or even
without those packages installed. A separate ``@pytest.mark.slow`` integration
test at the bottom of the file exercises the real model end-to-end but is
skipped unless ``CMDATA_DISCOVERY_SLOW=1`` is set.
"""
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.filterwarnings('ignore::pytest.PytestUnknownMarkWarning')

from chem_mat_data.discovery import classifier as classifier_mod
from chem_mat_data.discovery.classifier import (
    PROMPT_TEMPLATE,
    Classifier,
    ensure_classifier_model,
    format_prompt,
    is_ml_stack_available,
)


class TestFormatPrompt:
    """Tests for the prompt template used by the classifier."""

    def test_includes_title_and_abstract(self):
        prompt = format_prompt('My Paper', 'Cool dataset')
        assert 'My Paper' in prompt
        assert 'Cool dataset' in prompt

    def test_none_values_are_replaced_with_empty_string(self):
        # Should not crash on missing fields.
        prompt = format_prompt(None, None)
        assert isinstance(prompt, str)
        assert prompt

    def test_uses_template_constant(self):
        # Make sure PROMPT_TEMPLATE is what gets used so the exported constant
        # is the source of truth for users overriding it.
        assert '{title}' in PROMPT_TEMPLATE
        assert '{abstract}' in PROMPT_TEMPLATE


def _make_fake_archive(tmp_path: Path, include_dir_entry: bool = True) -> Path:
    """
    Create a tar.gz archive that mimics the expected classifier bundle
    (adapter_config.json, adapter_model.safetensors, tokenizer.json, ...).

    When ``include_dir_entry`` is True (the default, matching the real upstream
    archive) a top-level directory member ``classifier_512_tinyllama`` without
    a trailing slash is added before the file members. This reproduces the
    layout produced by ``tar czf`` on Linux and exercises the prefix-stripping
    branch of the extractor.
    """
    files = {
        'classifier_512_tinyllama/adapter_config.json': json.dumps(
            {'base_model_name_or_path': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'}
        ).encode('utf-8'),
        'classifier_512_tinyllama/adapter_model.safetensors': b'\x00' * 32,
        'classifier_512_tinyllama/tokenizer.json': b'{}',
        'classifier_512_tinyllama/tokenizer_config.json': b'{}',
        'classifier_512_tinyllama/special_tokens_map.json': b'{}',
    }

    archive_path = tmp_path / 'classifier_512_tinyllama.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as tar:
        if include_dir_entry:
            dir_info = tarfile.TarInfo(name='classifier_512_tinyllama')
            dir_info.type = tarfile.DIRTYPE
            dir_info.mode = 0o755
            tar.addfile(dir_info)
        for name, data in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive_path


class TestEnsureClassifierModel:
    """Tests for the lazy download-and-extract helper."""

    def test_skips_download_when_already_present(self, tmp_path):
        target = tmp_path / 'model'
        target.mkdir()
        (target / 'adapter_config.json').write_text('{}', encoding='utf-8')
        (target / 'adapter_model.safetensors').write_bytes(b'\x00')
        (target / 'tokenizer.json').write_text('{}', encoding='utf-8')

        # Any download call would raise — if the function tried to hit the
        # network the test would fail.
        def boom(*args, **kwargs):
            raise AssertionError('Should not download when model already exists')

        with patch(
            'chem_mat_data.discovery.classifier._download_to_file', side_effect=boom
        ):
            result = ensure_classifier_model(url='http://ignored', target_dir=target)

        assert result == target

    def test_downloads_and_extracts_archive(self, tmp_path):
        target = tmp_path / 'model'
        archive = _make_fake_archive(tmp_path)

        def fake_download(url, dest):
            # Simulate the download by copying the staged archive to ``dest``.
            Path(dest).write_bytes(archive.read_bytes())

        with patch(
            'chem_mat_data.discovery.classifier._download_to_file',
            side_effect=fake_download,
        ):
            result = ensure_classifier_model(
                url='http://example/classifier.tar.gz', target_dir=target
            )

        assert result == target
        assert (target / 'adapter_config.json').exists()
        assert (target / 'adapter_model.safetensors').exists()
        assert (target / 'tokenizer.json').exists()
        # Regression: the top-level directory entry from the tarball must not
        # leak through as a nested subdirectory.
        assert not (target / 'classifier_512_tinyllama').exists()

    def test_raises_when_url_is_none(self, tmp_path):
        with pytest.raises(ValueError, match='classifier_url'):
            ensure_classifier_model(url=None, target_dir=tmp_path / 'model')

    def test_cleans_up_partial_extraction_on_failure(self, tmp_path):
        target = tmp_path / 'model'

        def broken_download(url, dest):
            # Write invalid gzip bytes so the extract step fails.
            Path(dest).write_bytes(b'not a tar.gz')

        with patch(
            'chem_mat_data.discovery.classifier._download_to_file',
            side_effect=broken_download,
        ):
            with pytest.raises(Exception):
                ensure_classifier_model(
                    url='http://example/bad.tar.gz', target_dir=target
                )

        # Partial directory must have been removed so the next attempt
        # starts clean.
        assert not target.exists()


class TestClassifierPredict:
    """
    Tests for the :class:`Classifier` inference wrapper using a fake model +
    tokenizer. We do not exercise real torch here — the goal is to verify
    the glue (prompt construction, softmax, batching, threshold semantics).
    """

    def _fake_classifier(self):
        """
        Build a Classifier whose ``model`` and ``tokenizer`` are plain
        MagicMocks and whose internal ``_forward_logits`` is replaced with a
        deterministic stub keyed on the prompt text.
        """
        clf = Classifier.__new__(Classifier)
        clf.model = MagicMock()
        clf.tokenizer = MagicMock()
        clf.max_length = 512
        clf.device = 'cpu'
        return clf

    def test_predict_score_returns_float_in_unit_interval(self):
        clf = self._fake_classifier()
        with patch.object(clf, '_forward_logits', return_value=(0.1, 0.9)):
            score = clf.predict_score(title='T', abstract='A')
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_predict_score_reflects_positive_class_probability(self):
        clf = self._fake_classifier()
        # logits = (-1.0, 1.0) → softmax → positive prob ≈ 0.88
        with patch.object(clf, '_forward_logits', return_value=(-1.0, 1.0)):
            score = clf.predict_score(title='T', abstract='A')
        assert score > 0.5

    def test_predict_score_handles_missing_fields(self):
        clf = self._fake_classifier()
        with patch.object(clf, '_forward_logits', return_value=(0.0, 0.0)):
            # Should not raise even if title/abstract are None.
            score = clf.predict_score(title=None, abstract=None)
        assert 0.0 <= score <= 1.0

    def test_predict_batch_preserves_order(self):
        clf = self._fake_classifier()
        records = [
            {'title': 'A', 'abstract': 'x'},  # clear negative
            {'title': 'B', 'abstract': 'y'},  # clear positive
            {'title': 'C', 'abstract': 'z'},  # neutral
        ]
        # (neg_logit, pos_logit) → softmax → positive probability.
        # A: neg ≫ pos  → ~0.02,  B: neg ≪ pos → ~0.98,  C: equal → 0.5.
        with patch.object(
            clf,
            '_forward_logits',
            side_effect=[(2.0, -2.0), (-2.0, 2.0), (0.0, 0.0)],
        ):
            scores = clf.predict_batch(records)
        assert len(scores) == 3
        assert scores[0] < scores[2] < scores[1]


class TestMlStackAvailable:
    """Tests for the :func:`is_ml_stack_available` dependency probe."""

    def test_returns_false_when_torch_missing(self):
        with patch.dict(sys.modules, {'torch': None}):
            assert is_ml_stack_available() is False

    def test_import_hint_in_error_message(self):
        # load_classifier should raise ImportError with a friendly install hint
        # when the stack is unavailable.
        with patch(
            'chem_mat_data.discovery.classifier.is_ml_stack_available',
            return_value=False,
        ):
            with pytest.raises(ImportError, match=r'pip install.*dev'):
                classifier_mod.load_classifier(model_dir='/tmp/nope')


@pytest.mark.slow
def test_real_inference_end_to_end(tmp_path):
    """
    Optional integration test — downloads TinyLlama + adapter and runs a real
    forward pass. Skipped unless ``CMDATA_DISCOVERY_SLOW=1`` is set because it
    pulls several GB of weights.
    """
    if os.environ.get('CMDATA_DISCOVERY_SLOW') != '1':
        pytest.skip('Set CMDATA_DISCOVERY_SLOW=1 to run the slow integration test.')

    from chem_mat_data.config import Config

    config = Config()
    url = config.get_discovery_classifier_url()
    if not url:
        pytest.skip('No [discovery] classifier_url configured.')

    model_dir = ensure_classifier_model(url=url, target_dir=tmp_path / 'model')
    clf = classifier_mod.load_classifier(model_dir=str(model_dir), device='cpu')
    score = clf.predict_score(
        title='Open Quantum Materials Database for 2D Semiconductors',
        abstract='We release a dataset of 10,000 materials with computed band gaps.',
    )
    assert 0.0 <= score <= 1.0
