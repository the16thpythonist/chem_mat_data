"""
TinyLlama classifier inference for the dataset-discovery pipeline.

This module wraps the fine-tuned LoRA adapter from the upstream
Dataset-Discovery-Pipeline repository and exposes a small, friendly API:

- :data:`PROMPT_TEMPLATE` — the exact prompt template used during fine-tuning.
- :func:`format_prompt` — render the template for a (title, abstract) pair.
- :func:`ensure_classifier_model` — idempotent download-and-extract of the
  classifier archive into the user config directory.
- :func:`load_classifier` — construct a :class:`Classifier` from a local
  model directory. Uses 4-bit quantization when a CUDA device and
  ``bitsandbytes`` are available; otherwise falls back to standard fp32/fp16
  on CPU.
- :class:`Classifier` — thin inference wrapper with ``predict_score`` and
  ``predict_batch`` methods.

The module is intentionally importable even when the heavy ML stack
(``torch``, ``transformers``, ``peft``, optionally ``bitsandbytes``) is not
installed: heavy imports are deferred until :func:`load_classifier` is called,
and a clear :class:`ImportError` with an install hint is raised at that point.
This keeps ``cmmanage``'s startup path fast and CPU-friendly for users who
never touch the discovery pipeline.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import requests

#: Prompt template used for both training and inference. Kept in-sync with
#: the upstream ``train_classifier.py`` so scores remain comparable.
PROMPT_TEMPLATE: str = (
    'Classify whether the following scientific paper introduces a dataset '
    'of molecules or materials that is released for reuse.\n\n'
    'Title: {title}\n\n'
    'Abstract: {abstract}\n\n'
    'Answer:'
)

#: Files that must be present in a valid classifier directory.
_REQUIRED_FILES = (
    'adapter_config.json',
    'adapter_model.safetensors',
    'tokenizer.json',
)

PathLike = Union[str, os.PathLike]


def format_prompt(title: Optional[str], abstract: Optional[str]) -> str:
    """
    Render the classifier prompt for a given title and abstract.

    Missing fields are substituted with an empty string so that the caller
    does not have to filter records before scoring.

    :param title: The paper title.
    :param abstract: The paper abstract.

    :returns: The formatted prompt string.
    """
    return PROMPT_TEMPLATE.format(title=title or '', abstract=abstract or '')


def is_ml_stack_available() -> bool:
    """
    Return True if ``torch``, ``transformers`` and ``peft`` can be imported.

    ``bitsandbytes`` is optional and handled separately — CPU fallback works
    without it.
    """
    for name in ('torch', 'transformers', 'peft'):
        spec = importlib.util.find_spec(name)
        if spec is None:
            return False
    return True


def _raise_ml_stack_missing() -> None:
    raise ImportError(
        "The dataset-discovery classifier requires torch, transformers "
        "and peft. Install them via the 'dev' extras:\n\n"
        "    pip install chem_mat_database[dev]\n\n"
        'Then re-run the command.'
    )


# ---------------------------------------------------------------------------
# Model archive download / extraction
# ---------------------------------------------------------------------------


def _download_to_file(url: str, dest: PathLike) -> None:
    """
    Stream ``url`` to ``dest`` with a simple progress bar.

    Isolated from :func:`ensure_classifier_model` so that tests can
    monkey-patch the network boundary without touching ``requests``.
    """
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TransferSpeedColumn,
    )

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get('content-length') or 0)

    with Progress(
        TextColumn('[cyan]Downloading classifier…'),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
    ) as progress:
        task = progress.add_task('download', total=total or None)
        with dest_path.open('wb') as f:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                progress.update(task, advance=len(chunk))


def _extract_tarball(archive: PathLike, target_dir: PathLike) -> None:
    """
    Extract ``archive`` into ``target_dir``.

    The upstream archive is packed with a single top-level folder
    (``classifier_512_tinyllama/``); we flatten that folder so the caller gets
    the adapter files directly inside ``target_dir``.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, 'r:*') as tar:
        members = tar.getmembers()
        # Detect a single common top-level directory and strip it. We compare
        # against the prefix *without* trailing slash because tar stores the
        # top-level directory entry itself as e.g. ``classifier_512_tinyllama``
        # (no slash), while file entries use ``classifier_512_tinyllama/foo``.
        top_levels = {
            m.name.split('/', 1)[0] for m in members if m.name and not m.name.startswith('/')
        }
        strip_prefix = next(iter(top_levels)) if len(top_levels) == 1 else None

        for member in members:
            name = member.name
            if strip_prefix:
                if name == strip_prefix:
                    # The top-level directory entry itself — skip, we will
                    # recreate it by virtue of writing its children.
                    continue
                if name.startswith(strip_prefix + '/'):
                    name = name[len(strip_prefix) + 1:]
            if not name or name.startswith('..') or os.path.isabs(name):
                continue
            # Security: reject absolute or traversal paths.
            dest = target / name
            dest_resolved = dest.resolve()
            target_resolved = target.resolve()
            if target_resolved not in dest_resolved.parents and dest_resolved != target_resolved:
                continue

            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with dest.open('wb') as out:
                shutil.copyfileobj(src, out)


def _has_required_files(model_dir: Path) -> bool:
    return all((model_dir / name).exists() for name in _REQUIRED_FILES)


def ensure_classifier_model(
    url: Optional[str],
    target_dir: PathLike,
    force: bool = False,
) -> Path:
    """
    Ensure the fine-tuned classifier lives at ``target_dir`` and return its path.

    If ``target_dir`` already contains the expected adapter files the function
    is a no-op. Otherwise the archive at ``url`` is downloaded to a temporary
    file, extracted into ``target_dir`` and then the temp archive is removed.

    On any failure the partial target directory is cleaned up so that the
    next attempt starts from a clean slate.

    :param url: HTTPS URL of the ``.tar.gz`` classifier archive. Required
        unless the model is already present locally.
    :param target_dir: Destination directory (typically inside the user
        config folder).
    :param force: If True, re-download even when the target already exists.

    :returns: The resolved :class:`~pathlib.Path` to ``target_dir``.

    :raises ValueError: If ``url`` is ``None`` and the model is not already
        present.
    """
    target = Path(target_dir)

    if not force and target.exists() and _has_required_files(target):
        return target

    if not url:
        raise ValueError(
            'No classifier_url configured. Set [discovery] classifier_url in the '
            'config file or pass --model-url on the command line.'
        )

    target.mkdir(parents=True, exist_ok=True)
    archive_path = target.parent / (target.name + '.tar.gz.part')

    try:
        _download_to_file(url, archive_path)
        _extract_tarball(archive_path, target)
    except Exception:
        # Roll back the partial directory so the next run is clean.
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        if archive_path.exists():
            try:
                archive_path.unlink()
            except OSError:
                pass

    return target


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------


class Classifier:
    """
    Thin wrapper around a fine-tuned TinyLlama sequence-classification model.

    Instances are normally built via :func:`load_classifier` rather than
    constructed directly. The wrapper stores the underlying ``model`` and
    ``tokenizer`` objects plus the configured ``max_length`` and ``device``
    so that call sites do not have to thread those parameters through
    themselves.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_length: int = 512,
        device: str = 'cpu',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _forward_logits(self, prompt: str) -> Tuple[float, float]:
        """
        Run one forward pass and return the two-class logits as a plain tuple.

        Isolated so that tests can patch it out without having to mock torch.
        """
        import torch

        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        try:
            device = next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            device = self.device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits
        neg, pos = float(logits[0, 0].item()), float(logits[0, 1].item())
        return neg, pos

    @staticmethod
    def _softmax_positive(neg: float, pos: float) -> float:
        """Numerically stable softmax → probability of the positive class."""
        import math

        m = max(neg, pos)
        neg_e = math.exp(neg - m)
        pos_e = math.exp(pos - m)
        return pos_e / (neg_e + pos_e)

    def predict_score(self, title: Optional[str], abstract: Optional[str]) -> float:
        """
        Return the positive-class probability for a single paper.

        :param title: Paper title (may be ``None``).
        :param abstract: Paper abstract (may be ``None``).

        :returns: Probability in ``[0, 1]`` that the paper releases a
            molecular/materials dataset.
        """
        prompt = format_prompt(title, abstract)
        neg, pos = self._forward_logits(prompt)
        return self._softmax_positive(neg, pos)

    def predict_batch(self, records: Iterable[dict]) -> List[float]:
        """
        Score an iterable of ``{title, abstract}`` records in order.

        This is currently a simple loop over :meth:`predict_score`; the
        signature exists so that a future optimization can introduce real
        batched inference without breaking callers.

        :param records: Iterable of records with ``title`` and ``abstract``
            keys.

        :returns: List of probabilities, one per input record, in order.
        """
        return [
            self.predict_score(rec.get('title'), rec.get('abstract'))
            for rec in records
        ]


def load_classifier(
    model_dir: PathLike,
    device: Optional[str] = None,
    quantize: Optional[bool] = None,
    max_length: int = 512,
) -> Classifier:
    """
    Load the fine-tuned classifier from a local directory.

    Heavy imports (``torch``, ``transformers``, ``peft``) happen inside this
    function so that importing :mod:`chem_mat_data.discovery.classifier` does
    not pull them in. If any of the required packages are missing an
    :class:`ImportError` with an install hint is raised immediately.

    :param model_dir: Directory containing the LoRA adapter and tokenizer
        files (as produced by :func:`ensure_classifier_model`).
    :param device: ``"cpu"``, ``"cuda"``, or ``None`` to auto-detect. When
        CPU is chosen the 4-bit quantization path is skipped.
    :param quantize: Force-enable (``True``) or force-disable (``False``)
        4-bit quantization. ``None`` enables quantization only when CUDA is
        available and ``bitsandbytes`` is installed.
    :param max_length: Maximum tokenized input length.

    :returns: A fully-initialized :class:`Classifier` ready for inference.
    """
    if not is_ml_stack_available():
        _raise_ml_stack_missing()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir_path = Path(model_dir)
    adapter_cfg_path = model_dir_path / 'adapter_config.json'
    if not adapter_cfg_path.exists():
        raise FileNotFoundError(
            f'adapter_config.json not found in {model_dir_path}; '
            'run `cmmanage discover model-download` first.'
        )

    with adapter_cfg_path.open(encoding='utf-8') as f:
        base_ckpt = json.load(f).get('base_model_name_or_path')
    if not base_ckpt:
        raise RuntimeError(
            'adapter_config.json is missing base_model_name_or_path.'
        )

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if quantize is None:
        quantize = (
            device == 'cuda'
            and importlib.util.find_spec('bitsandbytes') is not None
        )

    # transformers renamed ``torch_dtype`` to ``dtype`` in 4.46.0; detect the
    # installed version so we avoid both TypeError on old installs and a
    # DeprecationWarning on new ones.
    import transformers as _tf
    from packaging.version import Version as _V

    dtype_kwarg = 'dtype' if _V(_tf.__version__) >= _V('4.46.0') else 'torch_dtype'

    load_kwargs: dict = {}
    if quantize:
        from transformers import BitsAndBytesConfig

        load_kwargs['device_map'] = 'auto'
        load_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        # CPU / plain-fp32 fallback.
        load_kwargs[dtype_kwarg] = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_ckpt, num_labels=2, **load_kwargs
    )
    model = PeftModel.from_pretrained(base_model, str(model_dir_path), is_trainable=False)
    model.eval()

    if not quantize:
        model.to(device)

    return Classifier(
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
    )
