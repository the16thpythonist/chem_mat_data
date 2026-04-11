"""
Orchestration logic for the dataset extraction agent.

This module provides the main agent workflows:

1. **Dataset identification** — given a publication URL or title, an LLM agent
   determines whether the paper publishes molecular/materials property data.
2. **Dataset discovery and download** — given a publication URL, an LLM agent
   finds and downloads the associated dataset files.
3. **Processing script generation** — given downloaded dataset files, an LLM
   agent generates a ChemMatData-compatible processing script.

All workflows are backend-agnostic: the concrete LLM backend (OpenCode,
Claude, ...) is selected at runtime via the ``backend_name`` parameter or
the ``[agent]`` section in the ChemMatData config file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from chem_mat_data.agent.backend import get_backend, EventCallback
from chem_mat_data.agent.prompts import get_prompt


ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Identification
IDENTIFY_DIR: Path = ARTIFACTS_DIR / "identify"
IDENTIFY_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Download
DOWNLOADS_DIR = (ARTIFACTS_DIR / "downloads").resolve()
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Script Generation
SCRIPTS_ARTIFACTS_DIR: Path = ARTIFACTS_DIR / "scripts"
SCRIPTS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLE_SCRIPTS_DIR: Path = Path(__file__).resolve().parent / "example_scripts"
BASE_TEMPLATE_PATH: Path = (
    Path(__file__).resolve().parents[1] / "scripts" / "create_graph_datasets.py"
)


def _is_url(value: str) -> bool:
    """Return ``True`` if *value* looks like a URL."""
    return value.startswith("http://") or value.startswith("https://")


def identify_dataset(
    query: str,
    backend_name: str | None = None,
    config=None,
    on_event: Optional[EventCallback] = None,
) -> dict:
    """
    Determine whether a publication publishes molecular/materials property data.

    The agent reads the paper (or searches for it by title), looks for data
    availability sections, repository links, supplementary tables, and DFT or
    experimental measurement collections. The result is saved as a JSON file in
    ``agent/artifacts/identify/``.

    :param query: A publication URL or paper title.
    :param backend_name: LLM backend to use. Falls back to the value from
        ``config``, then to ``"opencode"``.
    :param config: A :class:`~chem_mat_data.config.Config` instance. Uses the
        singleton if ``None``.
    :param on_event: Optional callback for intermediate agent events.

    :returns: A dict with ``"decision"`` (``"yes"`` or ``"no"``),
        ``"reasoning"`` (str), and ``"output_path"`` (:class:`~pathlib.Path`).
    :raises ValueError: If the agent response cannot be parsed as JSON.
    """
    import json
    import re

    from chem_mat_data.config import Config

    config = config or Config()
    backend_name = backend_name or config.get_agent_backend()
    backend_config = config.get_agent_backend_config(backend_name)

    backend = get_backend(backend_name, config=backend_config)

    identify_override = config.get_agent_prompt_override("identify_prompt")
    prompt_template = get_prompt(backend.PROMPT_VARIANT, "identify", override=identify_override)

    if _is_url(query):
        query_description = "Publication URL:"
    else:
        query_description = "Paper title (search for it first):"

    prompt = prompt_template.format(
        query_description=query_description,
        query_value=query,
    )

    result = backend.send_message(prompt, timeout=180, on_event=on_event)
    if not result:
        raise ValueError("Agent returned an empty response")

    # Parse the JSON from the agent response. The agent is instructed to return
    # raw JSON, but may include surrounding whitespace or markdown fences.
    cleaned = result.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse agent response as JSON: {exc}\n\nRaw response:\n{result}"
        )

    decision = str(parsed.get("decision", "")).lower().strip()
    reasoning = str(parsed.get("reasoning", ""))

    output = {
        "decision": decision,
        "reasoning": reasoning,
        "query": query,
    }

    # Derive a short filename from the query
    if _is_url(query):
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", query.split("//", 1)[-1])[:80].strip("_")
    else:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", query)[:80].strip("_")

    output_path = IDENTIFY_DIR / f"{slug}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    output["output_path"] = output_path
    return output


def discover_and_download(
    publication_url: str,
    backend_name: str | None = None,
    config=None,
    on_event: Optional[EventCallback] = None,
) -> str:
    """
    Discover and download dataset files from a scientific publication URL.

    An LLM agent navigates the publication page, searches for associated
    dataset repositories, and downloads the raw data files into
    ``agent/artifacts/downloads/``.

    :param publication_url: The URL of the scientific publication to process.
    :param backend_name: LLM backend to use. Falls back to the value from
        ``config``, then to ``"opencode"``.
    :param config: A :class:`~chem_mat_data.config.Config` instance. Uses the
        singleton if ``None``.
    :param on_event: Optional callback for intermediate agent events.

    :returns: The agent response text describing what was downloaded.
    :raises ValueError: If the agent finds no download links.
    """
    from chem_mat_data.config import Config

    config = config or Config()
    backend_name = backend_name or config.get_agent_backend()
    backend_config = config.get_agent_backend_config(backend_name)

    backend = get_backend(backend_name, config=backend_config, agent="dataset-links")

    link_override = config.get_agent_prompt_override("link_prompt")
    prompt_template = get_prompt(backend.PROMPT_VARIANT, "link", override=link_override)

    prompt = prompt_template.format(
        publication_link=publication_url,
        downloads_dir=DOWNLOADS_DIR,
    )

    result = backend.send_message(prompt, timeout=180, on_event=on_event)
    if not result:
        raise ValueError("No download links found")
    return result


def generate_processing_script(
    backend_name: str | None = None,
    config=None,
    on_event: Optional[EventCallback] = None,
    venv_python: Path | None = None,
) -> Path:
    """
    Generate a dataset processing script from downloaded dataset files.

    Scans ``agent/artifacts/downloads/`` for previously downloaded files,
    filters out READMEs and the download recipe, and delegates to
    :func:`generate_processing_script_for_dataset`.

    :param backend_name: LLM backend to use. Falls back to config default.
    :param config: A :class:`~chem_mat_data.config.Config` instance.
    :param on_event: Optional callback for intermediate agent events.
    :param venv_python: Path to the Python interpreter in the prepared venv.

    :returns: Path to the generated script.
    :raises FileNotFoundError: If no suitable downloaded dataset file exists.
    """
    downloaded_files = get_downloaded_files()
    skip_names = {"readme", "download_recipe"}
    dataset_files = [
        path for path in downloaded_files
        if not any(s in path.name.lower() for s in skip_names)
    ]
    if not dataset_files:
        raise FileNotFoundError(f"No dataset file in {DOWNLOADS_DIR}.")

    return generate_processing_script_for_dataset(
        dataset_files,
        backend_name=backend_name,
        config=config,
        on_event=on_event,
        venv_python=venv_python,
    )


def generate_processing_script_for_dataset(
    dataset_paths: Path | list[Path],
    target_dir: Path | None = None,
    backend_name: str | None = None,
    config=None,
    on_event: Optional[EventCallback] = None,
    venv_python: Path | None = None,
) -> Path:
    """
    Generate a dataset processing script for the provided dataset path(s).

    An LLM agent inspects the downloaded data files, reads example scripts
    for reference, and produces a new script that extends the base processing
    template. The agent then runs the script in the provided venv and iterates
    on errors up to 3 times.

    :param dataset_paths: Path to a dataset file/archive, or a list of paths.
    :param target_dir: Optional target directory for the generated script.
    :param backend_name: LLM backend to use. Falls back to config default.
    :param config: A :class:`~chem_mat_data.config.Config` instance.
    :param on_event: Optional callback for intermediate agent events.
    :param venv_python: Path to the Python interpreter in the prepared venv.
        If ``None``, the agent is not instructed to run the script.

    :returns: Path to the generated script.
    :raises FileNotFoundError: If any dataset path does not exist.
    """
    from chem_mat_data.config import Config

    config = config or Config()
    backend_name = backend_name or config.get_agent_backend()
    backend_config = config.get_agent_backend_config(backend_name)

    backend = get_backend(backend_name, config=backend_config, agent="script-generation")

    if isinstance(dataset_paths, list):
        dataset_paths = dataset_paths
    else:
        dataset_paths = [dataset_paths]

    for path in dataset_paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

    dataset_paths_text = "\n".join(str(path) for path in dataset_paths)

    script_override = config.get_agent_prompt_override("script_generation_prompt")
    prompt_template = get_prompt(
        backend.PROMPT_VARIANT, "script_generation", override=script_override
    )

    prompt = prompt_template.format(
        dataset_paths=dataset_paths_text,
        downloads_dir=DOWNLOADS_DIR,
        scripts_artifacts_dir=SCRIPTS_ARTIFACTS_DIR,
        example_scripts_dir=EXAMPLE_SCRIPTS_DIR,
        base_template_path=BASE_TEMPLATE_PATH,
        venv_python=str(venv_python) if venv_python else "python",
    )

    # The agent writes the script file directly and runs it — we increase
    # the timeout substantially to allow for run+iterate cycles.
    backend.send_message(prompt, timeout=600, on_event=on_event)

    # Find the generated script — the agent writes it into the output dir
    output_dir = target_dir or SCRIPTS_ARTIFACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Look for the most recently modified .py file in the output dir
    py_files = sorted(output_dir.glob("*.py"), key=lambda p: p.stat().st_mtime, reverse=True)
    if py_files:
        return py_files[0]

    # Fallback: construct expected name
    target_name = f"{dataset_paths[0].stem}_generated.py"
    target_path = output_dir / target_name
    if target_path.exists():
        return target_path

    raise FileNotFoundError(
        f"Agent did not create a script file in {output_dir}. "
        "Check the agent output for errors."
    )


def get_downloaded_files() -> list[Path]:
    """
    Return all downloaded files from the downloads artifacts folder.

    :returns: Sorted list of downloaded file paths.
    :raises FileNotFoundError: If no files exist in the downloads folder.
    """
    files: list[Path] = [
        path for path in sorted(DOWNLOADS_DIR.rglob("*")) if path.is_file()
    ]
    if files:
        return files
    raise FileNotFoundError(f"No files in {DOWNLOADS_DIR}")
