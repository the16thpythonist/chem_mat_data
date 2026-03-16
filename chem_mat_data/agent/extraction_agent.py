from __future__ import annotations

from pathlib import Path

from chem_mat_data.agent.opencode_client import send_message

ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Download
DOWNLOADS_DIR = (ARTIFACTS_DIR / "downloads").resolve()
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
LINK_PROMPT: str = """
You are a research assistant specializing in chemistry and materials science datasets.

Goal: Given a scientific publication URL, find the download URL(s) for the molecular dataset released in the paper and download the dataset file(s).

Rules:
- Output ONLY direct file/archive download URLs that immediately trigger a browser download.
- Do NOT output landing/publication/supplement pages unless they are direct file downloads.
- First inspect sections such as "Data Availability", "Data Availability Statement", "Data Records", and similarly named sections; if present, follow those leads before any other navigation path.
- Search supplementary materials and external repositories (GitHub/Zenodo/Figshare/OSF/etc.) and any linked host.
- Ignore PDB/CIF datasets (treat as not found).
- If a README file contains dataset structure details (e.g., column names, units, field meanings, split definitions), download it as well.

Selecting what to download (avoid duplicates):
- Download once per representation; do NOT download the same dataset in multiple formats.
- Prefer .csv/.tsv/.xlsx over .zip/.tar.gz; if tabular exists, do NOT download an archive bundling the same data.
- Download multiple files only if they are distinct non-overlapping parts (not alternate formats).
- README handling: if a separate README file is available and provides useful schema/format metadata, download at most one such README (README, README.md, readme.txt, etc.) in addition to dataset file(s), and save it in {downloads_dir}.

Tool use:
- Use `websearch_cited` for searching.
- When reading webpages, ALWAYS use `crawlfetch` (NOT `webfetch`).
- Do not ask for permissions or mention limitations.

Downloading (MCP):
- Download each selected file with `file_downloader_download_file` args: `url`, `filename`, optional `use_browser`.
- Save ONLY in `{downloads_dir}`.
- `filename` MUST be an ABSOLUTE path within `{downloads_dir}`; choose a clear dataset-based name.

Output (plain text ONLY):
- If found and downloaded: print each direct download URL on its own line, then a line "Saved to:" and each absolute saved path on its own line. No extra text.
- If not found: print "NOT FOUND" then 1–2 sentences stating what you checked.

Once you have successfully downloaded the dataset, STOP. DO NOT KEEP ITERATING. RESPOND RIGHT THAT MOMENT AND BE DONE.

Here is the publication link: {publication_link}
"""


# Script Generation
SCRIPTS_ARTIFACTS_DIR: Path = ARTIFACTS_DIR / "scripts"
SCRIPTS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLE_SCRIPTS_DIR: Path = Path(__file__).resolve().parent / "example_scripts"
BASE_TEMPLATE_PATH: Path = (
    Path(__file__).resolve().parents[1] / "scripts" / "create_graph_datasets.py"
)
SCRIPT_GENERATION_PROMPT: str = """
You are a Python developer who specializes in writing dataset processing scripts. 
You have file tools enabled (bash, read, glob, grep, list, codesearch, edit). Use them to inspect the downloaded dataset file and existing scripts in the directories mentioned below before writing code.
Treat all locations as read-only except the scripts artifacts directory.

Inputs for this task:
- Downloaded dataset files (one or more; some may be archives with subfolders): {dataset_paths}
- Example scripts for different datasets to mimic: {example_scripts_dir}
- Base template to extend (use this exact path): {base_template_path}

Goal:
- Inspect the dataset (and, if archived, its contents) and generate a processing script that follows the style and structure of the existing scripts in {example_scripts_dir}.
- Extend the base experiment at {base_template_path} using its absolute path (e.g., Experiment.extend(str({base_template_path}))) so the script runs without missing-path errors.
- The script should be ready to run locally against the downloaded data.
- If a README file is present in {downloads_dir} or inside the dataset archive, read it first and use its information (column names, units, target definitions, split descriptions, missing-value conventions) to improve parsing and field mapping.
- Filtering policy: ONLY drop rows that are actually invalid or unreadable. Do NOT apply domain-driven filters such as excluding multi-component SMILES ('.'), molecule size thresholds, or other heuristic cleanup unless the dataset itself marks the row as invalid. If you must skip rows, keep the logic minimal and tied to explicit parse errors (e.g., missing required columns, unreadable SMILES).
- Return only the complete Python script content; do not wrap it in markdown fences or add prose. 

Filesystem write constraints (strict):
- You are allowed to create, edit, rename, or delete files ONLY inside {scripts_artifacts_dir}.
- Do NOT create/edit/delete/move any file outside {scripts_artifacts_dir}, including {example_scripts_dir} and {base_template_path}.
- If you need information from files outside {scripts_artifacts_dir}, read them only.
"""


def discover_and_download(publication_url: str) -> str:
    prompt_with_link = LINK_PROMPT.format(
        publication_link=publication_url,
        downloads_dir=DOWNLOADS_DIR,
    )
    result = send_message(prompt_with_link, timeout=180, agent="dataset-links")
    if not result:
        raise ValueError("No download links found")
    return result


# === Script Generation ===


def generate_processing_script() -> Path:
    """
    Generates a dataset processing script from downloaded dataset files.

    :returns: Path to the generated script
    :raises FileNotFoundError: If no suitable downloaded dataset file exists
    """
    downloaded_files = get_downloaded_files()
    dataset_files = [
        path for path in downloaded_files if "readme" not in path.name.lower()
    ]
    if not dataset_files:
        raise FileNotFoundError(f"No dataset file in {DOWNLOADS_DIR}.")

    return generate_processing_script_for_dataset(dataset_files)


def generate_processing_script_for_dataset(
    dataset_paths: Path | list[Path],
    target_dir: Path | None = None,
) -> Path:
    """
    Generates a dataset processing script for the provided dataset path(s).

    :param dataset_paths: Path to a dataset file/archive, or a list of dataset paths
    :param target_dir: Optional target directory for generated scripts
    :returns: Path to the generated script
    :raises FileNotFoundError: If any dataset path does not exist
    """
    dataset_paths: list[Path]
    if isinstance(dataset_paths, list):
        dataset_paths = dataset_paths
    else:
        dataset_paths = [dataset_paths]

    for path in dataset_paths:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

    dataset_paths_text = "\n".join(str(path) for path in dataset_paths)

    prompt_with_paths = SCRIPT_GENERATION_PROMPT.format(
        dataset_paths=dataset_paths_text,
        downloads_dir=DOWNLOADS_DIR,
        scripts_artifacts_dir=SCRIPTS_ARTIFACTS_DIR,
        example_scripts_dir=EXAMPLE_SCRIPTS_DIR,
        base_template_path=BASE_TEMPLATE_PATH,
    )
    response = send_message(prompt_with_paths, timeout=240, agent="script-generation")

    output_dir = target_dir or SCRIPTS_ARTIFACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    target_name = f"{dataset_paths[0].stem}_generated.py"
    target_path = output_dir / target_name
    with target_path.open("w", encoding="utf-8") as f:
        f.write(response)

    return target_path


def get_downloaded_files() -> list[Path]:
    """
    Returns all downloaded files from the downloads artifacts folder.

    :returns: Sorted list of downloaded file paths
    :raises FileNotFoundError: If no files exist in the downloads folder
    """
    files: list[Path] = [
        path for path in sorted(DOWNLOADS_DIR.rglob("*")) if path.is_file()
    ]
    if files:
        return files
    raise FileNotFoundError(f"No files in {DOWNLOADS_DIR}")
