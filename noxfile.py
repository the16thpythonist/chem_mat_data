"""Nox configuration for chem_mat_data package."""

import nox

# Python versions to test against
PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12"]


@nox.session(python=PYTHON_VERSIONS[0], venv_backend="uv")
def test(session):
    """Run unit tests with pytest."""
    session.install(".[dev]")
    # Skip tests that require network access only
    session.run(
        "pytest",
        "tests/",
        "-v",
        "--ignore=tests/test_datasets.py",     # Network timeouts
        "-k", "not localonly"  # Skip local-only tests
    )

@nox.session(python="3.11", venv_backend="uv")
def lint(session):
    """Run linting with ruff."""
    session.install(".[dev]")
    session.run("ruff", "check", ".")


@nox.session(python="3.11", venv_backend="uv")
def format(session):
    """Format code with ruff."""
    session.install(".[dev]")
    session.run("ruff", "format", ".")


@nox.session(python="3.11", venv_backend="uv")
def type_check(session):
    """Run type checking with mypy."""
    session.install(".[dev]")
    session.run("mypy", "chem_mat_data", "--ignore-missing-imports")


@nox.session(python="3.11", venv_backend="uv")
def docs(session):
    """Build documentation."""
    session.install(".[dev]")
    session.run("sphinx-build", "-b", "html", "docs", "docs/_build/html")


@nox.session(python=PYTHON_VERSIONS, venv_backend="uv")
def install_check(session):
    """Check that the package can be installed and imported."""
    session.install(".")
    session.run("python", "-c", "import chem_mat_data; print('Import successful')")


@nox.session(python="3.11", venv_backend="uv")
def build(session):
    """Build the package."""
    session.install(".[dev]")
    session.run("python", "-m", "build")


@nox.session(python="3.11", venv_backend="uv")
def clean(session):
    """Clean build artifacts."""
    import shutil
    import os

    # Directories to clean
    clean_dirs = [
        "build",
        "dist",
        "*.egg-info",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        ".mypy_cache",
        ".ruff_cache"
    ]

    for pattern in clean_dirs:
        if "*" in pattern:
            import glob
            for path in glob.glob(pattern):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                elif os.path.isfile(path):
                    os.remove(path)
        else:
            if os.path.isdir(pattern):
                shutil.rmtree(pattern)
            elif os.path.isfile(pattern):
                os.remove(pattern)

    session.log("Cleaned build artifacts")