"""
Availability checks and bootstrap for the external page-reading tools.

The prepare stage of the ``dataset-extraction`` skill has to read publication and
repository landing pages before it can find a download URL. It does that with two
external command line tools that are deliberately *not* Python dependencies of this
project - they are a Node package and a separate Python application - and so nothing
in a normal ``pip install`` brings them along. They are missing on a fresh machine,
and the way they fail when missing is unhelpful enough that the skill previously
documented a fallback chain that did not survive contact with reality.

``crwl``
    Drives a real Chromium through Playwright, and is the default. It reads every page
    the other readers can, plus the ones they cannot: ChemRxiv returns 403 to
    ``defuddle`` and to ``WebFetch`` while serving ``crwl`` the full article.

``defuddle``
    Fetches over plain HTTP and extracts the article body. Roughly three seconds
    quicker per page because it starts no browser, and a reasonable fallback when
    ``crwl`` is unavailable or a page renders badly through it - but it fails outright
    on any host that bot-checks.

``crwl`` leads because the costs are asymmetric. Choosing it when ``defuddle`` would
have done wastes about three seconds; choosing ``defuddle`` on a defended host costs a
403, a diagnosis and a retry, which in an agentic pipeline is orders of magnitude more
expensive than the seconds saved. A default that always works beats a default that is
usually faster.

Both are still checked together, because the fallback has to actually be there on the
day ``crwl`` breaks.

Exposed through ``cmmanage agent tools``, which reports by default and only installs
when asked with ``--install``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from chem_mat_data.agent.environment import EnvironmentSetupError


@dataclass(frozen=True)
class ToolSpec:
    """
    A description of an external command line tool the prepare stage depends on.

    :param name: The executable name as it must appear on ``PATH``.
    :param purpose: One line on what the tool is for, shown in the status report.
    :param install_cmd: Command that installs the tool.
    :param requires: The executable that ``install_cmd`` itself needs, so a missing
        package manager can be reported as such rather than surfacing as a confusing
        ``FileNotFoundError``.
    :param version_args: Arguments that make the tool print its version, or ``None``
        when it has no such flag. ``crwl`` notably has none - it parses ``--version``
        as a URL to crawl - so presence on ``PATH`` is all that can be checked.
    """

    name: str
    purpose: str
    install_cmd: List[str]
    requires: str
    version_args: Optional[List[str]] = None


@dataclass
class ToolStatus:
    """
    The outcome of checking one tool.

    :param spec: The tool that was checked.
    :param available: Whether the tool is usable.
    :param version: Reported version, when the tool exposes one.
    :param path: Resolved path on ``PATH``, when found.
    :param problems: Human readable reasons the tool is not usable.
    """

    spec: ToolSpec
    available: bool
    version: Optional[str] = None
    path: Optional[str] = None
    problems: List[str] = field(default_factory=list)


#: The page readers the prepare stage depends on, in the order they should be reached
#: for: `crwl` first because it works everywhere, `defuddle` as the quicker fallback
#: for when it is unavailable.
PAGE_READERS: List[ToolSpec] = [
    ToolSpec(
        name='crwl',
        purpose='the default - real Chromium, so it reads bot-checked hosts too',
        install_cmd=['uv', 'tool', 'install', 'crawl4ai'],
        requires='uv',
        version_args=None,
    ),
    ToolSpec(
        name='defuddle',
        purpose='fallback - ~3s quicker, but 403s on any host that bot-checks',
        install_cmd=['npm', 'install', '-g', 'defuddle'],
        requires='npm',
        version_args=['--version'],
    ),
]


def playwright_cache_dir() -> Path:
    """
    Return the directory Playwright keeps its downloaded browsers in.

    Playwright uses a per-platform cache location, and ``crwl`` inherits it. Knowing
    the path lets the browser be checked for without paying the cost of starting one.

    :returns: The platform-appropriate ``ms-playwright`` cache directory.
    """
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Caches' / 'ms-playwright'
    if sys.platform == 'win32':
        return Path.home() / 'AppData' / 'Local' / 'ms-playwright'
    return Path.home() / '.cache' / 'ms-playwright'


def chromium_installed() -> bool:
    """
    Report whether Playwright has a Chromium build available.

    ``crwl`` fails with ``Executable doesn't exist at .../chromium-*`` when the browser
    was never downloaded, and that message does not say which command fixes it. This
    is checked separately from ``crwl`` itself so the report can name the actual
    missing piece.

    :returns: ``True`` if at least one Chromium build is present in the cache.
    """
    cache = playwright_cache_dir()
    if not cache.is_dir():
        return False
    return any(cache.glob('chromium-*'))


def check_tool(spec: ToolSpec) -> ToolStatus:
    """
    Check whether a single tool is present and runnable.

    Presence on ``PATH`` is necessary but not sufficient: a tool can resolve and still
    fail to execute, which is exactly what happens to a stale ``npm -g`` shim after an
    nvm version switch. The version probe therefore actually runs the binary when the
    tool exposes a version flag.

    :param spec: The tool to check.

    :returns: The resulting :class:`ToolStatus`.
    """
    path = shutil.which(spec.name)
    if path is None:
        return ToolStatus(
            spec=spec,
            available=False,
            problems=[f'`{spec.name}` is not on PATH'],
        )

    version: Optional[str] = None
    if spec.version_args is not None:
        try:
            result = subprocess.run(
                [spec.name, *spec.version_args],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return ToolStatus(
                spec=spec,
                available=False,
                path=path,
                problems=[f'`{spec.name}` resolves to {path} but cannot be run: {exc}'],
            )

        if result.returncode != 0:
            detail = (result.stderr.strip() or result.stdout.strip()).splitlines()
            first = detail[0] if detail else f'exit {result.returncode}'
            return ToolStatus(
                spec=spec,
                available=False,
                path=path,
                problems=[f'`{spec.name}` resolves to {path} but exits non-zero: {first}'],
            )

        version = result.stdout.strip() or None

    return ToolStatus(spec=spec, available=True, version=version, path=path)


def check_page_readers() -> List[ToolStatus]:
    """
    Check every tool the prepare stage depends on.

    The Chromium requirement is folded into ``crwl``'s status rather than reported as a
    tool of its own, because a ``crwl`` without a browser is not partially working - it
    fails on every page.

    Example:

    .. code-block:: python

        for status in check_page_readers():
            if not status.available:
                print(status.spec.name, status.problems)

    :returns: One :class:`ToolStatus` per entry in :data:`PAGE_READERS`.
    """
    statuses: List[ToolStatus] = []
    for spec in PAGE_READERS:
        status = check_tool(spec)
        if spec.name == 'crwl' and status.available and not chromium_installed():
            status.available = False
            status.problems.append(
                f'no Chromium build in {playwright_cache_dir()} - '
                '`crwl` will fail on every page until one is downloaded'
            )
        statuses.append(status)
    return statuses


def install_tool(spec: ToolSpec) -> None:
    """
    Install a single tool with its documented install command.

    The command is run with output inherited rather than captured, because both
    installs are slow enough that silence reads as a hang.

    :param spec: The tool to install.

    :raises EnvironmentSetupError: If the required package manager is missing or the
        install command fails.
    """
    if shutil.which(spec.requires) is None:
        raise EnvironmentSetupError(
            f'cannot install `{spec.name}`: `{spec.requires}` is not on PATH. '
            f'Install {spec.requires} first, then re-run.'
        )

    result = subprocess.run(spec.install_cmd)
    if result.returncode != 0:
        raise EnvironmentSetupError(
            f'`{" ".join(spec.install_cmd)}` failed (exit {result.returncode})'
        )


def install_chromium() -> None:
    """
    Download the Chromium build that ``crwl`` needs.

    ``crawl4ai-setup`` ships with crawl4ai and runs the browser download against the
    same Playwright installation crawl4ai itself uses, which a bare
    ``playwright install chromium`` from an unrelated environment does not necessarily
    do.

    :raises EnvironmentSetupError: If neither installer is available or the download
        fails.
    """
    if shutil.which('crawl4ai-setup') is not None:
        command = ['crawl4ai-setup']
    elif shutil.which('playwright') is not None:
        command = ['playwright', 'install', 'chromium']
    else:
        raise EnvironmentSetupError(
            'cannot install Chromium: neither `crawl4ai-setup` nor `playwright` is on '
            'PATH. Install crawl4ai first with `uv tool install crawl4ai`.'
        )

    result = subprocess.run(command)
    if result.returncode != 0:
        raise EnvironmentSetupError(
            f'`{" ".join(command)}` failed (exit {result.returncode})'
        )


def install_missing(statuses: List[ToolStatus]) -> List[ToolStatus]:
    """
    Install every tool in ``statuses`` that is not currently available, then re-check.

    A tool that is present but whose browser is missing is handled by downloading the
    browser rather than by reinstalling the tool, so re-running after a partial failure
    does the least work that could possibly help.

    :param statuses: The result of a previous :func:`check_page_readers` call.

    :returns: Freshly checked statuses reflecting the state after installing.
    :raises EnvironmentSetupError: If any install step fails.
    """
    chromium_attempted = False

    for status in statuses:
        if status.available:
            continue
        # A resolvable `crwl` that failed only its browser check needs the browser,
        # not another copy of crawl4ai.
        if status.path is not None and status.spec.name == 'crwl':
            install_chromium()
            chromium_attempted = True
            continue
        install_tool(status.spec)

    statuses = check_page_readers()

    # crawl4ai installed just now ships without a browser, so `crwl` only becomes
    # usable after this second step. Guarded so a browser download is never repeated.
    if not chromium_attempted and any(
        s.spec.name == 'crwl' and not s.available for s in statuses
    ):
        install_chromium()
        statuses = check_page_readers()

    return statuses
