import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from chem_mat_data.agent.environment import EnvironmentSetupError
from chem_mat_data.agent import tools
from chem_mat_data.agent.tools import (
    PAGE_READERS,
    ToolSpec,
    check_page_readers,
    check_tool,
    chromium_installed,
    install_chromium,
    install_missing,
    install_tool,
    playwright_cache_dir,
)


DEFUDDLE = ToolSpec(
    name='defuddle',
    purpose='fast HTTP article extraction',
    install_cmd=['npm', 'install', '-g', 'defuddle'],
    requires='npm',
    version_args=['--version'],
)

CRWL = ToolSpec(
    name='crwl',
    purpose='Chromium-backed crawling',
    install_cmd=['uv', 'tool', 'install', 'crawl4ai'],
    requires='uv',
    version_args=None,
)


def fake_which(available: Dict[str, str]):
    """
    Build a ``shutil.which`` replacement backed by a fixed mapping.

    :param available: Maps executable name to the path it should resolve to. Anything
        absent resolves to ``None``.

    :returns: A callable with the signature of ``shutil.which``.
    """
    def _which(name: str) -> Optional[str]:
        return available.get(name)

    return _which


def fake_run(outcomes: Dict[str, subprocess.CompletedProcess], calls: List[List[str]]):
    """
    Build a ``subprocess.run`` replacement that records calls and returns canned results.

    :param outcomes: Maps the first token of a command to the result to return. A
        command whose name is absent succeeds with empty output.
    :param calls: List that every invoked command is appended to, for assertions.

    :returns: A callable with the signature of ``subprocess.run``.
    """
    def _run(cmd, *args, **kwargs) -> subprocess.CompletedProcess:
        calls.append(list(cmd))
        return outcomes.get(
            cmd[0],
            subprocess.CompletedProcess(args=cmd, returncode=0, stdout='', stderr=''),
        )

    return _run


def make_cache(tmp_path: Path, with_chromium: bool = True) -> Path:
    """
    Create a fake Playwright browser cache directory.

    :param tmp_path: Base temporary directory.
    :param with_chromium: Whether to place a ``chromium-*`` build inside.

    :returns: The path that should be treated as the home directory.
    """
    home = tmp_path / 'home'
    cache = home / '.cache' / 'ms-playwright'
    cache.mkdir(parents=True)
    if with_chromium:
        (cache / 'chromium-1234').mkdir()
    return home


# -- check_tool --

def test_check_tool_reports_missing_when_not_on_path(monkeypatch):
    monkeypatch.setattr(tools.shutil, 'which', fake_which({}))

    status = check_tool(DEFUDDLE)

    assert not status.available
    assert status.path is None
    assert 'not on PATH' in status.problems[0]


def test_check_tool_reports_version_when_probe_succeeds(monkeypatch):
    monkeypatch.setattr(tools.shutil, 'which', fake_which({'defuddle': '/usr/bin/defuddle'}))
    monkeypatch.setattr(
        tools.subprocess,
        'run',
        fake_run(
            {'defuddle': subprocess.CompletedProcess([], 0, stdout='0.19.2\n', stderr='')},
            [],
        ),
    )

    status = check_tool(DEFUDDLE)

    assert status.available
    assert status.version == '0.19.2'
    assert status.path == '/usr/bin/defuddle'
    assert status.problems == []


def test_check_tool_rejects_a_resolvable_but_broken_binary(monkeypatch):
    """A stale `npm -g` shim resolves on PATH but cannot execute."""
    monkeypatch.setattr(tools.shutil, 'which', fake_which({'defuddle': '/stale/defuddle'}))
    monkeypatch.setattr(
        tools.subprocess,
        'run',
        fake_run(
            {
                'defuddle': subprocess.CompletedProcess(
                    [], 127, stdout='', stderr='env: node: No such file or directory'
                )
            },
            [],
        ),
    )

    status = check_tool(DEFUDDLE)

    assert not status.available
    assert status.path == '/stale/defuddle'
    assert 'node' in status.problems[0]


def test_check_tool_handles_oserror_from_the_probe(monkeypatch):
    monkeypatch.setattr(tools.shutil, 'which', fake_which({'defuddle': '/usr/bin/defuddle'}))

    def _boom(*args, **kwargs):
        raise OSError('Exec format error')

    monkeypatch.setattr(tools.subprocess, 'run', _boom)

    status = check_tool(DEFUDDLE)

    assert not status.available
    assert 'Exec format error' in status.problems[0]


def test_check_tool_skips_the_probe_when_the_tool_has_no_version_flag(monkeypatch):
    """`crwl` parses `--version` as a URL, so presence has to be the whole check."""
    calls: List[List[str]] = []
    monkeypatch.setattr(tools.shutil, 'which', fake_which({'crwl': '/usr/bin/crwl'}))
    monkeypatch.setattr(tools.subprocess, 'run', fake_run({}, calls))

    status = check_tool(CRWL)

    assert status.available
    assert status.version is None
    assert calls == []


# -- browser detection --

def test_playwright_cache_dir_is_platform_specific(monkeypatch, tmp_path):
    monkeypatch.setattr(tools.Path, 'home', staticmethod(lambda: tmp_path))

    monkeypatch.setattr(tools.sys, 'platform', 'linux')
    assert playwright_cache_dir() == tmp_path / '.cache' / 'ms-playwright'

    monkeypatch.setattr(tools.sys, 'platform', 'darwin')
    assert playwright_cache_dir() == tmp_path / 'Library' / 'Caches' / 'ms-playwright'

    monkeypatch.setattr(tools.sys, 'platform', 'win32')
    assert playwright_cache_dir() == tmp_path / 'AppData' / 'Local' / 'ms-playwright'


def test_chromium_installed_false_when_cache_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(tools.sys, 'platform', 'linux')
    monkeypatch.setattr(tools.Path, 'home', staticmethod(lambda: tmp_path / 'nowhere'))

    assert chromium_installed() is False


def test_chromium_installed_false_when_cache_has_no_chromium(monkeypatch, tmp_path):
    home = make_cache(tmp_path, with_chromium=False)
    monkeypatch.setattr(tools.sys, 'platform', 'linux')
    monkeypatch.setattr(tools.Path, 'home', staticmethod(lambda: home))

    assert chromium_installed() is False


def test_chromium_installed_true_when_a_build_is_present(monkeypatch, tmp_path):
    home = make_cache(tmp_path, with_chromium=True)
    monkeypatch.setattr(tools.sys, 'platform', 'linux')
    monkeypatch.setattr(tools.Path, 'home', staticmethod(lambda: home))

    assert chromium_installed() is True


# -- check_page_readers --

def test_check_page_readers_all_available(monkeypatch, tmp_path):
    home = make_cache(tmp_path, with_chromium=True)
    monkeypatch.setattr(tools.sys, 'platform', 'linux')
    monkeypatch.setattr(tools.Path, 'home', staticmethod(lambda: home))
    monkeypatch.setattr(
        tools.shutil, 'which',
        fake_which({'defuddle': '/usr/bin/defuddle', 'crwl': '/usr/bin/crwl'}),
    )
    monkeypatch.setattr(
        tools.subprocess, 'run',
        fake_run({'defuddle': subprocess.CompletedProcess([], 0, '0.19.2\n', '')}, []),
    )

    statuses = check_page_readers()

    assert [s.spec.name for s in statuses] == ['crwl', 'defuddle']
    assert all(s.available for s in statuses)


def test_check_page_readers_blames_the_browser_not_crwl(monkeypatch, tmp_path):
    """A `crwl` without a Chromium fails on every page; the report must say so."""
    home = make_cache(tmp_path, with_chromium=False)
    monkeypatch.setattr(tools.sys, 'platform', 'linux')
    monkeypatch.setattr(tools.Path, 'home', staticmethod(lambda: home))
    monkeypatch.setattr(
        tools.shutil, 'which',
        fake_which({'defuddle': '/usr/bin/defuddle', 'crwl': '/usr/bin/crwl'}),
    )
    monkeypatch.setattr(
        tools.subprocess, 'run',
        fake_run({'defuddle': subprocess.CompletedProcess([], 0, '0.19.2\n', '')}, []),
    )

    statuses = check_page_readers()
    crwl_status = next(s for s in statuses if s.spec.name == 'crwl')

    assert not crwl_status.available
    assert crwl_status.path == '/usr/bin/crwl'
    assert any('Chromium' in problem for problem in crwl_status.problems)


def test_page_readers_order_puts_crwl_first():
    """
    crwl is the documented default because it is the only reader that works everywhere.

    The order is what the skill's tool table is written against, so a change here is a
    change to the guidance and should have to be deliberate.
    """
    assert [spec.name for spec in PAGE_READERS] == ['crwl', 'defuddle']


def test_crwl_is_described_as_the_default_and_defuddle_as_the_fallback():
    """The status report is where a reader learns which to reach for first."""
    purposes = {spec.name: spec.purpose for spec in PAGE_READERS}

    assert 'default' in purposes['crwl']
    assert 'fallback' in purposes['defuddle']


# -- install_tool --

def test_install_tool_reports_a_missing_package_manager(monkeypatch):
    monkeypatch.setattr(tools.shutil, 'which', fake_which({}))

    with pytest.raises(EnvironmentSetupError) as exc:
        install_tool(DEFUDDLE)

    assert 'npm' in str(exc.value)


def test_install_tool_runs_the_documented_command(monkeypatch):
    calls: List[List[str]] = []
    monkeypatch.setattr(tools.shutil, 'which', fake_which({'npm': '/usr/bin/npm'}))
    monkeypatch.setattr(tools.subprocess, 'run', fake_run({}, calls))

    install_tool(DEFUDDLE)

    assert calls == [['npm', 'install', '-g', 'defuddle']]


def test_install_tool_raises_when_the_install_fails(monkeypatch):
    monkeypatch.setattr(tools.shutil, 'which', fake_which({'npm': '/usr/bin/npm'}))
    monkeypatch.setattr(
        tools.subprocess, 'run',
        fake_run({'npm': subprocess.CompletedProcess([], 1, '', '')}, []),
    )

    with pytest.raises(EnvironmentSetupError) as exc:
        install_tool(DEFUDDLE)

    assert 'npm install -g defuddle' in str(exc.value)


# -- install_chromium --

def test_install_chromium_prefers_crawl4ai_setup(monkeypatch):
    calls: List[List[str]] = []
    monkeypatch.setattr(
        tools.shutil, 'which',
        fake_which({'crawl4ai-setup': '/usr/bin/crawl4ai-setup', 'playwright': '/usr/bin/playwright'}),
    )
    monkeypatch.setattr(tools.subprocess, 'run', fake_run({}, calls))

    install_chromium()

    assert calls == [['crawl4ai-setup']]


def test_install_chromium_falls_back_to_playwright(monkeypatch):
    calls: List[List[str]] = []
    monkeypatch.setattr(
        tools.shutil, 'which', fake_which({'playwright': '/usr/bin/playwright'}),
    )
    monkeypatch.setattr(tools.subprocess, 'run', fake_run({}, calls))

    install_chromium()

    assert calls == [['playwright', 'install', 'chromium']]


def test_install_chromium_raises_when_no_installer_exists(monkeypatch):
    monkeypatch.setattr(tools.shutil, 'which', fake_which({}))

    with pytest.raises(EnvironmentSetupError) as exc:
        install_chromium()

    assert 'crawl4ai' in str(exc.value)


# -- install_missing --

def test_install_missing_leaves_available_tools_alone(monkeypatch, tmp_path):
    calls: List[List[str]] = []
    home = make_cache(tmp_path, with_chromium=True)
    monkeypatch.setattr(tools.sys, 'platform', 'linux')
    monkeypatch.setattr(tools.Path, 'home', staticmethod(lambda: home))
    monkeypatch.setattr(
        tools.shutil, 'which',
        fake_which({'defuddle': '/usr/bin/defuddle', 'crwl': '/usr/bin/crwl'}),
    )
    monkeypatch.setattr(
        tools.subprocess, 'run',
        fake_run({'defuddle': subprocess.CompletedProcess([], 0, '0.19.2\n', '')}, calls),
    )

    statuses = install_missing(check_page_readers())

    assert all(s.available for s in statuses)
    # Only version probes, no installs.
    assert all(call[0] == 'defuddle' for call in calls)


def test_install_missing_installs_the_browser_for_a_present_crwl(monkeypatch, tmp_path):
    """crwl on PATH with no browser needs `crawl4ai-setup`, not another crawl4ai."""
    calls: List[List[str]] = []
    home = make_cache(tmp_path, with_chromium=False)
    monkeypatch.setattr(tools.sys, 'platform', 'linux')
    monkeypatch.setattr(tools.Path, 'home', staticmethod(lambda: home))
    monkeypatch.setattr(
        tools.shutil, 'which',
        fake_which({
            'defuddle': '/usr/bin/defuddle',
            'crwl': '/usr/bin/crwl',
            'crawl4ai-setup': '/usr/bin/crawl4ai-setup',
        }),
    )
    monkeypatch.setattr(
        tools.subprocess, 'run',
        fake_run({'defuddle': subprocess.CompletedProcess([], 0, '0.19.2\n', '')}, calls),
    )

    install_missing(check_page_readers())

    install_calls = [call for call in calls if call[0] != 'defuddle']
    assert install_calls == [['crawl4ai-setup']]


def test_install_missing_installs_crawl4ai_then_its_browser(monkeypatch, tmp_path):
    """A freshly installed crawl4ai has no browser, so both steps must run - once each."""
    calls: List[List[str]] = []
    home = make_cache(tmp_path, with_chromium=False)
    resolvable = {'defuddle': '/usr/bin/defuddle', 'uv': '/usr/bin/uv'}

    monkeypatch.setattr(tools.sys, 'platform', 'linux')
    monkeypatch.setattr(tools.Path, 'home', staticmethod(lambda: home))
    monkeypatch.setattr(tools.shutil, 'which', lambda name: resolvable.get(name))

    def _run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        if cmd[0] == 'uv':
            # crawl4ai is on PATH from here on, but still has no browser.
            resolvable['crwl'] = '/usr/bin/crwl'
            resolvable['crawl4ai-setup'] = '/usr/bin/crawl4ai-setup'
        return subprocess.CompletedProcess(
            args=cmd, returncode=0,
            stdout='0.19.2\n' if cmd[0] == 'defuddle' else '', stderr='',
        )

    monkeypatch.setattr(tools.subprocess, 'run', _run)

    install_missing(check_page_readers())

    install_calls = [call for call in calls if call[0] not in ('defuddle',)]
    assert install_calls == [
        ['uv', 'tool', 'install', 'crawl4ai'],
        ['crawl4ai-setup'],
    ]


def test_install_missing_does_not_download_the_browser_twice(monkeypatch, tmp_path):
    """The browser never appears in the cache here, so a naive retry would repeat it."""
    calls: List[List[str]] = []
    home = make_cache(tmp_path, with_chromium=False)
    monkeypatch.setattr(tools.sys, 'platform', 'linux')
    monkeypatch.setattr(tools.Path, 'home', staticmethod(lambda: home))
    monkeypatch.setattr(
        tools.shutil, 'which',
        fake_which({
            'defuddle': '/usr/bin/defuddle',
            'crwl': '/usr/bin/crwl',
            'crawl4ai-setup': '/usr/bin/crawl4ai-setup',
        }),
    )
    monkeypatch.setattr(
        tools.subprocess, 'run',
        fake_run({'defuddle': subprocess.CompletedProcess([], 0, '0.19.2\n', '')}, calls),
    )

    statuses = install_missing(check_page_readers())

    assert calls.count(['crawl4ai-setup']) == 1
    # And the caller is told it is still not usable rather than being reassured.
    assert not next(s for s in statuses if s.spec.name == 'crwl').available
