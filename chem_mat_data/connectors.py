# PEP 563: keep annotations as strings so that PEP 604 union syntax (e.g. ``int | None``)
# used in some signatures is not evaluated at runtime, which would otherwise raise a
# TypeError on Python 3.9 (where ``type | None`` is not yet supported at runtime).
from __future__ import annotations

import os
import re
import time
import fnmatch
import tempfile
import zipfile
import shutil

import requests
from tqdm import tqdm

#: User-Agent sent with requests to Figshare. Figshare's bot protection answers clients
#: that claim to be a web browser - but cannot run its JavaScript challenge - with HTTP
#: 202 ("file is being prepared") on every attempt, so imitating a browser makes the
#: download impossible. An honest tool identifier is served the file directly.
USER_AGENT: str = 'chem_mat_data (+https://github.com/the16thpythonist/chem_mat_data)'


class AbsractOnlineSource:
    """
    Abstract base class for online data sources.
    """

    def prepare(self):
        """
        Optional steps that may be required to setup the online source such as
        authentication etc.
        """
        raise NotImplementedError('Subclasses must implement this method')

    def fetch(self) -> str:
        """
        Optional steps to fetch data from the online source. Should return the path
        to the downloaded data.
        """
        raise NotImplementedError('Subclasses must implement this method')


class ZenodoSource(AbsractOnlineSource):
    """
    Data source from a Zenodo repository.

    Downloads a ZIP file from Zenodo and extracts a specific file from it.
    Supports usage as a context manager for automatic cleanup.

    Example:

    .. code-block:: python

        # Basic usage
        source = ZenodoSource(
            url="https://zenodo.org/record/123456/files/dataset.zip",
            relative_path="data/molecules.csv"
        )
        file_path = source.fetch()
        # Remember to call source.cleanup() when done

        # Context manager usage (recommended)
        with ZenodoSource(
            url="https://zenodo.org/record/123456/files/dataset.zip",
            relative_path="data/molecules.csv"
        ) as source:
            file_path = source.fetch()
            # Automatic cleanup on exit

    :param url: URL to the Zenodo file (typically a ZIP archive).
    :param relative_path: Relative path to the target file within the archive.
    """

    def __init__(self, url: str, relative_path: str):
        """
        Initialize the ZenodoSource.

        :param url: URL to the Zenodo file to download.
        :param relative_path: Relative path to extract from the archive.
        """
        self.url = url
        self.relative_path = relative_path
        self.temp_dir = None

    def prepare(self):
        """
        Create temporary directory for extraction.

        No authentication is required for public Zenodo repositories.
        """
        self.temp_dir = tempfile.mkdtemp(prefix='zenodo_')

    def fetch(self) -> str:
        """
        Download ZIP file from Zenodo and extract the specified file.

        :return: Path to the extracted file.
        :raises: ValueError if the specified file is not found in the archive.
        :raises: Exception if download or extraction fails.
        """
        if self.temp_dir is None:
            self.prepare()

        # Download the ZIP file
        zip_path = os.path.join(self.temp_dir, 'download.zip')

        response = requests.get(self.url, stream=True)
        response.raise_for_status()

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract the ZIP file
        extract_dir = os.path.join(self.temp_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find the target file
        target_path = os.path.join(extract_dir, self.relative_path)

        # Handle case where relative_path might be relative to any subdirectory
        if not os.path.exists(target_path):
            # Search for the file in all extracted directories
            for root, dirs, files in os.walk(extract_dir):
                potential_path = os.path.join(root, self.relative_path)
                if os.path.exists(potential_path):
                    target_path = potential_path
                    break

                # Also check if the relative_path basename exists
                if os.path.basename(self.relative_path) in files:
                    target_path = os.path.join(
                        root, os.path.basename(self.relative_path)
                    )
                    break

        if not os.path.exists(target_path):
            raise ValueError(
                f"File '{self.relative_path}' not found in the downloaded archive"
            )

        return target_path

    def cleanup(self):
        """
        Clean up temporary files and directories.

        Should be called after the fetched file is no longer needed.
        """
        if self.temp_dir:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            self.temp_dir = None

    def __enter__(self):
        """
        Context manager entry.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - cleanup temporary files.
        """
        self.cleanup()

    def __del__(self):
        """
        Cleanup on object destruction.
        """
        try:
            self.cleanup()
        except:
            pass


class FigshareSource(AbsractOnlineSource):
    """
    Data source for downloading files from Figshare repositories.

    Figshare has several quirks that make a dedicated source useful:

    - **A paper links to the article, not to the file.** Data availability sections cite
      the DOI or the landing page, and neither can be downloaded: the request succeeds
      and hands back the HTML page. Only per-file ``ndownloader`` URLs are fetchable, and
      their ids appear nowhere in the citation. This class resolves an article or a DOI
      through the REST API and picks the file out of it.
    - The ``ndownloader`` URLs on some subdomains (e.g. ``springernature.figshare.com``)
      return HTTP 202 ("Accepted — file is being prepared") instead of the file content.
      This class normalises the URL to the canonical ``ndownloader.figshare.com`` host
      and retries on 202 responses.
    - Figshare's bot protection holds any client that claims to be a web browser at
      HTTP 202 indefinitely, so retrying cannot help. Requests therefore identify as
      chem_mat_data (:data:`USER_AGENT`) rather than imitating a browser.
    - Many Figshare datasets are distributed as ZIP archives. When ``relative_path`` is
      given, the archive is automatically extracted and only the requested file path is
      returned — similar to :class:`ZenodoSource`.

    An article usually holds several files, often with identical columns, so choosing
    the wrong one produces a valid dataset of the wrong thing. This class therefore
    refuses to guess: if ``name`` is not given and the article holds more than one file,
    the error lists what is actually there.

    Example:

    .. code-block:: python

        # The usual starting point: the DOI a paper cites, plus the file you want
        with FigshareSource(
            url='https://doi.org/10.6084/m9.figshare.25993060',
            name='QM40_main.csv',
        ) as src:
            path = src.fetch()

        # The article landing page works the same way, and `name` accepts a glob
        with FigshareSource(
            url='https://figshare.com/articles/dataset/QM40/25993060',
            name='*main*',
        ) as src:
            path = src.fetch()

        # Not sure what is in there? Ask, without downloading anything
        for entry in FigshareSource(article_id=25993060).list_files():
            print(entry['name'], entry['size'])

        # Download a single file by Figshare file ID
        with FigshareSource(file_id=1503990) as src:
            path = src.fetch()  # returns path to downloaded file

        # Download a ZIP and extract a specific CSV from it
        with FigshareSource(
            file_id=25766582,
            relative_path='PFAS-Map/test_data/PFAS_TTR_Binding.csv',
        ) as src:
            csv_path = src.fetch()

        # Alternatively pass a full ndownloader URL instead of a file_id
        with FigshareSource(
            url='https://ndownloader.figshare.com/files/25766582',
            relative_path='PFAS-Map/processed_data/Precalculated_EPA_PFAS_pca_classification.csv',
        ) as src:
            csv_path = src.fetch()

    :param file_id: Figshare numeric file ID.
    :param url: Any Figshare link - article page, DOI or per-file ndownloader URL.
    :param article_id: Figshare numeric article ID, as an alternative to *url*.
    :param version: Article version to resolve. Defaults to the latest.
    :param name: Which file to take from a multi-file article. Matched against the
        file names exactly, then case-insensitively, then as a glob, then as a
        substring - the first stage that yields exactly one file wins.
    :param relative_path: If given, treat the download as a ZIP archive and extract
        this path from it. If ``None``, return the raw downloaded file.
    :param verbose: Show a tqdm progress bar during download.
    :param max_retries: Number of retry attempts when Figshare returns HTTP 202.
    :param retry_delay: Seconds to wait between 202 retries.
    """

    #: Public REST API. Institutional portals (``springernature.figshare.com`` and
    #: friends) publish their articles through this same endpoint, so one base works
    #: for every host we have needed so far.
    API_BASE = 'https://api.figshare.com/v2'

    # Regex to extract the numeric file ID from various Figshare ndownloader URL styles.
    # Both spellings occur and both must match: the canonical host puts the id straight
    # after /files/ (``ndownloader.figshare.com/files/123``) while institutional portals
    # keep it under a path segment (``springernature.figshare.com/ndownloader/files/123``).
    # The original pattern required the path segment, so the canonical form - the one in
    # this class's own docstring - fell through to the "accept as-is" branch instead.
    _NDOWNLOADER_RE = re.compile(
        r'https?://(?:[\w.-]*\.)?figshare\.com/(?:ndownloader/)?files/(\d+)'
    )

    #: An article landing page: ``.../articles/<type>/<slug>/<id>`` with an optional
    #: trailing version. The type segment ("dataset", "journal_contribution", ...) and
    #: the slug are both optional, because Figshare serves short forms of this URL too.
    _ARTICLE_RE = re.compile(
        r'https?://(?:[\w.-]*\.)?figshare\.com/articles/'
        r'(?:[\w_]+/)?(?:[^/]+/)?(\d+)(?:/(\d+))?'
    )

    #: A Figshare DOI, bare or as a doi.org link. The ``.v2`` suffix is the version.
    _DOI_RE = re.compile(
        r'(?:https?://(?:dx\.)?doi\.org/)?10\.6084/m9\.figshare\.(\d+)(?:\.v(\d+))?'
    )

    def __init__(
        self,
        file_id: int | None = None,
        url: str | None = None,
        article_id: int | None = None,
        version: int | None = None,
        name: str | None = None,
        relative_path: str | None = None,
        verbose: bool = False,
        max_retries: int = 4,
        retry_delay: float = 5.0,
    ):
        if file_id is None and url is None and article_id is None:
            raise ValueError('One of file_id, article_id or url must be provided')
        if file_id is not None and url is not None:
            raise ValueError('Provide file_id or url, not both')
        if article_id is not None and url is not None:
            raise ValueError('Provide article_id or url, not both')

        if url is not None:
            file_id, article_id, version = self._parse_link(url, version)

        self.file_id = file_id
        self.article_id = article_id
        self.version = version
        self.name = name
        self.relative_path = relative_path
        self.verbose = verbose
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temp_dir = None

        #: Set once an article has been resolved, so the real file name can be used on
        #: disk instead of the numeric id that the ndownloader URL ends in.
        self.filename: str | None = None

        self.url = (
            f'https://ndownloader.figshare.com/files/{file_id}'
            if file_id is not None
            else None
        )

    @classmethod
    def _parse_link(cls, url: str, version: int | None = None) -> tuple:
        """
        Work out what a Figshare link actually points at.

        Three things get called "the figshare link" in a data availability section: the
        article page, the DOI, and the per-file download URL. Only the last can be
        fetched directly. The first two have to be resolved through the API, and the
        failure mode when they are not is silent - the request succeeds and returns the
        HTML landing page, which then either fails deep inside a zip reader or, worse,
        is handed back as though it were the data.

        :param url: Any Figshare link.
        :param version: Explicit version, which wins over one found in the URL.

        :returns: Tuple of ``(file_id, article_id, version)``, two of which are None.

        :raises ValueError: If the link is not a recognised Figshare address, rather
            than downloading whatever happens to be at the other end.
        """
        match = cls._NDOWNLOADER_RE.match(url)
        if match:
            return int(match.group(1)), None, None

        match = cls._DOI_RE.match(url)
        if match:
            found = int(match.group(2)) if match.group(2) else None
            return None, int(match.group(1)), version or found

        match = cls._ARTICLE_RE.match(url)
        if match:
            found = int(match.group(2)) if match.group(2) else None
            return None, int(match.group(1)), version or found

        raise ValueError(
            f'{url!r} is not a recognised Figshare link. Expected an article page '
            '(https://figshare.com/articles/dataset/<slug>/<id>), a DOI '
            '(10.6084/m9.figshare.<id>) or a file URL '
            '(https://ndownloader.figshare.com/files/<id>). Downloading an '
            'unrecognised URL would return the HTML landing page rather than the data; '
            'use FileDownloadSource if that is really what you want.'
        )

    def list_files(self) -> list:
        """
        Ask the API which files the article holds.

        :returns: List of the API's file records, each with ``id``, ``name``, ``size``
            and ``download_url``.

        :raises ValueError: If no article has been identified.
        """
        if self.article_id is None:
            raise ValueError('no article to list - this source points at a single file')

        endpoint = f'{self.API_BASE}/articles/{self.article_id}'
        if self.version is not None:
            endpoint += f'/versions/{self.version}'

        headers = {'User-Agent': USER_AGENT}
        response = requests.get(endpoint, headers=headers, timeout=(10, 60))
        response.raise_for_status()
        payload = response.json()

        # The article endpoint embeds the file list; the versioned one does too. Fall
        # back to the dedicated files endpoint if a future response drops it.
        files = payload.get('files')
        if files is None:
            response = requests.get(f'{endpoint}/files', headers=headers, timeout=(10, 60))
            response.raise_for_status()
            files = response.json()

        return files or []

    def _select_file(self, files: list) -> dict:
        """
        Pick the requested file out of an article, or refuse to guess.

        Choosing wrongly here is quiet and expensive: articles routinely ship several
        files with the same columns, so the wrong one yields a perfectly valid dataset
        of the wrong thing. When the choice is not determined, the error carries the
        actual file listing so the caller can make it without a second round trip.

        :param files: File records as returned by the API.

        :returns: The single matching record.

        :raises ValueError: If nothing matches, or more than one thing does.
        """
        if not files:
            raise ValueError(f'article {self.article_id} lists no files')

        if self.name is None:
            if len(files) == 1:
                return files[0]
            raise ValueError(
                f'article {self.article_id} holds {len(files)} files; '
                f'pass name= to pick one\n' + self._render_listing(files)
            )

        names = [str(f.get('name', '')) for f in files]
        wanted = self.name
        strategies = (
            [i for i, n in enumerate(names) if n == wanted],
            [i for i, n in enumerate(names) if n.lower() == wanted.lower()],
            [i for i, n in enumerate(names) if fnmatch.fnmatch(n.lower(), wanted.lower())],
            [i for i, n in enumerate(names) if wanted.lower() in n.lower()],
        )
        for matched in strategies:
            if len(matched) == 1:
                return files[matched[0]]
            if len(matched) > 1:
                raise ValueError(
                    f'name={wanted!r} matches {len(matched)} files in article '
                    f'{self.article_id}; be more specific\n'
                    + self._render_listing([files[i] for i in matched])
                )

        raise ValueError(
            f'no file named {wanted!r} in article {self.article_id}\n'
            + self._render_listing(files)
        )

    @staticmethod
    def _render_listing(files: list) -> str:
        """Format a file listing for an error message, names padded and sizes human."""
        width = max((len(str(f.get('name', ''))) for f in files), default=0)
        lines = []
        for entry in files:
            size = entry.get('size') or 0
            for unit in ('B', 'KB', 'MB', 'GB'):
                if size < 1024 or unit == 'GB':
                    break
                size /= 1024
            lines.append(f'  {str(entry.get("name", "")):<{width}}  {size:.1f} {unit}')
        return '\n'.join(lines)

    def resolve(self) -> str:
        """
        Turn whatever was passed in into a directly downloadable file URL.

        Safe to call more than once; the API is only consulted the first time.

        :returns: The ndownloader URL for the selected file.
        """
        if self.url is not None:
            return self.url

        selected = self._select_file(self.list_files())
        self.file_id = selected.get('id')
        self.filename = selected.get('name') or None
        self.url = selected.get('download_url') or (
            f'https://ndownloader.figshare.com/files/{self.file_id}'
        )
        return self.url

    def prepare(self):
        """Create temporary directory for download and extraction."""
        self.temp_dir = tempfile.mkdtemp(prefix='figshare_')

    def fetch(self) -> str:
        """
        Download the file from Figshare, retrying on HTTP 202 responses.

        If ``relative_path`` was specified, the download is treated as a ZIP
        archive and the requested file is extracted from it.

        :returns: Path to the downloaded (or extracted) file.
        :raises RuntimeError: If Figshare keeps returning 202 after all retries.
        :raises ValueError: If ``relative_path`` is not found inside the archive.
        """
        if self.temp_dir is None:
            self.prepare()

        # An article link has to become a file link before anything can be fetched.
        self.resolve()

        # Identify honestly: a browser User-Agent is held at HTTP 202 forever (see USER_AGENT).
        headers = {'User-Agent': USER_AGENT}

        # -- download with 202 retry logic --
        response = None
        for attempt in range(1, self.max_retries + 1):
            response = requests.get(
                self.url,
                stream=True,
                headers=headers,
                timeout=(10, 300),
                allow_redirects=True,
            )
            if response.status_code != 202:
                break
            response.close()
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)

        if response.status_code == 202:
            raise RuntimeError(
                f'Figshare returned HTTP 202 (file being prepared) after '
                f'{self.max_retries} attempts for {self.url}'
            )
        response.raise_for_status()

        # -- write to disk --
        # An ndownloader URL ends in the numeric file id, so without the API's name the
        # download lands on disk called something like "25993060" with no extension.
        filename = (
            self.filename
            or os.path.basename(self.url.split('?')[0])
            or 'downloaded_file'
        )
        download_path = os.path.join(self.temp_dir, filename)
        total_size = int(response.headers.get('content-length', 0))

        try:
            if self.verbose and total_size > 0:
                with open(download_path, 'wb') as f:
                    with tqdm(
                        total=total_size, unit='B', unit_scale=True, desc=filename
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            else:
                with open(download_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        finally:
            response.close()

        # -- optional ZIP extraction --
        if self.relative_path is None:
            return download_path

        extract_dir = os.path.join(self.temp_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(download_path, 'r') as zf:
            zf.extractall(extract_dir)

        target = os.path.join(extract_dir, self.relative_path)
        if not os.path.exists(target):
            # Search recursively as fallback
            basename = os.path.basename(self.relative_path)
            for root, _dirs, files in os.walk(extract_dir):
                if basename in files:
                    target = os.path.join(root, basename)
                    break

        if not os.path.exists(target):
            raise ValueError(
                f"File '{self.relative_path}' not found in the Figshare archive"
            )

        return target

    def cleanup(self):
        """Remove temporary download and extraction files."""
        if self.temp_dir:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            self.temp_dir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


class FileDownloadSource(AbsractOnlineSource):
    """
    Data source for downloading a single file from a URL.

    Downloads a file from the given URL and returns the local path to it.
    Supports usage as a context manager for automatic cleanup.

    Example:

    .. code-block:: python

        # Basic usage
        source = FileDownloadSource("https://example.com/data.csv")
        file_path = source.fetch()
        # Remember to call source.cleanup() when done

        # Context manager usage (recommended)
        with FileDownloadSource("https://example.com/data.csv") as source:
            file_path = source.fetch()
            # Automatic cleanup on exit

        # With progress bar for large downloads
        with FileDownloadSource("https://example.com/large_file.zip", verbose=True) as source:
            file_path = source.fetch()
            # Shows tqdm progress bar during download

        # Disable SSL verification for self-signed certificates
        with FileDownloadSource("https://self-signed.example.com/data.csv", ssl_verify=False) as source:
            file_path = source.fetch()
            # Downloads without SSL certificate verification

    :param url: URL to the file to download.
    :param verbose: If True, show download progress with tqdm progress bar.
    :param ssl_verify: If False, disable SSL certificate verification (use with caution).
    """

    def __init__(self, url: str, verbose: bool = False, ssl_verify: bool = True):
        """
        Initialize the FileDownloadSource.

        :param url: URL to the file to download.
        :param verbose: If True, show download progress with tqdm progress bar.
        :param ssl_verify: If False, disable SSL certificate verification (use with caution).
        """
        self.url = url
        self.verbose = verbose
        self.ssl_verify = ssl_verify
        self.temp_dir = None
        self.downloaded_file_path = None

    def prepare(self):
        """
        Create temporary directory for download.

        No authentication is required for public file downloads.
        """
        self.temp_dir = tempfile.mkdtemp(prefix='file_download_')

    def fetch(self) -> str:
        """
        Download the file from the URL.

        :return: Path to the downloaded file.
        :raises: Exception if download fails.
        """
        if self.temp_dir is None:
            self.prepare()

        # Extract filename from URL, or use a default name
        filename = os.path.basename(self.url.split('?')[0]) or 'downloaded_file'
        self.downloaded_file_path = os.path.join(self.temp_dir, filename)

        # Set up headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Download the file with timeout and proper headers
        try:
            response = requests.get(
                self.url,
                stream=True,
                headers=headers,
                timeout=(10, 30),  # (connection timeout, read timeout)
                allow_redirects=True,
                verify=self.ssl_verify,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise Exception(f'Failed to download file from {self.url}: {e}')

        # Get file size from headers for progress bar
        total_size = int(response.headers.get('content-length', 0))

        try:
            if self.verbose and total_size > 0:
                # Use tqdm progress bar for verbose downloads
                with open(self.downloaded_file_path, 'wb') as f:
                    with tqdm(
                        total=total_size, unit='B', unit_scale=True, desc=filename
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
            else:
                # Simple download without progress bar
                with open(self.downloaded_file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            # Clean up partial file on error
            if os.path.exists(self.downloaded_file_path):
                os.remove(self.downloaded_file_path)
            raise Exception(f'Failed to write downloaded file: {e}')
        finally:
            response.close()

        return self.downloaded_file_path

    def cleanup(self):
        """
        Clean up temporary files and directories.

        Should be called after the downloaded file is no longer needed.
        """
        if self.temp_dir:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            self.downloaded_file_path = None

    def __enter__(self):
        """
        Context manager entry.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - cleanup temporary files.
        """
        self.cleanup()

    def __del__(self):
        """
        Cleanup on object destruction.
        """
        try:
            self.cleanup()
        except:
            pass
