# PEP 563: keep annotations as strings so that PEP 604 union syntax (e.g. ``int | None``)
# used in some signatures is not evaluated at runtime, which would otherwise raise a
# TypeError on Python 3.9 (where ``type | None`` is not yet supported at runtime).
from __future__ import annotations

import os
import re
import time
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

    Example:

    .. code-block:: python

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

    :param file_id: Figshare numeric file ID (mutually exclusive with *url*).
    :param url: Full ndownloader URL (mutually exclusive with *file_id*).
    :param relative_path: If given, treat the download as a ZIP archive and extract
        this path from it. If ``None``, return the raw downloaded file.
    :param verbose: Show a tqdm progress bar during download.
    :param max_retries: Number of retry attempts when Figshare returns HTTP 202.
    :param retry_delay: Seconds to wait between 202 retries.
    """

    # Regex to extract the numeric file ID from various Figshare ndownloader URL styles.
    _NDOWNLOADER_RE = re.compile(
        r'https?://(?:[\w.-]*\.)?figshare\.com/ndownloader/files/(\d+)'
    )

    def __init__(
        self,
        file_id: int | None = None,
        url: str | None = None,
        relative_path: str | None = None,
        verbose: bool = False,
        max_retries: int = 4,
        retry_delay: float = 5.0,
    ):
        if file_id is None and url is None:
            raise ValueError('Either file_id or url must be provided')
        if file_id is not None and url is not None:
            raise ValueError('Provide file_id or url, not both')

        if url is not None:
            m = self._NDOWNLOADER_RE.match(url)
            if m:
                file_id = int(m.group(1))
            else:
                # Accept the URL as-is but warn the user
                self._raw_url = url
                file_id = None

        if file_id is not None:
            self._raw_url = f'https://ndownloader.figshare.com/files/{file_id}'

        self.file_id = file_id
        self.url = self._raw_url
        self.relative_path = relative_path
        self.verbose = verbose
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temp_dir = None

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
        filename = os.path.basename(self.url.split('?')[0]) or 'downloaded_file'
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
