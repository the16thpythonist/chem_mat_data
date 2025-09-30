
import os
import tempfile
import zipfile
import shutil
from pathlib import Path

import requests
from tqdm import tqdm


class AbsractOnlineSource:
    """
    Abstract base class for online data sources.
    """
    
    def prepare(self):
        """
        Optional steps that may be required to setup the online source such as 
        authentication etc.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def fetch(self) -> str:
        """
        Optional steps to fetch data from the online source. Should return the path
        to the downloaded data.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    
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
        self.temp_dir = tempfile.mkdtemp(prefix="zenodo_")

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
        zip_path = os.path.join(self.temp_dir, "download.zip")

        response = requests.get(self.url, stream=True)
        response.raise_for_status()

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract the ZIP file
        extract_dir = os.path.join(self.temp_dir, "extracted")
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
                    target_path = os.path.join(root, os.path.basename(self.relative_path))
                    break

        if not os.path.exists(target_path):
            raise ValueError(f"File '{self.relative_path}' not found in the downloaded archive")

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
        self.temp_dir = tempfile.mkdtemp(prefix="file_download_")

    def fetch(self) -> str:
        """
        Download the file from the URL.

        :return: Path to the downloaded file.
        :raises: Exception if download fails.
        """
        if self.temp_dir is None:
            self.prepare()

        # Extract filename from URL, or use a default name
        filename = os.path.basename(self.url.split('?')[0]) or "downloaded_file"
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
                verify=self.ssl_verify
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download file from {self.url}: {e}")

        # Get file size from headers for progress bar
        total_size = int(response.headers.get('content-length', 0))

        try:
            if self.verbose and total_size > 0:
                # Use tqdm progress bar for verbose downloads
                with open(self.downloaded_file_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
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
            raise Exception(f"Failed to write downloaded file: {e}")
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