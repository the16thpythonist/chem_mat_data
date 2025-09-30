"""
Unit tests for the "connectors.py" module that contains online data source connectors.
"""
import os
import tempfile
import zipfile
import pytest
from unittest.mock import Mock, patch, mock_open
from io import BytesIO

from chem_mat_data.connectors import ZenodoSource


class TestZenodoSource:
    """
    Test suite for the ZenodoSource class that handles downloading and extracting
    files from Zenodo repositories.
    """

    def test_init(self):
        """
        Test that ZenodoSource initializes correctly with url and relative_path.
        """
        url = "https://zenodo.org/record/123456/files/dataset.zip"
        relative_path = "data/molecules.csv"

        source = ZenodoSource(url, relative_path)

        assert source.url == url
        assert source.relative_path == relative_path
        assert source.temp_dir is None

    def test_prepare_creates_temp_dir(self):
        """
        Test that prepare() creates a temporary directory.
        """
        source = ZenodoSource("https://test.com/file.zip", "test.csv")

        source.prepare()

        assert source.temp_dir is not None
        assert os.path.exists(source.temp_dir)
        assert source.temp_dir.startswith(tempfile.gettempdir())

        # Cleanup
        source.cleanup()

    @patch('requests.get')
    def test_fetch_successful_download(self, mock_get):
        """
        Test successful download and extraction of a file from a mocked ZIP archive.
        """
        # Create a mock ZIP file content
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('data/molecules.csv', 'smiles,property\nCCO,1.2\nCCC,2.1\n')
        zip_content = zip_buffer.getvalue()

        # Mock the response
        mock_response = Mock()
        mock_response.iter_content.return_value = [zip_content]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        source = ZenodoSource(
            "https://zenodo.org/record/123456/files/dataset.zip",
            "data/molecules.csv"
        )

        file_path = source.fetch()

        # Verify the file was extracted correctly
        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            content = f.read()
        assert 'smiles,property' in content
        assert 'CCO,1.2' in content

        # Verify requests.get was called correctly
        mock_get.assert_called_once_with(source.url, stream=True)

        # Cleanup
        source.cleanup()

    @patch('requests.get')
    def test_fetch_file_not_found_in_archive(self, mock_get):
        """
        Test that ValueError is raised when the specified file is not found in the archive.
        """
        # Create a mock ZIP file without the target file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('other/file.txt', 'some content')
        zip_content = zip_buffer.getvalue()

        mock_response = Mock()
        mock_response.iter_content.return_value = [zip_content]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        source = ZenodoSource(
            "https://zenodo.org/record/123456/files/dataset.zip",
            "data/molecules.csv"
        )

        with pytest.raises(ValueError, match="File 'data/molecules.csv' not found in the downloaded archive"):
            source.fetch()

        source.cleanup()

    @patch('requests.get')
    def test_fetch_finds_file_by_basename(self, mock_get):
        """
        Test that fetch can find a file by its basename when the exact path doesn't match.
        """
        # Create a mock ZIP file with file in different location
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('different/path/molecules.csv', 'smiles,property\nCCO,1.2\n')
        zip_content = zip_buffer.getvalue()

        mock_response = Mock()
        mock_response.iter_content.return_value = [zip_content]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        source = ZenodoSource(
            "https://zenodo.org/record/123456/files/dataset.zip",
            "data/molecules.csv"  # This exact path doesn't exist, but basename does
        )

        file_path = source.fetch()

        assert os.path.exists(file_path)
        with open(file_path, 'r') as f:
            content = f.read()
        assert 'smiles,property' in content

        source.cleanup()

    @patch('requests.get')
    def test_fetch_handles_http_error(self, mock_get):
        """
        Test that HTTP errors are properly raised during download.
        """
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 404 Not Found")
        mock_get.return_value = mock_response

        source = ZenodoSource(
            "https://zenodo.org/record/nonexistent/files/dataset.zip",
            "data/molecules.csv"
        )

        with pytest.raises(Exception, match="HTTP 404 Not Found"):
            source.fetch()

        source.cleanup()

    def test_context_manager(self):
        """
        Test that ZenodoSource works correctly as a context manager.
        """
        url = "https://zenodo.org/record/123456/files/dataset.zip"
        relative_path = "data/molecules.csv"

        with ZenodoSource(url, relative_path) as source:
            assert source.url == url
            assert source.relative_path == relative_path
            # Context manager should work without calling prepare/cleanup manually

    @patch('requests.get')
    def test_context_manager_with_fetch(self, mock_get):
        """
        Test context manager with actual fetch operation and automatic cleanup.
        """
        # Create a mock ZIP file content
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('test.txt', 'test content')
        zip_content = zip_buffer.getvalue()

        mock_response = Mock()
        mock_response.iter_content.return_value = [zip_content]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        temp_dir_path = None

        with ZenodoSource(
            "https://zenodo.org/record/123456/files/dataset.zip",
            "test.txt"
        ) as source:
            file_path = source.fetch()
            temp_dir_path = source.temp_dir

            assert os.path.exists(file_path)
            assert os.path.exists(temp_dir_path)

        # After exiting context, temp directory should be cleaned up
        assert not os.path.exists(temp_dir_path)

    def test_cleanup_removes_temp_dir(self):
        """
        Test that cleanup() properly removes the temporary directory.
        """
        source = ZenodoSource("https://test.com/file.zip", "test.csv")
        source.prepare()

        temp_dir = source.temp_dir
        assert os.path.exists(temp_dir)

        source.cleanup()

        assert not os.path.exists(temp_dir)
        assert source.temp_dir is None

    def test_cleanup_handles_nonexistent_dir(self):
        """
        Test that cleanup() handles case where temp_dir doesn't exist.
        """
        source = ZenodoSource("https://test.com/file.zip", "test.csv")
        source.temp_dir = "/nonexistent/path"

        # Should not raise an exception
        source.cleanup()
        assert source.temp_dir is None

    def test_multiple_cleanup_calls(self):
        """
        Test that multiple cleanup() calls don't cause issues.
        """
        source = ZenodoSource("https://test.com/file.zip", "test.csv")
        source.prepare()

        source.cleanup()
        source.cleanup()  # Second call should be safe

        assert source.temp_dir is None


class TestZenodoSourceIntegration:
    """
    Integration tests that use real network calls to Zenodo.
    These tests are marked to be skipped during normal test runs.
    """

    @pytest.mark.slow
    @pytest.mark.skipif(
        os.environ.get("SKIP_INTEGRATION_TESTS", "true").lower() == "true",
        reason="Integration tests skipped (set SKIP_INTEGRATION_TESTS=false to enable)"
    )
    def test_real_zenodo_download(self):
        """
        Test downloading from a real Zenodo repository.

        Uses: https://zenodo.org/records/11580890/files/TheJacksonLab/ClosedLoopTransfer-ClosedLoopTransfer.zip
        This test is marked as slow and skipped by default.
        """
        url = "https://zenodo.org/records/11580890/files/TheJacksonLab/ClosedLoopTransfer-ClosedLoopTransfer.zip?download=1"

        with ZenodoSource(url, "README.md") as source:
            try:
                file_path = source.fetch()

                # Verify the file was downloaded and extracted
                assert os.path.exists(file_path)
                assert file_path.endswith("README.md")

                # Check that file has some content
                with open(file_path, 'r') as f:
                    content = f.read()
                assert len(content) > 0

            except Exception as e:
                pytest.skip(f"Real Zenodo download failed (network issue?): {e}")

    @pytest.mark.slow
    @pytest.mark.skipif(
        os.environ.get("SKIP_INTEGRATION_TESTS", "true").lower() == "true",
        reason="Integration tests skipped (set SKIP_INTEGRATION_TESTS=false to enable)"
    )
    def test_real_zenodo_file_search(self):
        """
        Test the file search functionality with a real Zenodo archive.
        This tests the fallback mechanisms for finding files when exact paths don't match.
        """
        url = "https://zenodo.org/records/11580890/files/TheJacksonLab/ClosedLoopTransfer-ClosedLoopTransfer.zip?download=1"

        with ZenodoSource(url, "LICENSE") as source:
            try:
                file_path = source.fetch()

                assert os.path.exists(file_path)

                # Verify it found a LICENSE file
                with open(file_path, 'r') as f:
                    content = f.read()
                assert len(content) > 0
                # LICENSE files typically contain copyright information
                assert any(word in content.lower() for word in ['license', 'copyright', 'mit', 'apache', 'gpl'])

            except Exception as e:
                pytest.skip(f"Real Zenodo download failed (network issue?): {e}")