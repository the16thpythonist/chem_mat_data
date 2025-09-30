import os
import tempfile
import unittest
from unittest import TestCase

import rdkit.Chem as Chem
from chem_mat_data.connectors import FileDownloadSource
from chem_mat_data.data import HOPV15Parser


class TestHOPV15FigshareDownload(TestCase):
    """
    Test suite for downloading HOPV15 dataset from Figshare and applying the HOPV15Parser.

    This test validates the complete workflow of downloading the actual HOPV15 dataset
    from its Figshare source and successfully parsing it using the HOPV15Parser.
    """

    def test_download_and_parse_hopv15_from_figshare(self):
        """
        Test downloading the HOPV15 dataset from Figshare and parsing it with the updated HOPV15Parser.

        This test:
        1. Downloads the HOPV15 .data file from the official Figshare URL
        2. Applies the new 2-step HOPV15Parser to parse the first molecule directly
        3. Validates that molecules are correctly parsed with expected data structure
        """
        # Figshare URL for HOPV15 dataset
        figshare_url = 'https://figshare.com/ndownloader/files/4513735'

        # Use context manager for automatic cleanup
        with FileDownloadSource(
            figshare_url,
            verbose=True,
            ssl_verify=False,
        ) as source:
            # Download the file
            downloaded_path = source.fetch()

            # Verify file was downloaded
            self.assertTrue(os.path.exists(downloaded_path))
            self.assertTrue(os.path.isfile(downloaded_path))

            # Check basic file properties
            with open(downloaded_path, 'r') as f:
                content = f.read()
                self.assertGreater(len(content), 1000000, "File should be substantial size")

                lines = content.split('\n')
                self.assertGreater(len(lines), 100000, "File should have many lines")

            # Parse the first molecule directly using the updated parser
            parser = HOPV15Parser(path=downloaded_path)
            mol, info = parser.parse()

            # Validate molecule structure
            self.assertIsInstance(mol, Chem.Mol)
            self.assertIsNotNone(mol)
            self.assertGreater(mol.GetNumAtoms(), 0)

            # Validate info dictionary
            self.assertIsInstance(info, dict)
            self.assertIn('smiles', info)

            # Validate SMILES string
            self.assertIsInstance(info['smiles'], str)
            self.assertGreater(len(info['smiles']), 0)

            # Test that RDKit molecule can be converted back to SMILES
            parsed_smiles = Chem.MolToSmiles(mol)
            self.assertIsInstance(parsed_smiles, str)
            self.assertGreater(len(parsed_smiles), 0)

            # Check conformers if present
            if 'conformers' in info and info['conformers']:
                self.assertIsInstance(info['conformers'], list)
                self.assertGreater(len(info['conformers']), 0)

                # Test first conformer structure
                first_conformer = info['conformers'][0]
                self.assertIsInstance(first_conformer, dict)

            # Check if experimental properties are present (full format molecule)
            if 'experimental_properties' in info:
                exp_props = info['experimental_properties']
                self.assertIsInstance(exp_props, dict)
                # Test for common experimental properties
                common_props = ['PCE', 'VOC', 'JSC', 'HOMO', 'LUMO', 'gap']
                for prop in common_props:
                    if prop in exp_props:
                        self.assertIsInstance(exp_props[prop], (int, float))

            # Log success information
            print(f"Successfully parsed first molecule from Figshare HOPV15 dataset using 2-step parser")
            print(f"Molecule SMILES: {info['smiles']}")
            print(f"Has conformers: {'conformers' in info and len(info.get('conformers', [])) > 0}")
            print(f"Has experimental properties: {'experimental_properties' in info}")
            print(f"Original file size: {len(content)} characters, {len(lines)} lines")

    def test_download_and_parse_multiple_molecules_from_figshare(self):
        """
        Test downloading HOPV15 from Figshare and parsing multiple molecules using parse_all.

        This test validates that the new 2-step parsing approach can handle
        multiple molecules from the Figshare dataset.
        """
        # Figshare URL for HOPV15 dataset
        figshare_url = 'https://figshare.com/ndownloader/files/4513735'

        with FileDownloadSource(
            figshare_url,
            verbose=True,
            ssl_verify=False,
        ) as source:
            # Download the file
            downloaded_path = source.fetch()

            # Initialize parser
            parser = HOPV15Parser(path=downloaded_path)

            # Parse multiple molecules using parse_all
            mol_tuples = parser.parse_all()

            # Basic validation
            self.assertIsInstance(mol_tuples, list)
            self.assertGreater(len(mol_tuples), 0, "Should parse at least one molecule")

            # Test first few molecules (limit to avoid long test time)
            molecules_to_test = min(5, len(mol_tuples))
            for i in range(molecules_to_test):
                mol, info = mol_tuples[i]

                # Validate molecule structure
                self.assertIsInstance(mol, Chem.Mol)
                self.assertIsNotNone(mol)
                self.assertGreater(mol.GetNumAtoms(), 0)

                # Validate info dictionary
                self.assertIsInstance(info, dict)
                self.assertIn('smiles', info)

                # Validate SMILES string
                self.assertIsInstance(info['smiles'], str)
                self.assertGreater(len(info['smiles']), 0)

                # Test that molecule can be converted to SMILES
                parsed_smiles = Chem.MolToSmiles(mol)
                self.assertIsInstance(parsed_smiles, str)

            print(f"Multiple molecule parsing successful. Parsed {len(mol_tuples)} molecules")
            print(f"First molecule SMILES: {mol_tuples[0][1]['smiles']}")
            if len(mol_tuples) > 1:
                print(f"Second molecule SMILES: {mol_tuples[1][1]['smiles']}")

    def test_figshare_file_format_consistency(self):
        """
        Test that the Figshare HOPV15 file has the expected format and content.

        This test validates that the downloaded file contains the expected
        patterns and structure without requiring the parser to handle the full file.
        """
        figshare_url = 'https://figshare.com/ndownloader/files/4513735'

        with FileDownloadSource(
            figshare_url,
            verbose=True,
            ssl_verify=False,
        ) as source:
            downloaded_path = source.fetch()

            # Analyze file content to verify expected patterns
            with open(downloaded_path, 'r') as f:
                content = f.read()

            with open(downloaded_path, 'r') as f:
                lines = f.readlines()

            # Basic file validation
            self.assertGreater(len(content), 15000000, "File should be substantial size")
            self.assertGreater(len(lines), 300000, "File should have many lines")

            # Check for expected patterns in the file
            # Count occurrences of key patterns
            smiles_pattern_count = content.count('InChI=')  # InChI lines should indicate molecule entries
            qchem_count = content.count('QChem ')
            conformer_count = content.count('Conformer ')

            # Verify we have substantial data
            self.assertGreater(smiles_pattern_count, 100, "Should have many InChI entries indicating molecules")
            self.assertGreater(qchem_count, 1000, "Should have many QChem calculation entries")
            self.assertGreater(conformer_count, 1000, "Should have many conformer entries")

            # Check that we have experimental data (CSV-like lines with numerical data)
            csv_pattern_count = 0
            for line in lines[:1000]:  # Check first 1000 lines for patterns
                if ',' in line and any(c.isdigit() for c in line):
                    # Look for lines that appear to be CSV with experimental data
                    parts = line.split(',')
                    if len(parts) > 5:  # Should have multiple comma-separated values
                        csv_pattern_count += 1

            self.assertGreater(csv_pattern_count, 10, "Should find CSV-like experimental data lines")

            # Test that we can parse the file using the new 2-step parser
            parser = HOPV15Parser(path=downloaded_path)
            mol, info = parser.parse()

            # Validate the molecule was parsed
            self.assertIsInstance(mol, Chem.Mol)
            self.assertIsNotNone(mol)
            self.assertGreater(mol.GetNumAtoms(), 0)

            # Check if experimental properties are present
            has_experimental_data = 'experimental_properties' in info

            print(f"Dataset file format validation successful using 2-step parser")
            print(f"File size: {len(content)} characters, {len(lines)} lines")
            print(f"InChI entries found: {smiles_pattern_count}")
            print(f"QChem calculations found: {qchem_count}")
            print(f"Conformer entries found: {conformer_count}")
            print(f"CSV-like experimental data lines found: {csv_pattern_count}")
            print(f"First molecule parsed successfully with experimental data: {has_experimental_data}")


if __name__ == '__main__':
    unittest.main()