import os
import sys
import yaml
import time
import difflib
from collections import defaultdict
from typing import List, Dict, Any

import rich_click as click
from prettytable import PrettyTable, TableStyle
from rich.style import Style
from rich.text import Text
from rich.pretty import pprint
from rich.console import Console, ConsoleOptions
from rich.panel import Panel
from rich.syntax import Syntax
from pycomex.functional.experiment import Experiment
from pycomex.utils import dynamic_import

from chem_mat_data.config import Config
from chem_mat_data.web import NextcloudFileShare
from chem_mat_data.utils import get_version
from chem_mat_data.utils import PATH
from chem_mat_data.utils import DOCS_PATH
from chem_mat_data.utils import METADATA_PATH
from chem_mat_data.utils import TEMPLATE_ENV
from chem_mat_data.utils import CsvListType
from chem_mat_data.utils import open_file_in_editor
from chem_mat_data.utils import RichMixin

# The path to the "scripts" folder of the experiment modules
SCRIPTS_PATH: str = os.path.join(PATH, 'scripts')
# The path to the "results" folder of the experiment module executions.
RESULTS_PATH: str = os.path.join(PATH, 'scripts', 'results')


class RichDiffDisplay(RichMixin):
    """
    Rich display element that shows the diff between two metadata.yml files.
    """
    def __init__(self, local_file: str, remote_file: str, diff_lines: List[str], changed_lines: int):
        self.local_file = local_file
        self.remote_file = remote_file
        self.diff_lines = diff_lines
        self.changed_lines = changed_lines
        
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Any:
        
        yield ''
        
        # Show file paths being compared
        from rich.table import Table
        comparison_table = Table(box=None)
        comparison_table.add_column("File", style="magenta", no_wrap=True)
        comparison_table.add_column("Path", style="cyan", no_wrap=False)
        comparison_table.add_row("Local", self.local_file)
        comparison_table.add_row("Remote", "metadata.yml (from server)")
        
        yield comparison_table
        
        # Show the actual diff if there are changes
        if self.changed_lines > 0 and self.diff_lines:
            yield ""
            
            # Limit diff output to first 50 lines to avoid overwhelming display
            display_lines = self.diff_lines[:50]
            if len(self.diff_lines) > 50:
                display_lines.append(f"... ({len(self.diff_lines) - 50} more lines truncated)")
            
            # Create a syntax-highlighted diff
            diff_text = '\n'.join(display_lines)
            syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=False)
            
            panel = Panel(
                syntax,
                title=f'File Differences (showing first {min(50, len(self.diff_lines))} lines)',
                expand=True
            )
            
            yield panel
            
        # Show diff summary
        yield ""
        if self.changed_lines == 0:
            yield Text(f"✅ Files are identical", style="green")
        else:
            yield Text(f"📊 {self.changed_lines} lines differ between local and remote files", style="yellow")


def is_experiment_archive(folder_path: str) -> bool:
    """
    Helper function to determine if a given ``folder_path`` represents a valid experiment archive.
    """
    if not os.path.isdir(folder_path):
        return False

    data_path = os.path.join(folder_path, 'experiment_data.json')
    meta_path = os.path.join(folder_path, 'experiment_meta.json')
    code_path = os.path.join(folder_path, 'experiment_code.py')
    metadata_path = os.path.join(folder_path, 'metadata.yml')
    if not os.path.exists(data_path) or not os.path.exists(meta_path) or not os.path.exists(code_path) or not os.path.exists(metadata_path):
        return False
    
    return True


class CLI(click.RichGroup):

    # This list contains the string identifiers for the possible strategies that can be used for 
    # the metadata collect method, which will collect the individual metadata files from the 
    # experiment results.
    EXPERIMENT_COLLECT_STRATEGIES: List[str] = ['recent']

    def __init__(self,
                 **kwargs):
        super(CLI, self).__init__(
            invoke_without_command=True,
            **kwargs
        )

        # This config file is a global singleton instance which allows access to the most 
        # important config parameter of the project such as the URL address of the fileshare 
        # server from which the datasets will be downloaded.
        self.config = Config()
        self.file_share = NextcloudFileShare(
            self.config.get_fileshare_url(), 
            **self.config.get_fileshare_parameters('nextcloud')
        )
        self.cache = self.config.cache
        
        self.help = (
            'This is the internal management command line interface for the ChemMatData package. '
            'This CLI will provide certain commands that are useful for maintaining and developing of the '
            'package and the database itself.'
        )
        
        ## -- Adding commands --
        
        # metadata command group
        self.add_command(self.metadata_group)
        self.metadata_group.add_command(self.metadata_collect_command)
        self.metadata_group.add_command(self.metadata_upload_command)
        self.metadata_group.add_command(self.metadata_edit_command)
        self.metadata_group.add_command(self.metadata_diff_command)
        self.metadata_group.add_command(self.metadata_remove_command)
        
        # dataset command group
        self.add_command(self.dataset_group)
        self.dataset_group.add_command(self.dataset_create_command)
        self.dataset_group.add_command(self.dataset_upload_command)

        # docs command group
        self.add_command(self.docs_group)
        self.docs_group.add_command(self.docs_collect_datasets_command)

    ## == METADATA COMMAND GROUP ==
    # This command group is used to manage the metadata.yml file both locally and on the remote server 

    @click.group('metadata', help='Commands for managing the metadata YML of the remote server.')
    @click.pass_obj
    def metadata_group(self,):
        """
        This command group contains commands that can be used to manage the metadata.yml file on the 
        remote file server. This metadata file contains all the necessary information about the 
        datasets and for example determines what information will be shown when using the
        ``list`` command.
        """
        pass
    
    def collect_dataset_archives_map(self, results_path: str) -> Dict[str, List[Experiment]]:
        """
        A helper function which creates a ``dataset_archives_map`` data structure if given the absolute 
        string ``results_path`` of where to search the experiment archives. Returns a dict data structure
        whose string keys are the dataset names and the values are the corresponding Experiment instances
        imported from the experiment archives.
        
        :param results_path: The absolute path to the folder where the experiment archives are stored.
        
        :return: A dict data structure mapping dataset names to Experiment instances.
        """
        click.secho('collecting experiment archives...', fg='bright_black')
        
        # ~ collecting experiment archives
        # We first want to generally collect all the experiment archives here for each dataset.
        experiments: List[Experiment] = []
        for root, folders, files in os.walk(results_path):
            
            for folder in folders:
                
                folder_path = os.path.join(root, folder)
                if is_experiment_archive(folder_path):
                    try:
                        experiment = Experiment.load(folder_path)
                    except Exception:
                        continue
                    
                    # for now we just add the loaded experiment to the list and will then later 
                    # sort out the metadata from it.
                    experiments.append(experiment)
                    
                    experiment_name = experiment.metadata['name']
                    experiment_id = os.path.basename(folder_path)
                    click.secho(f' * collecting experiment {experiment_name}/{experiment_id}', fg='bright_black')
        
        click.secho(f'collected {len(experiments)} experiments')
        
        # Now that we have all the experiments we can sort the experiments into lists of which dataset they belong to.
        # best case there is inly one dataest per experiment, but there are good chances that this will not be the 
        # case. And if there are multiple datasets per experiment, we will have to decide which one to use to get 
        # the metadata from based on the strategy.
        dataset_archives_map: Dict[str, List[Experiment]] = defaultdict(list)
        for experiment in experiments:
            dataset = experiment.parameters['DATASET_NAME']
            dataset_archives_map[dataset].append(experiment)
            
        return dataset_archives_map

    @click.command('collect')
    @click.option('--blank', is_flag=True, help=('If this flag is set, the initial metadata.yml file will be created from scratch.'
                                                 'Otherwise, the remote metadata.yml file will be downloaded and used as a base.'))
    @click.option('-p', '--path', help='The path at which to save the new metadata file', 
                  default=METADATA_PATH, show_default=True)
    @click.option('-s', '--source', help='The source path at which to search for dataset metadata.', default=RESULTS_PATH)
    @click.option('--strategy', help='The strategy to use for collecting the metadata.', 
                  type=click.Choice(EXPERIMENT_COLLECT_STRATEGIES), default='recent')
    @click.option('--verbose', is_flag=True, help='If this flag is set, the command will print additional information.')
    @click.option('--datasets', type=CsvListType(), help=(
        'A list of dataset names to collect metadata for. If not provided, all datasets will be collected.'
    ))
    @click.pass_obj
    def metadata_collect_command(self,
                                 blank: bool,
                                 path: str,
                                 source: str,
                                 strategy: str,
                                 verbose: bool,
                                 datasets: List[str],
                                 ) -> None:
        """
        This command collects the metadata of all datasets that are currently stored on the LOCAL system
        and bundles that information into a single metadata.yml file. Will use the current version of the 
        remote file share metadata as a base unless the `--blank` flag is set.
        """
        time_start: float = time.time()
        
        ## --- Creating the Base ---
        # If the blank flag is NOT explicitly set, we are going to try and download the metadata.yml file from the remote file 
        # share server and use that as a base version which is then updated with the local metadata.
        metadata_all: dict = {'datasets': []}
        if not blank:
            click.echo('downloading the metadata.yml file from the remote file share server...')
            metadata_all.update(self.file_share.fetch_metadata())
        
        ## --- Collecting the Metadata ---
        
        # This method will return a dict data structure that maps the dataset names to lists of Experiment instances
        # that have been loaded from the results folder of completed experiment. For each dataset name key the corresponding
        # value list consists of all experiment archives related to that dataset.
        dataset_archives_map: Dict[str, List[Experiment]] = self.collect_dataset_archives_map(source)
        
        if datasets:
            dataset_archives_map = {
                name: experiments
                for name, experiments in dataset_archives_map.items()
                if name in datasets
            }
        
        click.echo(f'updating information on {len(dataset_archives_map)} datasets: {" ".join(dataset_archives_map.keys())}')
        
        # This data structure will hold the metadata for each dataset that we have collected from the experiments.
        dataset_metadata_map: Dict[str, dict] = {}
        for dataset, experiments in dataset_archives_map.items():
            
            if strategy == 'recent':
                experiment = sorted(experiments, key=lambda e: e.metadata['end_time'])[-1]
        
            # The actual experiment metadata that we are interested in will be stored as an additional artifact 
            # called "metadata.yml" in the experiment folder. We will load this file and then add the metadata
            # to the dataset_metadata_map.
            metadata_path: str = os.path.join(experiment.path, 'metadata.yml')
            with open(metadata_path, mode='r') as file:
                metadata: dict = yaml.load(file, Loader=yaml.FullLoader)
            
            dataset_metadata_map[dataset] = metadata

        # --- merging the metadata ---
        # The structure of the overall metadata is such that there is an additional key for each dataset and the 
        # corresponding value is again a dict-like object that contains the metadata for that dataset.
        click.echo('merging the metadata...')
        for dataset, metadata in dataset_metadata_map.items():
            metadata_all['datasets'][dataset] = metadata
        
        # --- saving the file ---
        # In the end we can save the collected metadata into a file and place it at the specified path.
        path = os.path.expanduser(path)
        click.echo(f'saving the metadata file @ {path}...')
        with open(path, 'w') as file:
            yaml.dump(metadata_all, file, sort_keys=True, indent=4)
            
        time_end: float = time.time()
        
        click.secho(f'✅ collected metadata.yml file in {time_end - time_start:.2f} seconds', fg='green')
            
        if verbose:
            pprint(metadata_all, max_string=100)
            
        click.secho()
            
    @click.command('upload')
    @click.option('-p', '--path', help='The path to the metadata file that should be uploaded.', 
                  default=METADATA_PATH, show_default=True)
    @click.pass_obj
    def metadata_upload_command(self,
                                path: str
                                ) -> None:
        """
        Uploads a local metadata.yml file to the remore file share server. Note that this will overwrite the 
        version of the metadata file that is currently stored on the server.
        """
        if not os.path.exists(path):
            click.secho(f'Error: The specified metadata file "{path}" does not exist.', fg='red')
            sys.exit(1)

        click.secho('uploading the metadata file to the remote file share server...')        
        self.file_share.upload('metadata.yml', path)
        click.secho('✅ uploaded metadata.yml', fg='green')
        click.secho()
        
    @click.command('edit')
    @click.option('-p', '--path', help='The path to the metadata file that should be uploaded.', default='./metadata.yml')
    @click.pass_obj
    def metadata_edit_command(self,
                              path: str,
                              ) -> None:
        """
        Opens the metadata.yml file in the default editor of the system.
        """
        if not os.path.exists(path):
            click.secho(f'Error: The specified metadata file "{path}" does not exist.', fg='red')
            sys.exit(1)
        
        open_file_in_editor(path)

    @click.command('diff', short_help='Compare local metadata.yml with remote version')
    @click.option('--local-file', '-f', type=click.Path(exists=True, dir_okay=False, readable=True), 
                  default=METADATA_PATH, show_default=True,
                  help='Path to the local metadata.yml file to compare')
    @click.pass_obj
    def metadata_diff_command(self, local_file: str) -> None:
        """
        Compares a local metadata.yml file with the remote metadata.yml file from the server.
        Shows the number of different lines and displays a visual diff of the changes.
        
        By default, uses the package metadata.yml file for comparison.
        """
        
        ## --- Fetch Remote Metadata ---
        try:
            # Try to fetch the remote metadata
            self.file_share.fetch_metadata(force=True)
            
            if not hasattr(self.file_share, 'metadata') or self.file_share.metadata is None:
                click.secho('❌ Could not fetch remote metadata.yml file', fg='red')
                sys.exit(1)
                
        except Exception as e:
            click.secho(f'❌ Error fetching remote metadata: {str(e)}', fg='red')
            sys.exit(1)
        
        ## --- Read Local File ---
        try:
            with open(local_file, 'r', encoding='utf-8') as f:
                local_content = f.read()
        except Exception as e:
            click.secho(f'❌ Error reading local file: {str(e)}', fg='red')
            sys.exit(1)
        
        ## --- Normalize Both Files for Comparison ---
        try:
            # Parse local file to normalize it
            local_data = yaml.safe_load(local_content)
            # Convert both to the same YAML format for fair comparison
            local_normalized = yaml.dump(local_data, default_flow_style=False, sort_keys=True, indent=2)
            remote_normalized = yaml.dump(self.file_share.metadata, default_flow_style=False, sort_keys=True, indent=2)
        except Exception as e:
            click.secho(f'❌ Error normalizing YAML data: {str(e)}', fg='red')
            sys.exit(1)
        
        ## --- Calculate Diff ---
        local_lines = local_normalized.splitlines(keepends=False)
        remote_lines = remote_normalized.splitlines(keepends=False)
        
        # Generate unified diff
        diff_lines = list(difflib.unified_diff(
            local_lines,
            remote_lines,
            fromfile=f'local/{os.path.basename(local_file)}',
            tofile='remote/metadata.yml',
            lineterm=''
        ))
        
        # Count changed lines (lines starting with + or - but not +++ or ---)
        changed_lines = sum(1 for line in diff_lines 
                           if (line.startswith('+') and not line.startswith('+++')) or
                              (line.startswith('-') and not line.startswith('---')))
        
        ## --- Display Results ---
        rich_diff = RichDiffDisplay(
            local_file=local_file,
            remote_file="remote/metadata.yml",
            diff_lines=diff_lines,
            changed_lines=changed_lines
        )
        click.echo(rich_diff)

    @click.command('remove', short_help='Remove a dataset entry from the local metadata.yml file')
    @click.argument('dataset_name', type=str)
    @click.option('--metadata-path', '-f', type=click.Path(exists=True, dir_okay=False, readable=True, writable=True),
                  default=METADATA_PATH, show_default=True,
                  help='Path to the local metadata.yml file to modify')
    @click.option('--backup', is_flag=True, default=True, show_default=True,
                  help='Create a backup copy before modifying the file')
    @click.pass_obj
    def metadata_remove_command(self, dataset_name: str, metadata_path: str, backup: bool) -> None:
        """
        Removes a dataset entry from the local metadata.yml file.

        DATASET_NAME: Name of the dataset to remove from the metadata file
        """

        ## --- Read and Parse Local File ---
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_content = f.read()
            metadata = yaml.safe_load(metadata_content)
        except Exception as e:
            click.secho(f'❌ Error reading metadata file: {str(e)}', fg='red')
            sys.exit(1)

        ## --- Check if Dataset Exists ---
        if 'datasets' not in metadata:
            click.secho('❌ No datasets section found in metadata file', fg='red')
            sys.exit(1)

        if dataset_name not in metadata['datasets']:
            click.secho(f'❌ Dataset "{dataset_name}" not found in metadata file', fg='red')
            available_datasets = list(metadata['datasets'].keys())
            if available_datasets:
                click.secho(f'Available datasets: {", ".join(sorted(available_datasets))}', fg='yellow')
            sys.exit(1)

        ## --- Create Backup if Requested ---
        if backup:
            backup_path = f"{metadata_path}.backup"
            try:
                import shutil
                shutil.copy2(metadata_path, backup_path)
                click.secho(f'📋 Created backup at: {backup_path}', fg='bright_black')
            except Exception as e:
                click.secho(f'⚠️ Warning: Could not create backup: {str(e)}', fg='yellow')

        ## --- Remove Dataset ---
        removed_dataset = metadata['datasets'].pop(dataset_name)

        ## --- Write Updated Metadata ---
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                yaml.dump(metadata, f, default_flow_style=False, sort_keys=True, indent=4)
            click.secho(f'✅ Successfully removed dataset "{dataset_name}" from {metadata_path}', fg='green')

            # Show some info about what was removed
            compounds = removed_dataset.get('compounds', 'Unknown')
            targets = removed_dataset.get('targets', 'Unknown')
            click.secho(f'   Removed dataset had {compounds} compounds and {targets} targets', fg='bright_black')

        except Exception as e:
            click.secho(f'❌ Error writing updated metadata file: {str(e)}', fg='red')
            sys.exit(1)

    ## == DATASET COMMAND GROUP ==
    # all the commands related to the interaction with the local datasets - such as the (re)creation of
    # datasets or the uploading to the remote file share server.
        
    @click.group('dataset', help=('Commands for managing the datasets on the local system.'))
    @click.pass_obj
    def dataset_group(self,
                      ) -> None:
        pass
    
    def collect_dataset_experiment_map(experiments_path: str) -> Dict[str, Experiment]:
        """
        A helper function which creates a ``dataset_experiment_map`` data structure if given 
        the absolute string ``experiments_path`` of where to search the experiment modules. 
        Returns a dict data structure whose string keys are the dataset names and the values 
        are the corresponding Experiment instances imported from the experiment modules.
        
        :param experiments_path: The absolute path to the folder where the experiment modules are stored.
        
        :return: A dict data structure mapping dataset names to Experiment instances.
        """
        # In this data structure we are going to map the string dataset names to the imported
        # Experiment instances that can potentially be executed to create the corresponding 
        # datasets.
        dataset_experiment_map: Dict[str, Experiment] = {}
        
        file_names: List[str] = os.listdir(experiments_path)
        for file_name in file_names:
            file_path = os.path.join(experiments_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.py'):
                
                # Each python module we find, we'll import and check if it has the "DATASET_NAME"
                # attribute. If it does, we'll add it to the list of available experiment modules.
                module = dynamic_import(file_path)
                if hasattr(module, 'DATASET_NAME') and hasattr(module, '__experiment__'):
                    experiment = getattr(module, '__experiment__')
                    dataset_name = experiment.parameters['DATASET_NAME']
                    dataset_experiment_map[dataset_name] = experiment
                    
        return dataset_experiment_map
    
    @click.command('create', short_help='Creates a new dataset using the experiment modules.')
    @click.option('--scripts-path', help='The path to the scripts folder.', default=SCRIPTS_PATH)
    @click.option('--all', help='If this flag is set, all available datasets will be created.', is_flag=True)
    @click.argument('name', type=str)
    @click.pass_obj
    def dataset_create_command(self,
                               scripts_path: str,
                               name: str,
                               all: bool,
                               ) -> None:
        """
        This command can be used to trigger the execution of one or multiple experiment modules which 
        will re-create a dataset.
        
        The NAME argument may either be the string name of a single dataset to be created or a 
        comma separated list of dataset names.
        """
        # ~ collecting experiment modules
        # First of all we construct a data structure that tells us which experiment modules are actually 
        # available to create which datasets.
        click.secho('collecting available experiment modules...', color='bright_black')
        
        # This method creates a dict data structure that maps the dataset names to the Experiment instances
        # that are imported from the experiment modules.
        dataset_experiment_map: Dict[str, Experiment] = self.collect_dataset_experiment_map(scripts_path)
        click.secho(f'found {len(dataset_experiment_map)} experiment modules')

        # ~ parsing the name argument
        # The name argument may be a single dataset name or a comma separated list of dataset names.
        if all:
            dataset_names: List[str] = list(dataset_experiment_map.keys())
        else:
            dataset_names: List[str] = [n.replace(' ', '') for n in name.split(',')]

        # ~ executing the experiments
        # We can fetch the experiment instances from the map that we previously constructed and 
        # we can forcefully execute these experiements by using the "run" method.
        click.secho('running the experiment modules. Beware that this will take some time...')
        for dataset_name in dataset_names:
            click.secho(f'⭐ running "{dataset_name}"', fg='yellow')
            experiment = dataset_experiment_map[dataset_name]
            experiment.run()
        
        click.secho('✅ all experiments have been executed', fg='green')
        
    @click.command('upload', short_help='Uploads local datasets to the remote file share server.')
    @click.argument('name', type=str, required=False)
    @click.option('--results-path', help='The path to the results folder.', default=RESULTS_PATH)
    @click.option('-p', '--path', help='The path to the dataset file that should be uploaded.')
    @click.option('--strategy', help='The strategy to use for selecting the experiment to use for the upload.',
                    type=click.Choice(EXPERIMENT_COLLECT_STRATEGIES), default='recent')
    @click.option('--all', help='If this flag is set, all available datasets will be uploaded.', is_flag=True)
    @click.pass_obj
    def dataset_upload_command(self,
                               name: str,
                               results_path: str,
                               path: str,
                               strategy: str,
                               all: bool,
                               ) -> None:
        """
        This command will upload the graph dataset MPACK file and the raw CSV file for a given dataset
        NAME to the remote file share server. If the `--all` flag is set, all available datasets will be
        uploaded. Note that this will overwrite any existing dataset files on the remote server.

        The NAME argument may either be the string name of a single dataset to be uploaded or a
        comma-separated list of dataset names.
        """
        
        # This method will return a dict data structure that maps the dataset names to lists of Experiment instances
        # that have been loaded from the results folder of completed experiment. For each dataset name key the corresponding
        # value list consists of all experiment archives related to that dataset.
        dataset_archives_map: Dict[str, List[Experiment]] = defaultdict(list)
        dataset_archives_map.update(self.collect_dataset_archives_map(results_path))
        click.secho(f'found {len(dataset_archives_map)} datasets')
            
        # ~ parsing the name argument
        # The name argument may be a single dataset name or a comma separated list of dataset names.
        if all:
            dataset_names: List[str] = list(dataset_archives_map.keys())
        elif name:
            dataset_names: List[str] = [n.replace(' ', '') for n in name.split(',')]
        else:
            click.secho('❌ Error: Must specify either dataset NAME(s) or use --all flag', fg='red')
            sys.exit(1)
            
        # ~ uploading the datasets
        # Now we can go through the dataset archives map and upload the dataset files to the file share server.
        click.secho('uploading the datasets...')
        for dataset_name in dataset_names:
            
            experiments = dataset_archives_map[dataset_name]
            if len(experiments) == 0:
                click.secho(f'❗ no experiment archives found for dataset "{dataset_name}"', fg='red')
                continue
            
            click.secho(f'⭐ dataset "{dataset_name}"', fg='yellow')
            
            # First we need to select a single experiment to use as the basis of the upload for each dataset 
            # based on the selected strategy.
            if strategy == 'recent':
                experiment = sorted(experiments, key=lambda e: e.metadata['end_time'])[-1]
        
            # The actual dataset file that we are interested in will be stored as an additional artifact
            # called "{dataset_name}.xxx" in the experiment folder. We will load this file and then
            # upload it to the file share server.
            
            mpack_file_name = f'{dataset_name}.mpack.gz'
            mpack_file_path: str = os.path.join(experiment.path, f'{dataset_name}.mpack.gz')
            if os.path.exists(mpack_file_path):
                file_size = os.path.getsize(mpack_file_path)
                file_size_mb = file_size / (1024 * 1024)
                click.secho(f'   uploading "{mpack_file_name}" [{file_size_mb:.1f} MB]')
                self.file_share.upload(mpack_file_name, mpack_file_path)

            csv_file_name = f'{dataset_name}.csv.gz'
            csv_file_path: str = os.path.join(experiment.path, f'{dataset_name}.csv.gz')
            if os.path.exists(csv_file_path):
                file_size = os.path.getsize(csv_file_path)
                file_size_mb = file_size / (1024 * 1024)
                click.secho(f'   uploading "{csv_file_name}" [{file_size_mb:.1f} MB]')
                self.file_share.upload(csv_file_name, csv_file_path)

        click.secho('✅ datasets uploaded', fg='green')
        click.secho()

    ## == DOCS COMMAND GROUP ==
    # This command group is used to manage the documentation of the package.
    
    @click.group('docs', help='Commands for managing the documentation of the package.')
    @click.pass_obj
    def docs_group(self,
                     ) -> None:
          """
          This command group contains commands that can be used to manage the documentation of the package.
          This includes commands to build the documentation, open it in a browser, or edit it.
          """
          pass
      
    @click.command('collect-datasets', help='Compiles the datasets for the documentation from the metadata file.')
    @click.option('--metadata-path', help='The path to the metadata file to use for compiling the datasets.',
                  default=METADATA_PATH, show_default=True)
    @click.option('--output-path', help='The path to the output file where the compiled datasets will be written.',
                  default=os.path.join(DOCS_PATH, 'datasets.md'), show_default=True)
    @click.pass_obj
    def docs_collect_datasets_command(self,
                                      metadata_path: str,
                                      output_path: str,
                                      ) -> None:
        """
        This command will dynamically create the overview of the datasets that are available in the 
        remote file share server using the metadata.yml file. By default, this will collect the dataset information 
        from the results of the experiments that have been executed the most recently.
        """
        
        ## -- Getting Metadata --
        # This dictionary contains all the metadata information stored on the remote file share server.
        click.secho('fetching metadata from remote fileshare server...')
        metadata: dict = self.file_share.fetch_metadata(force=True)
        
        ## -- Creating Table --
        # Using prettytable to create the table in the markdown format
        click.secho('compiling datasets overview...')
        table = PrettyTable()
        table.align = 'l'
        table.field_names = [
            "Name",
            "Description",
            "No. Elements",
            "Target Type",
        ]
        for dataset_name, dataset_info in metadata['datasets'].items():
            table.add_row([
                f'**{dataset_name}**',
                dataset_info.get('verbose', '-'),
                str(dataset_info.get('compounds', 0)),
                ', '.join(dataset_info.get('target_type', [])),
            ])
            
        # This is important to set the style to markdown so that the string that is created from this is 
        # actually a valid markdown syntax.
        table.set_style(TableStyle.MARKDOWN)
        datasets_table: str = table.get_string()
        
        ## -- Writing the Docs File --
        # Here we use the jinja2 template to actually render the content of the documentation file 
        # mainly centered around the just created table of the datasets.
        click.secho('writing datasets overview to file...')
        template = TEMPLATE_ENV.get_template('docs_dataset.md.j2')
        content = template.render(
            file_share_url=self.file_share.url,
            datasets_table=datasets_table,
        )
        with open(output_path, 'w') as file:
            file.write(content)
        
        click.secho('✅ datasets overview written to file', fg='green')
        click.secho('')


@click.group(cls=CLI)
@click.option("-v", "--version", is_flag=True, help='Print the version of the package')
@click.option('--no-verify', is_flag=True, help='turn off ssl verification for the fileshare connection')
@click.pass_context
def cli(ctx: Any,
        version: bool,
        no_verify: bool,
        ) -> None:

    # 07.06.24
    # This is actually required to make the CLI class work as it is currently implemented.
    # The problem is that all the commands which are also methods need to be decorated with the 
    # @pass_obj in the position of the "self" argument and here we set the object to be passed 
    # as the CLI instance to make that happen. 
    ctx.obj = ctx.command
    ctx.obj.file_share.verfiy = not no_verify

    if version:
        version = get_version()
        click.echo(Text(version, style=Style(bold=True)))
        sys.exit(0)
        
    # 07.06.24
    # There was previously a bug here which caused the help text to be rendered for every 
    # command. After adding the additional condition "not ctx.invoked_subcommand" this was 
    # fixed.
    
    # This section will implement the small convenience feature that if no arguments are passed 
    # to the CLI at all, it will simply show the help information.
    elif not ctx.args and not ctx.invoked_subcommand:
        ctx.command.format_help(ctx, ctx.formatter_class())



if __name__ == "__main__":
    cli()  # pragma: no cover