import os
import sys
import datetime
import difflib
import gzip
import shutil
import typing as t
from typing import Any, List, Tuple, Dict, Union, Optional

import rich
import rich_click as click
from rich import box
from rich.console import Console, ConsoleOptions
from rich.panel import Panel
from rich.style import Style
from rich.padding import Padding
from rich.text import Text
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import Progress
from rich.rule import Rule
from rich.columns import Columns

from chem_mat_data.utils import get_version
from chem_mat_data.utils import TEMPLATE_PATH
from chem_mat_data.utils import RichMixin
from chem_mat_data.utils import open_file_in_editor
from chem_mat_data.config import Config
from chem_mat_data.web import NextcloudFileShare

from dotenv import load_dotenv


# Load environment variables from a .env file if it exits and get the server URL from environment variables
load_dotenv()
server_url = os.getenv("url")
if not server_url:
    raise ValueError("Server URL not found in environment variables")


# == RICH DISPLAY ELEMENTS ==
# The following classes are used to define the rich display elements that are used to render the
# different parts of the command line interface. These classes are all subclasses of the RichMixin
# class which is a simple class that defines the __rich_console__ method which is used to render
# the object to the console using the rich library.


class RichLogo:
    """
    A rich display which will show the ASlurmX logo in ASCII art when printed.
    """

    STYLE = Style(bold=True, color="white")

    def __rich_console__(self, console, options):
        text_path = os.path.join(TEMPLATE_PATH, "logo_text.txt")
        with open(text_path) as file:
            text_string: str = file.read()
            text = Text(text_string, style=self.STYLE)
            
        image_path = os.path.join(TEMPLATE_PATH, "logo_image.txt")
        with open(image_path) as file:
            image_string: str = file.read()
            # Replace \e with actual escape character and create Text from ANSI
            ansi_string = image_string.replace('\\e', '\033')
            image = Text.from_ansi(ansi_string)
            
        side_by_side = Columns([image, text], equal=True, padding=(0, 3))
        yield Padding(side_by_side, (1, 3, 0, 3))
            
            
class RichHelp(RichMixin):
    """
    Implements the "rich" console rendering of the help text for the command line interface. This help 
    section consists of a brief general description of the CLI and also a brief section containing 
    example use cases of the commands.
    """
    
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> t.Any:
        yield 'Command Line Interface to the ChemMatDatabase.\n'
        yield Text((
            'The database consists of a collection of datasets chemistry and materials sience. Each datasets contains '
            'various molecules and/or crystal structures associated with a specific property. The main purpose of these '
            'datasets is to be used for the training of machine learning models for the prediction of these properties.\n'
        ))
        yield Text((
            'You can print a list of the available datasets using the "list" command.'
        ), style=Style(color='bright_black'))
        yield Padding(Syntax((
            'cmdata list'
        ), lexer='bash', theme='monokai', line_numbers=False), (1, 5))
        yield Text((
            'To download a dataset, use the "download" command followed by the name of the dataset.'
        ), style=Style(color='bright_black'))
        yield(Padding(Syntax((
            'cmdata download "dataset_name"'
        ), lexer='bash', theme='monokai', line_numbers=False), (1, 5)))
        
        
class RichCommands(RichMixin):
    
    def __init__(self, commands: Dict[str, Union[click.RichCommand, click.RichGroup]]):
        self.commands = commands
    
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> t.Any:

        # Separate commands and groups
        commands = {}
        groups = {}

        for command_name, command in self.commands.items():
            # A command group has subcommands, while a regular command doesn't
            if hasattr(command, 'commands') and command.commands:
                groups[command_name] = command
            else:
                # Regular commands don't have subcommands
                commands[command_name] = command

        # Display regular commands
        if commands:
            table_commands = Table(title=None, box=None, show_header=False, leading=1)
            table_commands.add_column(justify='left', style='bold cyan', no_wrap=True)
            table_commands.add_column(justify='left', style='white', no_wrap=False)

            for command_name, command in commands.items():
                table_commands.add_row(
                    command_name,
                    command.short_help or ''
                )

            panel_commands = Panel(
                table_commands,
                title='[bright_black]Commands[/bright_black]',
                title_align='left',
                style='bright_black'
            )
            yield panel_commands

        # Display command groups
        if groups:
            table_groups = Table(title=None, box=None, show_header=False, leading=1)
            table_groups.add_column(justify='left', style='bold cyan', no_wrap=True)
            table_groups.add_column(justify='left', style='white', no_wrap=False)

            for command_name, command in groups.items():
                table_groups.add_row(
                    command_name,
                    command.help or ''
                )

            panel_groups = Panel(
                table_groups,
                title='[bright_black]Command Groups[/bright_black]',
                title_align='left',
                style='bright_black'
            )
            yield panel_groups
        

class RichDatasetInfo(RichMixin):
    """
    Implements the "rich" console rendering of the specific information about a single dataset.
    
    This display element will essentially show all the available information about the particular 
    dataset in a two-column format, where the first column contains the identifying names of the 
    different information fields and the second column contains the actual values of those fields.
    """
    
    def __init__(self, name: str, info: dict):
        self.name = name
        self.info = info
    
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> t.Any:
        
        yield ''
        
        table = Table(title=None, box=None, show_header=False, leading=1, padding=(0, 2), pad_edge=False)
        table.add_column(justify='left', style='magenta', no_wrap=True)
        table.add_column(justify='left', style='white', no_wrap=False)
        table.add_row('Name', f'[bold cyan]{self.name}[/bold cyan]')
        table.add_row('Type', ', '.join(self.info['target_type']))
        table.add_row('No. Elements', str(self.info['compounds']))
        table.add_row('No. Targets', str(self.info.get('targets', 'N/A')))
        panel = Panel(
            table, 
            title='[bright_black]Metadata[/bright_black]', 
            title_align='left',
            border_style='bright_black',
        )
        
        yield panel
        
        panel_description = Panel(
            self.info['description'],
            title='[bright_black]Description[/bright_black]',
            title_align='left',
            border_style='bright_black',
            style='white',
            padding=(0, 1),
        )
        yield panel_description
        
        # Show target descriptions if they exist
        if 'target_descriptions' in self.info and self.info['target_descriptions']:
            
            target_desc_table = Table(title=None, box=None, show_header=False, leading=1, padding=(0, 2), pad_edge=False)
            target_desc_table.add_column(justify='left', style='magenta', no_wrap=True)
            target_desc_table.add_column(justify='left', style='white', no_wrap=False)
            
            # Sort by target index for consistent display
            target_descriptions = self.info['target_descriptions']
            for target_idx in sorted(target_descriptions.keys(), key=int):
                target_desc_table.add_row(f'{target_idx}', target_descriptions[target_idx])
            
            target_desc_panel = Panel(
                target_desc_table,
                title='[bright_black]Target Descriptions[/bright_black]',
                title_align='left',
                border_style='bright_black',
                style='white',
                padding=(0, 1),
            )
            yield target_desc_panel
        
        if 'sources' in self.info and self.info['sources']:
            
            sources_content = '\n'.join([
                f':left_arrow_curving_right: [yellow]{s}[/yellow]' 
                for s in self.info['sources']
            ])
            sources_panel = Panel(
                sources_content,
                title='[bright_black]References[/bright_black]',
                title_align='left',
                border_style='bright_black',
                style='white',
                padding=(0, 1),
            )
            yield sources_panel
        
        tags_content = ', '.join([
            f':label:  {s}' 
            for s in self.info['tags']
        ])
        tags_panel = Panel(
            tags_content,
            title='[bright_black]Tags[/bright_black]',
            title_align='left',
            border_style='bright_black',
            style='bright_black',
            padding=(0, 1),
        )
        yield tags_panel
        
        
class RichDatasetList(RichMixin):
    """
    Implements the "rich" console rendering of the specific information about a single dataset.


    This display element will show a table which contains one row for each dataset and columns 
    that contain various pieces of information about the datasets.
    """
    def __init__(self, 
                 datasets: Dict[str, Dict],
                 sort: bool = False,
                 show_hidden: bool = False,
                 ):
        self.datasets = datasets
        self.sort = sort
        self.show_hidden = show_hidden
        
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> t.Any:
        
        table = Table(title='Available Datasets', expand=True, box=box.HORIZONTALS)
        
        table.add_column("Name", justify="left", style="cyan", no_wrap=True)
        table.add_column("Description", justify="left", style="white", no_wrap=False)
        table.add_column("No. Elements",justify="left", style="magenta")
        table.add_column("No. Targets", justify="left", style="green")
        table.add_column("Target type", justify="left", style="yellow")
        table.add_column("Tags", justify="left", style="bright_black")
        
        names = list(self.datasets.keys())
        # potentially we want to sort the items depending on the "sort" argument
        if self.sort:
            names.sort(key=lambda value: value.lower())
        
        for name in names:
            # Most importantly, if the name of the datasets starts with an underscore, we will consider
            # that a hidden dataset and dont want to show it unless the "show-hidden" flag is set as well
            if not self.show_hidden and name.startswith('_'):
                continue
            
            details = self.datasets[name]
            table.add_row(
                f'[bold]{name}[/bold]', 
                details.get('verbose', '-'),
                str(details['compounds']), 
                str(details.get('targets', 'N/A')),
                ', '.join(details.get('target_type', '')), 
                ', '.join(details.get('tags', '')),
            )
        
        yield table
        
        
class RichCacheList(RichMixin):

    """
    Implements the "rich" console rendering of all the lists that are stored in the cache.

    This display element will show a table which contains one row for each dataset and columns
    that contain various pieces of information about the datasets.
    """
    def __init__(self,
                 dataset_metadata_map: Dict[Tuple[str, str], Dict],
                 dataset_sizes: Dict[Tuple[str, str], int],
                 total_cache_size: int,
                 sort: bool = True,
                 show_hidden: bool = False,
                 ):
        self.dataset_metadata_map = dataset_metadata_map
        self.dataset_sizes = dataset_sizes
        self.total_cache_size = total_cache_size
        self.sort = sort
        self.show_hidden = show_hidden
        
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> t.Any:

        table = Table(title='Dataset Cache', expand=True, box=box.HORIZONTALS)

        table.add_column("Dataset", justify="left", style="cyan", no_wrap=True)
        table.add_column("Created", justify="left", style="bright_black")
        table.add_column("Size (MB)", justify="right", style="magenta")

        keys = list(self.dataset_metadata_map.keys())
        # potentially we want to sort the items depending on the "sort" argument
        if self.sort:
            keys.sort()

        for key in keys:

            info: dict = self.dataset_metadata_map[key]
            name, typ = key

            dt = datetime.datetime.fromtimestamp(info['_cache_time'])

            # Get dataset size in MB
            size_bytes = self.dataset_sizes.get(key, 0)
            size_mb = size_bytes / (1024 * 1024)

            table.add_row(
                f'{name}.{typ}',
                f'{dt:%d.%b %Y, %H:%M}',
                f'{size_mb:.2f}',
            )

        yield table

        # Add total cache size information
        total_mb = self.total_cache_size / (1024 * 1024)
        yield ''
        yield Text(f'Total cache size: {total_mb:.2f} MB', style='bold green')


class RichCacheInfo(RichMixin):
    """
    Implements the "rich" console rendering of cache information including location,
    size, and dataset statistics.
    """

    def __init__(self,
                 cache_path: str,
                 total_size: int,
                 total_datasets: int,
                 total_files: int,
                 ):
        self.cache_path = cache_path
        self.total_size = total_size
        self.total_datasets = total_datasets
        self.total_files = total_files

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> t.Any:

        table = Table(title=None, box=None, show_header=False, leading=1, padding=(0, 2), pad_edge=False)
        table.add_column(justify='left', style='magenta', no_wrap=True)
        table.add_column(justify='left', style='white', no_wrap=False)

        # Cache location
        table.add_row('Location', f'[cyan]{self.cache_path}[/cyan]')

        # Total size
        size_mb = self.total_size / (1024 * 1024)
        size_gb = size_mb / 1024
        if size_gb >= 1:
            size_display = f'{size_gb:.2f} GB ({size_mb:.2f} MB)'
        else:
            size_display = f'{size_mb:.2f} MB'
        table.add_row('Total Size', f'[green]{size_display}[/green]')

        # Number of datasets
        table.add_row('Datasets', f'[yellow]{self.total_datasets}[/yellow]')

        # Total files
        table.add_row('Total Files', f'[blue]{self.total_files}[/blue]')

        # Check if cache directory exists
        cache_exists = os.path.exists(self.cache_path)
        status = '[green]✅ Available[/green]' if cache_exists else '[red]❌ Not found[/red]'
        table.add_row('Status', status)

        panel = Panel(
            table,
            title='[bright_black]Cache Information[/bright_black]',
            title_align='left',
            border_style='bright_black',
        )

        yield ''
        yield panel


class RichConfig(RichMixin):
    
    def __init__(self, config_file_path: str):
        self.file_path = config_file_path
        self.file_name = os.path.basename(self.file_path)
        with open(self.file_path) as file:
            self.content = file.read()
        
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> t.Any:
            
        rule_top = Rule(
            title=f'[bright_black]{self.file_path}[/bright_black]', 
            align='center', 
            style='bright_black'
        )
        yield rule_top
        
        yield self.content.replace(r'[', r'\[')
            
        rule_bottom = Rule(title=None, style='bright_black', end=' ')
        yield rule_bottom


class RichRemoteInfo(RichMixin):
    """
    Rich display element that shows detailed information about the currently configured 
    remote file share server.
    """
    def __init__(self, config, show_all: bool = False, metadata_available: bool = False, dataset_count: Optional[int] = None):
        self.config = config
        self.show_all = show_all
        self.metadata_available = metadata_available
        self.dataset_count = dataset_count
        
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> t.Any:
        
        table = Table(title=None, expand=True, box=None)
        table.add_column("Setting", justify="left", style="magenta", no_wrap=True)
        table.add_column("Value", justify="left", style="white", no_wrap=False)
        
        # Basic configuration
        table.add_row('Fileshare Type', f'[bold cyan]{self.config.get_fileshare_type()}[/bold cyan]')
        table.add_row('Fileshare URL', f'[bold cyan]{self.config.get_fileshare_url()}[/bold cyan]')
        
        # Get fileshare parameters for the specific type
        fileshare_params = self.config.get_fileshare_parameters(self.config.get_fileshare_type())
        
        # DAV Configuration (if available)
        if 'dav_url' in fileshare_params:
            table.add_row('DAV URL', f'[cyan]{fileshare_params["dav_url"]}[/cyan]')
        else:
            table.add_row('DAV URL', '[bright_black]Not configured[/bright_black]')
            
        if 'dav_username' in fileshare_params:
            table.add_row('DAV Username', f'[cyan]{fileshare_params["dav_username"]}[/cyan]')
        else:
            table.add_row('DAV Username', '[bright_black]Not configured[/bright_black]')
        
        # Password handling - show masked unless --all flag is used
        if 'dav_password' in fileshare_params:
            if self.show_all:
                table.add_row('DAV Password', f'[yellow]{fileshare_params["dav_password"]}[/yellow]')
            else:
                table.add_row('DAV Password', '[yellow]***[/yellow] [bright_black](use --all to show)[/bright_black]')
        else:
            table.add_row('DAV Password', '[bright_black]Not configured[/bright_black]')
        
        # Metadata availability
        if self.metadata_available:
            table.add_row('Metadata Available', '[green]✅ Available[/green]')
        else:
            table.add_row('Metadata Available', '[red]❌ Not accessible[/red]')
        
        # Dataset count
        if self.dataset_count is not None:
            table.add_row('Available Datasets', f'[green]{self.dataset_count}[/green]')
        else:
            table.add_row('Available Datasets', '[bright_black]Unknown[/bright_black]')
            
        panel = Panel(
            table,
            title='Remote Configuration',
        )
            
        yield ''
        yield panel
        

class RichDiffDisplay(RichMixin):
    """
    Rich display element that shows the diff between two metadata.yml files.
    """
    def __init__(self, local_file: str, remote_file: str, diff_lines: List[str], changed_lines: int):
        self.local_file = local_file
        self.remote_file = remote_file
        self.diff_lines = diff_lines
        self.changed_lines = changed_lines
        
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> t.Any:
        
        yield ''
        
        # Show file paths being compared
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
        


# == ACTUAL CLI IMPLEMENTATION ==
# The following class is the actual implementation of the command line interface for the ChemMatData
# package. This class is a subclass of the RichGroup class from the rich_click package which is a
# subclass of the click.Group class. This class is used to define a group of commands that are all
# related to the same topic. In this case the topic is the ChemMatData package and the commands are
# all related to the downloading and managing of the datasets that are part of the package.

class CLI(click.RichGroup):

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
            url=self.config.get_fileshare_url(),
            **self.config.get_fileshare_parameters(fileshare_type='nextcloud'),
        )
        self.cache = self.config.cache

        # ~ Constructing the help string.
        # This is the string which will be printed when the help option is called.
        self.rich_help = RichHelp()
        
        # This is the rich object that can be used to print the ASCII art logo to the 
        # console.
        self.rich_logo = RichLogo()
        
        ## -- Adding Actual Commands --
        
        # first level commands
        self.add_command(self.download_command)
        self.add_command(self.list_command)
        self.add_command(self.info_command)
        
        # cache command group
        self.add_command(self.cache_group)
        self.cache_group.add_command(self.cache_list_command)
        self.cache_group.add_command(self.cache_clear_command)
        self.cache_group.add_command(self.cache_remove_command)
        self.cache_group.add_command(self.cache_info_command)
        
        # config command group
        self.add_command(self.config_group)
        self.config_group.add_command(self.config_show_command)
        self.config_group.add_command(self.config_edit_command)

        # remote command group
        self.add_command(self.remote_group)
        self.remote_group.add_command(self.remote_show_command)
        self.remote_group.add_command(self.remote_diff_command)
        self.remote_group.add_command(self.remote_upload_command)
        self.remote_group.add_command(self.remote_exists_command)

    # Here we override the default "format_help" method of the RichGroup base class.
    # This method is being called to actually render the "--help" option of the command group.
    # We override this here to add the custom behavior of prining the logo before actually
    # printing the help text.
    def format_help(self, ctx: t.Any, formatter: t.Any) -> None:

        # Before printing the help text we want to print the logo
        rich.print(self.rich_logo)

        self.format_usage(ctx, formatter)
        click.echo(self.rich_help)

        self.format_options(ctx, formatter)

        # Custom command display - only call this once
        rich_commands = RichCommands(self.commands)
        click.echo(rich_commands)

        self.format_epilog(ctx, formatter)
        
    def format_commands(self, ctx: Any, formatter: Any) -> None:
        # Override parent behavior - do nothing here since we handle it in format_help
        pass

    # --- commands ---
    # The following methods are actually the command implementations which are the specific commands 
    # that are part of the command group that is represented by the ExperimentCLI object instance itself.

    @click.command('download', short_help='Download a dataset form the remote file share server')
    @click.argument('name')
    @click.option('--full', is_flag=True, help='Download both the original and the processed datasets')
    @click.option('--path', default=os.getcwd(), type=click.Path(file_okay=False), help='Path to where the files will be downloaded')
    @click.pass_obj
    # Try with "clintox"
    def download_command(self, name: str, full: bool, path: str):
        """
        Downloads the dataset with the given NAME from the remote file share server to the local system. 
        """
        click.secho('Connecting to server...')
        self.file_share.fetch_metadata(force=True)
        
        # First of all we can check if the dataset is even registered in the metadata file 
        # from the file share server. If that is already not the case we dont even need to 
        # try and fetch the dataset files itself.
        if name not in self.file_share['datasets']:
            click.secho('Dataset not found! Use the "list" command to see available datasets...', fg='red')
            return 1
        
        dataset_metadata: dict = self.file_share['datasets'][name]
        
        with Progress() as progress:
            
            click.secho('Downloading raw dataset...')
            for file_extension in dataset_metadata['raw']:
                file_name = f'{name}.{file_extension}'

                # First try to download the gzipped version
                try:
                    file_name_compressed = f'{file_name}.gz'
                    file_path_compressed = self.file_share.download_file(
                        file_name_compressed,
                        folder_path=path,
                        progress=progress,
                    )

                    # Decompress the file
                    file_path = os.path.join(path, file_name)
                    with open(file_path, mode='wb') as file:
                        with gzip.open(file_path_compressed, mode='rb') as compressed_file:
                            shutil.copyfileobj(compressed_file, file)

                    # Remove the compressed file after decompression
                    os.remove(file_path_compressed)

                # If gzipped version fails, download uncompressed version
                except Exception:
                    file_path = self.file_share.download_file(
                        file_name,
                        folder_path=path,
                        progress=progress
                    )

                self.cache.add_dataset(
                    name=name,
                    typ=file_extension,
                    path=file_path,
                    metadata=dataset_metadata,
                )
                
            # We only want to download the full dataset (aka the message packed version of the 
            # graph dicts) when the corresponding flag is explicitly set for the command
            if full:
                
                # There is also the possibility that a full dataset is not even available for 
                # the dataset in question. In that case we want to inform the user about that
                if not dataset_metadata['full']:
                    click.secho('Full format dataset not available!', fg='red')
                
                path = self.file_share.download_dataset(
                    f'{name}.mpack', 
                    folder_path=path, 
                    progress=progress
                )
                
                # 01.11.24
                # After downloading the full dataset we want to store that dataset in the cache for 
                # future use. This way we can avoid downloading the dataset again in the future.
                # After each download we want to replace the version in the cache with the new one.
                self.cache.add_dataset(
                    name=name,
                    typ='mpack',
                    path=path,
                    metadata=dataset_metadata,
                )
        
        click.secho('Download complete!', fg='green')
        
    @click.command('list', short_help='List the available datasets')
    @click.option('-s', '--sort', is_flag=True, help='Sort the datasets by name.')
    @click.option('--show-hidden', is_flag=True, help='Show hidden datasets as well. Mainly for testing purposes')
    @click.pass_obj
    def list_command(self, sort: bool, show_hidden: bool):
        """
        This command will display a list of the available datasets that are available for download.
        The list will also display some basic information about each dataset such as the number of
        elements, the number of targets, the target type and some tags.
        """
        
        click.secho('Connecting to server...')
        self.file_share.fetch_metadata(force=True)
        
        rich_list = RichDatasetList(
            self.file_share['datasets'],
            sort=sort,
            show_hidden=show_hidden,    
        )
        click.echo(rich_list)
        
        return 0
           
    @click.command('info', short_help='Show detailed information about one of the datasets')
    @click.argument('name')
    @click.pass_obj
    def info_command(self, name: str):
        """
        This command will print detailed information about the dataset with the given NAME.
        """
        self.file_share.fetch_metadata()
        if name not in self.file_share['datasets']:
            click.secho('Dataset not found on the remote file share server!', fg='red')
            return
        
        dataset_info: dict = self.file_share['datasets'][name]
        rich_info = RichDatasetInfo(name, dataset_info)
        click.echo(rich_info)
        
        return 0
    
    @click.command('about', short_help='print additional information about the command line interface')
    @click.pass_obj
    def about(self):
        # TODO: Implement an "about" page which prints additional information about the command line interface
        pass
    
    # ~ CACHE COMMAND GROUP
    # The following methods are the implementations of the commands that are part of the cache command group 
    # which can be used to manage the local file system cache.
    
    @click.group('cache', help='Commands for managing the local dataset cache')
    @click.pass_obj
    def cache_group(self):
        """
        Commands for managing the local dataset cache. When downloading datasets from the remote cache they 
        will be stored in a local cache directory to avoid downloading the same dataset multiple times. This 
        command group allows to view & manipulate the data stored in this cache.
        """
        pass
    
    @click.command('list', short_help='List the datasets that are currently stored in the cache')
    @click.pass_obj
    def cache_list_command(self):
        """
        Shows a list of all the datasets that are currently stored in the cache along with some of the metadata
        that is stored along with the dataset.
        """
        if len(self.cache) == 0:
            click.secho('Cache is empty!', fg='yellow')

        else:
            click.echo('')

            # This method will return all the datasets that are stored in the cache as a list of tuples where the
            # first element of the tuple is the name of the dataset and the second element is the type of the dataset.
            datasets: List[Tuple[str, str]] = self.cache.list_datasets()
            dataset_metadata_map: Dict[Tuple[str, str], dict] = {}
            dataset_sizes: Dict[Tuple[str, str], int] = {}

            for name, typ in datasets:
                # Given the name and type of dataset, this function will simply read the metadata from the
                # corresponding yml file.
                metadata = self.cache.get_dataset_metadata(name, typ)
                dataset_metadata_map[(name, typ)] = metadata

                # Get the size of each dataset
                size_bytes = self.cache.get_dataset_size(name, typ)
                dataset_sizes[(name, typ)] = size_bytes

            # Get total cache size
            total_cache_size = self.cache.get_total_cache_size()

            rich_cache_list = RichCacheList(
                dataset_metadata_map,
                dataset_sizes,
                total_cache_size,
                sort=True
            )
            click.echo(rich_cache_list)
            
    @click.command('clear', short_help='Clear the entire cache')
    @click.option('-v', '--verbose', is_flag=True, help='print verbose output')
    @click.pass_obj
    def cache_clear_command(self, verbose: bool):
        """
        Clears the local cache directory which contains all the datasets that have been downloaded from the
        remote file share server.
        """
        click.echo('clearing cache...')
        num_elements: int = 0
        for file_name, file_path in self.cache.iterator_clear_():
            num_elements += 1
            click.secho(f' > remove "{file_name}"...', fg='bright_black')
            
        click.secho(f'cleared {num_elements} elements', fg='green')

    @click.command('remove', short_help='Remove a specific dataset from the cache')
    @click.argument('name')
    @click.option('--type', '-t', default='mpack', help='Type of dataset to remove (default: mpack)')
    @click.pass_obj
    def cache_remove_command(self, name: str, type: str):
        """
        Removes a specific dataset from the cache by NAME and optionally TYPE.

        The NAME should be the dataset identifier (e.g., 'clintox').
        The TYPE specifies the format of the dataset (e.g., 'mpack', 'csv').
        """
        if self.cache.contains_dataset(name, type):
            success = self.cache.remove_dataset(name, type)
            if success:
                click.secho(f'Successfully removed "{name}.{type}" from cache', fg='green')
            else:
                click.secho(f'Failed to remove "{name}.{type}" from cache', fg='red')
                return 1
        else:
            click.secho(f'Dataset "{name}.{type}" not found in cache', fg='yellow')

            # Show available datasets for helpful suggestions
            datasets = self.cache.list_datasets()
            if datasets:
                click.echo('\nAvailable datasets in cache:')
                for dataset_name, dataset_type in datasets:
                    click.echo(f'  - {dataset_name}.{dataset_type}')
            else:
                click.echo('Cache is empty.')
            return 1

    @click.command('info', short_help='Show information about the cache')
    @click.pass_obj
    def cache_info_command(self):
        """
        Displays detailed information about the cache including location, size,
        and statistics about the datasets stored in it.
        """
        # Get cache statistics
        total_size = self.cache.get_total_cache_size()
        datasets = self.cache.list_datasets()
        total_datasets = len(datasets)

        # Count total files in cache directory
        total_files = 0
        if os.path.exists(self.cache.path):
            for file_name in os.listdir(self.cache.path):
                file_path = os.path.join(self.cache.path, file_name)
                if os.path.isfile(file_path):
                    total_files += 1
                elif os.path.isdir(file_path):
                    # Count files in subdirectories too
                    for root, dirs, files in os.walk(file_path):
                        total_files += len(files)

        rich_cache_info = RichCacheInfo(
            cache_path=self.cache.path,
            total_size=total_size,
            total_datasets=total_datasets,
            total_files=total_files
        )
        click.echo(rich_cache_info)

    ## == CONFIG COMMAND GROUP ==
    # The following methods are the implementations of the commands that are part of the config command group
    # which can be used to manage the config file of the ChemMatData installation.
    
    @click.group('config', help='Commands for managing the configuration file')
    @click.pass_obj
    def config_group(self):
        pass
    
    @click.command('show', short_help='Show the current configuration settings')
    @click.pass_obj
    def config_show_command(self):
        """
        Prints the contents of the config file to the console.
        """
        rich_config = RichConfig(self.config.config_file_path)
        click.echo(rich_config, nl=False)
        
    @click.command('edit', short_help='Edit the configuration file')
    @click.pass_obj
    def config_edit_command(self):
        open_file_in_editor(self.config.config_file_path)

    ## == REMOTE GROUP ==
    # This command group exposes specific commands to interact with the remote file share server
    # which are more targeted towards the maintainers of the database to expose some utility 
    # functions that are not really meant to be used by the end users of the package.
    
    @click.group('remote', help='Commands for interacting with the remote file share server')
    @click.pass_obj
    def remote_group(self):
        pass
    
    def check_dav_credentials(self) -> None:
        """
        Checks if the DAV credentials are set up correctly in the config file and if not prints a 
        warning message to the user.
        
        :raises: AssertionError if the DAV credentials are not set up correctly
        
        :returns: None
        """
        try:
            self.file_share.assert_dav()
        except AssertionError:
            click.secho('⚠️ DAV credentials are not set up!'
                        'You need to add valid DAV credentials in the config file to interact '
                        'with files located on the remote file share server.')
    
    @click.command('upload', short_help='Upload a dataset or file to the remote file share server')
    @click.argument('file_path', type=click.Path(exists=True, dir_okay=False))
    @click.option('--name', default=None, help=(
        'The name of the file on the remote server. If not provided, the name of the file '
        'itself will be used.'
    ))
    @click.pass_obj
    def remote_upload_command(self, 
                              file_path: str,
                              name: Optional[str],
                              **kwargs
                              ) -> None:
        
        ## -- Checking Credentials --
        # This will make sure that the DAV credentials are set up correctly before trying to upload 
        # anything. If the credentials are missing, this will print to the user.
        self.check_dav_credentials()

        ## -- Uploading the file --
        if name is not None:
            file_name: str = name
        else: # If the name is not provided, we will use the name of the file itself
            file_name = os.path.basename(file_path)
        
        file_size: int = os.path.getsize(file_path)
        file_size_mb: int = file_size // (1024 * 1024)
        click.secho(f'... Uploading file "{file_name}" with {file_size_mb} MB')
        
        # This method will actually upload the file to the remote server.
        self.file_share.upload(file_name, file_path)

        ## -- Checking for the file --
        # After the upload is complete, we want to check if the file was actually uploaded.
        exists, meta = self.file_share.exists(file_name)
        if exists:
            click.secho('✅ File uploaded successfully!')
        else:
            click.secho('⚠️ File upload failed!')

    @click.command('exists', short_help='Check if a file exists on the rmote file share server')
    @click.argument('file_name')
    @click.pass_obj
    def remote_exists_command(self,
                              file_name: str,
                              ) -> None:
        """
        Checks if the file with the given FILE_NAME exists on the remote file share server.
        """
        ## -- Checking Credentials --
        # This will make sure that the DAV credentials are set up correctly before trying to upload
        # anything. If the credentials are missing, this will print to the user.
        self.check_dav_credentials()
        
        ## -- Checking for the file --
        # This method will check if the file with the given name exists on the remote server.
        exists, meta = self.file_share.exists(file_name)
        if exists:
            click.secho(f'✅ File "{file_name}" exists on the remote server!')
            click.secho(f'  - Size: {meta["size"]} bytes')
            click.secho(f'  - Last modified: {meta["last_modified"]}')
            click.secho(f'  - Content type: {meta["content_type"]}')
            
        else:
            click.secho(f'⚠️ File "{file_name}" does not exist on the remote server!', fg='red')
            sys.exit(1)
    
    @click.command('show', short_help='Display remote configuration information')
    @click.option('--all', is_flag=True, help='Show all information including sensitive data like passwords')
    @click.pass_obj
    def remote_show_command(self, all: bool) -> None:
        """
        Displays detailed information about the currently configured remote file share server,
        including URLs, DAV configuration, and metadata availability.
        """
        
        ## -- Check Metadata Availability --
        metadata_available = False
        dataset_count = None
        
        try:
            # Try to fetch metadata to see if it exists
            self.file_share.fetch_metadata(force=True)
            metadata_available = True
            
            # Get dataset count if available
            if 'datasets' in self.file_share.metadata:
                dataset_count = len(self.file_share.metadata['datasets'])
                
        except Exception:
            # Metadata is not accessible
            metadata_available = False
            dataset_count = None
        
        ## -- Creating Rich Display --
        rich_remote_info = RichRemoteInfo(
            self.config, 
            show_all=all, 
            metadata_available=metadata_available, 
            dataset_count=dataset_count
        )
        click.echo(rich_remote_info)
    
    @click.command('diff', short_help='Compare local metadata.yml with remote version')
    @click.argument('local_file', type=click.Path(exists=True, dir_okay=False, readable=True))
    @click.pass_obj
    def remote_diff_command(self, local_file: str) -> None:
        """
        Compares a local metadata.yml file with the remote metadata.yml file from the server.
        Shows the number of different lines and displays a visual diff of the changes.
        
        LOCAL_FILE: Path to the local metadata.yml file to compare
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
        import yaml
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


@click.group(cls=CLI)
@click.option("-v", "--version", is_flag=True, help='Print the version of the package')
@click.pass_context
def cli(ctx: t.Any,
        version: bool
        ) -> None:

    # 07.06.24
    # This is actually required to make the CLI class work as it is currently implemented.
    # The problem is that all the commands which are also methods need to be decorated with the 
    # @pass_obj in the position of the "self" argument and here we set the object to be passed 
    # as the CLI instance to make that happen. 
    ctx.obj = ctx.command

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
