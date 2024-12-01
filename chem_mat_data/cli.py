import os
import sys
import datetime
import typing as t
from typing import Any, List, Tuple, Dict, Union

import rich_click as click
from rich import box
from rich.console import Console, ConsoleOptions, RenderResult
from rich.panel import Panel
from rich.style import Style
from rich.padding import Padding
from rich.text import Text
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import Progress
from rich.rule import Rule

from chem_mat_data.utils import get_version
from chem_mat_data.utils import TEMPLATE_PATH
from chem_mat_data.utils import RichMixin
from chem_mat_data.utils import open_file_in_editor
from chem_mat_data.config import Config
from chem_mat_data.web import NextcloudFileShare

from typing import Dict
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


class RichLogo(RichMixin):
    """
    Implements the "rich" console rendering of the ASCII art logo of the ChemMatDatabase.
    """
    STYLE = Style(bold=True, color='white')
    
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> t.Any:
        logo_path = os.path.join(TEMPLATE_PATH, 'logo.txt')
        with open(logo_path, mode='r') as file:
            logo = file.read()
            text = Text(logo, style=self.STYLE)
            pad = Padding(text, (1, 1))
            yield pad
            
            
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
        
        table_commands = Table(title=None, box=None, show_header=False, leading=1)
        table_commands.add_column(justify='left', style='bold cyan', no_wrap=True)
        table_commands.add_column(justify='left', style='white', no_wrap=False)
        
        for command_name, command in self.commands.items():
            if isinstance(command, click.RichCommand) and not isinstance(command, click.RichGroup):
                table_commands.add_row(
                    command_name,
                    command.short_help
                )
        
        panel_commands = Panel(
            table_commands, 
            title='[bright_black]Commands[/bright_black]',
            title_align='left', 
            style='bright_black'
        )
        yield panel_commands
        
        table_groups = Table(title=None, box=None, show_header=False, leading=1)
        table_groups.add_column(justify='left', style='bold cyan', no_wrap=True)
        table_groups.add_column(justify='left', style='white', no_wrap=False)
        
        for command_name, command in self.commands.items():
            if isinstance(command, click.RichGroup):
                table_groups.add_row(
                    command_name, #click.style(command_name, fg='cyan', bold=True),
                    command.help
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
        table.add_row('No. Targets', str(self.info['targets']))
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
                str(details['compounds']), 
                str(details['targets']),
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
                 sort: bool = True,
                 show_hidden: bool = False,
                 ):
        self.dataset_metadata_map = dataset_metadata_map
        self.sort = sort
        self.show_hidden = show_hidden
        
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> t.Any:
        
        table = Table(title='Dataset Cache', expand=True, box=box.HORIZONTALS)
        
        table.add_column("Dataset", justify="left", style="cyan", no_wrap=True)
        table.add_column("Created",justify="left", style="bright_black")
        
        keys = list(self.dataset_metadata_map.keys())
        # potentially we want to sort the items depending on the "sort" argument
        if self.sort:
            keys.sort()
        
        for key in keys:
            
            info: dict = self.dataset_metadata_map[key]
            name, typ = key
            
            dt = datetime.datetime.fromtimestamp(info['_cache_time'])
            table.add_row(
                f'{name}.{typ}', 
                f'{dt:%d.%b %Y, %H:%M}', 
            )
        
        yield table
        
        
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
        self.file_share = NextcloudFileShare(self.config.get_fileshare_url())
        self.cache = self.config.cache

        # ~ Constructing the help string.
        # This is the string which will be printed when the help option is called.
        self.rich_help = RichHelp()
        
        # This is the rich object that can be used to print the ASCII art logo to the 
        # console.
        self.rich_logo = RichLogo()
        
        # ~ adding commands
        
        self.add_command(self.download_command)
        self.add_command(self.list_command)
        self.add_command(self.info_command)
        
        self.add_command(self.cache_group)
        self.cache_group.add_command(self.cache_list_command)
        self.cache_group.add_command(self.cache_clear_command)
        
        self.add_command(self.config_group)
        self.config_group.add_command(self.config_show_command)
        self.config_group.add_command(self.config_edit_command)


    # Here we override the default "format_help" method of the RichGroup base class.
    # This method is being called to actually render the "--help" option of the command group.
    # We override this here to add the custom behavior of prining the logo before actually
    # printing the help text.
    def format_help(self, ctx: t.Any, formatter: t.Any) -> None:
        
        formatter.config.command_groups.update({
            'cache': self.cache_group
        })
        
        # Before printing the help text we want to print the logo
        click.echo(self.rich_logo)
        
        self.format_usage(ctx, formatter)
        click.echo(self.rich_help)
        
        self.format_options(ctx, formatter)
        self.format_epilog(ctx, formatter)
        
    def format_commands(self, ctx: Any, formatter: Any) -> None:
        rich_commands = RichCommands(self.commands)
        click.echo(rich_commands)

    # -- commands
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
            for name, typ in datasets:
                # Given the name and type of dataset, this function will simply read the metadata from the 
                # corresponding yml file.
                metadata = self.cache.get_dataset_metadata(name, typ)
                dataset_metadata_map[(name, typ)] = metadata
                

            rich_cache_list = RichCacheList(dataset_metadata_map, sort=True)
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

    # ~ CONFIG COMMAND GROUP
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
