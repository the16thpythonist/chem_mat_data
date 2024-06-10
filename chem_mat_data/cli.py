import os
import sys
import typing as t

import rich_click as click
from rich.console import Console, ConsoleOptions
from rich.panel import Panel
from rich.layout import Layout
from rich.segment import Segment
from rich.style import Style
from rich.padding import Padding
from rich.text import Text
from rich.syntax import Syntax
from rich.table import Table
from rich.progress import Progress

from chem_mat_data.utils import download_dataset
from chem_mat_data.utils import get_version
from chem_mat_data.utils import TEMPLATE_PATH
from chem_mat_data.utils import RichMixin
from chem_mat_data.utils import RichHeader
from chem_mat_data.processing import MoleculeProcessing
from chem_mat_data.config import Config
from chem_mat_data.web import NextcloudFileShare

from dotenv import load_dotenv
import yaml
import requests


# Load environment variables from a .env file if it exits and get the server URL from environment variables
load_dotenv()
server_url = os.getenv("url")
if not server_url:
    raise ValueError("Server URL not found in environment variables")


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
            'chemdata list'
        ), lexer='bash', theme='monokai', line_numbers=False), (1, 5))
        yield Text((
            'To download a dataset, use the "download" command followed by the name of the dataset.'
        ), style=Style(color='bright_black'))
        yield(Padding(Syntax((
            'chemdata download "dataset_name"'
        ), lexer='bash', theme='monokai', line_numbers=False), (1, 5)))
        
        
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
        
        table = Table(title=None, box=None, show_header=False, leading=1)
        table.add_column(justify='left', style='yellow', no_wrap=True)
        table.add_column(justify='left', style='white', no_wrap=False)
        table.add_row('Name', self.name)
        table.add_row('Description', self.info['description'])
        table.add_row('Type', ', '.join(self.info['target_type']))
        table.add_row('#Elements', str(self.info['compounds']))
        table.add_row('#Targets', str(self.info['targets']))
        
        panel = Panel(table, title='Dataset Info', title_align='left')
        
        yield panel
        
        
class RichDatasetList(RichMixin):
    """
    Implements the "rich" console rendering of the specific information about a single dataset.


    This display element will show a table which contains one row for each dataset and columns 
    that contain various pieces of information about the datasets.
    """
    def __init__(self, 
                 datasets: dict[str, dict],
                 sort: bool = False,
                 show_hidden: bool = False,
                 ):
        self.datasets = datasets
        self.sort = sort
        self.show_hidden = show_hidden
        
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> t.Any:
        
        table = Table(title='Available Datasets', expand=True)
        
        table.add_column("Name", justify="left", style="cyan", no_wrap=True)
        table.add_column("Compounds",justify="left", style="magenta")
        table.add_column("Targets", justify="left", style="green")
        table.add_column("Target type", justify="left", style="yellow")
        table.add_column("Tags", justify="left", style="cyan")
        
        names = list(self.datasets.keys())
        # potentially we want to sort the items depending on the "sort" argument
        if self.sort:
            names.sort()
        
        for name in names:
            # Most importantly, if the name of the datasets starts with an underscore, we will consider
            # that a hidden dataset and dont want to show it unless the "show-hidden" flag is set as well
            if not self.show_hidden and name.startswith('_'):
                continue
            
            details = self.datasets[name]
            table.add_row(
                name, 
                str(details['compounds']), 
                str(details['targets']),
                ', '.join(details['target_type']), 
                ', '.join(details['tags']),
            )
        
        yield table
        

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

        # ~ Constructing the help string.
        # This is the string which will be printed when the help option is called.
        self.rich_help = RichHelp()
        
        # This is the rich object that can be used to print the ASCII art logo to the 
        # console.
        self.rich_logo = RichLogo()
        
        # ~ adding commands
        
        self.add_command(self.download)
        self.add_command(self.list)
        self.add_command(self.info)
        #self.add_command(self.about)

    # Here we override the default "format_help" method of the RichGroup base class.
    # This method is being called to actually render the "--help" option of the command group.
    # We override this here to add the custom behavior of prining the logo before actually
    # printing the help text.
    def format_help(self, ctx: t.Any, formatter: t.Any) -> None:
        
        # Before printing the help text we want to print the logo
        click.echo(self.rich_logo)
        
        self.format_usage(ctx, formatter)
        # self.format_help_text(ctx, formatter)
        click.echo(self.rich_help)
        self.format_options(ctx, formatter)
        self.format_epilog(ctx, formatter)

    # -- commands
    # The following methods are actually the command implementations which are the specific commands 
    # that are part of the command group that is represented by the ExperimentCLI object instance itself.

    @click.command('download', short_help='Download a dataset form the remote file share server')
    @click.argument('name')
    @click.option('--full', is_flag=True, help='Download both the original and the processed datasets')
    @click.option('--path', default=os.getcwd(), type=click.Path(file_okay=False), help='Path to where the files will be downloaded')
    @click.pass_obj
    # Try with "clintox"
    def download(self, name: str, full: bool, path: str):
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
        
        dataset = self.file_share['datasets'][name]
        with Progress() as progress:
            
            click.secho('Downloading raw dataset...')
            for file_extension in dataset['raw']:
                file_name = f'{name}.{file_extension}'
                self.file_share.download_file(file_name, folder_path=path, progress=progress)
                
            # We only want to download the full dataset (aka the message packed version of the 
            # graph dicts) when the corresponding flag is explicitly set for the command
            if full:
                # There is also the possibility that a full dataset is not even available for 
                # the dataset in question. In that case we want to inform the user about that
                if not dataset['full']:
                    click.secho('Full format dataset not available!', fg='red')
                
                self.file_share.download_dataset(name, folder_path=path, progress=progress)
        
        click.secho('Download complete!', fg='green')
        
    @click.command('list', short_help='List the available datasets')
    @click.option('-s', '--sort', is_flag=True, help='Sort the datasets by name.')
    @click.option('--show-hidden', is_flag=True, help='Show hidden datasets as well. Mainly for testing purposes')
    @click.pass_obj
    def list(self, sort: bool, show_hidden: bool):
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
        click.secho(rich_list)
        
        return 0
           
    @click.command('info', short_help='Show detailed information about one of the datasets')
    @click.argument('name')
    @click.pass_obj
    def info(self, name: str):
        """
        This command will print detailed information about the dataset with the given NAME.
        """
        self.file_share.fetch_metadata()
        if name not in self.file_share['datasets']:
            pass
        
        dataset_info: dict = self.file_share['datasets'][name]
        rich_info = RichDatasetInfo(name, dataset_info)
        click.echo(rich_info)
        
        return 0
    
    @click.command('about', short_help='print additional information about the command line interface')
    @click.pass_obj
    def about(self):
        # TODO: Implement an "about" page which prints additional information about the command line interface
        pass


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
