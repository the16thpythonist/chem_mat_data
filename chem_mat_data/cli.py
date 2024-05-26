import os
import sys
import typing as t

import rich_click as click
from rich.console import Console, ConsoleOptions
from rich.segment import Segment
from rich.style import Style
from rich.padding import Padding
from rich.text import Text
from rich.syntax import Syntax
from rich.table import Table

from chem_mat_data.utils import download_dataset
from chem_mat_data.utils import get_version
from chem_mat_data.utils import TEMPLATE_PATH
from chem_mat_data.utils import RichMixin
from chem_mat_data.utils import RichHeader
from chem_mat_data.processing import MoleculeProcessing

from dotenv import load_dotenv
import yaml
import requests


# Load environment variables from a .env file if it exits and get the server URL from environment variables
load_dotenv()
server_url = os.getenv("url")
if not server_url:
    raise ValueError("Server URL not found in environment variables")

class RichLogo(RichMixin):
    
    STYLE = Style(bold=True, color='white')
    
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> t.Any:
        logo_path = os.path.join(TEMPLATE_PATH, 'logo.txt')
        with open(logo_path, mode='r') as file:
            logo = file.read()
            text = Text(logo, style=self.STYLE)
            pad = Padding(text, (1, 1))
            yield pad
            
            
class RichHelp(RichMixin):
    
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
        


class CLI(click.RichGroup):

    def __init__(self,
                 **kwargs):
        super(CLI, self).__init__(
            invoke_without_command=True,
            **kwargs
        )

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
        Starts a new run of the experiment with the string identifier EXPERIMENT.

        

       #This gives errors. I dont know why..
       if self.context.terminal_width is not None:            
           length = self.context.terminal_width
        """
        click.secho('Downloading dataset...', bold=True)
        # TODO: Implement downloading the dataset.
        url = server_url + name
        if full:
            click.secho(f'Downloading original and processed dataset!', fg='yellow')
            destination = os.path.join(path, name + '.csv')
            download_dataset(url +'.csv', destination)
            click.secho(f'Dataset downloaded successfully! Location: {os.path.abspath(destination)}', fg='green')
        
        destination = os.path.join(path, name + '.json')
        download_dataset(url + '.json', destination)
        click.secho(f'Dataset downloaded successfully! Location: {os.path.abspath(destination)}', fg='green')


        
    @click.command('list', short_help='List the available datasets')
    @click.pass_obj
    def list(self):
        
        # TODO: Implement listing of the available datasets.
        response = requests.get(server_url + 'datasets.yml')
        if response.status_code != 200:
            click.secho(f"failed to fetch list of datasets. Status code: {response.status_code}", fg='red')
            return

        datasets = yaml.safe_load(response.text)
        table = Table(title="Available datasets")

        table.add_column("Name", justify="left", style="cyan", no_wrap=True)
        table.add_column("Compounds",justify="left", style="magenta")
        table.add_column("Targets", justify="left", style="green")
        table.add_column("Target type", justify="left", style="yellow")
        table.add_column("Tags", justify="left", style="cyan")

        
        for dataset in datasets['datasets']:
            for name, details in dataset.items():
                table.add_row(name, str(details.get('compounds')), str(details.get('targets')),', '.join(details.get('target type')), ', '.join(details.get('tags')))
        console = Console()
        console.print(table)
        """
        click.secho('Available datasets:', bold=True)
        for dataset in datasets['datasets']:
            for name, details in dataset.items():
                click.secho(f" - {name}:", fg = 'green', bold=True)
                click.secho(f"     Compounds: {details.get('compounds')}")
                click.secho(f"     Targets: {details.get('targets')}")
                target_type = details.get('target type')
                if isinstance(target_type, list):
                    target_type = ', '.join(target_type)
                click.secho(f"     Target type: {target_type}")
                tags = details.get('tags')
                if isinstance(tags, list):
                    tags = ', '.join(tags)
                click.secho(f"     Tags: {tags}")
    """
    
    @click.command('info', short_help='Show detailed information about one of the datasets')
    @click.pass_obj
    def info(self, name: str):
        # TODO: Implement a command that shows detailed information about a dataset.
        print('im here')
        pass
    
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

    if version:
        version = get_version()
        click.echo(Text(version, style=Style(bold=True)))
        sys.exit(0)
        
    elif not ctx.args:
        ctx.command.format_help(ctx, ctx.formatter_class())


if __name__ == "__main__":
    cli()  # pragma: no cover
