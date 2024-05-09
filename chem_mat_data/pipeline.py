import os
import logging
import requests
import click
from chem_mat_data.utils import download_dataset
from chem_mat_data.datasets import available_datasets


#This script runs the following way:
#In terminal type "python3 prototype_pipeline.py [name of dataset]"
#Right now there are only two datasets uploaded, HIV.csv and toxcast_data.csv
#Thus one has to execute "python3 prototype_pipeline.py HIV" if interested in that dataset.
#The .csv can be omitted.
#The names of the datasets were taken as given in original form.
# Where the datasets are stored
#base_url = 'https://bwsyncandshare.kit.edu/s/3XHQNptmyayD6f9/download?path=%2F&files=' # for smiles
base_url = 'https://bwsyncandshare.kit.edu/s/yqTsaqeXAw5nggs/download?path=&files='  # for json
@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument('filenames', nargs=-1, type=str, required=False)
@click.option('-check', is_flag=True, help= 'List available datasets')
@click.option('-a', is_flag=True, help = 'Downloads all available datasets')
def main(filenames, check, a):
    """
    Prototype Pipeline: A tool for downloading datasets.

    To download a dataset, provide its name as an argument. 
    For example: python3 pipeline.py HIV
    """

    if a:
        click.confirm('This will donwload the whole collection of datasets to your current working directory. Are you sure?', abort=True)
        try:
            download_dataset(base_url, 'Graphs.zip')
            click.echo(f'Collection downloaded succesfully')
        except Exception as e:
            logging.error(f'Error downloading collection : {e}')
            click.echo(f'Failed to download collection: {e}')
        return
    #This will print out the available datasets
    if check:
        response = requests.get(base_url + 'all_datasets.txt')
        if response.status_code == 200:
            click.echo('Available datasets:')
            click.echo(response.text)
        else:
            click.echo(f"Failed to fetch available datasets.Status code: {response.status_code}")
        return
  
    #If no argument is provided, it will prompt the user to give one or use an option flag
    if not filenames:
        click.echo('Please provide a dataset name or use the "check" option to see available datasets or use "--help" option.')
        return

    datasets_folder = os.path.join(os.path.expanduser('~'),'datasets')
    for file in filenames:
        #Strip it of whitespace 
        file = file.strip()

        # We need to check if the dataset actually exists
        if file not in available_datasets:
            click.echo(f'Dataset {file} not found. Available datasets can be seen via "-check" option')
            continue
        
        ### From here on out, the code aims to download the file as specified by the user ####
        
        # Add the ".csv" extension, wich is needed to construct the url
        file = file if file.endswith('.json') else f'{file}.json'
        # Create 'datasets' folder if it doesn't exist. All datasets are stored in that folder
        if not os.path.exists(datasets_folder):
            os.makedirs(datasets_folder)

        # This is the path where the dataset will be
        destination = os.path.join(datasets_folder, file)
        if os.path.exists(destination):
            click.confirm('File already exists. Do you want to overwrite?', abort=True)

        # Determine the url for the dataset
        dataset_url = base_url + file
        try:
            download_dataset(dataset_url, destination)
            click.echo(f'Dataset downloaded succesfully to {destination}')
        except Exception as e:
            logging.error(f'Error downloadign dataset: {e}')
            click.echo(f'Failed to download dataset: {e}')


cli.add_command(main, name='download')


if __name__ == '__main__':
    main()
