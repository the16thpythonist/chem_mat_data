from click.testing import CliRunner

from chem_mat_data.cli import cli


def test_help_command_basically_works():
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0


def test_list_command_basically_works():
    """
    The "list" command is supposed to simply print a list of all the available datasets on the 
    remote file share server.
    """
    # The CliRunner instance is able to simulate the input given to the command line 
    # interface and can therefore be used to test the CLI commands in a programmatic way.
    runner = CliRunner()
    result = runner.invoke(cli, ['list'])
    assert result.exit_code == 0
    assert 'available datasets' in result.output.lower()
    assert 'TypeError' not in result.output

    
def test_list_command_hidden_datasets_omitted():
    """
    The "list" command sources the list of available datasts from the metadata yml file which
    is also hosted on the file share server. All the datasets that start with an underscore 
    in that metadata file are supposed to be "hidden" and should therefore not be listed 
    unless the --show-hidden flag is explicitly set.
    """
    runner = CliRunner()
    result = runner.invoke(cli, ['list'])
    assert result.exit_code == 0
    # We know at least of one hidden dataset that should be in the metadata file (the testing
    # dataset) which is supposed to be hidden under normal circumstances.
    assert '_test' not in result.output.lower()
    
    # Now it should be in there if we use the --show-hidden flag!
    result = runner.invoke(cli, ['list', '--show-hidden'])
    assert '_test' in result.output.lower()
    
    
def test_download_command_basically_works():
    """
    The "download" command should be able to download a dataset from the remote file share server 
    and save it to the local file system.
    """
    runner = CliRunner()
    # The only dataset which we can assume exists during such a test run is the "_test" dataset.
    result = runner.invoke(cli, ['download', '_test'])
    assert result.exit_code == 0
    
    assert 'download complete' in result.output.lower()