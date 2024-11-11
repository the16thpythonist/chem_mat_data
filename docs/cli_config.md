# ``config`` - Interacting with the Local Config File

When using the ``cmdata`` CLI, a local ``config.yml`` file is automatically created in the user's ``.config`` 
folder. This config file contains certain options that configure the behavior of the package.

## Viewing the Config File

To view the content of the current config file, you can use the ``config show`` command:

```bash
cmdata config show
```

## Edit the Config File

To conveniently edit this config file, you can use the ``config edit`` command:

```bash
cmdata config edit
```

This command will open the config file using the system's default text editor.