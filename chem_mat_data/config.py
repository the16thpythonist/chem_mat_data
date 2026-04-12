import os
from typing import Optional, Dict, Any

import tomlkit
from dotenv import load_dotenv
from appdirs import AppDirs

from chem_mat_data.cache import Cache
from chem_mat_data.utils import config_file_from_template


class Singleton(type):
    """
    This is metaclass definition, which implements the singleton pattern. The objective is that whatever
    class uses this as a metaclass does not work like a traditional class anymore, where upon calling the
    constructor a NEW instance is returned. This class overwrites the constructor behavior to return the
    same instance upon calling the constructor. This makes sure that always just a single instance
    exists in the runtime!

    **USAGE**
    To implement a class as a singleton it simply has to use this class as the metaclass.
    .. code-block:: python
        class MySingleton(metaclass=Singleton):
            def __init__(self):
                # The constructor still works the same, after all it needs to be called ONCE to create the
                # the first and only instance.
                pass
        # All of those actually return the same instance!
        a = MySingleton()
        b = MySingleton()
        c = MySingleton()
        print(a is b) # true
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        return cls._instances[cls]


class Config(metaclass=Singleton):
    def __init__(self):

        # ~ loading environment variables

        # By using the override=False option the environment variable definitions in the .env file are
        # only used if definitions for those variables do not already exist in the environment! This
        # is useful to prevent overwriting of environment variables if the user wants to override some of
        # them with custom values.
        load_dotenv(override=False)
        self.fileshare_url = os.getenv('FILESHARE_URL')

        # This object can be used to access the absolute paths of application specific directories like
        # the config folder or a cache folder in an OS agnostic way.
        self.app_dirs = AppDirs('chem_mat_data')

        # ~ local dataset caching
        # The cache can be used to store datasets that have already been downloaded so that they don't
        # have to be downloaded every time. This path attribute will contain the absolute string path
        # to the cache folder.
        self.cache_path: Optional[str] = None
        # The Cache object instance is a wrapper that provides a more convenient way to interact with the
        # cache folder. It provides methods to add datasets to the cache, check if a dataset is already in
        # the cache, and to clear the cache, etc.
        self.cache: Optional[Cache] = None
        # This method will change the cache path to the given path, which also includes the construction
        # of a new Cache object instance to manage the access to that cache folder.
        self.set_cache_path(self.app_dirs.user_cache_dir)

        # ~ config files
        self.config_path: Optional[str] = None
        self.config_file_path: Optional[str] = None
        self.config_data: Optional[Dict[str, Any]] = None
        self.set_config_path(self.app_dirs.user_config_dir)

    def set_cache_path(self, cache_path: str) -> None:
        """
        This method will change the cache path to the given ``cache_path``, which also includes the construction
        of a new Cache object instance to manage the access to that cache folder.
        """
        self.cache_path = cache_path
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        self.cache = Cache(self.cache_path)

    def set_config_path(self, config_path: str) -> None:

        self.config_path = config_path
        if not os.path.exists(self.config_path):
            os.makedirs(self.config_path)

        # Inside this config folder there needs to be a "config.yml" file which contains the configuration
        # data for the application.
        self.config_file_path = os.path.join(self.config_path, 'config.toml')
        if not os.path.exists(self.config_file_path):
            # This util function will create a new default config file at the given file path
            config_file_from_template(self.config_file_path)

        # Finally, we can load the configuration data from the config file and load the results into the
        # "config_data" dict type attribute.
        with open(self.config_file_path) as file:
            self.config_data = tomlkit.loads(file.read())

    def save(self) -> None:
        """
        This method will save the current content of the "config_data" attribute to the config TOML file
        on the disk.

        :returns: None
        """
        with open(self.config_file_path, 'w') as file:
            file.write(tomlkit.dumps(self.config_data))

    def load(self):
        """
        This method will load the content of the config TOML file into the "config_data" attribute.

        :returns: None
        """
        with open(self.config_file_path) as file:
            self.config_data = tomlkit.loads(file.read())

    # ~ actual config values
    # The following methods define the interface to access the actual configuration values defined in the config
    # file.

    def get_fileshare_url(self) -> str:
        """
        Returns the fileshare URL which was loaded from the environment variables. This URL points to
        a cloud folder which contains all the actual dataset files and from where the datasets will
        be downloaded ultimately.

        :returns: string absolute URL
        """
        return self.config_data['remote']['fileshare_url']

    def get_fileshare_type(self) -> str:
        """
        Returns the fileshare type which was loaded from the config file. This type specifies the
        type of fileshare that should be used to download the datasets (e.g. Nextcloud, Zenodo, etc...)
        """
        return self.config_data['remote']['fileshare_type']

    def get_fileshare_parameters(self, fileshare_type: str) -> dict:
        """
        Returns a dict with the additional parameters specifically for the given "fileshare_type".

        :returns: dict
        """
        return self.config_data['remote'].get(fileshare_type, {})

    # ~ agent configuration
    # The following methods provide access to the ``[agent]`` section of the config file
    # which controls the LLM backend used for automated dataset discovery and processing.

    def get_agent_backend(self) -> str:
        """
        Returns the name of the configured agent backend.

        Falls back to ``"opencode"`` when the ``[agent]`` section is absent or
        does not contain a ``backend`` key, which ensures backwards compatibility
        with config files created before agent backend support was added.

        :returns: Backend identifier string (e.g. ``"opencode"`` or ``"claude"``).
        """
        return self.config_data.get('agent', {}).get('backend', 'opencode')

    def get_agent_backend_config(self, backend: str) -> dict:
        """
        Returns the backend-specific configuration dict for the given backend.

        This corresponds to the ``[agent.<backend>]`` subsection of the config
        file. For example, ``get_agent_backend_config("claude")`` returns the
        contents of ``[agent.claude]``.

        :param backend: The backend name (e.g. ``"opencode"`` or ``"claude"``).

        :returns: Dict of backend-specific settings, or empty dict if absent.
        """
        return self.config_data.get('agent', {}).get(backend, {})

    def get_agent_prompt_override(self, prompt_name: str) -> Optional[str]:
        """
        Returns a user-defined prompt override from the config file, or ``None``.

        When set, the override replaces the built-in prompt template for the
        given prompt name entirely. Valid prompt names are ``"link_prompt"``
        and ``"script_generation_prompt"``.

        :param prompt_name: The prompt key (e.g. ``"link_prompt"``).

        :returns: The override string, or ``None`` if not configured.
        """
        return self.config_data.get('agent', {}).get(prompt_name)

    # ~ discovery configuration
    # The following methods provide access to the ``[discovery]`` section of the
    # config file which controls the TinyLlama-based classifier used for the
    # dataset discovery pipeline (``cmmanage discover ...``).

    def get_discovery_classifier_url(self) -> Optional[str]:
        """
        Returns the download URL of the classifier archive, or ``None`` if unset.

        The archive is expected to be a ``.tar.gz`` bundle containing the
        fine-tuned LoRA adapter and tokenizer files. On first use, the archive
        is downloaded to the user config directory and cached there.

        :returns: The classifier URL string, or ``None`` if not configured.
        """
        return self.config_data.get('discovery', {}).get('classifier_url')

    def get_discovery_model_dir(self) -> str:
        """
        Returns the absolute path to the directory where the classifier is cached.

        The directory is nested inside the user config folder and is created
        lazily by the classifier loader when the archive is first extracted.

        :returns: Absolute path to the classifier model directory.
        """
        return os.path.join(self.config_path, 'discovery', 'classifier_512_tinyllama')
