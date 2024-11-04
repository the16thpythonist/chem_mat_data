import os
import tempfile

from chem_mat_data.cache import Cache
from chem_mat_data.config import Config


class TestConfig:
    """
    Unit tests for the Config class.
    """
    def test_construction_basically_works(self):
        """
        It should be possible to construct the config singleton and it should internally link to the config data dict 
        that was read from the config file and it should also link to the Cache wrapper object that manages the cache 
        folder.
        """
        config = Config()
        assert isinstance(config, Config)
        
        # This should be a singleton instance, which means that no matter how many times we create a new
        # instance of the Config class, it should always be the same object.
        config2 = Config()
        assert config is config2
        
        # Some config toml file should have been created
        assert os.path.exists(config.config_file_path)
        assert config.config_file_path.endswith('.toml')
        
        # The default version of the config data should be loaded from the config toml file
        assert isinstance(config.config_data, dict)
        assert len(config.config_data) != 0
        
        # Additionally, the cache wrapper object should be linked in the config object and 
        # it should point to a cache folder that exists
        assert isinstance(config.cache, Cache)
        assert os.path.exists(config.cache.path)
        
    def test_set_cache_path_works(self):
        """
        It should be possible to switch the cache folder by using the "set_cache_path" method on the config
        object.
        """
        with tempfile.TemporaryDirectory() as path:
            
            config = Config()
            # Before changing the path the cache folder should not be the new path
            prev_path = config.cache_path
            assert config.cache_path != path
            assert config.cache.path != path
            
            # Only after changing, the cache folder should be the new path. It is also important that 
            # the Cache wrapper object also points to this new path!
            config.set_cache_path(path)
            assert config.cache_path == path
            assert config.cache.path == path
            
            # After the test we need to reset the cache path to the previous path to avoid potential side effects
            config.set_cache_path(prev_path)
            
    def test_set_config_path_works(self):
        """
        It should be possible to switch the config folder by using the "set_config_path" method on the config
        object.
        """
        with tempfile.TemporaryDirectory() as path:
            
            config = Config()
            prev_path = config.config_path
            assert config.config_path != path
            
            config.set_config_path(path)
            assert config.config_path == path
            assert os.path.exists(config.config_file_path)
            # Since this operation should have created a new config file in the tempdir, this folder should now 
            # no longer be empty.
            assert len(os.listdir(path)) != 0
            
            # After the test we need to reset the config path to the previous path to avoid potential side effects
            config.set_config_path(prev_path)