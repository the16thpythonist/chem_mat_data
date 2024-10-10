import os

from dotenv import load_dotenv


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
        
    def get_fileshare_url(self) -> str:
        """
        Returns the fileshare URL which was loaded from the environment variables. This URL points to 
        a cloud folder which contains all the actual dataset files and from where the datasets will
        be downloaded ultimately.
        
        :returns: string absolute URL
        """
        return self.fileshare_url
