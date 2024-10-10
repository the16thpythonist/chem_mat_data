
from chem_mat_data.utils import get_version


# A function name needs to start with "test_" for pytest to be able to automatically 
# discover it.
def test_get_version():
    version = get_version()
    
    # With pytest - unlike unittest - it is sufficient to use the standard "assert" syntax.
    # the second string argument to the assert statement is the error message that will be 
    # printed in case the assertion fails.
    assert isinstance(version, str), 'the version is not a string!'
    assert version != '', 'the version is an empty string'
    

