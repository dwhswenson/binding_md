import os
import numpy as np
from pkg_resources import resource_filename

from numpy.testing import assert_array_equal, assert_allclose
import pytest

def find_testfile(fname):
    return resource_filename('binding_md', os.path.join('tests', fname))
