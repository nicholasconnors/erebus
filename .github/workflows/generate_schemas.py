import sys
import os
directory = os.path.abspath(os.path.dirname(os.getcwd()) + "/src/erebus")
sys.path.append(directory)
print(directory)

from .utility.planet import Planet
from .utility.run_cfg import ErebusRunConfig

Planet._save_schema(directory + "/erebus/src/erebus/schema/planet_schema.json")
ErebusRunConfig._save_schema(directory + "/erebus/src/erebus/schema/run_cfg_schema.json")
