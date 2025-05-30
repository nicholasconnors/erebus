import sys
import os
directory = os.path.abspath(os.path.dirname(os.getcwd()) + "/src")
sys.path.append(directory)
print(directory)

from erebus.utility.planet import Planet
from erebus.utility.run_cfg import ErebusRunConfig

Planet._save_schema(directory + "/erebus/src/schema/planet_schema.json")
ErebusRunConfig._save_schema(directory + "/erebus/src/schema/run_cfg_schema.json")
