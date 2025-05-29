import sys
import os
directory = os.path.abspath(os.path.dirname(os.getcwd()) + "/src")
sys.path.append(directory)
print(directory)

from src.utility.planet import Planet
from src.utility.run_cfg import ErebusRunConfig

Planet.save_schema(directory + "/erebus/src/schema/planet_schema.json")
ErebusRunConfig.save_schema(directory + "/erebus/src/schema/run_cfg_schema.json")
