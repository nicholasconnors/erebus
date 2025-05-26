import sys
import os
directory = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(directory)
print(directory)

from .src.utility.planet import Planet
from .src.utility.run_cfg import ErebusRunConfig

Planet.save_schema(directory + "/src/schema/planet_schema.json")
ErebusRunConfig.save_schema(directory + "/src/schema/run_cfg_schema.json")
