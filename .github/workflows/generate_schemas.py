import sys
import os

from erebus.utility.planet import Planet
from erebus.utility.run_cfg import ErebusRunConfig

Planet._save_schema(os.path.dirname(os.getcwd()) + "/erebus/src/erebus/schema/planet_schema.json")
ErebusRunConfig._save_schema(os.path.dirname(os.getcwd()) + "/erebus/src/erebus/schema/run_cfg_schema.json")
