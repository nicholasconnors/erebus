import sys
import os
directory = os.path.abspath(os.path.dirname(os.getcwd()) + "erebus/src/erebus/utility")
sys.path.append(directory)
print(directory)

from planet import Planet
from run_cfg import ErebusRunConfig

Planet._save_schema(os.path.dirname(os.getcwd()) + "erebus/src/erebus/schema/planet_schema.json")
ErebusRunConfig._save_schema(os.path.dirname(os.getcwd()) + "erebus/src/erebus/schema/run_cfg_schema.json")
