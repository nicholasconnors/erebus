import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import glob
from src.erebus import Erebus
from src.utility.run_cfg import ErebusRunConfig

runs = glob.glob("./run_cfgs/*")
for run in runs:
    print(f"Running {run}")
    erebus = Erebus(ErebusRunConfig.load(run))
    erebus.run()
    del erebus