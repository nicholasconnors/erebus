import glob
from src.erebus import Erebus
from src.utility.run_cfg import ErebusRunConfig

runs = glob.glob("./run_cfgs/*.yaml")
for run in runs:
    print(f"Running {run}")
    cfg = ErebusRunConfig.load(run)
    erebus = Erebus(cfg)
    erebus.run()
    del erebus