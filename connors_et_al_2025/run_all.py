import glob
from src.erebus.erebus import Erebus
from src.erebus.utility.run_cfg import ErebusRunConfig

runs = glob.glob("./run_cfgs/*.yaml")
for run in runs:
    print(f"Running {run}")
    cfg = ErebusRunConfig.load(run)
    erebus = Erebus(cfg)
    erebus.run()
    del erebus