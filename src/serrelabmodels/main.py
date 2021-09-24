import os
from serrelabmodels.ops import experiment_tools
from serrelabmodels.utils import py_utils
import sys
import logging

def main(cfg):
    logging.info(cfg.pretty())

if __name__ == "__main__":
    cfg = py_utils.get_config(sys.argv[1:])
    py_utils.setup_logging(cfg.dir)
    print('PID:',os.getpid())
    if 'gpus' in cfg and cfg.gpus is not None:
        py_utils.allocate_gpus(cfg.gpus)
    main(cfg)