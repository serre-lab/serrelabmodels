import sys
import os
import numpy as np
from omegaconf import OmegaConf as oc
from omegaconf import DictConfig
from time import gmtime, strftime
import yaml
import logging


def allocate_gpus(num_gpus):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    free_memory = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
    gpus = [str(x) for x in np.argsort(free_memory)[::-1]]
    lf = (','.join(gpus[:num_gpus]))    
    os.environ["CUDA_VISIBLE_DEVICES"] = lf
    print('allocating gpus:', lf)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.history.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def import_module(module, prepath=''):
    """Dynamically import a module."""
    if prepath!='':
        module = prepath+'.'+module 
    return __import__(module, fromlist=[''])

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_config(cfg_path, cfg_folder='config', sep='-'):

    if oc.get_resolver('now') is None:
        oc.register_resolver('now',lambda x:strftime(x, gmtime()))

    def recursive_load(cfg):
        for k in cfg:
            if isinstance(cfg[k],str) and cfg[k].startswith(sep):
                path = os.path.join(cfg_folder, cfg[k].replace(sep,'').replace('.','/') + '.yaml')
                cfg[k] = oc.load(path)
                cfg[k] = recursive_load(cfg[k])
            elif isinstance(cfg[k],DictConfig):
                cfg[k] = recursive_load(cfg[k])
        return cfg

    cfg = oc.load(cfg_path)
    cfg = recursive_load(cfg)

    return cfg

def get_config(args, cfg_folder='config', sep='-'):
    
    if oc.get_resolver('now') is None:
        oc.register_resolver('now',lambda x:strftime(x, gmtime()))

    def recursive_load(cfg):
        for k in cfg:
            
            if isinstance(cfg[k],str) and cfg[k].startswith(sep):
                path = os.path.join(cfg_folder, cfg[k].replace(sep,'').replace('.','/') + '.yaml')
                cfg[k] = oc.load(path)
                cfg[k] = recursive_load(cfg[k])
            elif isinstance(cfg[k],DictConfig):
                cfg[k] = recursive_load(cfg[k])
            
        return cfg

    base_path = os.path.join(cfg_folder,'base.yaml')
    exp = args[0] + '.yaml'
    cfg = oc.load(exp)
    cfg = recursive_load(cfg)
    overrides = args[1:]
    if len(args) >1:
        args = args[1:]
        overrides_1 = []
        overrides_2 = []
        for i in range(len(args)):
            if '='+sep in args[i]:
                overrides_1.append(args[i])
            else:
                overrides_2.append(args[i])
        if len(overrides_1) >0:
            for o_r in overrides_1:
                cfg = oc.merge(cfg, oc.from_dotlist([o_r]))
                cfg = recursive_load(cfg)

        if len(overrides_2)>0:
            cfg = oc.merge(cfg, oc.from_dotlist(overrides_2))
    
    ensure_dir(cfg.dir)

    with open(os.path.join(cfg.dir, 'config.yaml'), 'w') as outfile:
        cf = oc.to_container(cfg,resolve=True)
        yaml.dump(cf, outfile, default_flow_style=False)

    with open(os.path.join(cfg.dir, 'overrides.yaml'), 'w') as outfile:
        yaml.dump(overrides, outfile, default_flow_style=False)
    return cfg

def setup_logging(path):
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(path,"out.log")),
                                logging.StreamHandler(sys.stdout)])