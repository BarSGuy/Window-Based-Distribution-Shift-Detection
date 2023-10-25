import yaml
import logging
from easydict import EasyDict as edict
from ImageNet_exp import *

import yaml
from utils import *


if __name__ == '__main__':
    args = parse_args()

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)
        cfg = edict(cfg)
    cfg = override_cfg_with_args_from_command_line(vars(args), cfg)

    cfg.DEVICE = torch.device("cuda" + ':' + str(cfg.IMAGENET.DEVICE_INDEX) if torch.cuda.is_available() else "cpu")

    # Set the seed for random module
    random.seed(cfg.IMAGENET.EXPERIMENT.SEED)
    np.random.seed(cfg.IMAGENET.EXPERIMENT.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.IMAGENET.EXPERIMENT.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Define a custom logger
    logger = logging.getLogger(__name__)

    # Set the logger's level to DEBUG
    logger.setLevel(logging.DEBUG)

    # Define a custom formatter
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    # Define a custom console handler and set its formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)
    cfg['logging'] = logger

    average_all(cfg)