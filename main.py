import yaml
import logging
from easydict import EasyDict as edict
from ImageNet_exp import *
import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='Script to process arguments')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to configuration file')
    parser.add_argument('--IMAGENET.GET_ACCURACY_ONLY', type=bool, default=None,
                        help='To get accuracy only')
    parser.add_argument('--IMAGENET.DATASETS.PATH_TO_IMAGENET', type=str, default=None, help='Path to ImageNet dataset')
    parser.add_argument('--IMAGENET.DATASETS.PATH_TO_IMAGENET_O', type=str, default=None,
                        help='Path to ImageNet-O dataset')
    parser.add_argument('--IMAGENET.DATASETS.PATH_TO_IMAGENET_A', type=str, default=None,
                        help='Path to ImageNet-A dataset')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS', type=float, default=None,
                        help='FGSM epsilon value')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.FGSM.BATCH_SIZE', type=int, default=None,
                        help='FGSM batch size')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.CW.C', type=float, default=None, help='CW C value')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.CW.KAPPA', type=float, default=None, help='CW kappa value')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.CW.STEPS', type=int, default=None, help='CW steps')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.CW.LR', type=float, default=None, help='CW learning rate')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.PGD.EPS', type=float, default=None,
                        help='PGD epsilon value')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.PGD.STEPS', type=int, default=None, help='PGD steps')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.PGD.ALPHA', type=float, default=None,
                        help='PGD alpha value')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.PGD.RANDOM_START', type=bool, default=None,
                        help='PGD random start')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD', type=float, default=None,
                        help='Gaussian noise std')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE', type=int, default=None,
                        help='Rotation angle')
    parser.add_argument('--IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR', type=float, default=None, help='Zoom factor')
    parser.add_argument('--IMAGENET.EXPERIMENT.SEED', type=int, default=None, help='Experiment seed')
    parser.add_argument('--IMAGENET.EXPERIMENT.OOD', type=str, default=None, help='Out-of-distribution dataset name')
    parser.add_argument('--IMAGENET.EXPERIMENT.MMD_FIT_SIZE', type=int, default=None, help='MMD fit size')
    parser.add_argument('--IMAGENET.EXPERIMENT.KS_FIT_SIZE', type=int, default=None, help='KS fit size')
    parser.add_argument('--IMAGENET.EXPERIMENT.TRAIN_SIZE', type=int, default=None, help='Training size')
    parser.add_argument('--IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS', type=int, default=None, help='Number of random runs')
    parser.add_argument('--IMAGENET.EXPERIMENT.NUM_BOOTSTRAP_RUNS', type=int, default=None,
                        help='Number of bootstrap runs')
    parser.add_argument('--IMAGENET.EXPERIMENT.WINDOW_SIZES', nargs='+', type=int, default=None, help='Window sizes')
    parser.add_argument('--IMAGENET.EXPERIMENT.DETECTORS', nargs='+', default=None, help='Detectors')
    parser.add_argument('--IMAGENET.EXPERIMENT.PATH_TO_RESULTS', type=str, default=None,
                        help='Path to results directory')
    parser.add_argument('--IMAGENET.MODEL', type=str, default=None, help='Model name')
    parser.add_argument('--IMAGENET.PATH_TO_SAVE_OUTPUTS', type=str, default=None, help='Path to save outputs')
    parser.add_argument('--IMAGENET.BATCH_SIZE', type=int, default=None, help='Batch size')
    parser.add_argument('--IMAGENET.PATH_TO_SAVE_ACCURACIES', type=str, default=None, help='Path to save accuracies')
    parser.add_argument('--MY_DETECTOR.C_NUM', type=int, default=None, help='C num')
    parser.add_argument('--MY_DETECTOR.DELTA', type=float, default=None, help='Delta')
    parser.add_argument('--MY_DETECTOR.TEMP', type=float, default=None, help='Temperature')
    parser.add_argument('--MY_DETECTOR.UC_MECH', type=str, default=None, help='UC mechanism')
    parser.add_argument('--MY_DETECTOR.SIGNIFICANCE_LEVEL', type=float, default=None, help='Significance level')
    parser.add_argument('--IMAGENET.DEVICE_INDEX', type=int, default=None, help='Device index')
    parser.add_argument('--IMAGENET.NUM_WORKERS', type=int, default=None, help='Number of workers')

    parser.add_argument('--ablation', action='store_true', default=False, help='whether it is an ablation of mine or not')


    args = parser.parse_args()
    return args


def change_cfg_with_args_from_command_line(args, config):
    # Update configuration file with command line arguments
    if args['IMAGENET.GET_ACCURACY_ONLY'] is not None:
        config['IMAGENET']['GET_ACCURACY_ONLY'] = args['IMAGENET.GET_ACCURACY_ONLY']
    if args['IMAGENET.DATASETS.PATH_TO_IMAGENET'] is not None:
        config['IMAGENET']['DATASETS']['PATH_TO_IMAGENET'] = args['IMAGENET.DATASETS.PATH_TO_IMAGENET']
    if args['IMAGENET.DATASETS.PATH_TO_IMAGENET_O'] is not None:
        config['IMAGENET']['DATASETS']['PATH_TO_IMAGENET_O'] = args['IMAGENET.DATASETS.PATH_TO_IMAGENET_O']
    if args['IMAGENET.DATASETS.PATH_TO_IMAGENET_A'] is not None:
        config['IMAGENET']['DATASETS']['PATH_TO_IMAGENET_A'] = args['IMAGENET.DATASETS.PATH_TO_IMAGENET_A']
    if args['IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['FGSM']['EPS'] = args['IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS']
    if args['IMAGENET.DATASETS.GENERATING_OOD.FGSM.BATCH_SIZE'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['FGSM'][
            'BATCH_SIZE'] = args['IMAGENET.DATASETS.GENERATING_OOD.FGSM.BATCH_SIZE']
    if args['IMAGENET.DATASETS.GENERATING_OOD.CW.C'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['CW']['C'] = args['IMAGENET.DATASETS.GENERATING_OOD.CW.C']
    if args['IMAGENET.DATASETS.GENERATING_OOD.CW.KAPPA'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['CW']['KAPPA'] = args['IMAGENET.DATASETS.GENERATING_OOD.CW.KAPPA']
    if args['IMAGENET.DATASETS.GENERATING_OOD.CW.STEPS'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['CW']['STEPS'] = args['IMAGENET.DATASETS.GENERATING_OOD.CW.STEPS']
    if args['IMAGENET.DATASETS.GENERATING_OOD.CW.LR'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['CW']['LR'] = args['IMAGENET.DATASETS.GENERATING_OOD.CW.LR']
    if args['IMAGENET.DATASETS.GENERATING_OOD.PGD.EPS'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['PGD']['EPS'] = args['IMAGENET.DATASETS.GENERATING_OOD.PGD.EPS']
    if args['IMAGENET.DATASETS.GENERATING_OOD.PGD.STEPS'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['PGD'][
            'STEPS'] = args['IMAGENET.DATASETS.GENERATING_OOD.PGD.STEPS']
    if args['IMAGENET.DATASETS.GENERATING_OOD.PGD.ALPHA'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['PGD'][
            'ALPHA'] = args['IMAGENET.DATASETS.GENERATING_OOD.PGD.ALPHA']
    if args['IMAGENET.DATASETS.GENERATING_OOD.PGD.RANDOM_START'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['PGD'][
            'RANDOM_START'] = args['IMAGENET.DATASETS.GENERATING_OOD.PGD.RANDOM_START']
    if args['IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['GAUSS_NOISE'][
            'STD'] = args['IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD']
    if args['IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['ROTATE'][
            'ANGLE'] = args['IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE']
    if args['IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR'] is not None:
        config['IMAGENET']['DATASETS']['GENERATING_OOD']['ZOOM'][
            'FACTOR'] = args['IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR']
    if args['IMAGENET.EXPERIMENT.SEED'] is not None:
        config['IMAGENET']['EXPERIMENT']['SEED'] = args['IMAGENET.EXPERIMENT.SEED']
    if args['IMAGENET.EXPERIMENT.OOD'] is not None:
        config['IMAGENET']['EXPERIMENT']['OOD'] = args['IMAGENET.EXPERIMENT.OOD']
    if args['IMAGENET.EXPERIMENT.MMD_FIT_SIZE'] is not None:
        config['IMAGENET']['EXPERIMENT']['MMD_FIT_SIZE'] = args['IMAGENET.EXPERIMENT.MMD_FIT_SIZE']
    if args['IMAGENET.EXPERIMENT.KS_FIT_SIZE'] is not None:
        config['IMAGENET']['EXPERIMENT']['KS_FIT_SIZE'] = args['IMAGENET.EXPERIMENT.KS_FIT_SIZE']
    if args['IMAGENET.EXPERIMENT.TRAIN_SIZE'] is not None:
        config['IMAGENET']['EXPERIMENT']['TRAIN_SIZE'] = args['IMAGENET.EXPERIMENT.TRAIN_SIZE']
    if args['IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS'] is not None:
        config['IMAGENET']['EXPERIMENT']['NUM_RANDOM_RUNS'] = args['IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS']
    if args['IMAGENET.EXPERIMENT.NUM_BOOTSTRAP_RUNS'] is not None:
        config['IMAGENET']['EXPERIMENT']['NUM_BOOTSTRAP_RUNS'] = args['IMAGENET.EXPERIMENT.NUM_BOOTSTRAP_RUNS']
    if args['IMAGENET.EXPERIMENT.WINDOW_SIZES'] is not None:
        config['IMAGENET']['EXPERIMENT']['WINDOW_SIZES'] = args['IMAGENET.EXPERIMENT.WINDOW_SIZES']

    if args['IMAGENET.EXPERIMENT.DETECTORS'] is not None:
        config['IMAGENET']['EXPERIMENT']['DETECTORS'] = args['IMAGENET.EXPERIMENT.DETECTORS']

    if args['IMAGENET.EXPERIMENT.PATH_TO_RESULTS'] is not None:
        config['IMAGENET']['EXPERIMENT']['PATH_TO_RESULTS'] = args['IMAGENET.EXPERIMENT.PATH_TO_RESULTS']

    if args['IMAGENET.MODEL'] is not None:
        config['IMAGENET']['MODEL'] = args['IMAGENET.MODEL']

    if args['IMAGENET.PATH_TO_SAVE_OUTPUTS'] is not None:
        config['IMAGENET']['PATH_TO_SAVE_OUTPUTS'] = args['IMAGENET.PATH_TO_SAVE_OUTPUTS']

    if args['IMAGENET.BATCH_SIZE'] is not None:
        config['IMAGENET']['BATCH_SIZE'] = args['IMAGENET.BATCH_SIZE']

    if args['IMAGENET.PATH_TO_SAVE_ACCURACIES'] is not None:
        config['IMAGENET']['PATH_TO_SAVE_ACCURACIES'] = args['IMAGENET.PATH_TO_SAVE_ACCURACIES']

    if args['MY_DETECTOR.C_NUM'] is not None:
        config['MY_DETECTOR']['C_NUM'] = args['MY_DETECTOR.C_NUM']

    if args['MY_DETECTOR.DELTA'] is not None:
        config['MY_DETECTOR']['DELTA'] = args['MY_DETECTOR.DELTA']

    if args['MY_DETECTOR.TEMP'] is not None:
        config['MY_DETECTOR']['TEMP'] = args['MY_DETECTOR.TEMP']

    if args['MY_DETECTOR.UC_MECH'] is not None:
        config['MY_DETECTOR']['UC_MECH'] = args['MY_DETECTOR.UC_MECH']

    if args['MY_DETECTOR.SIGNIFICANCE_LEVEL'] is not None:
        config['MY_DETECTOR']['SIGNIFICANCE_LEVEL'] = args['MY_DETECTOR.SIGNIFICANCE_LEVEL']

    if args['IMAGENET.DEVICE_INDEX'] is not None:
        config['IMAGENET']['DEVICE_INDEX'] = args['IMAGENET.DEVICE_INDEX']
    if args['IMAGENET.NUM_WORKERS'] is not None:
        config['IMAGENET']['NUM_WORKERS'] = args['IMAGENET.NUM_WORKERS']


    return config

if __name__ == '__main__':
    args = parse_args()

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)
        cfg = edict(cfg)
    cfg = change_cfg_with_args_from_command_line(vars(args), cfg)

    cfg.DEVICE = torch.device("cuda" + ':' + str(cfg.IMAGENET.DEVICE_INDEX) if torch.cuda.is_available() else "cpu")

    # pl.seed_everything(cfg.IMAGENET.EXPERIMENT.SEED)
    # Set the seed for random module
    random.seed(cfg.SEED)

    # Set the seed for NumPy
    np.random.seed(cfg.SEED)

    # Set the seed for PyTorch
    torch.manual_seed(cfg.SEED)

    # If you're using CUDA, set the seed for CUDA as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.SEED)
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
    # save_detection_times(cfg)
    # load_and_plot_detection_times(cfg)
    # exit()
    if args.ablation:
        print("running ablation study")
        ablation_study(cfg)
    else:
        run_experiment(cfg)
    exit()
    #
    # ablation_study(cfg)
    # exit()
    # average_all_ablations(cfg)
    exit()



    ## getting detection times
    # save_detection_times(cfg)
    # load_and_plot_detection_times(cfg)
    exit()

    logits = inference_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET, suffix='_imagenet_',
                               threshold=5)
    logits = attack_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET)
    print("guy")

    # =================================================================
    # Define a custom logger
    # logger = logging.getLogger(__name__)
    #
    # # Set the logger's level to DEBUG
    # logger.setLevel(logging.DEBUG)
    #
    # # Define a custom formatter
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    #
    # # Define a custom console handler and set its formatter
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    #
    # # Add the console handler to the logger
    # logger.addHandler(console_handler)
    # cfg['logging'] = logger
    # ablation_all_oods(cfg)
    # get_average_of_ablations(cfg)
    # exit()
    # run_experiment(cfg)
    # exit()
    # evaluating_all_cifar10(cfg)
    # exit()
    #
    # evaluating_all_cifar10(cfg)
    # exit()
    #
    # run_experiment(cfg)
    # exit()
    # shift_baselines.run_and_analyze(cfg)
    # exit()
    #
    # run_and_analyze(cfg)
    #
    # exit()
    # oods = ['Cifar100', 'Cifar10_FGSM', 'Cifar10_CW', 'Cifar10_PGD', 'Cifar10_Gauss', 'Cifar10_Rotation',
    #         'Cifar10_Zoom']
    # for ood in oods:
    #     cfg.OUT_OF_DISTRIBUTION = ood
    #     if cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Gauss':
    #         stds = [0.01, 0.02, 0.03, 0.04, 0.05]
    #         for std in stds:
    #             cfg.GENERATING_OOD.GAUSS_NOISE.STD = std
    #             run_experiment(cfg)
    #
    #             analyze_experiment(cfg)
    #     elif cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Rotation':
    #         angles = [2, 5, 10, 15, 20, 25]
    #         for angle in angles:
    #             cfg.GENERATING_OOD.ROTATE.ANGLE = angle
    #             run_experiment(cfg)
    #
    #             analyze_experiment(cfg)
    #     elif cfg.OUT_OF_DISTRIBUTION == 'Cifar10_Zoom':
    #         factors = [1.3, 1.2, 1.1, 1.05, 0.95, 0.9, 0.8, 0.7]
    #         for factor in factors:
    #             cfg.GENERATING_OOD.ZOOM.FACTOR = factor
    #             run_experiment(cfg)
    #
    #             analyze_experiment(cfg)
    #     else:
    #         run_experiment(cfg)
    #
    #         analyze_experiment(cfg)
