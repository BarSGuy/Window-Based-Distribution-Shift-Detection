from copy import deepcopy
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from utils import *
import torchattacks
from timeit import default_timer as timer
import sys
from methods import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import statistics
import numpy as np
import torchvision
import pprint
import matplotlib.pyplot as plt
import numpy as np


## ================================ inference ================================ ##


def infer_images(cfg, dataloader, model, threshold=100000000):
    """
    Infers all images in the dataloader through the model and returns the logits.
    Args:
        dataloader: The PyTorch dataloader containing the images to be inferred.
        model: The PyTorch model to use for inference.
    Returns:
        logits: A numpy array of logits for each image in the dataloader.
    """
    # Set the model to evaluation mode
    model.eval()

    # Check if a GPU is available
    device = cfg.DEVICE
    # Move the model to the device
    model.to(device)

    # Initialize an empty list to store the logits
    logits = []
    labels = []

    # Disable gradient calculation to speed up inference
    with torch.no_grad():
        # Loop over each batch of images in the dataloader
        counter = 0
        for batch in tqdm.tqdm(dataloader, desc='Inference'):
            counter += 1
            # Move the batch of images to the device used by the model
            inputs = batch[0].to(device)
            targets = batch[1]

            # Perform inference on the batch of images
            batch_logits = model(inputs).cpu()

            # Append the batch of logits to the list of logits
            logits.append(batch_logits)
            labels.append(targets)
            if counter == threshold:
                break
    # Concatenate the list of logits into a single numpy array
    logits = torch.cat(logits)
    labels = torch.cat(labels)
    predictions = torch.argmax(logits, dim=1)
    correct_predictions = (predictions == labels)
    accuracy = np.mean(correct_predictions.cpu().numpy())

    cfg.logging.info(f'Accuracy is {accuracy}')
    return logits, accuracy


def data_loader(cfg, path_to_dataset, batch_size, shuffle, transform):
    '''
    this function should load the dataset form the given path_to_dataset, and create a dataloder
    Args:
        path_to_dataset: A path to a dataset - assumes there are a lot of folders in this path, and in each folder, there are the images
        batch_size: The batch size of the dataset
        shuffle: The shuffle
        transform: The transform
    Returns: a dataloader object

    '''
    # Load the dataset from the specified path
    dataset = datasets.ImageFolder(path_to_dataset, transform)

    # Create a DataLoader object with the specified batch size and shuffle option
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cfg.IMAGENET.NUM_WORKERS)

    # Return the DataLoader object
    return dataloader


def load_model(model_name, embs_model=False):
    # Load the model from the Timm repository
    model = timm.create_model(model_name, pretrained=True)
    if embs_model:
        model.fc = torch.nn.Identity()
    # Save the corresponding transformations
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


def fgsm_inference(cfg, model_arch, dataloader, epsilon, device, mean, std, model_to_eval=None, threshold=10000000):
    # Instantiate the model and move it to the device
    if model_to_eval != None:
        model_to_eval = model_to_eval.to(device)
        model_to_eval.eval()
    model = model_arch.to(device)
    model.eval()

    # Instantiate the attack method
    attack = torchattacks.FGSM(model, eps=float(epsilon))

    # Disable gradient calculation since we're only doing inference
    logits = []
    targets = []
    correct = 0

    # Loop over the data in the dataloader and add a progress bar

    with tqdm.tqdm(desc="Attacking", total=len(dataloader), file=sys.stdout) as pbar:
        timer_start = timer()
        counter = 0
        for data, target in dataloader:
            if counter == threshold:
                break
            counter += 1
            # Move the data and target tensors to the device
            data, target = data.to(device), target.to(device)

            # Compute the accuracy before the attack

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Perform the FGSM attack on the batch
            attack.set_normalization_used(mean=mean.detach().numpy(), std=std.detach().numpy())
            adv_data = attack(data, target)

            # only inference
            with torch.no_grad():
                # Compute the logits on the adversarial batch
                # output = model(adv_data)
                output = -1
                if model_to_eval != None:
                    output = model_to_eval(adv_data)

            # Save the logits and targets
            logits.append(output.cpu())
            targets.append(target.cpu())
            pbar.set_description(f'Attacking: {timer() - timer_start:.3f} sec')
            pbar.update()
    # Compute the accuracy after the attack
    logits = torch.cat(logits)
    targets = torch.cat(targets)
    acc_before, acc_after = 0, 0
    if model_to_eval != None:
        pred = logits.argmax(dim=1, keepdim=True)
        correct_adv = pred.eq(targets.view_as(pred)).sum().item()
        acc_before = 100. * correct / len(dataloader.dataset)
        acc_after = 100. * correct_adv / len(dataloader.dataset)
        cfg.logging.info(f'Accuracy before attack is {acc_before}, accuracy after attack is {acc_after}')
    return logits, targets, acc_before, acc_after


def cw_inference(cfg, model_arch, dataloader, c, kappa, steps, lr, device, mean, std, model_to_eval=None,
                 threshold=100000000):
    # Instantiate the model and move it to the device
    if model_to_eval != None:
        model_to_eval = model_to_eval.to(device)
        model_to_eval.eval()
    model = model_arch.to(device)
    model.eval()

    # Instantiate the attack method
    attack = torchattacks.CW(model, c=float(c), kappa=float(kappa), steps=int(steps), lr=float(lr))

    # Disable gradient calculation since we're only doing inference
    logits = []
    targets = []
    correct = 0
    # threshold = 20
    # Loop over the data in the dataloader and add a progress bar
    with tqdm.tqdm(desc="Attacking", total=len(dataloader), file=sys.stdout) as pbar:
        timer_start = timer()
        counter = 0
        for data, target in dataloader:
            if counter == threshold:
                break
            counter += 1
            # Move the data and target tensors to the device
            data, target = data.to(device), target.to(device)

            # Compute the accuracy before the attack

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Perform the FGSM attack on the batch
            attack.set_normalization_used(mean=mean.detach().numpy(), std=std.detach().numpy())
            adv_data = attack(data, target)

            # only inference
            with torch.no_grad():
                # Compute the logits on the adversarial batch
                # output = model(adv_data)
                output = -1
                if model_to_eval != None:
                    output = model_to_eval(adv_data)

            # Save the logits and targets
            logits.append(output.cpu())
            targets.append(target.cpu())
            pbar.set_description(f'Attacking: {timer() - timer_start:.3f} sec')
            pbar.update()

    # Compute the accuracy after the attack
    logits = torch.cat(logits)
    targets = torch.cat(targets)
    acc_before, acc_after = 0, 0
    if model_to_eval != None:
        pred = logits.argmax(dim=1, keepdim=True)
        correct_adv = pred.eq(targets.view_as(pred)).sum().item()
        acc_before = 100. * correct / len(dataloader.dataset)
        acc_after = 100. * correct_adv / len(dataloader.dataset)
        cfg.logging.info(f'Accuracy before attack is {acc_before}, accuracy after attack is {acc_after}')
    return logits, targets, acc_before, acc_after


def pgd_inference(cfg, model_arch, dataloader, epsilon, alpha, steps, random_start, device, mean, std,
                  model_to_eval=None, threshold=10000000):
    # Instantiate the model and move it to the device
    if model_to_eval != None:
        model_to_eval = model_to_eval.to(device)
        model_to_eval.eval()
    model = model_arch.to(device)
    model.eval()

    # Instantiate the attack method
    attack = torchattacks.PGD(model, eps=float(epsilon), alpha=float(alpha), steps=int(steps),
                              random_start=random_start)

    # Disable gradient calculation since we're only doing inference
    logits = []
    targets = []
    correct = 0

    # Loop over the data in the dataloader and add a progress bar
    with tqdm.tqdm(desc="Attacking", total=len(dataloader), file=sys.stdout) as pbar:
        timer_start = timer()
        counter = 0
        for data, target in dataloader:
            if counter == threshold:
                break
            counter += 1
            # Move the data and target tensors to the device
            data, target = data.to(device), target.to(device)

            # Compute the accuracy before the attack

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Perform the FGSM attack on the batch
            attack.set_normalization_used(mean=mean.detach().numpy(), std=std.detach().numpy())
            adv_data = attack(data, target)

            # only inference
            with torch.no_grad():
                # Compute the logits on the adversarial batch
                # output = model(adv_data)
                output = -1
                if model_to_eval != None:
                    output = model_to_eval(adv_data)

            # Save the logits and targets
            logits.append(output.cpu())
            targets.append(target.cpu())
            pbar.set_description(f'Attacking: {timer() - timer_start:.3f} sec')
            pbar.update()
    # Compute the accuracy after the attack
    logits = torch.cat(logits)
    targets = torch.cat(targets)
    acc_before, acc_after = 0, 0
    if model_to_eval != None:
        pred = logits.argmax(dim=1, keepdim=True)
        correct_adv = pred.eq(targets.view_as(pred)).sum().item()
        acc_before = 100. * correct / len(dataloader.dataset)
        acc_after = 100. * correct_adv / len(dataloader.dataset)
        cfg.logging.info(f'Accuracy before attack is {acc_before}, accuracy after attack is {acc_after}')
    return logits, targets, acc_before, acc_after


# ================================ inference - high level ================================ #
def inference_dataset(cfg, path_to_dataset, get_embs=False, threshold=100000000, suffix=''):
    # model, transform = load_model(cfg.IMAGENET.MODEL, embs_model=get_embs)
    model, transform = load_model(cfg.IMAGENET.MODEL)
    if get_embs:
        model.fc = torch.nn.Identity()

    if suffix == '_imagenet_gaussian_noise_' + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD):
        def add_noise(x):
            return x + torch.randn_like(x) * cfg.IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD

        transform.transforms.append(add_noise)

    if suffix == '_imagenet_rotate_' + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE):
        def rotate(x):
            return torchvision.transforms.functional.rotate(x, cfg.IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE)

        transform.transforms.append(rotate)

    if suffix == '_imagenet_zoom_' + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR):
        def zoom(x):
            return torchvision.transforms.functional.affine(x, scale=cfg.IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR,
                                                            angle=0, translate=(0, 0), shear=0)

        transform.transforms.append(zoom)

    dataloader = data_loader(cfg, path_to_dataset=path_to_dataset, batch_size=cfg.IMAGENET.BATCH_SIZE, shuffle=True,
                             transform=transform)

    if get_embs:
        path = cfg.IMAGENET.PATH_TO_SAVE_OUTPUTS + "/" + '_embs_' + suffix + cfg.IMAGENET.MODEL
        if not os.path.exists(path):
            cfg.logging.info(f"Inferenceing")
            outputs, accuracy = infer_images(cfg, dataloader, model, threshold=threshold)
            save_data_to_pickle_file(
                file_path=path,
                data=outputs)
        else:
            cfg.logging.info(f"Skipping_inference")
    else:
        path = cfg.IMAGENET.PATH_TO_SAVE_OUTPUTS + "/" + '_logits_' + suffix + cfg.IMAGENET.MODEL
        if not os.path.exists(path):
            cfg.logging.info(f"Inferenceing")
            outputs, accuracy = infer_images(cfg, dataloader, model, threshold=threshold)
            save_data_to_pickle_file(
                file_path=path,
                data=outputs)
            append_csv(row_name=suffix + cfg.IMAGENET.MODEL, value=accuracy,
                       file_name=cfg.IMAGENET.PATH_TO_SAVE_ACCURACIES)
        else:
            cfg.logging.info(f"Skipping_inference")
    return path


def attack_dataset(cfg, path_to_dataset, get_embs=False, attack='fgsm'):
    model_attack, transform = load_model(cfg.IMAGENET.MODEL)
    model_for_eval = deepcopy(model_attack)
    if get_embs:
        model_for_eval.fc = torch.nn.Identity()
    # model_for_eval, _ = load_model(cfg.IMAGENET.MODEL, embs_model=get_embs)
    dataloader = data_loader(cfg, path_to_dataset=path_to_dataset,
                             batch_size=cfg.IMAGENET.DATASETS.GENERATING_OOD.FGSM.BATCH_SIZE,
                             shuffle=True,
                             transform=transform)
    mean = transform.transforms[-1].mean
    std = transform.transforms[-1].std
    if attack == 'fgsm':
        if get_embs:
            path = cfg.IMAGENET.PATH_TO_SAVE_OUTPUTS + "/" + '_embs_' + '_fgsm_' + cfg.IMAGENET.MODEL + str(
                cfg.IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS)
            if not os.path.exists(path):
                cfg.logging.info(f"Inferenceing")
                outputs, _, _, _ = fgsm_inference(cfg=cfg, model_arch=model_attack, dataloader=dataloader,
                                                  epsilon=cfg.IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS,
                                                  device=cfg.DEVICE,
                                                  model_to_eval=model_for_eval, mean=mean, std=std)
                save_data_to_pickle_file(file_path=path,
                                         data=outputs)
            else:
                cfg.logging.info(f"Skipping_inference")

        else:
            path = cfg.IMAGENET.PATH_TO_SAVE_OUTPUTS + "/" + '_logits_' + '_fgsm_' + cfg.IMAGENET.MODEL + str(
                cfg.IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS)
            if not os.path.exists(path):
                cfg.logging.info(f"Inferenceing")
                outputs, targets, acc_before, acc_after = fgsm_inference(cfg=cfg, model_arch=model_attack,
                                                                         dataloader=dataloader,
                                                                         epsilon=cfg.IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS,
                                                                         device=cfg.DEVICE,
                                                                         model_to_eval=model_for_eval, mean=mean,
                                                                         std=std)
                save_data_to_pickle_file(file_path=path,
                                         data=outputs)
                append_csv(row_name='_fgsm_' + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS) + cfg.IMAGENET.MODEL, value=acc_after,
                           file_name=cfg.IMAGENET.PATH_TO_SAVE_ACCURACIES)
            else:
                cfg.logging.info(f"Skipping_inference")
    elif attack == 'pgd':
        if get_embs:
            path = cfg.IMAGENET.PATH_TO_SAVE_OUTPUTS + "/" + '_embs_' + '_pgd_' + cfg.IMAGENET.MODEL + str(
                cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.EPS) + str(
                cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.STEPS) + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.ALPHA)
            print(path)
            if not os.path.exists(path):
                cfg.logging.info(f"Inferenceing")
                outputs, _, _, _ = pgd_inference(cfg=cfg, model_arch=model_attack, dataloader=dataloader,
                                                 epsilon=cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.EPS,
                                                 alpha=cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.ALPHA,
                                                 steps=cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.STEPS,
                                                 random_start=cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.RANDOM_START,
                                                 device=cfg.DEVICE,
                                                 model_to_eval=model_for_eval, mean=mean, std=std)
                save_data_to_pickle_file(file_path=path,
                                         data=outputs)
            else:
                cfg.logging.info(f"Skipping_inference")
        else:

            path = cfg.IMAGENET.PATH_TO_SAVE_OUTPUTS + "/" + '_logits_' + '_pgd_' + cfg.IMAGENET.MODEL + str(
                cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.EPS) + str(
                cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.STEPS) + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.ALPHA)
            print(path)
            if not os.path.exists(path):
                cfg.logging.info(f"Inferenceing")
                outputs, targets, acc_before, acc_after = pgd_inference(cfg=cfg, model_arch=model_attack,
                                                                        dataloader=dataloader,
                                                                        epsilon=cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.EPS,
                                                                        alpha=cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.ALPHA,
                                                                        steps=cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.STEPS,
                                                                        random_start=cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.RANDOM_START,
                                                                        device=cfg.DEVICE,
                                                                        model_to_eval=model_for_eval, mean=mean,
                                                                        std=std)
                save_data_to_pickle_file(file_path=path,
                                         data=outputs)
                append_csv(row_name='_pgd_' + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.EPS) + cfg.IMAGENET.MODEL, value=acc_after,
                           file_name=cfg.IMAGENET.PATH_TO_SAVE_ACCURACIES)
            else:
                cfg.logging.info(f"Skipping_inference")
    elif attack == 'cw':
        if get_embs:
            path = cfg.IMAGENET.PATH_TO_SAVE_OUTPUTS + "/" + '_embs_' + '_cw_' + cfg.IMAGENET.MODEL + str(
                cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.C) + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.KAPPA) + str(
                cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.STEPS) + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.LR)
            if not os.path.exists(path):
                cfg.logging.info(f"Inferenceing")
                outputs, _, _, _ = cw_inference(cfg=cfg, model_arch=model_attack, dataloader=dataloader,
                                                c=cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.C,
                                                kappa=cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.KAPPA,
                                                steps=cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.STEPS,
                                                lr=cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.LR,
                                                device=cfg.DEVICE,
                                                model_to_eval=model_for_eval, mean=mean, std=std)
                save_data_to_pickle_file(file_path=path,
                                         data=outputs)
            else:
                cfg.logging.info(f"Skipping_inference")
        else:
            path = cfg.IMAGENET.PATH_TO_SAVE_OUTPUTS + "/" + '_logits_' + '_cw_' + cfg.IMAGENET.MODEL + str(
                cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.C) + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.KAPPA) + str(
                cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.STEPS) + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.LR)
            if not os.path.exists(path):
                cfg.logging.info(f"Inferenceing")
                outputs, targets, acc_before, acc_after = cw_inference(cfg=cfg, model_arch=model_attack,
                                                                       dataloader=dataloader,
                                                                       c=cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.C,
                                                                       kappa=cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.KAPPA,
                                                                       steps=cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.STEPS,
                                                                       lr=cfg.IMAGENET.DATASETS.GENERATING_OOD.CW.LR,
                                                                       device=cfg.DEVICE,
                                                                       model_to_eval=model_for_eval, mean=mean, std=std)
                save_data_to_pickle_file(file_path=path,
                                         data=outputs)
                append_csv(row_name='_cw_' + cfg.IMAGENET.MODEL, value=acc_after,
                           file_name=cfg.IMAGENET.PATH_TO_SAVE_ACCURACIES)
            else:
                cfg.logging.info(f"Skipping_inference")
    else:
        raise ValueError("Unknown attack")

    return path


# ================================ getting detection times - for user ================================ #
def save_detection_times(cfg):
    # Inference and load in dist and out dist
    path = inference_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET, suffix='_imagenet_')
    logits_in = load_pickle_file(path)
    softmaxes_in = F.softmax(logits_in, dim=1)
    path = inference_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET, suffix='_imagenet_',
                             get_embs=True)
    embs_in = load_pickle_file(path)
    (logits_in, softmaxes_in, embs_in) = (
        shuffle_tensor_rows(logits_in),
        shuffle_tensor_rows(softmaxes_in),
        shuffle_tensor_rows(embs_in),
    )
    logits_in_train, logits_in_val = split_array(logits_in, cfg.IMAGENET.EXPERIMENT.TRAIN_SIZE)
    softmaxes_in_train, softmaxes_in_val = split_array(softmaxes_in, cfg.IMAGENET.EXPERIMENT.TRAIN_SIZE)
    embs_in_train, embs_in_val = split_array(embs_in, cfg.IMAGENET.EXPERIMENT.TRAIN_SIZE)
    sizes = [10, 100, 1000, 5000, 10000, 20000, 30000, 40000, 1000000]
    results_dicts = []
    cfg.IMAGENET.EXPERIMENT.MMD_FIT_SIZE = 1000
    for i in range(3):
        print('run number ' + str(i))
        # Initialize results_dict
        results_dict = edict({})
        results_dict.in_vs_in = edict({})
        for size in sizes:
            print(f'        {size=}')
            cfg.logging.info(f"Fitting detectors")
            # Mine
            # size = 1000000
            if size > 40000:
                import numpy as np
                def concatenate_array(a, N):
                    K, L = a.shape
                    if K >= N:
                        return a[:N, :]
                    else:
                        repetitions = (N + K - 1) // K
                        return torch.tensor(np.concatenate([a] * repetitions, axis=0)[:N, :])

                logits_in_train = concatenate_array(logits_in_train, sizes[-1])
                softmaxes_in_train = concatenate_array(softmaxes_in_train, sizes[-1])
                embs_in_train = concatenate_array(embs_in_train, sizes[-1])

            my_detector = MyDetector(c_num=cfg.MY_DETECTOR.C_NUM, delta=cfg.MY_DETECTOR.DELTA,
                                     temprature=cfg.MY_DETECTOR.TEMP, uncertainty_mechanism=cfg.MY_DETECTOR.UC_MECH)
            my_detector.fit(logits_in_train[:size])
            # Ks - embs
            drift_ks_embs = Ks()
            drift_ks_embs.fit(embs_in_train[:size])
            # Ks - logits
            drift_ks_logits = Ks()
            drift_ks_logits.fit(logits_in_train[:size])
            # Ks - softmax
            drift_ks_softmaxes = Ks()
            drift_ks_softmaxes.fit(softmaxes_in_train[:size])
            # Single instance - softmax
            drift_single_softmaxes = Single_SR()
            drift_single_softmaxes.fit(logits_in_train[:size])
            # Single instance - entropy
            drift_single_entropies = Single_Ent()
            drift_single_entropies.fit(logits_in_train[:size])

            if size <= cfg.IMAGENET.EXPERIMENT.MMD_FIT_SIZE:
                # Mmd - embs
                drift_mmd_embs = Mmd()
                drift_mmd_embs.fit(embs_in_train[:size].numpy())
                # Mmd - logits
                drift_mmd_logits = Mmd()
                drift_mmd_logits.fit(logits_in_train[:size].numpy())
                # Mmd - softmax
                drift_mmd_softmaxes = Mmd()
                drift_mmd_softmaxes.fit(softmaxes_in_train[:size].numpy())

            # detect in_vs_in
            window_size = 10
            cfg.logging.info(f"Detecting")
            windowed_embs_in_val, windowed_logits_in_val, windowed_softmaxes_in_val = embs_in_val[
                                                                                      :window_size], logits_in_val[
                                                                                                     :window_size], softmaxes_in_val[
                                                                                                                    :window_size]
            results_dict.in_vs_in[str(size)] = edict({
                'my_detector': my_detector.detect(windowed_logits_in_val)['detection_time'],

                'drift_ks_embs': drift_ks_embs.detect(windowed_embs_in_val)['detection_time'],
                'drift_ks_logits': drift_ks_logits.detect(windowed_logits_in_val)['detection_time'],
                'drift_ks_softmaxes': drift_ks_softmaxes.detect(windowed_softmaxes_in_val)['detection_time'],

                'drift_single_softmaxes': drift_single_softmaxes.detect(windowed_logits_in_val)['detection_time'],
                'drift_single_entropies': drift_single_entropies.detect(windowed_logits_in_val)['detection_time'],

            })
            if size <= cfg.IMAGENET.EXPERIMENT.MMD_FIT_SIZE:
                results_dict.in_vs_in[str(size)]['drift_mmd_embs'] = \
                    drift_mmd_embs.detect(windowed_embs_in_val.numpy())['detection_time']
                results_dict.in_vs_in[str(size)]['drift_mmd_logits'] = \
                    drift_mmd_logits.detect(windowed_logits_in_val.numpy())['detection_time']
                results_dict.in_vs_in[str(size)]['drift_mmd_softmaxes'] = \
                    drift_mmd_softmaxes.detect(windowed_softmaxes_in_val.numpy())['detection_time']

        pprint.pprint(results_dict)
        results_dicts.append(results_dict)
    save_data_to_pickle_file(file_path=cfg.IMAGENET.EXPERIMENT.PATH_TO_RESULTS + '/detection_times', data=results_dicts)


def load_and_plot_detection_times(cfg):
    path = cfg.IMAGENET.EXPERIMENT.PATH_TO_RESULTS + '/detection_times'
    x = load_pickle_file(path)
    new_x = []
    for dct in x:
        new_x.append(dct['in_vs_in'])

    def compute_averages(data):
        """
        Computes the average and standard deviation of each method's running time
        for each number in each run in the given list of dictionaries.

        Arguments:
        - data: a list of dictionaries, where each dictionary corresponds to a different run
                and has the format {number: {method: time}}

        Returns:
        - a dictionary with the same format as the input dictionaries, where each method's
          average and standard deviation are included as a tuple (average, std)
        """
        result = {}
        for d in data:
            for number, methods in d.items():
                if number not in result:
                    result[number] = {}

                for method, times in methods.items():
                    if method not in result[number]:
                        result[number][method] = []

                    result[number][method].append(times)

        for number, methods in result.items():
            for method, times in methods.items():
                avg = statistics.mean(times)
                std = statistics.stdev(times) if len(times) > 1 else 0
                result[number][method] = (avg, std)

        return result

    data = compute_averages(new_x)

    def apply_mapping(mapping_dict, input_str):
        """
        Apply a mapping dictionary to an input string.

        Args:
            mapping_dict (dict): A dictionary that maps strings to strings.
            input_str (str): The input string to be mapped.

        Returns:
            str: The mapped string, or the input string if no mapping is found.
        """
        if input_str in mapping_dict:
            return mapping_dict[input_str]
        else:
            return input_str

    # create lists for x values and y values for each method
    x_values = []
    y_values = {}
    std_values = {}

    mapping_dict = {
        'my_detector': 'Ours',

        'drift_ks_embs': 'KS-BBSD-EMB',
        'drift_ks_softmaxes': 'KS-BBSD-S',

        'drift_mmd_embs': 'MMD-BBSD-EMB',
        'drift_mmd_softmaxes': 'MMD-BBSD-S',

        'drift_single_softmaxes': 'Single-SR',
        'drift_single_entropies': 'Single-Ent',

    }
    for num, methods in data.items():
        x_values.append(int(num))
        for method, values in methods.items():
            if method not in y_values:
                y_values[method] = []
                std_values[method] = []
            y_values[method].append(values[0])
            std_values[method].append(values[1])

    fig, ax = plt.subplots()
    for method in y_values:
        if 'logits' in method:
            continue
        print(f'{method=}_{x_values[:len(y_values[method])]=}_{y_values[method]=}_{std_values[method]=}')
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][len(ax.lines) % len(plt.rcParams['axes.prop_cycle'])]
        ax.plot(x_values[:len(y_values[method])], y_values[method], label=apply_mapping(mapping_dict, method),
                marker='o', markersize=3, color=color)
        ax.fill_between(x_values[:len(y_values[method])],
                        [y - std for y, std in zip(y_values[method], std_values[method])],
                        [y + std for y, std in zip(y_values[method], std_values[method])], alpha=0.3)

        # Add text for the last value of y_values[method]
        last_y_value = y_values[method][-1]
        log_last_y_value = np.log10(last_y_value)
        log_max_y_value = np.log10(max(y_values[method]))

        # Determine text position based on the flag
        text_position = 'above'  # set to 'above' or 'below'

        text_y_offset = 0.05 if text_position == 'above' else -0.05
        text_y_pos = 10 ** (log_last_y_value + text_y_offset * (log_max_y_value - log_last_y_value))

        # Format the last value in scientific notation
        mantissa, exponent = f'{last_y_value:.2e}'.split('e')
        mantissa = float(mantissa)
        exponent = int(exponent)

        # Add text to the plot
        text_label = f'{mantissa:.2f}$\\times10^{{{exponent}}}$' + '(s)'
        ha = 'center'
        va = 'bottom' if text_position == 'above' else 'top'
        fontsize = 12
        if method == 'my_detector' or method == 'drift_ks_softmaxes' or method == 'drift_single_softmaxes' or method == 'drift_mmd_embs':
            ax.text(x_values[len(y_values[method]) - 1], text_y_pos, text_label,
                    ha=ha, va=va, fontsize=fontsize, color=color)

    # Add arrow between method1 and method2

    # y1 = y_values['my_detector'][-1]
    # y2 = y_values['drift_ks_softmaxes'][-1]
    # order_of_mag_diff = math.ceil(np.log10(y2 / y1))
    # diff_text = f''
    # # Add arrow and text
    # arrow_style = dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=0.3")
    # ax.annotate(diff_text, xy=(x_values[-1], y1), xytext=(x_values[-1], y2),
    #             arrowprops=arrow_style, ha='center', va='center')

    # # diff = y2 - y1
    #
    # order_of_mag_diff = math.ceil(np.log10(y2 / y1))
    # diff_text = f'X{order_of_mag_diff}'
    # diff_x_pos = x_values[len(y_values['my_detector']) - 1] + 0.1
    # diff_y_pos = 10 ** ((np.log10(y1) + np.log10(y2)) / 2.0)
    # # diff_y_pos = (y1 + y2) / 2.0
    # ax.annotate(diff_text, xy=(x_values[len(y_values['my_detector']) - 1], y2), xytext=(diff_x_pos, diff_y_pos),
    #             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black', lw=1), fontsize=12)
    #
    # # plot the data with error bars
    # fig, ax = plt.subplots()
    # for method in y_values:
    #     if 'logits' in method:
    #         continue
    #     print(f'{method=}_{x_values[:len(y_values[method])]=}_{y_values[method]=}_{std_values[method]=}')
    #     color = plt.rcParams['axes.prop_cycle'].by_key()['color'][len(ax.lines) % len(plt.rcParams['axes.prop_cycle'])]
    #     ax.plot(x_values[:len(y_values[method])], y_values[method], label=apply_mapping(mapping_dict, method),
    #             marker='o', markersize=3, color=color)
    #     ax.fill_between(x_values[:len(y_values[method])],
    #                     [y - std for y, std in zip(y_values[method], std_values[method])],
    #                     [y + std for y, std in zip(y_values[method], std_values[method])], alpha=0.3)
    #
    #     # Add text for the last value of y_values[method]
    #     last_y_value = y_values[method][-1]
    #     log_max_y_value = np.log10(max(y_values[method]))
    #
    #     # Add text for the last value of y_values[method]
    #     text_y_pos = 10 ** (log_last_y_value + 0.05 * (log_max_y_value - log_last_y_value))
    #
    #     # Format the last value in scientific notation
    #     mantissa, exponent = f'{last_y_value:.2e}'.split('e')
    #     mantissa = float(mantissa)
    #     exponent = int(exponent)
    #     if apply_mapping(mapping_dict, method) == 'KS-BBSD-S' or apply_mapping(mapping_dict, method) == 'Single-SR' or apply_mapping(mapping_dict, method) == 'Ours':
    #         ax.text(x_values[len(y_values[method]) - 1], text_y_pos, f'{mantissa:.2f}$\\times10^{{{exponent}}}$' + '(s)',
    #                 ha='center', va='bottom', fontsize=8, color=color)
    #
    #     # # Add text for the last value of y_values[method]
    #     # last_y_value = y_values[method][-1]
    #     # ax.text(x_values[len(y_values[method])-1], last_y_value + 0.1, f'{last_y_value:.2f} (s)',
    #     #         ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Source size')
    ax.set_ylabel('Run time (s)')
    ax.set_xticks(x_values)
    ax.set_xscale('log')  # set the x-axis to log scale
    ax.set_ylim(0.0001, 300)  # manually set the limits of the y-axis
    ax.set_yscale('log')  # set the y-axis to log scale
    ax.grid(True)  # add grid
    legend = ax.legend(frameon=True, fancybox=True, shadow=True, borderpad=1, fontsize='small', loc='upper left')
    legend.get_frame().set_alpha(0.3)  # set the transparency of the legend box
    legend.set_title('Methods', prop={
        'size': 'small'})  # set the legend title

    # Save the plot
    path_to_save = cfg.IMAGENET.EXPERIMENT.PATH_TO_RESULTS + '/Run_times_updated'
    plt.savefig(path_to_save)
    plt.show()

    #
    # def plot_data(cfg, data):
    #     # Create the plot
    #     fig, ax = plt.subplots()
    #
    #     # Set the x-axis ticks and labels
    #     xticks = list(data.keys())
    #     xticks = [int(x) for x in xticks]
    #     ax.set_xticks(xticks)
    #     ax.set_xticklabels(xticks)
    #
    #     # Set the y-axis ticks and labels
    #     yticks = np.arange(0, 7, 0.5)
    #     ax.set_yticks(yticks)
    #     ax.set_yticklabels([f'{t:.1f}' for t in yticks])
    #
    #     # Iterate over each method and plot its data
    #
    #     for i, method in enumerate(data['10'].keys()):
    #         if 'mmd' in method.split('_'):
    #             x = list(data.keys())[:4]
    #         else:
    #             x = list(data.keys())
    #         # Set the x-axis ticks and labels
    #         xticks = list(data.keys())
    #         xticks = [int(x) for x in xticks]
    #         ax.set_xticks(xticks)
    #         ax.set_xticklabels(xticks)
    #         y = [data[n][method][0] for n in x]
    #         yerr = [data[n][method][1] for n in x]
    #
    #         ax.fill_between(x, [a-b for a,b in zip(y, yerr)], [a+b for a,b in zip(y, yerr)], alpha=0.3, label=method)
    #         ax.plot(x, y, '-o', markersize=3, label=None)
    #
    #     # Add labels and legend
    #     ax.set_xlabel('Number')
    #     ax.set_ylabel('Running time (s)')
    #     ax.legend(title='Method')
    #
    #
    #     # Set the plot title and axis scales
    #     ax.set_title('Running Time by Method')
    #     # ax.set_yscale('log')
    #
    #     # Save the plot
    #     path_to_save = cfg.IMAGENET.EXPERIMENT.PATH_TO_RESULTS + '/run_times' + 'png'
    #     plt.savefig(path_to_save)
    #
    #     # Show the plot
    #     plt.show()
    #
    # plot_data(cfg, data)


# ================================ experiments - for user ================================ #
def run_experiment(cfg):
    # Inference and load in dist and out dist
    path = inference_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET, suffix='_imagenet_')
    logits_in = load_pickle_file(path)
    softmaxes_in = F.softmax(logits_in, dim=1)
    path = inference_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET, suffix='_imagenet_',
                             get_embs=True)
    embs_in = load_pickle_file(path)

    if cfg.IMAGENET.EXPERIMENT.OOD == 'imagenet_o':
        path = inference_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET_O, suffix='_imagenet_o_')
        logits_out = load_pickle_file(path)
        softmaxes_out = F.softmax(logits_out, dim=1)
        path = inference_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET_O, suffix='_imagenet_o_',
                                 get_embs=True)
        embs_out = load_pickle_file(path)
    if cfg.IMAGENET.EXPERIMENT.OOD == 'imagenet_a':
        path = inference_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET_A, suffix='_imagenet_a_')
        logits_out = load_pickle_file(path)
        softmaxes_out = F.softmax(logits_out, dim=1)
        path = inference_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET_A, suffix='_imagenet_a_',
                                 get_embs=True)
        embs_out = load_pickle_file(path)
    if cfg.IMAGENET.EXPERIMENT.OOD == 'fgsm' or cfg.IMAGENET.EXPERIMENT.OOD == 'pgd' or cfg.IMAGENET.EXPERIMENT.OOD == 'cw':
        path = attack_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET,
                              attack=cfg.IMAGENET.EXPERIMENT.OOD)
        logits_out = load_pickle_file(path)
        softmaxes_out = F.softmax(logits_out, dim=1)
        path = attack_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET, get_embs=True,
                              attack=cfg.IMAGENET.EXPERIMENT.OOD)
        embs_out = load_pickle_file(path)
    if cfg.IMAGENET.EXPERIMENT.OOD == 'imagenet_gaussian_noise':
        path = inference_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET,
                                 suffix='_imagenet_gaussian_noise_' + str(
                                     cfg.IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD))
        logits_out = load_pickle_file(path)
        softmaxes_out = F.softmax(logits_out, dim=1)
        path = inference_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET,
                                 suffix='_imagenet_gaussian_noise_' + str(
                                     cfg.IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD),
                                 get_embs=True)
        embs_out = load_pickle_file(path)
    if cfg.IMAGENET.EXPERIMENT.OOD == 'imagenet_rotate':
        path = inference_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET,
                                 suffix='_imagenet_rotate_' + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE))
        logits_out = load_pickle_file(path)
        softmaxes_out = F.softmax(logits_out, dim=1)
        path = inference_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET,
                                 suffix='_imagenet_rotate_' + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE),
                                 get_embs=True)
        embs_out = load_pickle_file(path)
    if cfg.IMAGENET.EXPERIMENT.OOD == 'imagenet_zoom':
        path = inference_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET,
                                 suffix='_imagenet_zoom_' + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR))
        logits_out = load_pickle_file(path)
        softmaxes_out = F.softmax(logits_out, dim=1)
        path = inference_dataset(cfg, path_to_dataset=cfg.IMAGENET.DATASETS.PATH_TO_IMAGENET,
                                 suffix='imagenet_zoom_' + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR),
                                 get_embs=True)
        embs_out = load_pickle_file(path)

    if cfg.IMAGENET.GET_ACCURACY_ONLY:
        return
    # detecting shifts #
    result_dicts = []
    for i in range(cfg.IMAGENET.EXPERIMENT.NUM_RANDOM_RUNS):
        cfg.logging.info(f"Running experiment number {i}")
        # Shuffle input lists to ensure randomness
        (logits_in, logits_out, softmaxes_in, softmaxes_out, embs_in, embs_out) = (
            shuffle_tensor_rows(logits_in),
            shuffle_tensor_rows(logits_out),
            shuffle_tensor_rows(softmaxes_in),
            shuffle_tensor_rows(softmaxes_out),
            shuffle_tensor_rows(embs_in),
            shuffle_tensor_rows(embs_out)
        )
        # Split the in-distribution data into train and validation sets
        logits_in_train, logits_in_val = split_array(logits_in, cfg.IMAGENET.EXPERIMENT.TRAIN_SIZE)
        softmaxes_in_train, softmaxes_in_val = split_array(softmaxes_in, cfg.IMAGENET.EXPERIMENT.TRAIN_SIZE)
        embs_in_train, embs_in_val = split_array(embs_in, cfg.IMAGENET.EXPERIMENT.TRAIN_SIZE)
        cfg.logging.info(f"Fitting detectors")

        # fitting detectors
        # Mine
        if 'my_detector' in cfg.IMAGENET.EXPERIMENT.DETECTORS:
            my_detector = MyDetector(c_num=cfg.MY_DETECTOR.C_NUM, delta=cfg.MY_DETECTOR.DELTA,
                                     temprature=cfg.MY_DETECTOR.TEMP, uncertainty_mechanism=cfg.MY_DETECTOR.UC_MECH)
            my_detector.fit(logits_in_train)
        # Ks - embs
        if 'drift_ks_embs' in cfg.IMAGENET.EXPERIMENT.DETECTORS:
            drift_ks_embs = Ks()
            drift_ks_embs.fit(embs_in_train[:cfg.IMAGENET.EXPERIMENT.KS_FIT_SIZE])
        # Ks - logits
        if 'drift_ks_logits' in cfg.IMAGENET.EXPERIMENT.DETECTORS:
            drift_ks_logits = Ks()
            drift_ks_logits.fit(logits_in_train[:cfg.IMAGENET.EXPERIMENT.KS_FIT_SIZE])
        # Ks - softmax
        if 'drift_ks_softmaxes' in cfg.IMAGENET.EXPERIMENT.DETECTORS:
            drift_ks_softmaxes = Ks()
            drift_ks_softmaxes.fit(softmaxes_in_train[:cfg.IMAGENET.EXPERIMENT.KS_FIT_SIZE])

        # Mmd - embs
        if 'drift_mmd_embs' in cfg.IMAGENET.EXPERIMENT.DETECTORS:
            drift_mmd_embs = Mmd(device=cfg.DEVICE)
            drift_mmd_embs.fit(embs_in_train[:cfg.IMAGENET.EXPERIMENT.MMD_FIT_SIZE].numpy())
        # Mmd - logits
        if 'drift_mmd_logits' in cfg.IMAGENET.EXPERIMENT.DETECTORS:
            drift_mmd_logits = Mmd(device=cfg.DEVICE)
            drift_mmd_logits.fit(logits_in_train[:cfg.IMAGENET.EXPERIMENT.MMD_FIT_SIZE].numpy())
        # Mmd - softmax
        if 'drift_mmd_softmaxes' in cfg.IMAGENET.EXPERIMENT.DETECTORS:
            drift_mmd_softmaxes = Mmd(device=cfg.DEVICE)
            drift_mmd_softmaxes.fit(softmaxes_in_train[:cfg.IMAGENET.EXPERIMENT.MMD_FIT_SIZE].numpy())

        # Single instance - softmax
        if 'drift_single_softmaxes' in cfg.IMAGENET.EXPERIMENT.DETECTORS:
            drift_single_softmaxes = Single_SR()
            drift_single_softmaxes.fit(logits_in_train)
        # Single instance - entropy
        if 'drift_single_entropies' in cfg.IMAGENET.EXPERIMENT.DETECTORS:
            drift_single_entropies = Single_Ent()
            drift_single_entropies.fit(logits_in_train)
        # Initialize results_dict
        results_dict = edict({})
        results_dict.window_sizes = cfg.IMAGENET.EXPERIMENT.WINDOW_SIZES
        results_dict.in_vs_in = edict({})
        results_dict.in_vs_out = edict({})
        for window_size in cfg.IMAGENET.EXPERIMENT.WINDOW_SIZES:
            # detect in_vs_out
            cfg.logging.info(f"Detecting shifts for window size {window_size}")
            windowed_embs_out, windowed_logits_out, windowed_softmaxes_out = embs_out[:window_size], logits_out[
                                                                                                     :window_size], softmaxes_out[
                                                                                                                    :window_size]
            results_dict.in_vs_out[str(window_size)] = edict({
                'my_detector': my_detector.detect(
                    windowed_logits_out) if 'my_detector' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,

                'drift_ks_embs': drift_ks_embs.detect(
                    windowed_embs_out) if 'drift_ks_embs' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,
                'drift_ks_logits': drift_ks_logits.detect(
                    windowed_logits_out) if 'drift_ks_logits' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,
                'drift_ks_softmaxes': drift_ks_softmaxes.detect(
                    windowed_softmaxes_out) if 'drift_ks_softmaxes' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,

                'drift_mmd_embs': drift_mmd_embs.detect(
                    windowed_embs_out.numpy()) if 'drift_mmd_embs' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,
                'drift_mmd_logits': drift_mmd_logits.detect(
                    windowed_logits_out.numpy()) if 'drift_mmd_logits' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,
                'drift_mmd_softmaxes': drift_mmd_softmaxes.detect(
                    windowed_softmaxes_out.numpy()) if 'drift_mmd_softmaxes' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,

                'drift_single_softmaxes': drift_single_softmaxes.detect(
                    windowed_logits_out) if 'drift_single_softmaxes' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,
                'drift_single_entropies': drift_single_entropies.detect(
                    windowed_logits_out) if 'drift_single_entropies' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,

            })
            # detect in_vs_in
            windowed_embs_in_val, windowed_logits_in_val, windowed_softmaxes_in_val = embs_in_val[
                                                                                      :window_size], logits_in_val[
                                                                                                     :window_size], softmaxes_in_val[
                                                                                                                    :window_size]
            results_dict.in_vs_in[str(window_size)] = edict({
                'my_detector': my_detector.detect(
                    windowed_logits_in_val) if 'my_detector' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,

                'drift_ks_embs': drift_ks_embs.detect(
                    windowed_embs_in_val) if 'drift_ks_embs' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,
                'drift_ks_logits': drift_ks_logits.detect(
                    windowed_logits_in_val) if 'drift_ks_logits' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,
                'drift_ks_softmaxes': drift_ks_softmaxes.detect(
                    windowed_softmaxes_in_val) if 'drift_ks_softmaxes' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,

                'drift_mmd_embs': drift_mmd_embs.detect(
                    windowed_embs_in_val.numpy()) if 'drift_mmd_embs' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,
                'drift_mmd_logits': drift_mmd_logits.detect(
                    windowed_logits_in_val.numpy()) if 'drift_mmd_logits' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,
                'drift_mmd_softmaxes': drift_mmd_softmaxes.detect(
                    windowed_softmaxes_in_val.numpy()) if 'drift_mmd_softmaxes' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,

                'drift_single_softmaxes': drift_single_softmaxes.detect(
                    windowed_logits_in_val) if 'drift_single_softmaxes' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,
                'drift_single_entropies': drift_single_entropies.detect(
                    windowed_logits_in_val) if 'drift_single_entropies' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None,

            })

        result_dicts.append(results_dict)

    # evaluating detections
    cfg.logging.info(f"Starting evaluation")
    for window_size in cfg.IMAGENET.EXPERIMENT.WINDOW_SIZES:
        cfg.logging.info(f"Evaluating shifts for window size {window_size}")
        shift_extra_params = ''
        if cfg.IMAGENET.EXPERIMENT.OOD == 'imagenet_gaussian_noise':
            shift_extra_params = '_' + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD) + '_'
        if cfg.IMAGENET.EXPERIMENT.OOD == 'imagenet_zoom':
            shift_extra_params = '_' + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR) + '_'
        if cfg.IMAGENET.EXPERIMENT.OOD == 'imagenet_rotate':
            shift_extra_params = '_' + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE) + '_'
        if cfg.IMAGENET.EXPERIMENT.OOD == 'fgsm' or cfg.IMAGENET.EXPERIMENT.OOD == 'pgd':
            if cfg.IMAGENET.EXPERIMENT.OOD == 'fgsm':
                shift_extra_params = '_' + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.FGSM.EPS) + '_'
            if cfg.IMAGENET.EXPERIMENT.OOD == 'pgd':
                shift_extra_params = '_' + str(cfg.IMAGENET.DATASETS.GENERATING_OOD.PGD.EPS) + '_'
        save_data_my_detector_imagenet(cfg=cfg, result_dicts=result_dicts, window_size=window_size,
                                       threshold=1 - cfg.MY_DETECTOR.SIGNIFICANCE_LEVEL,
                                       suffix=f'_temp_{cfg.MY_DETECTOR.TEMP}_c_num_{cfg.MY_DETECTOR.C_NUM}_delta_{cfg.MY_DETECTOR.DELTA}_uc_mech_{cfg.MY_DETECTOR.UC_MECH}',
                                       shift_extra_params=shift_extra_params) if 'my_detector' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None

        save_data_drift_ks_embs_imagenet(cfg=cfg, result_dicts=result_dicts, window_size=window_size,
                                         threshold=1 - cfg.MY_DETECTOR.SIGNIFICANCE_LEVEL / embs_in_train.shape[
                                             1],
                                         shift_extra_params=shift_extra_params) if 'drift_ks_embs' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None
        save_data_drift_ks_logits_imagenet(cfg=cfg, result_dicts=result_dicts, window_size=window_size,
                                           threshold=1 - cfg.MY_DETECTOR.SIGNIFICANCE_LEVEL / logits_in_train.shape[
                                               1],
                                           shift_extra_params=shift_extra_params) if 'drift_ks_logits' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None
        save_data_drift_ks_softmaxes_imagenet(cfg=cfg, result_dicts=result_dicts, window_size=window_size,
                                              threshold=1 - cfg.MY_DETECTOR.SIGNIFICANCE_LEVEL /
                                                        softmaxes_in_train.shape[
                                                            1],
                                              shift_extra_params=shift_extra_params) if 'drift_ks_softmaxes' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None

        save_data_drift_mmd_embs_imagenet(cfg=cfg, result_dicts=result_dicts, window_size=window_size,
                                          threshold=1 - cfg.MY_DETECTOR.SIGNIFICANCE_LEVEL,
                                          shift_extra_params=shift_extra_params) if 'drift_mmd_embs' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None
        save_data_drift_mmd_logits_imagenet(cfg=cfg, result_dicts=result_dicts, window_size=window_size,
                                            threshold=1 - cfg.MY_DETECTOR.SIGNIFICANCE_LEVEL,
                                            shift_extra_params=shift_extra_params) if 'drift_mmd_logits' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None
        save_data_drift_mmd_softmaxes_imagenet(cfg=cfg, result_dicts=result_dicts, window_size=window_size,
                                               threshold=1 - cfg.MY_DETECTOR.SIGNIFICANCE_LEVEL,
                                               shift_extra_params=shift_extra_params) if 'drift_mmd_softmaxes' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None

        save_data_drift_single_softmaxes_imagenet(cfg=cfg, result_dicts=result_dicts, window_size=window_size,
                                                  threshold=1 - cfg.MY_DETECTOR.SIGNIFICANCE_LEVEL,
                                                  shift_extra_params=shift_extra_params) if 'drift_single_softmaxes' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None
        save_data_drift_single_entropies_imagenet(cfg=cfg, result_dicts=result_dicts, window_size=window_size,
                                                  threshold=1 - cfg.MY_DETECTOR.SIGNIFICANCE_LEVEL,
                                                  shift_extra_params=shift_extra_params) if 'drift_single_entropies' in cfg.IMAGENET.EXPERIMENT.DETECTORS else None


def ablation_study(cfg):
    cfg.IMAGENET.EXPERIMENT.DETECTORS = ['my_detector']
    # oods = ['imagenet_a', 'fgsm', 'pgd', 'cw', 'imagenet_o', 'imagenet_zoom', 'imagenet_gaussian_noise',
    #         'imagenet_rotate']
    c_nums = [50]
    deltas = [0.01, 0.001]
    temps = [100, 10, 5, 1, 0.1]
    uc_mechs = ['SR']
    # c_nums = [100]
    # deltas = [0.001]
    # temps = [10]
    # uc_mechs = ['SR']
    c_nums = [50]
    deltas = [0.01, 0.001]
    temps = [100, 10, 5, 1, 0.1]
    uc_mechs = ['SR', 'Ent']

    # ====
    c_nums = [10, 25, 50, 100]
    deltas = [0.0001, 0.001, 0.01, 0.05]
    temps = [1]
    uc_mechs = ['SR', 'Ent']
    for uc_mech in uc_mechs:
        for c_num in c_nums:
            for delta in deltas:
                for tmp in temps:

                    cfg.MY_DETECTOR.C_NUM = c_num
                    cfg.MY_DETECTOR.DELTA = delta
                    cfg.MY_DETECTOR.TEMP = tmp
                    cfg.MY_DETECTOR.UC_MECH = uc_mech
                    print(f'{c_num=}, {delta=}, {tmp=}, {uc_mech=}')
                    # for ood in oods:
                    #     cfg.IMAGENET.EXPERIMENT.OOD = ood
                    ood = cfg.IMAGENET.EXPERIMENT.OOD
                    if ood == 'imagenet_gaussian_noise':
                        # stds = [0.01, 0.02, 0.03, 0.04, 0.05]
                        # for std in stds:
                        #     cfg.IMAGENET.DATASETS.GENERATING_OOD.GAUSS_NOISE.STD = std
                        run_experiment(cfg)
                    elif ood == 'imagenet_zoom':
                        # factors = [0.8, 0.9, 0.95, 1.05, 1.1, 1.2]
                        # for factor in factors:
                        #     cfg.IMAGENET.DATASETS.GENERATING_OOD.ZOOM.FACTOR = factor
                        run_experiment(cfg)
                    elif ood == 'imagenet_rotate':
                        # angles = [5, 10, 15, 20, 25]
                        # for angle in angles:
                        #     cfg.IMAGENET.DATASETS.GENERATING_OOD.ROTATE.ANGLE = angles
                        run_experiment(cfg)
                    else:
                        run_experiment(cfg)


def average_all_baselines(cfg):
    baselines = ['drift_ks_embs', 'drift_ks_logits', 'drift_ks_softmaxes',
                 'drift_mmd_embs', 'drift_mmd_logits', 'drift_mmd_softmaxes', 'drift_single_softmaxes',
                 'drift_single_entropies']
    for baseline in baselines:
        for window_size in cfg.IMAGENET.EXPERIMENT.WINDOW_SIZES:
            calculate_averages(parent_path='/home/guy.b/code_updated/imagenet_experiments/imagenet',
                               file_name=str(baseline) + '_window_' + str(
                                   window_size) + '.csv',
                               new_file_name='AVG_ALL_' + str(baselines) + '_window_' + str(
                                   window_size) + '.csv')


def average_all_ablations(cfg):
    c_nums = [5, 10, 20, 30]
    deltas = [0.01, 0.001]
    temps = [2, 1, 0.5]
    uc_mechs = ['SR', 'Ent']
    for window_size in cfg.IMAGENET.EXPERIMENT.WINDOW_SIZES:
        for uc_mech in uc_mechs:
            for c_num in c_nums:
                for delta in deltas:
                    for tmp in temps:
                        suffix_for_file_name = f'_temp_{tmp}_c_num_{c_num}_delta_{delta}_uc_mech_{uc_mech}'
                        # suffix_for_file_name = f'_temp_{cfg.MY_DETECTOR.TEMP}_c_num_{cfg.MY_DETECTOR.C_NUM}_delta_{cfg.MY_DETECTOR.DELTA}_uc_mech_{cfg.MY_DETECTOR.UC_MECH}'
                        calculate_averages(parent_path='/home/guy.b/code_updated/imagenet_experiments/imagenet',
                                           file_name='my_detector_window_' + str(
                                               window_size) + suffix_for_file_name + '.csv',
                                           new_file_name='AVG_ALL_my_detector_window_' + str(
                                               window_size) + suffix_for_file_name + '.csv')
