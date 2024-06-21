from inspect import isfunction
import os
import argparse
import glob

VIT_MODEL = 'vit_huge_patch14_224_clip_laion2b'
VIT_FEATURE_CHANNEL = 1280
VIEW_IMAGE_RES = 224
VIT_PATCH_NUMBER = 256
CLIP_MODEL = 'ViT-L/14'
CLIP_FEATURE_CHANNEL = 768



def get_tensorboard_dir():
    if 'TENSORBOARD_LOG_DIR' in os.environ:
        tensorboard_dir = os.environ['TENSORBOARD_LOG_DIR']
    elif 'DLTS_JOB_ID' in os.environ:
        tensorboard_dir = os.path.join(os.path.expanduser(
            '~/tensorboard/{}/logs'.format(os.environ['DLTS_JOB_ID'])))
    else:
        if os.path.exists(os.path.expanduser('~/tensorboard')) is False:
            ensure_directory(os.path.expanduser('~/tensorboard/1/logs'))
        tensorboard_dir = os.path.join(
            glob(os.path.expanduser('~/tensorboard/*'))[0], 'logs')

    return tensorboard_dir


def find_best_epoch(ckpt_folder):
    try:
        ckpt_files = os.listdir(ckpt_folder)
        epochs = [int(filename.split(".")[0].split("=")[1])
                  for filename in ckpt_files]
        if len(epochs) > 0:
            return max(epochs)
        else:
            return 0
    except Exception as e:
        print(str(e))
        return None


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


def run(cmd, verbose=True):
    if verbose:
        print(cmd)
    os.system(cmd)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def update_moving_average(ma_model, current_model, ema_updater):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
