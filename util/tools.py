import os
import time
import torch
import random
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from datetime import datetime


def init_training(cfg):
    # cudnn
    if not torch.cuda.is_available():
        print("No GPU available !")
        exit(0)

    torch.cuda.empty_cache()
    cudnn.deterministic = True
    cudnn.benchmark = False

    # dist
    cfg.dist = True
    cfg.world_size, cfg.rank, cfg.local_rank = 1, 0, 0
    cfg.ngpus_per_node = torch.cuda.device_count()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.world_size = int(os.environ['WORLD_SIZE'])
        cfg.rank = int(os.environ["RANK"])
        cfg.local_rank = int(os.environ['LOCAL_RANK'])
        cfg.nnodes = cfg.world_size // cfg.ngpus_per_node
    elif 'SLURM_PROCID' in os.environ:
        cfg.rank = int(os.environ['SLURM_PROCID'])
        cfg.local_rank = cfg.rank % cfg.ngpus_per_node
        cfg.nnodes = cfg.world_size // cfg.ngpus_per_node
    else:
        cfg.dist = False
        cfg.nnodes = 1
    if cfg.dist:
        torch.cuda.set_device(cfg.local_rank)
        cfg.master = cfg.rank == cfg.logger.logger_rank
        cfg.dist_backend = 'nccl'
        dist.init_process_group(backend=cfg.dist_backend, init_method='env://')
        dist.barrier()
    else:
        cfg.master = True

    # seed
    seed = cfg.seed + cfg.local_rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # dataset
    cfg.trainer.batch_size_per_gpu = cfg.trainer.batch_size // cfg.world_size
    assert cfg.trainer.batch_size_per_gpu * cfg.world_size == cfg.trainer.batch_size
    cfg.trainer.batch_size_per_gpu_test = cfg.trainer.batch_size_test // cfg.world_size
    assert cfg.trainer.batch_size_per_gpu_test * cfg.world_size == cfg.trainer.batch_size_test
    cfg.trainer.num_workers = cfg.trainer.num_workers_per_gpu * cfg.world_size


def distribute_bn(model, world_size, dist_bn):
    # ensure every node has the same running bn stats
    model = model.module if hasattr(model, 'module') else model
    for bn_name, bn_buf in model.named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if dist_bn == 'reduce':
                # average bn stats across whole group
                torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                bn_buf /= float(world_size)
            elif dist_bn == 'broadcast':
                # broadcast bn stats from rank 0 to whole group
                torch.distributed.broadcast(bn_buf, 0)
            else:
                pass


def get_timepc(cuda_synchronize=False):
    if torch.cuda.is_available() and cuda_synchronize:
        torch.cuda.synchronize()
    return time.perf_counter()


def load_weights(model, checkpoint_path, device='cpu'):
    model_state_dict = model.state_dict()
    pretrained_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    for k, v in pretrained_dict.items():
        if k.startswith('module.'):
            k = k[7:]
            pretrained_dict[k] = v
        else:
            pretrained_dict[k] = v

    fail_load_keys, temp_dict = [], {}
    for k, v in pretrained_dict.items():
        if k in model_state_dict.keys() and np.shape(model_state_dict[k]) == np.shape(v):
            temp_dict[k] = v
        elif k in model_state_dict.keys():
            fail_load_keys.append(k)

    model_state_dict.update(temp_dict)
    model.load_state_dict(model_state_dict)


def trans_state_dict(state_dict, is_dist=True):
    state_dict_modify = dict()
    if is_dist:
        for k, v in state_dict.items():
            k = k if k.startswith('module') else 'module.'+k
            state_dict_modify[k] = v
    else:
        for k, v in state_dict.items():
            k = k[7:] if k.startswith('module') else k
            state_dict_modify[k] = v
    return state_dict_modify


def save_weights(model, path, text=None, verbose=False):
    if not os.path.exists(path):
        os.makedirs(path)
    save_filename = 'epoch_%s.pth' % text if text is not None else 'model_latest.pth'
    save_path = path + '/' + save_filename
    state_dict = model.state_dict()
    torch.save(state_dict, save_path)

    if verbose:
        print('\n{:}\nsave state dict to file:{:}\n'.format(datetime.now(), save_path))


def get_net_params(model):
    num_params, frozen_num_params = 0, 0
    for p in model.parameters():
        if not p.requires_grad:
            frozen_num_params += p.numel()
        num_params += p.numel()
    return num_params, frozen_num_params
