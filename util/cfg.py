import yaml
from argparse import Namespace


class Config:
    def __init__(
            self,
            model_name: str = 'ConvNet-lite',
            trainer_name: str = 'cls_name',
            task: str = 'cls',
            seed: int = 42,
            total_epochs: int = 100,
            warmup_epochs: int = 5,
            test_start_epoch: int = 50,
            batch_size: int = 32,
            test_batch_size: int = 32,
            optimizer: str = 'Adam',
            lr: float = 5e-4,
            weight_decay: float = 5e-2,
            betas: tuple = (0.9, 0.999),
            loss_fn: str = 'CrossEntropyLoss',
            dataset_type: str = "custom",
            dataset_dir: str = None,
            ckpt_dir: str = 'checkpoints',
            resume_dir: str = None,
            log_dir: str = 'logs'
    ):
        model = Namespace()
        model.name = model_name
        model.task = task
        self.model = model

        data = Namespace()
        data.dir = dataset_dir
        data.dataset_type = dataset_type
        data.train_transforms = None
        data.test_transforms = None
        self.data = data

        optim = Namespace()
        optim.optimizer = dict(name=optimizer, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)
        optim.scheduler = None
        self.optim = optim

        loss = Namespace()
        loss.loss_terms = dict(name=loss_fn)
        self.loss = loss

        trainer = Namespace()
        trainer.name = trainer_name
        trainer.ckpt_dir = ckpt_dir
        trainer.resume_dir = resume_dir
        trainer.save_per_epoch = 5
        trainer.test_per_epoch = 1
        trainer.batch_size = batch_size
        trainer.batch_size_per_gpu = None
        trainer.batch_size_test = test_batch_size
        trainer.batch_size_per_gpu_test = None
        trainer.num_workers_per_gpu = 8
        trainer.drop_last = True
        trainer.pin_memory = True
        trainer.persistent_workers = False
        trainer.scaler = "native"
        self.trainer = trainer

        logger = Namespace()
        logger.dir = log_dir
        logger.freq = 100
        logger.logger_rank = 0
        self.logger = logger

        self.seed = seed
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.test_start_epoch = test_start_epoch

    def get_cfg_dict(self):
        cfg_dict = dict()
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                for k, v in value.__dict__.items():
                    cfg_dict[f'{key} | {k}'] = v
            else:
                cfg_dict[key] = value
        return cfg_dict


def read_yaml_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Not Found [{file_path}].")
    except yaml.YAMLError as e:
        print(f"Error: {e}")
    return None


def get_cfg(cfg_path=None, **kwargs):

    base_cfg = Config(**kwargs)
    if cfg_path is None:
        return base_cfg

    update_cfg = read_yaml_config(cfg_path)
    yaml_cfg_keys = update_cfg.keys()

    update_cfg_keys = ['model', 'data', 'optim', 'loss', 'trainer', 'logger']
    for cfg_k in update_cfg_keys:
        if cfg_k in yaml_cfg_keys:
            _update_cfg = update_cfg.pop(cfg_k)
            for k, v in _update_cfg.items():
                base_cfg.__dict__[cfg_k].__setattr__(k, v)

    for k, v in update_cfg.items():
        if isinstance(v, dict):
            base_cfg.__setattr__(k, Namespace())
            for _k, _v in v.items():
                base_cfg.__dict__[k].__setattr__(_k, _v)
        else:
            base_cfg.__setattr__(k, v)

    return base_cfg


if __name__ == "__main__":

    configs = get_cfg(r"F:\DL\My_Github_Open_Source\DLTrainer-main\configs\test.yaml")

    for k, v in configs.__dict__.items():
        if hasattr(v, '__dict__'):
            v_dict = v.__dict__
            for _k, _v in v_dict.items():
                print(f"{k:<20}{_k}: {_v}")
        else:
            print(k, v)
