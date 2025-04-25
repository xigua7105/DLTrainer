import argparse
from util import get_cfg, init_training
from datetime import datetime
from src.trainer import get_trainer


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cfg_path', type=str, default='configs/resnet50-cifar100.yaml')
    cfg = get_cfg(**parser.parse_args().__dict__)
    cfg.__setattr__("task_start_time", datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S'))
    init_training(cfg)
    trainer = get_trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
