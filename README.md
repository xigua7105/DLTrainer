<div align="center">
<h2>🔎A Simple Deep Learning Trainer</h2>

[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
</div>


## 🚀 Usage 
- This is a PyTorch-based deep learning trainer designed for beginners to get started quickly.

### 1. Installation
- **Clone the repo**:
    ```
    git clone https://github.com/xigua7105/DLTrainer.git
    cd DLTrainer
    ```

- **Environment setup**: DLTrainer works with Python 3.8+ and PyTorch 2.0+.
    ```
    conda create -n DLTrainer python=3.8
    conda activate DLTrainer
    pip install -r requirements.txt
    ```

### 2. QuickStart
- For example, training ResNet-50 on the CIFAR-100 dataset:
    ```
     torchrun --nproc_per_node=1 --nnodes=1 --standalone train.py --c configs/resnet50-cifar100.yaml
    ``` 

## 📝 Configs Structure
- The configs mainly consist of six major parts: model, data, optim, loss, trainer, and logger. The following is the explanation of the usage of different parameters for each part.

### 1. Model
- name (optional): Your custom naming for the network, such as convnet-lite, convnet-base, convnet-large, etc. 
- task (optional): A brief introduction to the task.
- struct (required): Parameters required for defining the model architecture.

### 2. Data
- dir (required): The path of the dataset.
- dataset_type (required): The type of the dataset, such as IRDataset (image restoration dataset), ImageFolderDefault (image classification dataset), etc. You can define the dataset in [dataset](src/data/dataset.py) and register it in [_register](src/data/_register.py). 
- is_multi_loader (required): Whether there are multiple training sets or test sets.
- train_transforms: 
  - name (required): The names of the defined transforms function. You can define and register it in [transforms](src/data/transforms.py). 
  - kwargs (optional):
- test_transforms (required):
- train_target_transforms (required):
- test_target_transforms (required):

### 3. Optim
- optimizer:
  - name (required): Such as Adam, SGD, etc. You can register them in [_register](src/optim/_register.py).
  - lr (required): Learning rate.
- scheduler:
  - name (required): Such as Cosine, MultiStepLR, etc.

### 4. Loss
- loss_terms:
  - name (required): Such as CrossEntropyLoss, IRLoss, etc. You can define and register it in [loss](src/loss).

### 5. Trainer
- name (required): The type of the Trainer, such as CLSTrainer, IRTrainer, etc. You can define it in [trainer](src/trainer) and register it in [_register](src/trainer/_register.py).
- ckpt_dir (required): The folder for saving the checkpoint.
- batch_size (required): Global train batch size.
- batch_size_test (required): Global test batch size.
- num_workers_per_gpu (required): Dataloaders parameters.
- drop_last (required): Dataloaders parameters.
- pin_memory (required): Dataloaders parameters.
- save_freq (required): The frequency of saving the checkpoint.
- amp (required): Whether to enable mixed precision to accelerate the training of the model.

### 6. Logger
- dir: The folder for saving the log.
- log_freq: The frequency of saving the log.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
