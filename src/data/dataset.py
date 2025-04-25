import os
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS


def pil_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolderDefault(ImageFolder):
    def __init__(self, root, cfg=None, is_train: bool = False, transform=None, target_transform=None):
        img_loader = pil_loader

        self.root = root
        self.is_train = is_train
        super(ImageFolderDefault, self).__init__(
            root=self.root,
            loader=img_loader,
            transform=transform,
            target_transform=target_transform,
        )

    def __getitem__(self, index: int) -> dict:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'img': sample, 'target': target}

    def __len__(self) -> int:
        return len(self.samples)


class SynDataset(Dataset):
    def __init__(self, root, cfg, is_train: bool = False, transform=None, target_transform=None):
        r"""
        This Dataset is defined for Synthetic Image Detection.
        Dataset struct are as fellow:
        |---txt
            |---train
                |---ProGAN.txt
            |---test
                |---ProGAN.txt
                |--- ...
                |---StyleGAN.txt
        |---train
            |---ProGAN
                |---0_real
                    |---xxx.png
                    |--- ...
                    |---xxx.png
                |---1_fake
                    |---xxx.png
                    |--- ...
                    |---xxx.png
        |---test
            |---ProGAN
                |---0_real
                |---1_fake
            |--- ...
            |---StyleGAN
                |---0_real
                |---1_fake
        """

        self.root = root
        self.txt_root = "{}/txt/{}/{}.txt".format(cfg.data.dir, "train" if is_train else "test", os.path.basename(root))
        self.is_train = is_train

        # self.class_to_idx = cfg.data.custom_class_to_idx if cfg.data.custom_class_to_idx is not None else None
        self.class_to_idx = {"0_real": 0, "1_fake": 1}

        self.loader = pil_loader
        self.transform = transform
        self.target_transform = target_transform

        self.samples = self.make_dataset()

    def find_classes(self):
        classes = sorted(entry.name for entry in os.scandir(self.root) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {self.root}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return class_to_idx

    def make_dataset(self):
        instance = []

        if not os.path.exists(self.txt_root):
            if not os.path.exists(os.path.dirname(self.txt_root)):
                os.makedirs(os.path.dirname(self.txt_root))

            self.class_to_idx = self.class_to_idx if self.class_to_idx is not None else self.find_classes()
            with open(self.txt_root, 'a', encoding='utf-8') as f:
                for root, _, files in os.walk(self.root):
                    for file in files:
                        if file.lower().endswith(IMG_EXTENSIONS):
                            path = os.path.join(root, file)
                            cls_name = path.replace("\\", "/").split('/')[-2]
                            assert cls_name in self.class_to_idx.keys()
                            label = self.class_to_idx[cls_name]
                            f.write(f"{path}|{label}\n")

                            item = path, label
                            instance.append(item)
        else:
            with open(self.txt_root, 'r') as f:
                for line in f:
                    item = line.strip().split('|')
                    instance.append(item)
        return instance

    def __getitem__(self, index: int) -> dict:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'img': sample, 'target': int(target)}

    def __len__(self) -> int:
        return len(self.samples)


class ImageFolderLMDB(Dataset):
    def __init__(self, cfg, is_train: bool = False, transform=None, target_transform=None):
        super().__init__()


class CIFAR100(datasets.CIFAR100):
    def __init__(self, root, cfg, is_train: bool = False, transform=None, target_transform=None):
        root = cfg.data.dir
        super().__init__(root=root, train=is_train, transform=transform, target_transform=target_transform, download=True)

    def __getitem__(self, index: int) -> dict:
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'img': img, 'target': target}


class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, cfg, is_train: bool = False, transform=None, target_transform=None):
        root = cfg.data.dir
        super().__init__(root=root, train=is_train, transform=transform, target_transform=target_transform, download=True)

    def __getitem__(self, index: int) -> dict:
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'img': img, 'target': target}


class IRDataset(Dataset):
    def __init__(self, root, cfg, is_train: bool = False, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.is_train = is_train
        self.txt_root = "{}/txt/{}/{}.txt".format(cfg.data.dir, "train" if is_train else "test", os.path.basename(root))

        self.loader = pil_loader
        self.transform = transform
        self.target_transform = target_transform

        self.samples = self.make_dataset()

    def make_dataset(self):
        instance = []

        if not os.path.exists(self.txt_root):
            if not os.path.exists(os.path.dirname(self.txt_root)):
                os.makedirs(os.path.dirname(self.txt_root))

            with open(self.txt_root, 'a', encoding='utf-8') as f:
                for root, dirs, files in os.walk(self.root):
                    for file in files:
                        if file.lower().endswith(IMG_EXTENSIONS):
                            input_type = root.split('/')[-1]
                            assert input_type in ['input', 'target']
                            if input_type == 'input':
                                input_path = os.path.join(root, file)
                                target_path = os.path.join(os.path.dirname(root), 'target', file)
                                assert os.path.exists(target_path)

                                f.write(f"{input_path}|{target_path}\n")
                                item = input_path, target_path
                                instance.append(item)
        else:
            with open(self.txt_root, 'r') as f:
                for line in f:
                    item = line.strip().split('|')
                    instance.append(item)
        return instance

    def __getitem__(self, index: int) -> dict:
        path, target = self.samples[index]
        sample = self.loader(path)
        target = self.loader(target)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'img': sample, 'target': target}

    def __len__(self) -> int:
        return len(self.samples)
