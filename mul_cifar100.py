from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data.dataset import ConcatDataset


class mul_CIFAR100DataLoader(data.DataLoader):

    def __init__(self, root: str, image_size: int, train: bool, batch_size: int, shuffle: bool, **kwargs):
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))

        if train:
            transform1 = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                # transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            transform2 = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4),
                transforms.RandomResizedCrop(image_size, scale=(0.3, 1.0), ratio=( 0.8, 1.2 )),
                # transforms.RandomResizedCrop(image_size,scale=(0.8, 1.2), ratio=( 4./5., 5./4. )),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            transform3 = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                # transforms.RandomResizedCrop(image_size,scale=(0.8, 1.2), ratio=( 4./5., 5./4. )),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        else:
            transform = transforms.Compose([
                # transforms.Resize(int(image_size*1.143)),
                # transforms.Resize(int(256)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])
        if train:
            dataset1 = datasets.CIFAR100(root, train=train, transform=transform1, download=False)
            dataset2 = datasets.CIFAR100(root, train=train, transform=transform2, download=False)
            # dataset3 = datasets.CIFAR100(root, train=train, transform=transform3, download=False)
            # dataset = ConcatDataset(( dataset1,dataset2, dataset3))
            dataset = ConcatDataset(( dataset1,dataset2))
        else:
            dataset = datasets.CIFAR100(root, train=train, transform=transform, download=False)
        super(mul_CIFAR100DataLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
