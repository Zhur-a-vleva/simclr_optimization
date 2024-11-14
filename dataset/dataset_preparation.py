from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms, datasets


class SimCLRTransform:
    def __init__(self, size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class Dataset:

    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.train = None
        self.val = None
        self.test = None
        self.test_loader = None
        self.val_loader = None
        self.train_loader = None

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train = datasets.CIFAR10(root='./dataset/data', train=True, transform=SimCLRTransform(), download=True)
        test = datasets.CIFAR10(root='./dataset/data', train=False, transform=test_transform, download=True)

        val_size = int(0.1 * len(train))
        train_size = len(train) - val_size
        train, val = random_split(train, [train_size, val_size])

        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                                  drop_last=True, pin_memory=True)
        val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        self.train, self.val, self.test = train, val, test
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader

    def get_data(self):
        return self.train, self.val, self.test

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader
