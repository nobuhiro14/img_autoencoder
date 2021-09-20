from torch.utils.data import DataLoader , random_split
from torchvision import datasets, transforms


def load_cifar10(batch=128):
    trainval = datasets.CIFAR10datasets.CIFAR10('./data',
                     train=True,
                     download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize(
                            [0.5, 0.5, 0.5],  # RGB 平均
                            [0.5, 0.5, 0.5]   # RGB 標準偏差
                            )
                     ]))
    n_sample = len(trainval)
    train_size = int(len(trainval)*0.8)
    val_size = n_sample - train_size
    train_data , val_data  = random_split(trainval, [train_size, val_size])
    train_loader = DataLoader(
        train_data,
        batch_size=batch,
        shuffle=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch,
        shuffle=True
    )

    test_loader = DataLoader(
        datasets.CIFAR10('./data',
                         train=False,
                         download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 [0.5, 0.5, 0.5],  # RGB 平均
                                 [0.5, 0.5, 0.5]  # RGB 標準偏差
                             )
                         ])),
        batch_size=batch,
        shuffle=True
    )

    return {'train': train_loader, 'val',val_loader,'test': test_loader}
