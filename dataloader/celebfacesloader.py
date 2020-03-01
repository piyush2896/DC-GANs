import torch
from torchvision import datasets
from torchvision import transforms

class CelebFacesDataloaders:
    def __init__(self, root, img_size, batch_size, num_workers):
        self.dataset = datasets.ImageFolder(
            root=root,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers
        )
