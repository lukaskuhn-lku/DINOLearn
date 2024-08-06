import timm
import wandb

import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

from dino import DINOAugmentations, DINOHead, DINOLoss, MultiCropWrapper
from utils import compute_knn, gradient_clipping
from dataset import load_isic_subset

def main():
    print("DINO Training started")
    print(r"""  
    ^..^      /
    /_/\_____/
       /\   /\
      /  \ /  \  
          """)
    
    run = wandb.init(
        project="DINO Cifar10",
        tags=["dino", "cifar10", "vit"],
    )

    vit_name, dim = "deit_tiny_patch16_224", 192

    device = "cuda" if torch.cuda.is_available() else "mps"

    n_workers = 4
    n_local_crops = 8
    batch_size = 32
    out_dim = 1024
    weight_decay = 1e-4
    epochs = 100
    momentum_teacher = 0.995

    transform_aug = DINOAugmentations(local_crops_number=n_local_crops, size=224)

    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(224)
    ])

    #dataset_train = CIFAR10(root='data', download=True, train=True, transform=transform_aug)
    #dataset_train_knn = CIFAR10(root='data', download=True, train=True, transform=transform_norm)
    #dataset_val_knn =  CIFAR10(root='data', download=True, train=False, transform=transform_norm)

    dataset_train, dataset_train_knn, dataset_val_knn = load_isic_subset(10000, transform_aug, transform_norm)

    data_loader_train_aug = DataLoader(dataset_train, batch_size, shuffle=True, drop_last=True, num_workers=n_workers, pin_memory=True)
    data_loader_train_plain = DataLoader(dataset_train_knn, batch_size, shuffle=True, drop_last=False, num_workers=n_workers)
    data_loader_val_plain = DataLoader(dataset_val_knn, batch_size, shuffle=True, drop_last=False, num_workers=n_workers)

    student_vit = timm.create_model(vit_name)
    teacher_vit = timm.create_model(vit_name)

    student = MultiCropWrapper(
        student_vit,
        DINOHead(
            in_dim=dim,
            out_dim=out_dim,
            norm_last_layer=True,
        )
    )

    teacher = MultiCropWrapper(
        teacher_vit,
        DINOHead(
            in_dim=dim,
            out_dim=out_dim,
        )
    )

    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    student, teacher = student.to(device), teacher.to(device)

    loss_inst = DINOLoss(
        out_dim
    ).to(device)

    lr = 1e-3
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    for e in range(epochs):
        wandb.log({"epochs": e})
        student.train()
        for images, _ in tqdm(data_loader_train_aug):
            images = [img.to(device) for img in images]

            teacher_output = teacher(images[:2])
            student_output = student(images)
            
            loss = loss_inst(student_output, teacher_output)
            wandb.log({"loss": loss})

            optimizer.zero_grad()
            loss.backward()
            gradient_clipping(student)
            optimizer.step()

            with torch.no_grad():
                for student_ps, teacher_ps in zip(student.parameters(), teacher.parameters()):
                    teacher_ps.data.mul_(momentum_teacher)
                    teacher_ps.data.add_(
                        (1-momentum_teacher) + student_ps.detach().data
                    )

        student.eval()
        current_acc = compute_knn(
            student.backbone,
            data_loader_train_plain,
            data_loader_val_plain
        )

        wandb.log({"knn_acc": current_acc})

        #torch.save(student.state_dict(), f"models/student_{e}.pth")
        #torch.save(teacher.state_dict(), f"models/teacher_{e}.pth")
        #torch.save(student.backbone.state_dict(), f"models/student_backbone_{e}.pth")

        #run.log_model(path="models/student_backbone_{e}.pth", name=f"student_backbone_{e}")
        #run.log_model(path="models/student_{e}.pth", name=f"student_{e}")
        #run.log_model(path="models/teacher_{e}.pth", name=f"teacher_{e}")

if __name__ == "__main__":
    main()