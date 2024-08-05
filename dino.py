import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOAugmentations:
    def __init__(self, global_crops_scale=(0.4, 1), local_crops_scale=(0.05, 0.4), local_crops_number=4, size=224):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.image_size = size

        RandomGaussianBlur = lambda p: transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=p)

        flip_and_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)]),
            transforms.RandomGrayscale(p=0.2)
        ])

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.global_1 = transforms.Compose([
            transforms.RandomResizedCrop(size=self.image_size, scale=self.global_crops_scale),
            flip_and_jitter,
            RandomGaussianBlur(1.0), # always apply
            normalize
        ])

        self.global_2 = transforms.Compose([
            transforms.RandomResizedCrop(size=self.image_size, scale=self.global_crops_scale),
            flip_and_jitter,
            RandomGaussianBlur(0.1),
            transforms.RandomSolarize(170, p=0.2), # always skip
            normalize
        ])

        self.local = transforms.Compose([
            transforms.RandomResizedCrop(size=self.image_size, scale=self.local_crops_scale),
            flip_and_jitter,
            RandomGaussianBlur(0.5),
            normalize
        ])

    def __call__(self, x):
        all_crops = []
        all_crops.append(self.global_1(x))
        all_crops.append(self.global_2(x))

        all_crops.extend([self.local(x) for _ in range(self.local_crops_number)])

        return all_crops
    
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, bottleneck_dim=256, n_layers=3, norm_last_layer=False):
        super().__init__()

        if n_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(n_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))

        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)

        return x
    

class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, new_head):
        super().__init__()
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.new_head = new_head

    def forward(self, x):
        n_crops = len(x)
        concatenated = torch.cat(x, dim=0)
        cls_embedding = self.backbone(concatenated)
        logits = self.new_head(cls_embedding)
        chunks = logits.chunk(n_crops)

        return chunks
    
class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        student_temp = [s / self.student_temp for s in student_output]
        teacher_temp = [(t - self.center) / self.teacher_temp for t in teacher_output]

        student_sm = [F.log_softmax(s, dim=-1) for s in student_temp]
        teacher_sm = [F.softmax(t, dim=-1).detach() for t in teacher_temp]

        total_loss = 0
        n_loss_terms = 0

        for t_ix, t in enumerate(teacher_sm):
            for s_ix, s in enumerate(student_sm):
                if t_ix == s_ix:
                    continue

                loss = torch.sum(-t * s, dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)

        self.center = self.center * self.center_momentum + batch_center * (1-self.center_momentum)