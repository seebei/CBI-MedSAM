import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryRefineBlock(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1)
        )

    def forward(self, coarse_mask):
        edge = self.compute_edge(coarse_mask)
        correction = self.conv(edge)
        return coarse_mask + correction

    def compute_edge(self, x):
        # Sobel边缘检测（近似）
        sobel_x = torch.tensor([[[[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]]]], device=x.device, dtype=x.dtype)
        sobel_y = torch.tensor([[[[-1, -2, -1],
                                  [0,  0,  0],
                                  [1,  2,  1]]]], device=x.device, dtype=x.dtype)
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return grad
