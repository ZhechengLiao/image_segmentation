import torch
import torch.nn as nn
import torchvision
from torch.nn.modules.conv import Conv2d

class Up(nn.Module):
  """Upsampling, use torch.nn.ConvTranspose2d()"""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.up = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=2, stride=2)

  def __call__(self, x):
    return self.up(x)

class Down(nn.Module):
  """Sampling, use torch.nn.MaxPool2d() with size 2"""
  def __init__(self):
    super().__init__()
    self.down = nn.MaxPool2d(2)

  def __call__(self, x):
    return self.down(x)

class DoubleConv(nn.Module):
  """
  Convlution network, 2 convnet, conv1(input_dim, output_dim); conv2(output_dim, output_dim)
  kernel_size=3, padding=1
  """
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        nn.ReLU()
    )

  def __call__(self, x):
    return self.net(x)

class Concat(nn.Module):
  """
  Help to contact x and prev_x
  """
  def __call__(self, x, contracting_x):
    contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
    x = torch.cat([x, contracting_x], dim=1)

    return x