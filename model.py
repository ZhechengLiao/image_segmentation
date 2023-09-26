from model_part import *

class UNet(nn.Module):
  def __init__(self, n_classes):
    super(UNet, self).__init__()

    # sampling
    down_size = [(3, 64), (64, 128), (128, 256), (256, 512)]
    # use ModuleList can help to write less
    self.down_conv = nn.ModuleList([DoubleConv(i, o) for i, o in down_size])
    self.down = nn.ModuleList([Down() for _ in range(4)]) # 4 times

    # middle conv
    self.middle_conv = DoubleConv(512, 1024)

    # unsampling
    up_size = [(1024, 512), (512, 256), (256, 128), (128, 64)]
    self.up_conv = nn.ModuleList([DoubleConv(i, o) for i, o in up_size])
    self.up = nn.ModuleList([Up(i, o) for i, o in up_size])

    # out
    self.out = nn.Conv2d(64, n_classes, 1)

    # concat
    self.concat = nn.ModuleList([Concat() for _ in range(4)])

  def __call__(self, x):
    passing_through = []

    for i in range(len(self.down_conv)):
      x = self.down_conv[i](x)
      passing_through.append(x)
      x = self.down[i](x)

    x = self.middle_conv(x)

    for i in range(len(self.up_conv)):
      x = self.up[i](x)
      x = self.concat[i](x, passing_through.pop())
      x = self.up_conv[i](x)

    out = self.out(x)

    return out