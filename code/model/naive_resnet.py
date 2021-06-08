import megengine.module as M
import megengine.functional as F


class BaselineBlock(M.Module):
    def __init__(self, in_channels, expansion, norm=M.BatchNorm2d):
        super().__init__()
        self.expansion = expansion
        self.channels = in_channels * self.expansion
        self.conv0 = M.Conv2d(in_channels, self.channels, kernel_size=(2, 1), stride=1, padding=0)
        self.conv1 = M.Conv2d(self.channels, self.channels, kernel_size=(1, 2), stride=1, padding=0)

    def forward(self, x):
        identity = x[:, :, 1:, 1:]
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x += identity
        x = F.relu(x)
        return x


class BasicBlock(M.Module):
    expansion = 1

    def __init__(
            self,
            in_channels,
            channels,
            stride=1,
            groups=1,
            base_width=64,
            dilation=1,
            norm=M.BatchNorm2d,
    ):
        super().__init__()
        self.conv1 = M.Conv2d(
            in_channels, channels, 3, stride, padding=dilation, bias=False
        )
        self.bn1 = norm(channels)
        self.conv2 = M.Conv2d(channels, channels, 3, 1, padding=1, bias=False)
        self.bn2 = norm(channels)
        self.downsample = (
            M.Identity()
            if in_channels == channels and stride == 1
            else M.Sequential(
                M.Conv2d(in_channels, channels, 1, stride, bias=False),
                norm(channels),
            )
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        identity = self.downsample(identity)
        x += identity
        x = F.relu(x)
        return x


class NaiveResNet(M.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = M.Conv2d(16, 128, kernel_size=(2, 1), stride=1, padding=1)
        self.conv1 = M.Conv2d(128, 128, kernel_size=(1, 2), stride=1, padding=0)
        self.bb0 = BaselineBlock(128, 1)
        self.bb1 = BaselineBlock(128, 1)
        self.bb2 = BaselineBlock(128, 1)
        self.dense0 = M.Linear(2 * 2 * 128, 24)
        self.dense1 = M.Linear(24, 3)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bb0(x)
        x = self.bb1(x)
        x = self.bb2(x)
        x = F.flatten(x, 1)
        x = self.dense0(x)
        x = F.relu(x)
        x = self.dense1(x)
        return x
