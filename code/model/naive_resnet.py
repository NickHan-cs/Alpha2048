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


class NaiveResNet(M.Module):
    def __init__(self):
        self.conv0 = M.Conv2d(16, 128, kernel_size=(2, 1), stride=1, padding=1)
        self.conv1 = M.Conv2d(128, 128, kernel_size=(1, 2), stride=1, padding=0)

        self.bb0 = BaselineBlock(128, 1)
        self.bb1 = BaselineBlock(128, 1)

        self.conv2 = M.Conv2d(128, 256, kernel_size=2, stride=1, padding=0)

        self.dense0 = M.Linear(2 * 2 * 256, 16)

        self.dense1 = M.Linear(16, 3)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(x)

        x = self.conv1(x)
        x = F.relu(x)

        x = self.bb0(x)
        x = self.bb1(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.flatten(x, 1)

        x = self.dense0(x)
        x = F.relu(x)

        x = self.dense1(x)
        return x
