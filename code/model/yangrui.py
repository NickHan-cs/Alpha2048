import megengine.module as M
import megengine.functional as F


class YangRuiNet(M.Module):
    def __init__(self):
        super(YangRuiNet, self).__init__()
        self.conv0 = M.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu0 = M.ReLU()
        self.conv1 = M.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = M.ReLU()
        self.fc1 = M.Linear(1024, 512)
        self.relu2 = M.ReLU()
        self.fc2 = M.Linear(512, 128)
        self.relu3 = M.ReLU()
        self.fc3 = M.Linear(128, 3)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = F.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.fc3(x)
        x = F.reshape(x, (-1, 3))
        return x
