import megengine.module as M
import megengine.functional as F


class Net(M.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = M.Conv2d(16, 256, kernel_size=(2, 1), stride=1, padding=0)
        self.relu0 = M.ReLU()
        self.conv1 = M.Conv2d(
            256, 256, kernel_size=(1, 2), stride=1, padding=0)
        self.relu1 = M.ReLU()
        self.conv2 = M.Conv2d(
            256, 256, kernel_size=(2, 1), stride=1, padding=0)
        self.relu2 = M.ReLU()
        self.conv3 = M.Conv2d(
            256, 256, kernel_size=(1, 2), stride=1, padding=0)
        self.relu3 = M.ReLU()
        self.fc1 = M.Linear(1024, 16)
        self.relu5 = M.ReLU()
        self.fc2 = M.Linear(16, 3)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = F.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = F.reshape(x, (-1, 3))
        return x
