import megengine.module as M
import megengine.functional as F

class JPCNN(M.Module):
    def __init__(self):
        super().__init__()
        W = 222
        self.conv0 = M.Conv2d(16, W, kernel_size=2, stride=1, padding=1)
        self.relu0 = M.ReLU()
        self.conv1 = M.Conv2d(W, W, kernel_size=2, stride=1, padding=1)
        self.relu1 = M.ReLU()
        self.conv2 = M.Conv2d(W, W, kernel_size=2, stride=1, padding=1)
        self.relu2 = M.ReLU()
        self.conv3 = M.Conv2d(W, W, kernel_size=2, stride=1, padding=1)
        self.relu3 = M.ReLU()
        self.conv4 = M.Conv2d(W, W, kernel_size=2, stride=1, padding=1)
        self.relu4 = M.ReLU()
#         self.conv5 = M.Conv2d(W, W, kernel_size=2, stride=1, padding=1, bias=False)
#         self.relu5 = M.ReLU()
#         self.conv6 = M.Conv2d(W, W, kernel_size=2, stride=1, padding=1, bias=False)
#         self.relu6 = M.ReLU()
#         self.conv7 = M.Conv2d(W, W, kernel_size=2, stride=1, padding=1)
#         self.relu7 = M.ReLU()
#         self.conv8 = M.Conv2d(W, W, kernel_size=2, stride=1, padding=1)
#         self.relu8 = M.ReLU()
#         self.fc0 = M.Linear(4 * 4 * W, 3)
        self.fc0 = M.Linear(4 * 4 * W, 192)
        self.relu9 = M.ReLU()
        self.fc1 = M.Linear(192, 3)
    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        x = x[: , : , : x.shape[2] - 1, : x.shape[3] - 1]
        x = self.conv1(x)
        x = self.relu1(x)
        x = x[: , : , : x.shape[2] - 1, : x.shape[3] - 1]
        x = self.conv2(x)
        x = self.relu2(x)
        x = x[: , : , : x.shape[2] - 1, : x.shape[3] - 1]
        x = self.conv3(x)
        x = self.relu3(x)
        x = x[: , : , : x.shape[2] - 1, : x.shape[3] - 1]
        x = self.conv4(x)
        x = self.relu4(x)
        x = x[: , : , : x.shape[2] - 1, : x.shape[3] - 1]
#         x = self.conv5(x)
#         x = self.relu5(x + 0.1)
#         x = x[: , : , : x.shape[2] - 1, : x.shape[3] - 1]
#         x = self.conv6(x)
#         x = self.relu6(x + 0.1)
#         x = x[: , : , : x.shape[2] - 1, : x.shape[3] - 1]
#         x = self.conv7(x)
#         x = self.relu7(x)
#         x = x[: , : , : x.shape[2] - 1, : x.shape[3] - 1]
#         x = self.conv8(x)
#         x = self.relu8(x)
#         x = x[: , : , : x.shape[2] - 1, : x.shape[3] - 1]
        x = F.flatten(x, 1)
        x = self.fc0(x)
        x = self.relu9(x)
        x = self.fc1(x)
#         x = F.reshape(x, (-1, 3))
        return x