import megengine.module as M
import megengine.functional as F


class NavjinNet(M.Module):
    def __init__(self):
        super(NavjinNet, self).__init__()
        self.conv1 = M.Conv2d(16, 128, kernel_size=(1, 2), stride=1, padding=0)
        self.relu1 = M.ReLU()
        self.conv2 = M.Conv2d(16, 128, kernel_size=(2, 1), stride=1, padding=0)
        self.relu2 = M.ReLU()
        
        self.conv11 = M.Conv2d(128, 128, kernel_size=(1, 2), stride=1, padding=0)
        self.relu11 = M.ReLU()
        self.conv12 = M.Conv2d(128, 128, kernel_size=(2, 1), stride=1, padding=0)
        self.relu12 = M.ReLU()
        self.conv21 = M.Conv2d(128, 128, kernel_size=(1, 2), stride=1, padding=0)
        self.relu21 = M.ReLU()
        self.conv22 = M.Conv2d(128, 128, kernel_size=(2, 1), stride=1, padding=0)
        self.relu22 = M.ReLU()
        
        expand_size = 128*4*3*2 + 128*4*2*2 + 128*3*3*2
        self.fc1 = M.Linear(expand_size, 256)
        self.relu_fc = M.ReLU()
        self.fc2 = M.Linear(256, 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        x2 = self.conv2(x)
        x2 = self.relu2(x2)
        
        x11 = self.conv11(x1)
        x11 = self.relu11(x11)
        x12 = self.conv12(x1)
        x12 = self.relu12(x12)
        x21 = self.conv21(x2)
        x21 = self.relu21(x21)
        x22 = self.conv22(x2)
        x22 = self.relu22(x22)
        
        hidden1 = F.flatten(x1,1)
        hidden2 = F.flatten(x2,1)
        hidden11 = F.flatten(x11,1)
        hidden12 = F.flatten(x12,1)
        hidden21 = F.flatten(x21,1)
        hidden22 = F.flatten(x22,1)
        hidden = F.concat([hidden1,hidden2,hidden11,hidden12,hidden21,hidden22],axis=1)
        
        hidden = self.fc1(hidden)
        hidden = self.relu_fc(hidden)
        output = self.fc2(hidden)
        output = F.reshape(output,(-1,3))
        return output
