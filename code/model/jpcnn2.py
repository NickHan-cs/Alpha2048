import megengine.module as M
import megengine.functional as F


class JPCNN(M.Module):
    widths = {2: 436, 3: 312, 4: 256, 5: 222, 6: 200, 7: 182, 8: 168, 9: 158}

    def __init__(self, layer_number):
        super().__init__()
        if layer_number < 2 or layer_number > 9:
            print('layer_number must between 2 and 9.')
        hidden_channel = self.widths[layer_number]
        layers = list()
        layers.append(M.Conv2d(16, hidden_channel, kernel_size=2, stride=1, padding=1))
        for i in range(layer_number - 1):
            layers.append(M.Conv2d(hidden_channel, hidden_channel, kernel_size=2, stride=1, padding=1))
        self.convs = M.Sequential(*layers)
        self.dense0 = M.Linear(4 * 4 * hidden_channel, hidden_channel)
        self.dense1 = M.Linear(hidden_channel, 3)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
            x = x[:, :, 1:, 1:, ]
        x = F.flatten(x, 1)
        x = self.dense0(x)
        x = F.relu(x)
        x = self.dense1(x)
        #         x = F.reshape(x, (-1, 3))
        return x
