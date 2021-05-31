import random
import megengine.functional as F


class rpm(object):
    '''
    定义记忆回放类并实例化
    在记录一次决策过程后，我们储存到该类中，并在训练时选择一部分记忆进行训练
    '''

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0

    def append(self, obj):
        if self.size() > self.buffer_size:
            print('buffer size larger than set value, trimming...')
            self.buffer = self.buffer[(self.size() - self.buffer_size):]
        elif self.size() == self.buffer_size:
            self.buffer[self.index] = obj
            self.index += 1
            self.index %= self.buffer_size
        else:
            self.buffer.append(obj)

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size):
        if self.size() < batch_size:
            batch = random.sample(self.buffer, self.size())
        else:
            batch = random.sample(self.buffer, batch_size)

        res = []
        for i in range(5):
            k = F.stack(tuple(item[i] for item in batch), axis=0)
            res.append(k)
        return res[0], res[1], res[2], res[3], res[4]
