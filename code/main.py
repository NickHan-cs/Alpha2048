from .rpm import rpm
from .model.megvii import MegviiNet
import numpy as np
import megengine as mge
from megengine import tensor
from megengine.autodiff import GradManager
from megengine.optimizer import Adam
import megengine.functional as F
from .console2048 import Game, any_possible_moves
import tqdm
import json
import os


def make_input(grid):
    g0 = grid
    r = np.zeros(shape=(16, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            v = g0[i, j]
            r[table[v], i, j] = 1
    return r


def dict_to_json(save_path, dict_):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as jo:
        json.dump(dict_, jo)


if __name__ == "__main__":
    loss_dict = {}
    Q_dict = {}
    reward_dict = {}
    avg_score_dict = {}

data = rpm(5000)

model = MegviiNet()
model_target = MegviiNet()

table = {2 ** i: i for i in range(1, 16)}
table[0] = 0

opt = Adam(model.parameters(), lr=1e-4)

maxscore = 0
avg_score = 0
epochs = 10000

game = []
'''Play 32 games at the same time'''
for i in range(32):
    game.append(Game())

with tqdm(total=epochs * 5, desc="epoch") as tq:
    for epoch in range(epochs):

        '''double DQN'''
        if epoch % 10 == 0:
            mge.save(model, "1.mge")
            model_target = mge.load("1.mge")

        grid = []
        for k in range(32):

            '''Check if the game is over'''
            if any_possible_moves(game[k].grid) == False:
                if avg_score == 0:
                    avg_score = game[k].score
                else:
                    avg_score = avg_score * 0.99 + game[k].score * 0.01
                game[k] = Game()

            tmp = make_input(game[k].grid)
            grid.append(tensor(tmp))

        status = F.stack(grid, 0)

        '''Choose the action with the highest probability'''
        a = F.argmax(model(status).detach(), 1)
        a = a.numpy()

        for k in range(32):
            pre_score = game[k].score
            pre_grid = game[k].grid.copy()
            game[k].move(a[k])
            after_score = game[k].score
            if game[k].score > maxscore:
                maxscore = game[k].score
            action = a[k]

            '''In some situations, some actions are meaningless, try another'''
            while (game[k].grid == pre_grid).all():
                action = (action + 1) % 4
                game[k].move(action)

            score = after_score - pre_score
            done = tensor(any_possible_moves(game[k].grid) == False)
            grid = tensor(make_input(game[k].grid.copy()))

            '''Record to memory'''
            '''(status, next_status, action, score, if_game_over)'''
            data.append((tensor(make_input(pre_grid)), tensor(
                grid), tensor(a[k]), tensor(score / 128), done))

        for j in range(5):
            gm = GradManager().attach(model.parameters())
            with gm:
                s0, s1, a, reward, d = data.sample_batch(32)

                '''double DQN'''
                pred_s0 = model(s0)
                pred_s1 = F.max(model_target(s1), axis=1)

                loss = 0
                total_Q = 0
                total_reward = 0
                for i in range(32):
                    Q = pred_s0[i][a[i]]
                    total_Q += Q
                    total_reward += reward[i]
                    loss += F.loss.square_loss(
                        Q, pred_s1[i].detach() * 0.99 * (1 - d[i]) + reward[i])
                loss /= 32
                total_Q /= 32
                total_reward = total_reward / 32 * 128
                tq.set_postfix(
                    {
                        "loss": "{0:1.5f}".format(loss.numpy().item()),
                        "Q": "{0:1.5f}".format(total_Q.numpy().item()),
                        "reward": "{0:1.5f}".format(total_reward.numpy().item()),
                        "avg_score": "{0:1.5f}".format(avg_score),
                    }
                )
                tq.update(1)
                gm.backward(loss)

                if epoch % 100 == 0 and j == 0:
                    loss_dict[epoch * 5] = loss.numpy().item()
                    Q_dict[epoch * 5] = total_Q.numpy().item()
                    reward_dict[epoch * 5] = total_reward.numpy().item()
                    avg_score_dict[epoch * 5] = avg_score

                    save_dir = './output/'
                    dict_to_json(save_dir + 'loss.json', loss_dict)
                    dict_to_json(save_dir + 'Q.json', Q_dict)
                    dict_to_json(save_dir + 'reward.json', reward_dict)
                    dict_to_json(save_dir + 'avg_score.json', avg_score_dict)

        opt.step()
        opt.clear_grad()

print("maxscore:{}".format(maxscore))
print("avg_score:{}".format(avg_score))
