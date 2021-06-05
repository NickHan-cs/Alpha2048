import megengine as mge
from megengine import tensor
import megengine.functional as F
from console2048 import Game, any_possible_moves
from main import make_input

model_name_list = []
best_model_name = ""
game_num = 100

model_list = []
for model_name in model_name_list:
    model_list.append(mge.load(f"./{model_name}/1.mge"))
best_model = mge.load(f"./{best_model_name}/1.mge")
game_list = []
for i in range(game_num):
    game_list.append(Game())

score = 0
max_score = 0
for j in range(game_num):
    while any_possible_moves(game_list[j].grid):
        grid = []
        tmp = make_input(game_list[j].grid)
        grid.append(tensor(tmp))
        status = F.stack(grid, 0)
        actions = [0, 0, 0, 0]
        for model in model_list:
            a = F.argmax(model(status).detach(), 1)
            a = a.numpy()
            actions[a[0]] += 1
        sorted_actions_with_idx = sorted(enumerate(actions), key=lambda x: x[1], reverse=True)
        action = sorted_actions_with_idx[0][0]
        if sorted_actions_with_idx[0][1] == sorted_actions_with_idx[1][1]:
            a = F.argmax(best_model(status).detach(), 1)
            a = a.numpy()
            action = a[0]
        pre_score = game_list[j].score
        pre_grid = game_list[j].grid.copy()
        game_list[j].move(action)
        after_score = game_list[j].score
        max_score = max(max_score, after_score)
        if not any_possible_moves(game_list[j].grid):
            break
        while (game_list[j].grid == pre_grid).all():
            action = (action + 1) % 4
            game_list[j].move(action)
    print(f"Game {j} got {game_list[j].score} scores")
    score += game_list[j].score
model_name = " + ".join(model_name_list)
print(f"{model_name} play {game_num} games.")
print(f"The average score is {score / game_num}")
print(f"Tha max score is {max_score}")
