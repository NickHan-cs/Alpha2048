import megengine as mge
from megengine import tensor
import megengine.functional as F
from console2048 import Game, any_possible_moves
from main import make_input


model_name = ""
game_num = 100

model = mge.load("./" + model_name + "/1.mge")
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

        a = F.argmax(model(status).detach(), 1)
        a = a.numpy()

        pre_score = game_list[j].score
        pre_grid = game_list[j].grid.copy()
        game_list[j].move(a[0])
        after_score = game_list[j].score
        max_score = max(max_score, after_score)

        if not any_possible_moves(game_list[j].grid):
            break
        action = a[0]
        while (game_list[j].grid == pre_grid).all():
            action = (action + 1) % 4
            game_list[j].move(action)

    print(f"Game {j} got {game_list[j].score} scores")
    score += game_list[j].score

print(f"{model_name} play {game_num} games.")
print(f"The average score is {score / game_num}")
print(f"Tha max score is {max_score}")
