from pycatan import Game, DevelopmentCard, Resource
from pycatan.board import BeginnerBoard, BoardRenderer, BuildingType
import string

game = Game(BeginnerBoard())
renderer = BoardRenderer(game.board)


def get_coord_sort_by_xy(c):
    x, y = renderer.get_coords_as_xy(c)
    return 1000 * x + y


label_letters = string.ascii_lowercase + string.ascii_uppercase + "123456789"


def choose_intersection(intersection_coords, choice):
    intersection_list = [game.board.intersections[i] for i in intersection_coords]
    return intersection_list[choice].coords


def choose_path(path_coords, choice):
    # Label all the paths with a letter
    path_list = [game.board.paths[i] for i in path_coords]
    return path_list[choice].path_coords


def choose_hex(hex_coords, choice):  # choice: 0-143
    hex_list = [game.board.hexes[i] for i in hex_coords]
    return hex_list[choice].coords


def choose_resource(choice):
    resources = [res for res in Resource]
    return resources[choice]


def move_robber(player):
    # Don't let the player move the robber back onto the same hex
    hex_coords = choose_hex([c for c in game.board.hexes if c != game.board.robber],
                            "Where do you want to move the robber? ")
    game.board.robber = hex_coords
    # Choose a player to steal a card from
    potential_players = list(game.board.get_players_on_hex(hex_coords))
    print("Choose who you want to steal from:")
    for p in potential_players:
        i = game.players.index(p)
        print("%d: Player %d" % (i + 1, i + 1))
    p = int(input('->  ')) - 1
    # If they try and steal from another player they lose their chance to steal
    to_steal_from = game.players[p] if game.players[p] in potential_players else None
    if to_steal_from:
        resource = to_steal_from.get_random_resource()
        player.add_resources({resource: 1})
        to_steal_from.remove_resources({resource: 1})
        print("Stole 1 %s for player %d" % (resource, p + 1))
