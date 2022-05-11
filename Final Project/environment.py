from pycatan import Game, DevelopmentCard, Resource
from pycatan.board import BeginnerBoard, BoardRenderer, BuildingType, Coords
from enums import GameStage
import funcs
import random

from typing import Optional
import numpy as np

import gym
from gym import spaces


class CatanEnvironmentCoop(gym.Env):
    game: Game
    renderer: BoardRenderer
    stage: GameStage
    current_player_number: int
    rolled: bool
    turns: int
    # illegal_acts: int
    bought_dev_card: bool
    played_dev_card: bool
    connected_intersection: Coords

    def __init__(self):
        super(CatanEnvironmentCoop, self).__init__()
        self.resets = 0

        # self.action_space = spaces.Discrete(72)
        self.action_space = spaces.MultiDiscrete([10, 25, 54])

        self.observation_space = spaces.Dict({
            'stage': spaces.Discrete(13),  # GameStage
            'robber location': spaces.Discrete(54),  # Robber location
            'buildings': spaces.Box(low=0, high=1, shape=(4, 180), dtype=np.byte),
            # Location of Settlements, Cities, Roads
            'cards in hand': spaces.Box(low=0, high=256, shape=(4, 11), dtype=np.byte)
            # Num cards in hand AND knights played
        })

    def _all_hex_coords(self):
        return self.game.board.hexes.keys()

    def _all_intersection_coords(self):
        return self.game.board.intersections.keys()

    def _all_path_coords(self):
        return self.game.board.paths.keys()

    def _next_observation(self):
        obs = {'stage': self.stage.value}

        hexes = {hex_coord: i for i, hex_coord in enumerate(self._all_hex_coords())}
        obs['robber location'] = hexes[self.game.board.robber]

        buildings = np.zeros((4, 180))
        for i in range(len(self.game.players)):
            for j, coord in enumerate(self._all_intersection_coords()):
                building = self.game.board.intersections[coord].building
                if building is None:
                    continue
                if building.owner == self.game.players[(self.current_player_number + i) % 4]:
                    col = j
                    if building.building_type == BuildingType.CITY:
                        col += 54
                    buildings[i, col] = 1

            for j, coord in enumerate(self._all_path_coords()):
                building = self.game.board.paths[coord].building
                if building is None:
                    continue
                if building.owner == self.game.players[(self.current_player_number + i) % 4]:
                    buildings[i, j + 108] = 1
        obs['buildings'] = buildings

        cards = np.zeros((4, 11))
        for i in range(len(self.game.players)):
            for j, res in enumerate(Resource):
                cards[i, j] = self.game.players[(self.current_player_number + i) % 4].resources[res]
            for j, dev in enumerate(DevelopmentCard):
                cards[i, j + 5] = self.game.players[(self.current_player_number + i) % 4].development_cards[dev]
            cards[i, 10] = self.game.players[(self.current_player_number + i) % 4].number_played_knights
        obs['cards in hand'] = cards

        return obs

    def step(self, action):
        next_stage: Optional[GameStage] = None
        reward = 0
        next_player_number = self.current_player_number
        current_player = self.game.players[self.current_player_number]
        pass_turn = False
        assert isinstance(action, np.ndarray)
        try:
            match self.stage:

                case GameStage.NOT_ROLLED:
                    if (action[0] % 2 == 0 or
                            current_player.development_cards[DevelopmentCard.KNIGHT] == 0 or
                            self.played_dev_card):
                        dice = random.randint(1, 6) + random.randint(1, 6)
                        self.rolled = True
                        if dice == 7:
                            next_stage = GameStage.MOVING_ROBBER
                        else:
                            self.game.add_yield_for_roll(dice)
                            next_stage = GameStage.ROLLED
                    else:
                        self.game.play_development_card(current_player, DevelopmentCard.KNIGHT)
                        next_stage = GameStage.MOVING_ROBBER

                case GameStage.ROLLED:
                    if action[0] == 0:
                        pass_turn = True
                    elif action[0] == 1:
                        if current_player.development_cards[DevelopmentCard.KNIGHT] != 0 and not self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.KNIGHT)
                            next_stage = GameStage.MOVING_ROBBER
                        else:
                            pass_turn = True
                    elif action[0] == 2:
                        if current_player.development_cards[DevelopmentCard.YEAR_OF_PLENTY] != 0 and not \
                                self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.YEAR_OF_PLENTY)
                            resource = [res for res in Resource][action[1] % len(Resource)]
                            current_player.add_resources({resource: 1})
                            resource = [res for res in Resource][action[2] % len(Resource)]
                            current_player.add_resources({resource: 1})
                        else:
                            pass_turn = True
                    elif action[0] == 3:
                        if current_player.development_cards[DevelopmentCard.MONOPOLY] != 0 and not \
                                self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.MONOPOLY)
                            resource = [res for res in Resource][action[1] % 5]
                            for player in self.game.players:
                                if player is not current_player:
                                    amount = player.resources[resource]
                                    player.remove_resources({resource: amount})
                                    current_player.add_resources({resource: amount})
                        else:
                            pass_turn = True
                    elif action[0] == 4:
                        if current_player.development_cards[DevelopmentCard.ROAD_BUILDING] != 0 and not \
                                self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.ROAD_BUILDING)
                            valid_coords = self.game.board.get_valid_road_coords(current_player)
                            coord = funcs.choose_path(valid_coords, action[1] % len(valid_coords))
                            self.game.build_road(current_player, coord, cost_resources=False)
                            valid_coords = self.game.board.get_valid_road_coords(current_player)
                            coord = funcs.choose_path(valid_coords, action[2] % len(valid_coords))
                            self.game.build_road(current_player, coord, cost_resources=False)
                        else:
                            pass_turn = True
                    elif action[0] == 5:
                        if len(current_player.get_possible_trades()) == 0:
                            pass_turn = True
                        next_stage = GameStage.TRADING
                    elif action[0] == 6:
                        valid_coords = self.game.board.get_valid_settlement_coords(current_player)
                        if current_player.has_resources(BuildingType.SETTLEMENT.get_required_resources()) \
                                and valid_coords:
                            coord = funcs.choose_intersection(valid_coords, action[1] % len(valid_coords))
                            self.game.build_settlement(current_player, coord)
                            reward += 100
                        else:
                            pass_turn = True
                    elif action[0] == 7:
                        valid_coords = self.game.board.get_valid_city_coords(current_player)
                        if current_player.has_resources(BuildingType.CITY.get_required_resources()) \
                                and valid_coords:
                            coord = funcs.choose_intersection(valid_coords, action[1] % len(valid_coords))
                            self.game.upgrade_settlement_to_city(current_player, coord)
                            reward += 100
                        else:
                            pass_turn = True
                    elif action[0] == 8:
                        valid_coords = self.game.board.get_valid_road_coords(current_player)
                        if current_player.has_resources(BuildingType.ROAD.get_required_resources()) \
                                and valid_coords:
                            coord = funcs.choose_path(valid_coords, action[1] % len(valid_coords))
                            self.game.build_road(current_player, coord)
                            reward += 10
                        else:
                            pass_turn = True
                    elif action[0] == 9:
                        if current_player.has_resources(DevelopmentCard.get_required_resources()) and \
                                not self.bought_dev_card:
                            if self.game.build_development_card(current_player) == DevelopmentCard.VICTORY_POINT:
                                reward += 10
                        else:
                            pass_turn = True
                    else:
                        print("This shouldn't happen")
                        raise Exception

                case GameStage.MOVING_ROBBER:
                    valid_hexes = [coord for coord in self._all_hex_coords() if coord != self.game.board.robber]
                    self.game.board.robber = funcs.choose_hex(valid_hexes, action[2] % len(valid_hexes))
                    next_stage = GameStage.STEALING

                case GameStage.STEALING:
                    targets = list(self.game.board.get_players_on_hex(self.game.board.robber))
                    if len(targets) > 0:
                        target = targets[action[0] % len(targets)]
                        resource = target.get_random_resource()
                        if resource is not None:
                            target.remove_resources({resource: 1})
                            current_player.add_resources({resource: 1})
                    next_stage = GameStage.ROLLED if self.rolled else GameStage.NOT_ROLLED

                case GameStage.TRADING:
                    valid_trades = current_player.get_possible_trades()
                    trade = valid_trades[action[1] % len(valid_trades)]
                    current_player.add_resources(trade)
                    next_stage = GameStage.ROLLED

            if pass_turn:
                self.rolled = False
                next_player_number = (next_player_number + 1) % 4
                if next_player_number == 0:
                    self.turns += 1
                    # print(f"turn {self.turns}, {self.illegal_acts} illegal actions")
                    # self.illegal_acts = 0
                next_stage = GameStage.NOT_ROLLED
                self.bought_dev_card = False
                self.played_dev_card = False
                reward = -30

        except Exception:
            reward -= 3
            # self.illegal_acts += 1
            print(self.stage, action)

        # if random.random() < 0.005:
        #     print(self.current_player_number, self.game.get_victory_points(current_player), self.stage, action)

        if next_stage is not None:
            self.stage = next_stage
        self.current_player_number = next_player_number
        done = self.game.get_victory_points(current_player) >= 10
        if done:
            print(f"Done in {self.turns} turns!")
            reward += 10000 // self.turns
        return self._next_observation(), reward, done, {}

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        self.game = Game(BeginnerBoard())
        self.renderer = BoardRenderer(self.game.board)
        self.stage = GameStage.NOT_ROLLED
        self.current_player_number = 0
        self.turns = 0
        # self.illegal_acts = 0
        self.rolled = False
        self.bought_dev_card = False
        self.played_dev_card = False
        print(f"Reset {self.resets} times")
        self.resets += 1

        starting_settlements = [Coords(-3, 2), Coords(-2, 3), Coords(1, 2), Coords(3, -1),
                                Coords(1, -3), Coords(-2, 0), Coords(2, -2), Coords(-1, -2)]
        starting_roads = [{Coords(-3, 2), Coords(-2, 2)}, {Coords(-1, 3), Coords(-2, 3)},
                          {Coords(1, 2), Coords(2, 1)}, {Coords(2, 0), Coords(3, -1)},
                          {Coords(1, -3), Coords(0, -2)}, {Coords(-2, 0), Coords(-3, 1)},
                          {Coords(2, -3), Coords(2, -2)}, {Coords(-1, -2), Coords(-2, -1)}]

        for i in range(8):
            player = self.game.players[min(i, 7 - i)]
            self.game.build_settlement(player, starting_settlements[i], cost_resources=False, ensure_connected=False)
            self.game.build_road(player, starting_roads[i], cost_resources=False, ensure_connected=False)
            if i >= 4:
                player.add_resources(self.game.board.get_hex_resources_for_intersection(starting_settlements[i]))

        return self._next_observation()

    def render(self, mode="human"):
        pass


class CatanEnvironmentComp(gym.Env):
    game: Game
    renderer: BoardRenderer
    stage: GameStage
    current_player_number: int
    rolled: bool
    turns: int
    # illegal_acts: int
    bought_dev_card: bool
    played_dev_card: bool
    connected_intersection: Coords

    def __init__(self, models_to_beat):
        super(CatanEnvironmentComp, self).__init__()
        self.resets = 0
        self.models_to_beat = models_to_beat

        # self.action_space = spaces.Discrete(72)
        self.action_space = spaces.MultiDiscrete([10, 25, 54])

        self.observation_space = spaces.Dict({
            'stage': spaces.Discrete(13),  # GameStage
            'robber location': spaces.Discrete(54),  # Robber location
            'buildings': spaces.Box(low=0, high=1, shape=(4, 180), dtype=np.byte),
            # Location of Settlements, Cities, Roads
            'cards in hand': spaces.Box(low=0, high=256, shape=(4, 11), dtype=np.byte)
            # Num cards in hand AND knights played
        })

    def _all_hex_coords(self):
        return self.game.board.hexes.keys()

    def _all_intersection_coords(self):
        return self.game.board.intersections.keys()

    def _all_path_coords(self):
        return self.game.board.paths.keys()

    def _next_observation(self):
        obs = {'stage': self.stage.value}

        hexes = {hex_coord: i for i, hex_coord in enumerate(self._all_hex_coords())}
        obs['robber location'] = hexes[self.game.board.robber]

        buildings = np.zeros((4, 180))
        for i in range(len(self.game.players)):
            for j, coord in enumerate(self._all_intersection_coords()):
                building = self.game.board.intersections[coord].building
                if building is None:
                    continue
                if building.owner == self.game.players[(self.current_player_number + i) % 4]:
                    col = j
                    if building.building_type == BuildingType.CITY:
                        col += 54
                    buildings[i, col] = 1

            for j, coord in enumerate(self._all_path_coords()):
                building = self.game.board.paths[coord].building
                if building is None:
                    continue
                if building.owner == self.game.players[(self.current_player_number + i) % 4]:
                    buildings[i, j + 108] = 1
        obs['buildings'] = buildings

        cards = np.zeros((4, 11))
        for i in range(len(self.game.players)):
            for j, res in enumerate(Resource):
                cards[i, j] = self.game.players[(self.current_player_number + i) % 4].resources[res]
            for j, dev in enumerate(DevelopmentCard):
                cards[i, j + 5] = self.game.players[(self.current_player_number + i) % 4].development_cards[dev]
            cards[i, 10] = self.game.players[(self.current_player_number + i) % 4].number_played_knights
        obs['cards in hand'] = cards

        return obs

    def _perform_action(self, action):
        next_stage: Optional[GameStage] = None
        reward = 0
        next_player_number = self.current_player_number
        current_player = self.game.players[self.current_player_number]
        pass_turn = False
        try:
            match self.stage:

                case GameStage.NOT_ROLLED:
                    if (action[0] % 2 == 0 or
                            current_player.development_cards[DevelopmentCard.KNIGHT] == 0 or
                            self.played_dev_card):
                        dice = random.randint(1, 6) + random.randint(1, 6)
                        self.rolled = True
                        if dice == 7:
                            next_stage = GameStage.MOVING_ROBBER
                        else:
                            self.game.add_yield_for_roll(dice)
                            next_stage = GameStage.ROLLED
                    else:
                        self.game.play_development_card(current_player, DevelopmentCard.KNIGHT)
                        next_stage = GameStage.MOVING_ROBBER

                case GameStage.ROLLED:
                    if action[0] == 0:
                        pass_turn = True
                    elif action[0] == 1:
                        if current_player.development_cards[DevelopmentCard.KNIGHT] != 0 and not self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.KNIGHT)
                            next_stage = GameStage.MOVING_ROBBER
                        else:
                            pass_turn = True
                    elif action[0] == 2:
                        if current_player.development_cards[DevelopmentCard.YEAR_OF_PLENTY] != 0 and not \
                                self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.YEAR_OF_PLENTY)
                            resource = [res for res in Resource][action[1] % len(Resource)]
                            current_player.add_resources({resource: 1})
                            resource = [res for res in Resource][action[2] % len(Resource)]
                            current_player.add_resources({resource: 1})
                        else:
                            pass_turn = True
                    elif action[0] == 3:
                        if current_player.development_cards[DevelopmentCard.MONOPOLY] != 0 and not \
                                self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.MONOPOLY)
                            resource = [res for res in Resource][action[1] % 5]
                            for player in self.game.players:
                                if player is not current_player:
                                    amount = player.resources[resource]
                                    player.remove_resources({resource: amount})
                                    current_player.add_resources({resource: amount})
                        else:
                            pass_turn = True
                    elif action[0] == 4:
                        if current_player.development_cards[DevelopmentCard.ROAD_BUILDING] != 0 and not \
                                self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.ROAD_BUILDING)
                            valid_coords = self.game.board.get_valid_road_coords(current_player)
                            if valid_coords:
                                coord = funcs.choose_path(valid_coords, action[1] % len(valid_coords))
                                self.game.build_road(current_player, coord, cost_resources=False)
                            valid_coords = self.game.board.get_valid_road_coords(current_player)
                            if valid_coords:
                                coord = funcs.choose_path(valid_coords, action[2] % len(valid_coords))
                                self.game.build_road(current_player, coord, cost_resources=False)
                        else:
                            pass_turn = True
                    elif action[0] == 5:
                        if len(current_player.get_possible_trades()) == 0:
                            pass_turn = True
                        next_stage = GameStage.TRADING
                    elif action[0] == 6:
                        valid_coords = self.game.board.get_valid_settlement_coords(current_player)
                        if current_player.has_resources(BuildingType.SETTLEMENT.get_required_resources()) \
                                and valid_coords:
                            coord = funcs.choose_intersection(valid_coords, action[1] % len(valid_coords))
                            self.game.build_settlement(current_player, coord)
                            reward += 100
                        else:
                            pass_turn = True
                    elif action[0] == 7:
                        valid_coords = self.game.board.get_valid_city_coords(current_player)
                        if current_player.has_resources(BuildingType.CITY.get_required_resources()) \
                                and valid_coords:
                            coord = funcs.choose_intersection(valid_coords, action[1] % len(valid_coords))
                            self.game.upgrade_settlement_to_city(current_player, coord)
                            reward += 100
                        else:
                            pass_turn = True
                    elif action[0] == 8:
                        valid_coords = self.game.board.get_valid_road_coords(current_player)
                        if current_player.has_resources(BuildingType.ROAD.get_required_resources()) \
                                and valid_coords:
                            coord = funcs.choose_path(valid_coords, action[1] % len(valid_coords))
                            self.game.build_road(current_player, coord)
                            reward += 10
                        else:
                            pass_turn = True
                    elif action[0] == 9:
                        if current_player.has_resources(DevelopmentCard.get_required_resources()) and \
                                not self.bought_dev_card:
                            if self.game.build_development_card(current_player) == DevelopmentCard.VICTORY_POINT:
                                reward += 10
                        else:
                            pass_turn = True
                    else:
                        print("This shouldn't happen")
                        raise Exception

                case GameStage.MOVING_ROBBER:
                    valid_hexes = [coord for coord in self._all_hex_coords() if coord != self.game.board.robber]
                    self.game.board.robber = funcs.choose_hex(valid_hexes, action[2] % len(valid_hexes))
                    next_stage = GameStage.STEALING

                case GameStage.STEALING:
                    targets = list(self.game.board.get_players_on_hex(self.game.board.robber))
                    if len(targets) > 0:
                        target = targets[action[0] % len(targets)]
                        resource = target.get_random_resource()
                        if resource is not None:
                            target.remove_resources({resource: 1})
                            current_player.add_resources({resource: 1})
                    next_stage = GameStage.ROLLED if self.rolled else GameStage.NOT_ROLLED

                case GameStage.TRADING:
                    valid_trades = current_player.get_possible_trades()
                    trade = valid_trades[action[1] % len(valid_trades)]
                    current_player.add_resources(trade)
                    next_stage = GameStage.ROLLED

        except Exception:
            reward -= 3
            # self.illegal_acts += 1
            print(self.stage, action)
            pass_turn = True

        if pass_turn:
            self.rolled = False
            next_player_number = (next_player_number + 1) % 4
            if next_player_number == 0:
                self.turns += 1
                # print(f"turn {self.turns}, {self.illegal_acts} illegal actions")
                # self.illegal_acts = 0
            next_stage = GameStage.NOT_ROLLED
            self.bought_dev_card = False
            self.played_dev_card = False
            reward = -30

        done = self.game.get_victory_points(current_player) >= 10

        return next_stage, reward, next_player_number, done

    def step(self, action):
        assert isinstance(action, np.ndarray)
        next_stage, reward, next_player_number, done = self._perform_action(action)

        # if random.random() < 0.005:
        #     print(self.current_player_number, self.game.get_victory_points(current_player), self.stage, action)

        if next_stage is not None:
            self.stage = next_stage
        self.current_player_number = next_player_number

        while self.current_player_number != 0 and not done:
            model = self.models_to_beat[self.current_player_number - 1]
            opp_action, _state = model.predict(self._next_observation(), deterministic=False)
            next_stage, _, next_player_number, done = self._perform_action(opp_action)
            if next_stage is not None:
                self.stage = next_stage
            self.current_player_number = next_player_number

        if done:
            vps = self.game.get_victory_points(self.game.players[0])
            place = sum([vps > self.game.get_victory_points(player) for player in self.game.players])  # 3 = winner
            print(f"Done in {self.turns} turns, earned {vps} points, in place {4-place}!")

            reward += 1000 * place
            if place == 4:
                reward += 100000 / self.turns
        return self._next_observation(), reward, done, {}

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        self.game = Game(BeginnerBoard())
        self.renderer = BoardRenderer(self.game.board)
        self.stage = GameStage.NOT_ROLLED
        self.current_player_number = 0
        self.turns = 0
        # self.illegal_acts = 0
        self.rolled = False
        self.bought_dev_card = False
        self.played_dev_card = False
        print(f"Reset {self.resets} times")
        self.resets += 1

        starting_settlements = [Coords(-3, 2), Coords(-2, 3), Coords(1, 2), Coords(3, -1),
                                Coords(1, -3), Coords(-2, 0), Coords(2, -2), Coords(-1, -2)]
        starting_roads = [{Coords(-3, 2), Coords(-2, 2)}, {Coords(-1, 3), Coords(-2, 3)},
                          {Coords(1, 2), Coords(2, 1)}, {Coords(2, 0), Coords(3, -1)},
                          {Coords(1, -3), Coords(0, -2)}, {Coords(-2, 0), Coords(-3, 1)},
                          {Coords(2, -3), Coords(2, -2)}, {Coords(-1, -2), Coords(-2, -1)}]

        for i in range(8):
            player = self.game.players[min(i, 7 - i)]
            self.game.build_settlement(player, starting_settlements[i], cost_resources=False, ensure_connected=False)
            self.game.build_road(player, starting_roads[i], cost_resources=False, ensure_connected=False)
            if i >= 4:
                player.add_resources(self.game.board.get_hex_resources_for_intersection(starting_settlements[i]))

        return self._next_observation()

    def render(self, mode="human"):
        pass


class CatanEnvironmentDQN(gym.Env):
    game: Game
    renderer: BoardRenderer
    stage: GameStage
    current_player_number: int
    rolled: bool
    turns: int
    # illegal_acts: int
    bought_dev_card: bool
    played_dev_card: bool
    connected_intersection: Coords

    def __init__(self, models_to_beat, opp_uses_dict):
        super(CatanEnvironmentDQN, self).__init__()
        self.resets = 0
        self.models_to_beat = models_to_beat
        self.opp_uses_dict = opp_uses_dict

        self.first = 0
        self.second = 0
        self.third = 0
        self.fourth = 0

        # self.action_space = spaces.Discrete(72)
        self.action_space = spaces.Discrete(10*25*54)
        #
        # self.observation_space = spaces.Dict({
        #     'stage': spaces.Discrete(13),  # GameStage
        #     'robber location': spaces.Discrete(54),  # Robber location
        #     'buildings': spaces.Box(low=0, high=1, shape=(4, 180), dtype=np.byte),
        #     # Location of Settlements, Cities, Roads
        #     'cards in hand': spaces.Box(low=0, high=256, shape=(4, 11), dtype=np.byte)
        #     # Num cards in hand AND knights played
        # })

        self.observation_space = spaces.Box(low=0, high=256, shape=(4, 253))

    def _all_hex_coords(self):
        return self.game.board.hexes.keys()

    def _all_intersection_coords(self):
        return self.game.board.intersections.keys()

    def _all_path_coords(self):
        return self.game.board.paths.keys()

    def _next_observation(self):
        obs = np.zeros((4, 253))
        hexes = {hex_coord: i for i, hex_coord in enumerate(self._all_hex_coords())}

        # Stage
        obs[..., self.stage.value] = 255

        # Robber Location
        obs[..., 8 + hexes[self.game.board.robber]] = 255

        # Buildings and cards
        for i in range(len(self.game.players)):
            for j, coord in enumerate(self._all_intersection_coords()):
                building = self.game.board.intersections[coord].building
                if building is None:
                    continue
                if building.owner == self.game.players[(self.current_player_number + i) % 4]:
                    col = 62 + j
                    if building.building_type == BuildingType.CITY:
                        col += 54
                    obs[i, j] = 255

            for j, coord in enumerate(self._all_path_coords()):
                building = self.game.board.paths[coord].building
                if building is None:
                    continue
                if building.owner == self.game.players[(self.current_player_number + i) % 4]:
                    obs[i, 170 + j] = 255

            for j, res in enumerate(Resource):
                obs[i, 242 + j] = self.game.players[(self.current_player_number + i) % 4].resources[res]
            for j, dev in enumerate(DevelopmentCard):
                obs[i, 247 + j] = self.game.players[(self.current_player_number + i) % 4].development_cards[dev]
            obs[i, 252] = self.game.players[(self.current_player_number + i) % 4].number_played_knights

        return obs

    def _next_observation_dict(self):
        obs = {'stage': self.stage.value}

        hexes = {hex_coord: i for i, hex_coord in enumerate(self._all_hex_coords())}
        obs['robber location'] = hexes[self.game.board.robber]

        buildings = np.zeros((4, 180))
        for i in range(len(self.game.players)):
            for j, coord in enumerate(self._all_intersection_coords()):
                building = self.game.board.intersections[coord].building
                if building is None:
                    continue
                if building.owner == self.game.players[(self.current_player_number + i) % 4]:
                    col = j
                    if building.building_type == BuildingType.CITY:
                        col += 54
                    buildings[i, col] = 1

            for j, coord in enumerate(self._all_path_coords()):
                building = self.game.board.paths[coord].building
                if building is None:
                    continue
                if building.owner == self.game.players[(self.current_player_number + i) % 4]:
                    buildings[i, j + 108] = 1
        obs['buildings'] = buildings

        cards = np.zeros((4, 11))
        for i in range(len(self.game.players)):
            for j, res in enumerate(Resource):
                cards[i, j] = self.game.players[(self.current_player_number + i) % 4].resources[res]
            for j, dev in enumerate(DevelopmentCard):
                cards[i, j + 5] = self.game.players[(self.current_player_number + i) % 4].development_cards[dev]
            cards[i, 10] = self.game.players[(self.current_player_number + i) % 4].number_played_knights
        obs['cards in hand'] = cards

        return obs

    def _perform_action(self, action):
        next_stage: Optional[GameStage] = None
        reward = 0
        next_player_number = self.current_player_number
        current_player = self.game.players[self.current_player_number]
        pass_turn = False
        try:
            match self.stage:

                case GameStage.NOT_ROLLED:
                    if (action[0] % 2 == 0 or
                            current_player.development_cards[DevelopmentCard.KNIGHT] == 0 or
                            self.played_dev_card):
                        dice = random.randint(1, 6) + random.randint(1, 6)
                        self.rolled = True
                        if dice == 7:
                            next_stage = GameStage.MOVING_ROBBER
                        else:
                            self.game.add_yield_for_roll(dice)
                            next_stage = GameStage.ROLLED
                    else:
                        self.game.play_development_card(current_player, DevelopmentCard.KNIGHT)
                        next_stage = GameStage.MOVING_ROBBER

                case GameStage.ROLLED:
                    if action[0] == 0:
                        pass_turn = True
                    elif action[0] == 1:
                        if current_player.development_cards[DevelopmentCard.KNIGHT] != 0 and not self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.KNIGHT)
                            next_stage = GameStage.MOVING_ROBBER
                        else:
                            pass_turn = True
                    elif action[0] == 2:
                        if current_player.development_cards[DevelopmentCard.YEAR_OF_PLENTY] != 0 and not \
                                self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.YEAR_OF_PLENTY)
                            resource = [res for res in Resource][action[1] % len(Resource)]
                            current_player.add_resources({resource: 1})
                            resource = [res for res in Resource][action[2] % len(Resource)]
                            current_player.add_resources({resource: 1})
                        else:
                            pass_turn = True
                    elif action[0] == 3:
                        if current_player.development_cards[DevelopmentCard.MONOPOLY] != 0 and not \
                                self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.MONOPOLY)
                            resource = [res for res in Resource][action[1] % 5]
                            for player in self.game.players:
                                if player is not current_player:
                                    amount = player.resources[resource]
                                    player.remove_resources({resource: amount})
                                    current_player.add_resources({resource: amount})
                        else:
                            pass_turn = True
                    elif action[0] == 4:
                        if current_player.development_cards[DevelopmentCard.ROAD_BUILDING] != 0 and not \
                                self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.ROAD_BUILDING)
                            valid_coords = self.game.board.get_valid_road_coords(current_player)
                            if valid_coords:
                                coord = funcs.choose_path(valid_coords, action[1] % len(valid_coords))
                                self.game.build_road(current_player, coord, cost_resources=False)
                            valid_coords = self.game.board.get_valid_road_coords(current_player)
                            if valid_coords:
                                coord = funcs.choose_path(valid_coords, action[2] % len(valid_coords))
                                self.game.build_road(current_player, coord, cost_resources=False)
                        else:
                            pass_turn = True
                    elif action[0] == 5:
                        if len(current_player.get_possible_trades()) == 0:
                            pass_turn = True
                        next_stage = GameStage.TRADING
                    elif action[0] == 6:
                        valid_coords = self.game.board.get_valid_settlement_coords(current_player)
                        if current_player.has_resources(BuildingType.SETTLEMENT.get_required_resources()) \
                                and valid_coords:
                            coord = funcs.choose_intersection(valid_coords, action[1] % len(valid_coords))
                            self.game.build_settlement(current_player, coord)
                            reward += 100
                        else:
                            pass_turn = True
                    elif action[0] == 7:
                        valid_coords = self.game.board.get_valid_city_coords(current_player)
                        if current_player.has_resources(BuildingType.CITY.get_required_resources()) \
                                and valid_coords:
                            coord = funcs.choose_intersection(valid_coords, action[1] % len(valid_coords))
                            self.game.upgrade_settlement_to_city(current_player, coord)
                            reward += 100
                        else:
                            pass_turn = True
                    elif action[0] == 8:
                        valid_coords = self.game.board.get_valid_road_coords(current_player)
                        if current_player.has_resources(BuildingType.ROAD.get_required_resources()) \
                                and valid_coords:
                            coord = funcs.choose_path(valid_coords, action[1] % len(valid_coords))
                            self.game.build_road(current_player, coord)
                            reward += 10
                        else:
                            pass_turn = True
                    elif action[0] == 9:
                        if current_player.has_resources(DevelopmentCard.get_required_resources()) and \
                                not self.bought_dev_card:
                            if self.game.build_development_card(current_player) == DevelopmentCard.VICTORY_POINT:
                                reward += 10
                        else:
                            pass_turn = True
                    else:
                        print("This shouldn't happen")
                        raise Exception

                case GameStage.MOVING_ROBBER:
                    valid_hexes = [coord for coord in self._all_hex_coords() if coord != self.game.board.robber]
                    self.game.board.robber = funcs.choose_hex(valid_hexes, action[2] % len(valid_hexes))
                    next_stage = GameStage.STEALING

                case GameStage.STEALING:
                    targets = list(self.game.board.get_players_on_hex(self.game.board.robber))
                    if len(targets) > 0:
                        target = targets[action[0] % len(targets)]
                        resource = target.get_random_resource()
                        if resource is not None:
                            target.remove_resources({resource: 1})
                            current_player.add_resources({resource: 1})
                    next_stage = GameStage.ROLLED if self.rolled else GameStage.NOT_ROLLED

                case GameStage.TRADING:
                    valid_trades = current_player.get_possible_trades()
                    trade = valid_trades[action[1] % len(valid_trades)]
                    current_player.add_resources(trade)
                    next_stage = GameStage.ROLLED

        except Exception:
            reward -= 3
            # self.illegal_acts += 1
            print(self.stage, action)
            pass_turn = True

        if pass_turn:
            self.rolled = False
            next_player_number = (next_player_number + 1) % 4
            if next_player_number == 0:
                self.turns += 1
                # print(f"turn {self.turns}, {self.illegal_acts} illegal actions")
                # self.illegal_acts = 0
            next_stage = GameStage.NOT_ROLLED
            self.bought_dev_card = False
            self.played_dev_card = False
            reward = -30

        done = self.game.get_victory_points(current_player) >= 10

        return next_stage, reward, next_player_number, done

    def step(self, action):
        assert isinstance(action, np.int32) or isinstance(action, np.int64)
        action = [action//(54*25), (action//54) % 25, action % 54]
        next_stage, reward, next_player_number, done = self._perform_action(action)

        # if random.random() < 0.005:
        #     print(self.current_player_number, self.game.get_victory_points(current_player), self.stage, action)

        if next_stage is not None:
            self.stage = next_stage
        self.current_player_number = next_player_number

        while self.current_player_number != 0 and not done:
            model = self.models_to_beat[self.current_player_number - 1]
            obs = self._next_observation_dict if self.opp_uses_dict else self._next_observation
            opp_action, _state = model.predict(obs(), deterministic=False)
            next_stage, _, next_player_number, done = self._perform_action(opp_action)
            if next_stage is not None:
                self.stage = next_stage
            self.current_player_number = next_player_number

        if done:
            vps = self.game.get_victory_points(self.game.players[0])
            place = sum([vps > self.game.get_victory_points(player) for player in self.game.players])  # 3 = winner
            print(f"Done in {self.turns} turns, earned {vps} points, in place {4-place}!")

            reward += 1000 * place
            if place == 3:
                reward += 100000 / self.turns
                self.first += 1
            elif place == 2:
                self.second += 1
            elif place == 1:
                self.third += 1
            else:
                self.fourth += 1
        return self._next_observation(), reward, done, {}

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        self.game = Game(BeginnerBoard())
        self.renderer = BoardRenderer(self.game.board)
        self.stage = GameStage.NOT_ROLLED
        self.current_player_number = 0
        self.turns = 0
        # self.illegal_acts = 0
        self.rolled = False
        self.bought_dev_card = False
        self.played_dev_card = False
        print(f"Reset {self.resets} times")
        if self.resets % 10 == 0:
            print(f"{self.first} first place, {self.second} second place, "
                  f"{self.third} third place, {self.fourth} fourth place")
        self.resets += 1

        starting_settlements = [Coords(-3, 2), Coords(-2, 3), Coords(1, 2), Coords(3, -1),
                                Coords(1, -3), Coords(-2, 0), Coords(2, -2), Coords(-1, -2)]
        starting_roads = [{Coords(-3, 2), Coords(-2, 2)}, {Coords(-1, 3), Coords(-2, 3)},
                          {Coords(1, 2), Coords(2, 1)}, {Coords(2, 0), Coords(3, -1)},
                          {Coords(1, -3), Coords(0, -2)}, {Coords(-2, 0), Coords(-3, 1)},
                          {Coords(2, -3), Coords(2, -2)}, {Coords(-1, -2), Coords(-2, -1)}]

        for i in range(8):
            player = self.game.players[min(i, 7 - i)]
            self.game.build_settlement(player, starting_settlements[i], cost_resources=False, ensure_connected=False)
            self.game.build_road(player, starting_roads[i], cost_resources=False, ensure_connected=False)
            if i >= 4:
                player.add_resources(self.game.board.get_hex_resources_for_intersection(starting_settlements[i]))

        return self._next_observation()

    def render(self, mode="human"):
        pass


class CatanEnvironmentDQNSettlements(gym.Env):
    game: Game
    renderer: BoardRenderer
    stage: GameStage
    current_player_number: int
    rolled: bool
    turns: int
    # illegal_acts: int
    bought_dev_card: bool
    played_dev_card: bool
    connected_intersection: Coords

    def __init__(self, model_to_use, models_to_beat, opp_uses_dict=True):
        super(CatanEnvironmentDQNSettlements, self).__init__()
        self.resets = 0
        self.model_to_use = model_to_use
        self.models_to_beat = models_to_beat
        self.opp_uses_dict = opp_uses_dict

        self.first = 0
        self.second = 0
        self.third = 0
        self.fourth = 0

        # self.action_space = spaces.Discrete(72)
        self.action_space = spaces.Discrete(54)
        #
        # self.observation_space = spaces.Dict({
        #     'stage': spaces.Discrete(13),  # GameStage
        #     'robber location': spaces.Discrete(54),  # Robber location
        #     'buildings': spaces.Box(low=0, high=1, shape=(4, 180), dtype=np.byte),
        #     # Location of Settlements, Cities, Roads
        #     'cards in hand': spaces.Box(low=0, high=256, shape=(4, 11), dtype=np.byte)
        #     # Num cards in hand AND knights played
        # })

        self.observation_space = spaces.Box(low=0, high=255, shape=(4, 253), dtype=np.byte)

    def _all_hex_coords(self):
        return self.game.board.hexes.keys()

    def _all_intersection_coords(self):
        return self.game.board.intersections.keys()

    def _all_path_coords(self):
        return self.game.board.paths.keys()

    def _next_observation(self):
        obs = np.zeros((4, 253))
        hexes = {hex_coord: i for i, hex_coord in enumerate(self._all_hex_coords())}

        # Stage
        obs[..., self.stage.value] = 255

        # Robber Location
        obs[..., 8 + hexes[self.game.board.robber]] = 255

        # Buildings and cards
        for i in range(len(self.game.players)):
            for j, coord in enumerate(self._all_intersection_coords()):
                building = self.game.board.intersections[coord].building
                if building is None:
                    continue
                if building.owner == self.game.players[(self.current_player_number + i) % 4]:
                    col = 62 + j
                    if building.building_type == BuildingType.CITY:
                        col += 54
                    obs[i, j] = 255

            for j, coord in enumerate(self._all_path_coords()):
                building = self.game.board.paths[coord].building
                if building is None:
                    continue
                if building.owner == self.game.players[(self.current_player_number + i) % 4]:
                    obs[i, 170 + j] = 255

            for j, res in enumerate(Resource):
                obs[i, 242 + j] = self.game.players[(self.current_player_number + i) % 4].resources[res]
            for j, dev in enumerate(DevelopmentCard):
                obs[i, 247 + j] = self.game.players[(self.current_player_number + i) % 4].development_cards[dev]
            obs[i, 252] = self.game.players[(self.current_player_number + i) % 4].number_played_knights

        return obs

    def _next_observation_dict(self):
        obs = {'stage': self.stage.value}

        hexes = {hex_coord: i for i, hex_coord in enumerate(self._all_hex_coords())}
        obs['robber location'] = hexes[self.game.board.robber]

        buildings = np.zeros((4, 180))
        for i in range(len(self.game.players)):
            for j, coord in enumerate(self._all_intersection_coords()):
                building = self.game.board.intersections[coord].building
                if building is None:
                    continue
                if building.owner == self.game.players[(self.current_player_number + i) % 4]:
                    col = j
                    if building.building_type == BuildingType.CITY:
                        col += 54
                    buildings[i, col] = 1

            for j, coord in enumerate(self._all_path_coords()):
                building = self.game.board.paths[coord].building
                if building is None:
                    continue
                if building.owner == self.game.players[(self.current_player_number + i) % 4]:
                    buildings[i, j + 108] = 1
        obs['buildings'] = buildings

        cards = np.zeros((4, 11))
        for i in range(len(self.game.players)):
            for j, res in enumerate(Resource):
                cards[i, j] = self.game.players[(self.current_player_number + i) % 4].resources[res]
            for j, dev in enumerate(DevelopmentCard):
                cards[i, j + 5] = self.game.players[(self.current_player_number + i) % 4].development_cards[dev]
            cards[i, 10] = self.game.players[(self.current_player_number + i) % 4].number_played_knights
        obs['cards in hand'] = cards

        return obs

    def _perform_action(self, action):
        next_stage: Optional[GameStage] = None
        reward = 0
        next_player_number = self.current_player_number
        current_player = self.game.players[self.current_player_number]
        pass_turn = False
        try:
            match self.stage:

                case GameStage.NOT_ROLLED:
                    if (action[0] % 2 == 0 or
                            current_player.development_cards[DevelopmentCard.KNIGHT] == 0 or
                            self.played_dev_card):
                        dice = random.randint(1, 6) + random.randint(1, 6)
                        self.rolled = True
                        if dice == 7:
                            next_stage = GameStage.MOVING_ROBBER
                        else:
                            self.game.add_yield_for_roll(dice)
                            next_stage = GameStage.ROLLED
                    else:
                        self.game.play_development_card(current_player, DevelopmentCard.KNIGHT)
                        next_stage = GameStage.MOVING_ROBBER

                case GameStage.ROLLED:
                    if action[0] == 0:
                        pass_turn = True
                    elif action[0] == 1:
                        if current_player.development_cards[DevelopmentCard.KNIGHT] != 0 and not self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.KNIGHT)
                            next_stage = GameStage.MOVING_ROBBER
                        else:
                            pass_turn = True
                    elif action[0] == 2:
                        if current_player.development_cards[DevelopmentCard.YEAR_OF_PLENTY] != 0 and not \
                                self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.YEAR_OF_PLENTY)
                            resource = [res for res in Resource][action[1] % len(Resource)]
                            current_player.add_resources({resource: 1})
                            resource = [res for res in Resource][action[2] % len(Resource)]
                            current_player.add_resources({resource: 1})
                        else:
                            pass_turn = True
                    elif action[0] == 3:
                        if current_player.development_cards[DevelopmentCard.MONOPOLY] != 0 and not \
                                self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.MONOPOLY)
                            resource = [res for res in Resource][action[1] % 5]
                            for player in self.game.players:
                                if player is not current_player:
                                    amount = player.resources[resource]
                                    player.remove_resources({resource: amount})
                                    current_player.add_resources({resource: amount})
                        else:
                            pass_turn = True
                    elif action[0] == 4:
                        if current_player.development_cards[DevelopmentCard.ROAD_BUILDING] != 0 and not \
                                self.played_dev_card:
                            self.game.play_development_card(current_player, DevelopmentCard.ROAD_BUILDING)
                            valid_coords = self.game.board.get_valid_road_coords(current_player)
                            if valid_coords:
                                coord = funcs.choose_path(valid_coords, action[1] % len(valid_coords))
                                self.game.build_road(current_player, coord, cost_resources=False)
                            valid_coords = self.game.board.get_valid_road_coords(current_player)
                            if valid_coords:
                                coord = funcs.choose_path(valid_coords, action[2] % len(valid_coords))
                                self.game.build_road(current_player, coord, cost_resources=False)
                        else:
                            pass_turn = True
                    elif action[0] == 5:
                        if len(current_player.get_possible_trades()) == 0:
                            pass_turn = True
                        next_stage = GameStage.TRADING
                    elif action[0] == 6:
                        valid_coords = self.game.board.get_valid_settlement_coords(current_player)
                        if current_player.has_resources(BuildingType.SETTLEMENT.get_required_resources()) \
                                and valid_coords:
                            coord = funcs.choose_intersection(valid_coords, action[1] % len(valid_coords))
                            self.game.build_settlement(current_player, coord)
                            reward += 100
                        else:
                            pass_turn = True
                    elif action[0] == 7:
                        valid_coords = self.game.board.get_valid_city_coords(current_player)
                        if current_player.has_resources(BuildingType.CITY.get_required_resources()) \
                                and valid_coords:
                            coord = funcs.choose_intersection(valid_coords, action[1] % len(valid_coords))
                            self.game.upgrade_settlement_to_city(current_player, coord)
                            reward += 100
                        else:
                            pass_turn = True
                    elif action[0] == 8:
                        valid_coords = self.game.board.get_valid_road_coords(current_player)
                        if current_player.has_resources(BuildingType.ROAD.get_required_resources()) \
                                and valid_coords:
                            coord = funcs.choose_path(valid_coords, action[1] % len(valid_coords))
                            self.game.build_road(current_player, coord)
                            reward += 10
                        else:
                            pass_turn = True
                    elif action[0] == 9:
                        if current_player.has_resources(DevelopmentCard.get_required_resources()) and \
                                not self.bought_dev_card:
                            if self.game.build_development_card(current_player) == DevelopmentCard.VICTORY_POINT:
                                reward += 10
                        else:
                            pass_turn = True
                    else:
                        print("This shouldn't happen")
                        raise Exception

                case GameStage.MOVING_ROBBER:
                    valid_hexes = [coord for coord in self._all_hex_coords() if coord != self.game.board.robber]
                    self.game.board.robber = funcs.choose_hex(valid_hexes, action[2] % len(valid_hexes))
                    next_stage = GameStage.STEALING

                case GameStage.STEALING:
                    targets = list(self.game.board.get_players_on_hex(self.game.board.robber))
                    if len(targets) > 0:
                        target = targets[action[0] % len(targets)]
                        resource = target.get_random_resource()
                        if resource is not None:
                            target.remove_resources({resource: 1})
                            current_player.add_resources({resource: 1})
                    next_stage = GameStage.ROLLED if self.rolled else GameStage.NOT_ROLLED

                case GameStage.TRADING:
                    valid_trades = current_player.get_possible_trades()
                    trade = valid_trades[action[1] % len(valid_trades)]
                    current_player.add_resources(trade)
                    next_stage = GameStage.ROLLED

        except Exception:
            reward -= 3
            # self.illegal_acts += 1
            print(self.stage, action)
            pass_turn = True

        if pass_turn:
            self.rolled = False
            next_player_number = (next_player_number + 1) % 4
            if next_player_number == 0:
                self.turns += 1
                # print(f"turn {self.turns}, {self.illegal_acts} illegal actions")
                # self.illegal_acts = 0
            next_stage = GameStage.NOT_ROLLED
            self.bought_dev_card = False
            self.played_dev_card = False
            reward = -30

        done = self.game.get_victory_points(current_player) >= 10

        return next_stage, reward, next_player_number, done

    def step(self, settlement_loc):
        assert isinstance(settlement_loc, np.int32) or isinstance(settlement_loc, np.int64)

        action, _state = self.model_to_use.predict(self._next_observation_dict(), deterministic=False)
        if action[0] == 6 and \
                self.game.board.get_valid_settlement_coords(self.game.players[0]) and \
                self.game.players[0].has_resources(BuildingType.SETTLEMENT.get_required_resources()):
            action[1] = settlement_loc
        next_stage, reward, next_player_number, done = self._perform_action(action)

        # if random.random() < 0.005:
        #     print(self.current_player_number, self.game.get_victory_points(current_player), self.stage, action)

        if next_stage is not None:
            self.stage = next_stage
        self.current_player_number = next_player_number

        while self.current_player_number != 0 and not done:
            model = self.models_to_beat[self.current_player_number - 1]
            obs = self._next_observation_dict if self.opp_uses_dict else self._next_observation
            opp_action, _state = model.predict(obs(), deterministic=False)
            next_stage, _, next_player_number, done = self._perform_action(opp_action)
            if next_stage is not None:
                self.stage = next_stage
            self.current_player_number = next_player_number

        if done:
            vps = self.game.get_victory_points(self.game.players[0])
            place = sum([vps > self.game.get_victory_points(player) for player in self.game.players])  # 3 = winner
            print(f"Done in {self.turns} turns, earned {vps} points, in place {4-place}!")

            reward += 1000 * place
            if place == 3:
                reward += 100000 / self.turns
                self.first += 1
            elif place == 2:
                self.second += 1
            elif place == 1:
                self.third += 1
            else:
                self.fourth += 1
        return self._next_observation(), reward, done, {}

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        self.game = Game(BeginnerBoard())
        self.renderer = BoardRenderer(self.game.board)
        self.stage = GameStage.NOT_ROLLED
        self.current_player_number = 0
        self.turns = 0
        # self.illegal_acts = 0
        self.rolled = False
        self.bought_dev_card = False
        self.played_dev_card = False
        print(f"Reset {self.resets} times")
        if self.resets % 10 == 0:
            print(f"{self.first} first place, {self.second} second place, "
                  f"{self.third} third place, {self.fourth} fourth place")
        self.resets += 1

        starting_settlements = [Coords(-3, 2), Coords(-2, 3), Coords(1, 2), Coords(3, -1),
                                Coords(1, -3), Coords(-2, 0), Coords(2, -2), Coords(-1, -2)]
        starting_roads = [{Coords(-3, 2), Coords(-2, 2)}, {Coords(-1, 3), Coords(-2, 3)},
                          {Coords(1, 2), Coords(2, 1)}, {Coords(2, 0), Coords(3, -1)},
                          {Coords(1, -3), Coords(0, -2)}, {Coords(-2, 0), Coords(-3, 1)},
                          {Coords(2, -3), Coords(2, -2)}, {Coords(-1, -2), Coords(-2, -1)}]

        for i in range(8):
            player = self.game.players[min(i, 7 - i)]
            self.game.build_settlement(player, starting_settlements[i], cost_resources=False, ensure_connected=False)
            self.game.build_road(player, starting_roads[i], cost_resources=False, ensure_connected=False)
            if i >= 4:
                player.add_resources(self.game.board.get_hex_resources_for_intersection(starting_settlements[i]))

        return self._next_observation()

    def render(self, mode="human"):
        pass
