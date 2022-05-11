
from enum import Enum


class GameStage(Enum):

    NOT_ROLLED = 0
    ROLLED = 1

    PLACING_ROAD = 2
    PLACING_SETTLEMENT = 3
    PLACING_CITY = 4

    MOVING_ROBBER = 5
    STEALING = 6

    TRADING = 7
