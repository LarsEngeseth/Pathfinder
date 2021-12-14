# -*- encoding: utf-8 -*-

"""
Programming AI to find an optimal path is a classic problem in AI.
What seems intuitive to humans is less intuitive to an agent that relies on its vision
and a rule-base to find the way through the maze.

To solve this assignment, I programmed a lot of overhead around the agent.
I made 4 classes:
1. Coord2D
2. WorldMap
3. Robot
4. Simulation

View README.md for details.
"""

__author__ = "Lars Engesæth, NMBU/SDSU"
__email__ = 'laen@nmbu.no / lengesaeth3430@sdsu.edu'

import logging
import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import numba as nb

from random import randint
from math import sqrt
from numba import njit
from numba import types
from numba.typed import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
matplotlib.use('Tkagg')


class Coord2D:
    def __init__(self, row, column=None) -> None:
        if isinstance(row, int):
            self.row = row
            self.column = column
        elif isinstance(row, float):
            self.row = int(row)
            self.column = int(column)
        elif isinstance(row, Coord2D):
            self.row = row.row
            self.column = row.column
        else:
            self.row = int(row[0])
            self.column = int(row[1])

    def __add__(self, b):
        return Coord2D(self.row + b.row, self.column + b.column)

    def __sub__(self, b):
        return Coord2D(self.row - b.row, self.column - b.column)

    def __eq__(self, b):
        return ((self.row == b.row) and (self.column == b.column))

    def __repr__(self):
        return "Coord2D({},{})".format(self.row, self.column)

    def __str__(self):
        return "({},{})".format(self.row, self.column)

    def distanceTo(self, b):
        return sqrt((float(self.row - b.row)) ** 2 + float((self.column - b.column)) ** 2.0)

    def __mul__(self, b):
        return Coord2D(self.row * b, self.column * b)

    def copy(self):
        return Coord2D(self.row, self.column)


class WorldMap:
    def __getitem__(self, index):
        index = self._makeC2D(index)
        if self._isWithinMap(index):
            return self.map[index.row, index.column]
        else:
            return self._oob

    def __setitem__(self, index, item):
        index = self._makeC2D(index)
        if self._isWithinMap(index):
            self.map[index.row, index.column] = item

    def __array__(self):
        return self.map

    def __eq__(self, b):
        return self.map == b

    def __add__(self, b):
        self.map + b

    def __mul__(self, b):
        self.map * b

    def __div__(self, b):
        self.map / b

    def __minus__(self, b):
        self.map - b

    @staticmethod
    def _makeC2D(index):
        if not isinstance(index, Coord2D):
            index = Coord2D(index)
        return index

    def _isWithinMap(self, tile: Coord2D):
        tile = self._makeC2D(tile)
        return (
                tile.row >= 0 and tile.row < self.maxRow and tile.column >= 0 and tile.column < self.maxColumn)

    def __init__(self, shape, defaultOOB, dtype=None, warnOOB=False):
        self.map = np.zeros(shape, dtype=dtype)
        self.maxRow, self.maxColumn = self.map.shape
        self._oob = defaultOOB
        self._warnOOB = warnOOB
        self.shape = self.map.shape

    @property
    def dtype(self):
        return self.map.dtype

    def copy(self):
        newMap = WorldMap(self.shape, self._oob, dtype=self.dtype, warnOOB=self._warnOOB)
        newMap.map = self.map.copy()
        return newMap


class Robot:
    CLOCKWISE_MAP = {
        str(Coord2D(1, 0)): Coord2D(1, -1),
        str(Coord2D(1, -1)): Coord2D(0, -1),
        str(Coord2D(0, -1)): Coord2D(-1, -1),
        str(Coord2D(-1, -1)): Coord2D(-1, 0),
        str(Coord2D(-1, 0)): Coord2D(-1, 1),
        str(Coord2D(-1, 1)): Coord2D(0, 1),
        str(Coord2D(0, 1)): Coord2D(1, 1),
        str(Coord2D(1, 1)): Coord2D(1, 0),

    }

    ANTICLOCKWISE_MAP = {
        str(Coord2D(1, 0)): Coord2D(1, 1),
        str(Coord2D(1, 1)): Coord2D(0, 1),
        str(Coord2D(0, 1)): Coord2D(-1, 1),
        str(Coord2D(-1, 1)): Coord2D(-1, 0),
        str(Coord2D(-1, 0)): Coord2D(-1, -1),
        str(Coord2D(-1, -1)): Coord2D(0, -1),
        str(Coord2D(0, -1)): Coord2D(1, -1),
        str(Coord2D(1, -1)): Coord2D(1, 0),

    }

    def __init__(self, row: int, column: int, direction: Coord2D, cognitiveMap) -> None:
        self.position = Coord2D(row, column)
        if isinstance(direction, list):
            direction = Coord2D(direction[0], direction[1])

        self.direction = direction
        self.cognitiveMap = cognitiveMap
        self.directionLeft = None
        self.directionMid = self.direction
        self.directionRight = None
        self.visionLeft = None
        self.visionMid = None
        self.visionRight = None
        self._setVision()
        self.goal = Coord2D(self._getGoalFromMap())
        self.move = 0
        self.pathPenaltyDict = {}
        self.penalty = 20.0
        self.lastPosition = self.position
        self._buildFog()

    def _buildFog(self):
        for row in range(self.cognitiveMap.shape[0]):
            for column in range(self.cognitiveMap.shape[1]):
                tile = self._getTileValue(Coord2D(row, column))
                if tile == Simulation.Open_cell:
                    self.cognitiveMap[row, column] = Simulation.Fog_cell

    def markVision(self, left, mid, right):
        visionLeft, visionMid, visionRight = self.getVision()
        self._markTile(visionLeft, left)
        self._markTile(visionMid, mid)
        self._markTile(visionRight, right)

    def markLineOfSight(self, lefts, mids, rights):
        losTiles = ((self.directionLeft, lefts), (self.directionMid, mids), (self.directionRight, rights))
        for dir, tileValues in losTiles:
            dist = 0
            for tileValue in tileValues:
                dist += 1
                tile = self.position + dir * dist
                self._markTile(tile, tileValue)

    def _markTile(self, vision, tileValue):
        # we don't want to overwrite path tiles with free tiles
        if tileValue == Simulation.Open_cell:
            if self.cognitiveMap[vision.row, vision.column] == Simulation.Path_cell:
                return
        self.cognitiveMap[vision.row, vision.column] = tileValue

    def _rotateLeft(self, direction: Coord2D):
        direction2D = Robot.CLOCKWISE_MAP[str(direction)]
        return direction2D

    def _rotateRight(self, direction: Coord2D):
        direction2D = Robot.ANTICLOCKWISE_MAP[str(direction)]
        return direction2D

    def getVision(self):
        left = self.position + self.directionLeft
        mid = self.position + self.directionMid
        right = self.position + self.directionRight
        return left, mid, right

    def getVisionDir(self):
        return self.directionLeft, self.directionMid, self.directionRight

    def _setVision(self):
        self.directionLeft = self._rotateLeft(self.direction)
        self.directionMid = self.direction
        self.directionRight = self._rotateRight(self.direction)
        self.visionLeft, self.visionMid, self.visionRight = self.getVision()

    def _getGoalFromMap(self):
        index = np.ravel(np.asarray(self.cognitiveMap == Simulation.Goal_cell).nonzero())
        logging.info("Goal coords: {}".format(index))
        return index

    def _makeDistDict(self, tileTuples):
        tileDists = {}
        for tile, dist in tileTuples:
            if self.penalty > 0.0:
                penalty = self.pathPenaltyDict.get(str(tile), 0.0)
                dist += penalty
            tileList = tileDists.get(dist, [])
            tileList.append(tile)
            tileDists.update({dist: tileList})
        dists = list(tileDists)
        dists.sort()
        return tileDists, dists

    def _getVisionTileDistances(self):
        left, mid, right = self.getVision()
        dLeft = left.distanceTo(self.goal)
        dMid = mid.distanceTo(self.goal)
        dRight = right.distanceTo(self.goal)
        tileTuples = ((left, dLeft), (mid, dMid), (right, dRight))
        tileDists, dists = self._makeDistDict(tileTuples)
        return tileDists, dists

    def _isObstacle(self, tile):
        return self.cognitiveMap[tile.row, tile.column] == Simulation.Obstacle_cell

    def _getTileValue(self, tile):
        return self.cognitiveMap[tile.row, tile.column]

    def _fullScan(self):
        tiles = []
        direction = self.directionLeft
        directionsTested = 0
        while True:
            scan = self.position + direction
            tile = self._getTileValue(scan)
            directionsTested += 1
            if ((tile == Simulation.Open_cell) or (tile == Simulation.Path_cell) or (
                    tile == Simulation.Fog_cell)):
                dist = scan.distanceTo(self.goal)
                tiles.append((scan, dist))
            direction = self._rotateLeft(direction)
            if direction == self.directionLeft:
                break
        tileDists, dists = self._makeDistDict(tiles)
        logging.info("Directions tested:{}".format(directionsTested))
        return tileDists, dists

    def decideMove(self):
        # update map if we moved somewhere new
        if self.lastPosition != self.position:
            self._markTile(self.lastPosition, Simulation.Path_cell)
            self.lastPosition = self.position.copy()

        # look up possible paths
        tileDists, dists = self._getVisionTileDistances()
        moveTo = None
        currentDistance = self.position.distanceTo(self.goal)

        # Consider options:
        for dist in dists:
            tiles = tileDists.get(dist, None)
            if tiles is not None:
                freeTiles = []
                for tile in tiles:
                    # add only viable paths
                    if not self._isObstacle(tile):
                        freeTiles.append(tile)

                if len(freeTiles) == 1:
                    # move to the only open tile
                    moveTo = freeTiles[0]
                    if self._getTileValue(moveTo) == Simulation.Path_cell:
                        # we have already tried this cell
                        moveTo = None
                        continue
                    break
                elif len(freeTiles) == 0:
                    # if there are no open tiles
                    continue
                else:
                    # in case of equidistant options
                    moveTo = freeTiles[randint(0, len(freeTiles) - 1)]
                    # we may miss something here, but it is unlikely,
                    # and we can catch it on the full scan
                    if self._getTileValue(moveTo) == Simulation.Path_cell:
                        # we have already tried this path
                        moveTo = None
                        continue
                    break

        if moveTo is not None:
            # if we cant move towards goal
            # get a second opinion if we are moving away
            moveToDist = moveTo.distanceTo(self.goal)
            if moveToDist > currentDistance:
                moveTo = None

        if moveTo is None:
            logging.info("Help, we must reconsider...!")
            # all tiles we can see are obstacles or already visited
            # we will have to decide where to turn
            tileDists, dists = self._fullScan()
            logging.info("Distances: {}".format(tileDists))
            for dist in dists:
                tiles = tileDists.get(dist, None)
                if tiles is not None:
                    tile = tiles[0]
                    if len(tiles) > 1:
                        # in case of equidistant options, we choose random
                        tile = tiles[randint(0, len(tiles) - 1)]
                    distLeft = self.visionLeft.distanceTo(tile)
                    distRight = self.visionRight.distanceTo(tile)

                    # turn towards exit cell, or back up if it is behind us
                    if distLeft < distRight:
                        return self.visionLeft
                    elif distRight < distLeft:
                        return self.visionLeft
                    else:
                        # reverse backwards
                        moveTo = self.getBack()
                        tile = self._getTileValue(moveTo)
                        if tile != Simulation.Obstacle_cell:
                            # Reverse into new cell that is not a obstacle
                            logging.info("Reversing...")
                            return moveTo
                        else:
                            rngTiles = (self.visionLeft, self.visionRight)
                            return rngTiles[randint(0, 1)]
            logging.info("Now it is real trouble!")
            # We mostly return some values instead of getting this far.
            return None
        else:
            # we can choose freely, so we take the shortest path to the goal
            return moveTo

    def getBack(self) -> Coord2D:
        direction = self.direction * -1
        back = self.position + direction
        return back

    def backedIntoWall(self, oldPos: Coord2D):
        self.cognitiveMap[self.position.row, self.position.column] = Simulation.Obstacle_cell
        self.position = oldPos.copy()

    def DoAction(self, moveTo):
        logging.info(
            "Robot:{} New tile:{} Sight:{} {} {}".format(self.position, moveTo, self.visionLeft,
                                                         self.visionMid, self.visionRight))
        if moveTo == self.visionMid:
            return self._DoMove(moveTo)
        elif ((moveTo == self.visionRight) or (moveTo == self.visionLeft)):
            return self._DoTurn(moveTo)
        elif moveTo == self.getBack():
            return self._DoMove(moveTo)

    def _addPathPenalty(self):
        position = str(self.position)
        penalty = self.pathPenaltyDict.get(position, None)

        if penalty is None:
            self.pathPenaltyDict.update({position: self.penalty})
        else:
            self.pathPenaltyDict[position] += self.penalty

    def _DoMove(self, moveTo):
        logging.info('Moving 1 cell...')
        self.cognitiveMap[self.position.row, self.position.column] = Simulation.Open_cell
        if self.penalty > 0.0:
            self._addPathPenalty()
        self.position = moveTo
        self.cognitiveMap[self.position.row, self.position.column] = Simulation.Robot_Start
        self._setVision()

    def _DoTurn(self, turnTo):
        logging.info("Turning on the spot...")
        self.move += 1
        newDirection = turnTo - self.position
        logging.info(
            "Robot pos:{} Direction:{} {} New direction:{}".format(self.position, self.direction,
                                                                   self.directionMid, newDirection))
        self.direction = newDirection
        self._setVision()

    def getPosition(self):
        return self.position


class Simulation:
    # TODO: User should be able to be change standard values dynamically.
    """
    This class takes care of the simulation. It contains all default values and several methods to
    make the simulation run.
    Lots of the methods are overhead, runSim is the important method controlling the robot until
    it reaches it's destination.
    """

    Default_Heading = Coord2D(1, 0)

    # Standard start location.
    # Can be changed. Takes integers only.
    Standard_Init_Row = 6
    Standard_Init_Column = 6

    # Standard locations of goal position.
    # Can be changed. Takes integers only.
    Standard_Goal_Row = 32
    Standard_Goal_Column = 37

    # These are fixed,
    # but can be altered to increase or decrease the size of the board.
    Standard_Map_Row = 35
    Standard_Map_Column = 40

    # Colouring cells
    Standard_Open_Color = 'white'
    Standard_Robot_Color = 'orange'
    Standard_Path_Color = 'red'
    Standard_Goal_Color = 'green'
    Standard_Obstacle_Colour = 'black'
    Standard_FogofWar_Colour = np.array([47, 79, 79])  # Gray

    # As requested by assignment
    Open_cell: int = 0
    Obstacle_cell: int = 1
    Robot_Start: int = 2
    Goal_cell: int = 3
    Path_cell: int = 4
    Fog_cell: int = 5

    # How high the % of obstacles are.
    # Should be able to be changed by user. TODO
    Obstacle_percentage = 0.20

    @classmethod
    def makeDefaultWorld(cls, obstacles: float = Obstacle_percentage):
        map = WorldMap((cls.Standard_Map_Row, cls.Standard_Map_Column), cls.Obstacle_cell)
        obsmap = WorldMap((cls.Standard_Map_Row, cls.Standard_Map_Column), cls.Obstacle_cell)
        cogMap = WorldMap((cls.Standard_Map_Row, cls.Standard_Map_Column), cls.Obstacle_cell)

        map[cls.Standard_Init_Row, cls.Standard_Init_Column] = cls.Robot_Start
        map[cls.Standard_Goal_Row, cls.Standard_Goal_Column] = cls.Goal_cell

        cogMap[cls.Standard_Init_Row, cls.Standard_Init_Column] = cls.Robot_Start
        cogMap[cls.Standard_Goal_Row, cls.Standard_Goal_Column] = cls.Goal_cell

        cls.addObstacles(map, obsmap, obstacles=obstacles)
        return map, obsmap, cogMap

    @classmethod
    def addObstacles(cls, map, obsmap, obstacles: float):
        # This method counts the number of tiles and adds the specified % of obstacles.
        mapSize = map.shape[0] * map.shape[1]
        rowmax = map.shape[0] - 1
        colmax = map.shape[1] - 1
        obstacleNum = int(mapSize * obstacles)
        obstaclePlaced = 0
        while obstaclePlaced < obstacleNum:
            rngRow = randint(0, rowmax)
            rngCol = randint(0, colmax)
            if map[rngRow, rngCol] == cls.Open_cell:
                map[rngRow, rngCol] = cls.Obstacle_cell
                obsmap[rngRow, rngCol] = cls.Obstacle_cell
                obstaclePlaced += 1

    def __init__(self):
        self.map, self.obstacleMap, cognitiveMap = self.makeDefaultWorld(
            Simulation.Obstacle_percentage)
        self.historyMap = self.map.copy()
        self.agent = Robot(Simulation.Standard_Init_Row, Simulation.Standard_Init_Column,
                           Simulation.Default_Heading, cognitiveMap)
        self.goal = Coord2D(Simulation.Standard_Goal_Row, Simulation.Standard_Goal_Column)

        self.colourMap: Dict = Simulation.makeColorGrid()

        self._figure = plt.figure()
        self.axis1 = self._figure.add_subplot(221)
        self.axis2 = self._figure.add_subplot(222)

    @classmethod
    def makeColorGrid(cls):
        # Creates the world the robot moves in.

        nbColourMap = Dict.empty(
            key_type=types.int32,
            value_type=types.int32[:],
        )

        colourMap = {
            cls.Open_cell: cls.Standard_Open_Color,
            cls.Robot_Start: cls.Standard_Robot_Color,
            cls.Obstacle_cell: cls.Standard_Obstacle_Colour,
            cls.Goal_cell: cls.Standard_Goal_Color,
            cls.Path_cell: cls.Standard_Path_Color,
            cls.Fog_cell: cls.Standard_FogofWar_Colour,
        }

        # colouring the grid
        for value in colourMap.keys():
            colour = colourMap[value]
            if isinstance(colour, str):
                if colour == 'white':
                    colour = np.array([255, 255, 255])
                elif colour == 'red':
                    colour = np.array([255, 0, 0])
                elif colour == 'black':
                    colour = np.array([0, 0, 0])
                elif colour == 'orange':
                    colour = np.array([255, 165, 0])
                elif colour == 'green':
                    colour = np.array([0, 255, 0])

            nbColourMap[value] = colour

        return nbColourMap

    @staticmethod
    @njit(parallel=True)
    def generateRGB(map, colourMap: Dict):
        worldmap = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.int32)
        for row in nb.prange(map.shape[0]):
            for column in range(map.shape[1]):
                value = int(map[row, column])
                colour = colourMap[value]
                worldmap[row, column, 0] = colour[0]
                worldmap[row, column, 1] = colour[1]
                worldmap[row, column, 2] = colour[2]
        return worldmap

    def getTile(self, tile):
        return self.obstacleMap[tile.row, tile.column]

    def isTileFree(self, tile):
        tile = self.getTile(tile)
        return (tile == Simulation.Open_cell or tile == Simulation.Robot_Start)

    def _getLineOfSightDirection(self, dir):
        tileValues = []
        dist = 0
        while True:
            dist += 1
            tile = self.agent.position + dir * dist
            tileValue = self.obstacleMap[tile.row, tile.column]
            tileValues.append(tileValue)
            if tileValue == Simulation.Obstacle_cell:
                break
            pass
        return tileValues

    def getLineOfSight(self, directionLeft, directionMid, directionRight):
        lefts = self._getLineOfSightDirection(directionLeft)
        mids = self._getLineOfSightDirection(directionMid)
        rights = self._getLineOfSightDirection(directionRight)
        return lefts, mids, rights

    def runSim(self):
        stepHistory = []

        turn = 0
        illegal_moves = 0
        while True:
            turn += 1
            logging.info(
                "Action number: {} Robot:{} DIR:{}".format(turn, self.agent.position,
                                                           self.agent.direction))
            oldPos = self.agent.position.copy()

            # use sensors
            directionLeft, directionMid, directionRight = self.agent.getVisionDir()
            lefts, mids, rights = self.getLineOfSight(directionLeft, directionMid, directionRight)
            self.agent.markLineOfSight(lefts, mids, rights)

            # decide and do new action
            action = self.agent.decideMove()
            stepHistory.append(action)
            self.agent.DoAction(action)

            if not self.isTileFree(self.agent.position):
                logging.info("Backed into a wall!")
                self.agent.backedIntoWall(oldPos)
                illegal_moves += 1

            # map updates
            newPos = self.agent.position
            self.map[oldPos.row, oldPos.column] = Simulation.Open_cell
            self.map[newPos.row, newPos.column] = Simulation.Robot_Start
            self.historyMap[oldPos.row, oldPos.column] = Simulation.Path_cell
            self.historyMap[newPos.row, newPos.column] = Simulation.Robot_Start

            # Check if we have reached the goal
            if self.agent.position == self.goal:
                logging.info("Turn: {} ({}) Robot coord:{}".format(turn, illegal_moves,
                                                                   self.agent.position))
                logging.info("Homecoming! I made it to the goal.")
                break

            # more map updates
            rgbMap = Simulation.generateRGB(self.historyMap.map, self.colourMap)
            rgbCog = Simulation.generateRGB(self.agent.cognitiveMap.map, self.colourMap)
            self.axis1.cla()
            self.axis1.imshow(rgbMap)
            self.axis2.cla()
            self.axis2.imshow(rgbCog)

            plt.pause(1e-7)
            logging.info("Number of actual moves: {}".format(turn - self.agent.move + 1))

        plt.show()

        # Pickle the file into an output file.
        # The path is stored as 'stepHistory' in memory
        with open('output.pkl', 'wb') as f:
            pickle.dump(stepHistory, f)


if __name__ == '__main__':
    # initiates a Simulation instance
    sim = Simulation()
    sim.runSim()
