import time
import heapq
import sys
from collections import deque

MOVES = (('u', -1, 0),
         ('d', 1, 0),
         ('l', 0, -1),
         ('r', 0, 1))
DIRS = {char: (d_row, d_col) for char, d_row, d_col in MOVES}   #made the moves into a dict for path rebuild

class Solver:
    def __init__(self, width, height, walls, goals):
        self.width = width
        self.height = height
        self.walls = walls
        self.goals = goals
        self.dead = set()
        self.minDist = {}
        self.goalDist = []          #distance table per goal
        self.hCache = {}            #heuristic cache, per state (to make sure same states do not get their h recalculated)
        self.buildDistanceTable()
        self.findDeadSquares()

    #states are defined per push of a crate
    def successors(self, player, crates):
        result = []                                         #list of all possible new states
        dist = self.movableRegion(player, crates)           #call the method to get the region player can move to

        for crate in crates:
            for char, d_row, d_col in MOVES:
                standing = (crate[0] - d_row, crate[1] - d_col)     #where the player has to be standing to make the push
                pushed = (crate[0] + d_row, crate[1] + d_col)       #where the crate will be pushed to
                
                if standing not in dist:                            #make sure player can reach the position
                    continue
                if not self.isValidCell(pushed):                    #make sure the push crate is a valid cell
                    continue    
                if pushed in crates:                                #make sure the pushed crate does not bump into another crate
                    continue
                
                newCrates = (crates - {crate}) | {pushed}           #new set of crates after the push
                if not self.isSafePush(pushed, newCrates):          #make sure the push does not end in a deadlock
                    continue

                newState = (crate, newCrates)                       #the player will be in the old position of the crate
                cost = dist[standing] + 1                           #cost is the walk towards there + 1 for the push
                result.append((char, newState, cost))               #char for the direction input, newState and cost self-explanatory

        return result

    def manhattanH(self, crates):
        total = 0

        for crate in crates:
            best = float('inf')
            for goal in self.goals:
                dist = abs(crate[0] - goal[0]) + abs(crate[1] - goal[1])
                if dist < best:
                    best = dist

            total += best

        return total

    def bfsFromGoal(self, goal):
        dist = {goal: 0}
        queue = deque([goal])

        while queue:
            cell = queue.popleft()
            for _, d_row, d_col in MOVES:
                newCell = (cell[0] - d_row, cell[1] - d_col)
                playerCell = (cell[0] - 2 * d_row, cell[1] - 2 * d_col)

                if not (self.isValidCell(newCell) and self.isValidCell(playerCell)):
                    continue
                if newCell in dist:
                    continue

                dist[newCell] = dist[cell] + 1
                queue.append(newCell)

        return dist

    def aStar(self, startState):
        startCrates = startState[1]

        startH = self.heuristic(startCrates)

        if self.isGoal(startCrates):
            return ""
        if startH == float('inf'):
            return ""

        counter = 0

        frontier = []
        heapq.heappush(frontier, (startH, counter, 0, startState))

        bestG = {startState: 0}
        parent = {startState: None}
        expanded = 0

        while frontier:
            f, _, g, state = heapq.heappop(frontier)
            if g > bestG[state]:
                continue

            expanded += 1

            if self.isGoal(state[1]):
                print("expanded", expanded, file=sys.stderr)
                return self.rebuildPath(parent, state)

            player, crates = state

            for moveChar, child, cost in self.successors(player, crates):
                newG = g + cost
                if not newG < bestG.get(child, float('inf')):
                    continue

                h = self.heuristic(child[1])
                if h == float('inf'):
                    continue

                bestG[child] = newG
                parent[child] = (state, moveChar)

                counter += 1

                heapq.heappush(frontier, (newG + h, counter, newG, child))

        return ""

    #heuristic that gets all the mindist of each crate to a goal and adds them
    def heuristicSimple(self, crates):
        total = 0
        for crate in crates:
            d = self.minDist.get(crate)

            if d == None:
                return float('inf')

            total += d

        return total

    def heuristic(self, crates):
        cached = self.hCache.get(crates)        #try to check if the state already has a calculated heuristic
        if cached is not None:                  #if there is, return corresponding heuristic
            return cached
        cost = self.getStateHCost(crates)       #else run the method to get the heuristic
        self.hCache[crates] = cost              #and store it to the hCache for possible future use
        return cost 


    #dist given, one to one mapping of crate to a goal, as a heuristic. tighter bound but still admissible since it still underestimates the real cost
    def getStateHCost(self, crates):
        INF = float('inf')
        rowList = []

        for crate in crates:
            row = [table.get(crate, INF) for table in self.goalDist]    #turn the tableDist of each crate into a list

            if min(row) == INF:                                         #if the minimum is inf, means no goal is reachable for crate
                return INF
            rowList.append(row)                                         #add to the list of rows
        
        #easy case; assuming that each crate has a different goal
        total = 0                                   #total distance between all crates and goals
        picked = set()                              #set for each picked goal
        distinct = True                             #we assume each crate has a different goal
        for row in rowList:
            best = min(row)                         #get the min dist
            j = row.index(best)                     #get the index 
            if j in picked:                         #if the index already picked then, the crates share the same min goal
                distinct = False
                break                               
            picked.add(j)                          
            total += best

        if distinct:                               #only return the total if every crate has its own goal
            return total


        #if crates have the same goal; we use bitmask to get the combination of row indices that gives minimum heuristic value
        bitmask = {0:0}                                 #no goals taken yet and cost 0
        for row in rowList:                             #get the disttable of every crate
            newBitmask = {}                             #create a temp bitmask for each crate
            for mask, cost in bitmask.items():          #every partial bitmask made so far
                for j in range(len(row)):               #try every goal of this crate
                    if row[j] == INF:                   #means the crate cant reach goal[j]
                        continue
                    if (mask >> j) & 1:                 #shift the mask right by j so bit j is lowest, then bitwise and with 1 to test whether goal j is taken
                        continue                        #if both are 1, then that means goal[j] is already taken
                    newMask = mask | (1 << j)           #means to shift 1 to the left by j and do a bitwise or to mark goal[j] as taken
                    newCost = cost + row[j]             #add cost of the partial bitmask with the added cost of the new marked goal
                    if newCost < newBitmask.get(newMask, INF):          #compare if the new cost is lower compared to old combination or if its new
                        newBitmask[newMask] = newCost
            bitmask = newBitmask                        #replace the partial bitmask with the new combinations made
            if not bitmask:                             #means nothing valid is left
                return INF
        
        return min(bitmask.values())    #return the heuristic cost of the combination that yields the least heuristic (meaning minDist but no overlap of goals)


    def isValidCell(self, cell):
        row, col = cell
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        if cell in self.walls:
            return False

        return True

    def isCorner(self, cell):
        row, col = cell
        vertCheck = False
        horiCheck = False

        if (row - 1, col) in self.walls or (row + 1, col) in self.walls:
            vertCheck = True
        if (row, col - 1) in self.walls or (row, col + 1) in self.walls:
            horiCheck = True

        if vertCheck and horiCheck:
            return True
        else:
            return False

    def isGoal(self, crates):
        return crates <= self.goals

    def isFrozenBlock(self, pos, crates):
        row, col = pos
        walls = self.walls
        goals = self.goals

        for d_row in (-1, 1):
            for d_col in (-1, 1):   
                vert = (row + d_row, col)                               #to check above/below
                hori = (row, col + d_col)                               #to check left/right
                diag = (row + d_row, col + d_col)                       #to check diagonally

                if vert not in crates and vert not in walls:
                    continue
                if hori not in crates and hori not in walls:
                    continue
                if diag not in crates and diag not in walls:
                    continue

                if pos not in goals:                                    #if everything is blocked and pos is not in a goal, instantly means a deadlock
                    return True
                
                #changed the for loop check into if statement
                if (vert in crates and vert not in goals) or \
                   (hori in crates and hori not in goals) or \
                   (diag in crates and diag not in goals):
                    return True
                
        return False                                                    #else its not in a deadlock

    def isSafePush(self, dest, newCrates):
        if dest in self.dead:
            return False
        if self.isFrozenBlock(dest, newCrates):
            return False

        return True

    def buildDistanceTable(self):
            for goal in self.goals:
                table = self.bfsFromGoal(goal)
                self.goalDist.append(table)             #save the distance table of each goal
                for cell, d in table.items():
                    self.minDist[cell] = min(d, self.minDist.get(cell, float('inf')))

    #since the states now are per push, the trail we follow are just the edges and need to find the actual input of the path
    def rebuildPath(self, parent, goalState):
        edges = []
        state = goalState

        while parent[state] is not None:                                #gets the edge trail
            previousState, moveChar = parent[state]
            edges.append((previousState, moveChar, state))              #all the necessary details we need to rebuild path
            state = previousState

        edges.reverse()

        path = []
        for previousState, moveChar, state in edges:                    
            prevPlayer, prevCrates = previousState
            newPlayer = state[0]
            d_row, d_col = DIRS[moveChar]                               #tells us how the player got to the new position
            standing = (newPlayer[0] - d_row, newPlayer[1] - d_col)     #recalculate where the player stands before push based on moveChar
            walk = self.findPath(prevPlayer, standing, prevCrates)      #use find path to find the path from prevEdge to right before the push
            path.append(walk)                                           #append the entire walk to the path
            path.append(moveChar)                                       #then append the moveChar since thats the push that leads to the next state
        return ''.join(path)

        

    def findDeadSquares(self):
        for row in range(self.height):
                for col in range(self.width):
                    cell = (row, col)

                    if cell not in self.walls and cell not in self.minDist:
                        self.dead.add(cell)

    #added a bfs from player position to get all possible position player can reach without obstruction
    def movableRegion(self, player, crates):
        dist = {player: 0}                                      #dict of a cell tuple corresponding to the distance
        queue = deque([player])                                 #bfs queue 
        walls = self.walls                                      #walls

        while queue:
            cell = queue.popleft()

            #iterates the moves instead for speed
            #check all directions from the cell
            for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                newCell = (cell[0] + d_row, cell[1] + d_col)
                if newCell in crates or newCell in dist:        #make sure its not blocked by crates or already explored
                    continue
                #make sure its a valid cell
                if not (0 <= newCell[0] < self.height and 0 <= newCell[1] < self.width):
                    continue
                if newCell in walls:                            #make sure its not blocked by walls
                    continue

                dist[newCell] = dist[cell] + 1                  #distance of newCell is always +1 from its starting cell
                queue.append(newCell)                           #add to the queue

        return dist

    #used to find the path going from one state to another
    def findPath(self, player, target, crates):
        if target == player:                                    #return empty string when player already standing in the correct position to push
            return ""

        parent = {player: None}
        queue = deque([player])

        while queue:                                            #same logic as movableRegion                            
            cell = queue.popleft()
            for char, d_row, d_col in MOVES:
                newCell = (cell[0] + d_row, cell[1] + d_col)

                if not self.isValidCell(newCell):
                    continue
                if newCell in crates or newCell in parent:
                    continue

                parent[newCell] = (cell, char)                  #adds the previous cell postion and move used to get there    
                queue.append(newCell)

                if newCell == target:                           #if we already reach the target, do the same logic as old rebuild path
                    moves = []
                    state = target
                    while parent[state] is not None:            #keep following the previous postion until we reach the starting position
                        previousState, moveChar = parent[state]
                        moves.append(moveChar)
                        state = previousState
                    moves.reverse()                             #reverse the list
                    return ''.join(moves)                       #return string of moves

        return None                                             #happens when target is unreachable; wont really happen but just a safety net



class SokoBot:
    def solveSokobanPuzzle(self, width, height, mapData, itemsData):
        # YOU NEED TO REWRITE THE IMPLEMENTATION OF THIS METHOD TO MAKE THE BOT SMARTER
        # Default stupid behavior: Think (sleep) for 3 seconds, and then return a
        # sequence
        # that just moves left and right repeatedly.
        # try:
        #     time.sleep(3)
        # except Exception as ex:
        #     print(ex)
        # return "lrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlrlr"

        walls = set()
        goals = set()
        crates = set()
        player = None

        for i in range(height):
            for j in range(width):
                if mapData[i][j] == '#':
                    walls.add((i, j))
                elif mapData[i][j] == '.':
                    goals.add((i, j))
                if itemsData[i][j] == '$':
                    crates.add((i, j))
                elif itemsData[i][j] == '@':
                    player = ((i, j))

        solver = Solver(width, height, walls, goals)
        startState = (player, frozenset(crates))

        return solver.aStar(startState)
