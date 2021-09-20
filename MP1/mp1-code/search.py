# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Rahul Kunji (rahulsk2@illinois.edu) on 01/16/2019

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod)(maze)

class Queue:
    def __init__(self):
        self.items = []
    
    def is_empty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop(-1)

    def bottom(self):
        return self.items[0]

    def top(self):
        return self.items[-1]

    def length(self):
        return len(self.items)

class Stack:
    def __init__(self):
        self.items = []
    
    def is_empty(self):
        return self.items == []

    def push(self, item):
        self.items.insert(-1,item)

    def pop(self):
        return self.items.pop(0)

    def bottom(self):
        return self.items[0]

    def top(self):
        return self.items[-1]

    def length(self):
        return len(self.items)

def bfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    
    # initialization
    possible_paths = Queue() 
    current_path = [] 
    path = [] 
    start_point = maze.getStart()
    points_visited = []

    # enqueue the start poiht
    possible_paths.enqueue([start_point])

    while(possible_paths.is_empty() == False):
        # get the current path to be explored
        current_path = possible_paths.dequeue()
        # get the current point
        current_point = current_path[-1]
        # continue if the current point has been visited
        if (current_point in points_visited): continue
        # check if the current point is the objective point
        if (maze.isObjective(current_point[0], current_point[1])):
            points_visited.append(current_point)
            path = current_path
            break
        # explore the neighbors of the current point
        for neighbor_point in maze.getNeighbors(current_point[0], current_point[1]):
            if (neighbor_point not in points_visited):
                points_visited.append(current_point)
                possible_paths.enqueue(current_path + [neighbor_point])

    return path, len(points_visited)

def greedy(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    
    # return value
    path = []
    explored = 0
    # the next point to be expended will be selected from here
    CurrQueue = []
    
    # we can see if the point have been visited from the map below
    dim = maze.getDimensions()
    visitMap = []
    for i in range(dim[0]):
        k = []
        for j in range(dim[1]):
            k.append([0,0])
        visitMap.append(k)
        
    ss = maze.getStart()
    startPoint = [ss[0], ss[1]]
    endPoints = maze.getObjectives()
    finished = []
    # all values in CurrQueue will be in form: [point position, distance to endpoint]
    start = [startPoint, (abs(startPoint[0] - endPoints[0][0]) + abs(startPoint[1] - endPoints[0][1]))]
    
    # insert start point into path
    visitMap[startPoint[0]][startPoint[1]] = startPoint
    #insert start point to CurrQueue
    CurrQueue.insert(0,start)
    # print(CurrQueue)
    # search for goal
    curr = start
    while(len(endPoints) != 0 and len(CurrQueue) != 0):
        explored += 1
        # put the first value in CurrQueue to path
        curr = CurrQueue.pop(0)
        CC = (curr[0][0], curr[0][1])
        if (CC in endPoints):
            endPoints.remove(CC)
            finished.append(CC)
            break
        # print(CC)
        # find all neighbors of curr, examine them and put them into CurrQueue
        news = maze.getNeighbors(curr[0][0], curr[0][1])
        for new in news:
            if (visitMap[new[0]][new[1]] == [0,0]):
                visitMap[new[0]][new[1]] = curr[0]
                newPoint = [[new[0], new[1]], (abs(new[0] - endPoints[0][0]) + abs(new[1] - endPoints[0][1]))]
                # print(newPoint[0], ",", newPoint[1])
                i = 0
                while (1):
                    if (i == len(CurrQueue)):
                        break
                    if(CurrQueue[i][1] >= newPoint[1]):
                        break
                    i += 1
                CurrQueue.insert(i, newPoint)
        
        # print(news)
        # print(path)
    CurrP = curr[0]
    while (CurrP != startPoint):
        path.insert(0, CurrP)
        CurrP = visitMap[CurrP[0]][CurrP[1]]
    path.insert(0, startPoint)
    # print(path)
    return path, explored

def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    
    # return value
    path = []
    explored = 0
    # the next point to be expended will be selected from here
    CurrQueue = []
    
    # we can see if the point have been visited from the map below
    dim = maze.getDimensions()
    visitMap = []
    for i in range(dim[0]):
        k = []
        for j in range(dim[1]):
            k.append([[0,0], 0])
        visitMap.append(k)
    ss = maze.getStart()
    endPoints = maze.getObjectives()
    allPoints = endPoints
    allPoints.insert(0, ss)
    distances = []
    for i in range(len(allPoints)):
        k = []
        for j in range(len(allPoints)):
            k.append([0, []])
        distances.append(k)
    # find all distance between points
    for i in range(len(allPoints)):
        for j in range(len(allPoints)):
            end = [allPoints[j][0], allPoints[j][1]]
            startPoint = [allPoints[i][0], allPoints[i][1]]
            if (distances[j][i] == [0, []]):
                CurrQueue = []
                for a in range(dim[0]):
                    for b in range(dim[1]):
                        visitMap[a][b] = [[0,0], 0]
            # all values in CurrQueue will be in form: [point position, path length, distance to endpoint]
                start = [startPoint, 0, abs(startPoint[0] - end[0]) + abs(startPoint[1] - end[1])]
                # insert start point into path
                visitMap[start[0][0]][start[0][1]] = [start[0], 0]
                #insert start point to CurrQueue
                CurrQueue.insert(0,start)
                
                # search for goal
                curr = start
                while(len(CurrQueue) != 0):
                    explored += 1
                    # put the first value in CurrQueue to path
                    curr = CurrQueue.pop(0)
                    if (curr[0] == end):
                        break
                    # find all neighbors of curr, examine them and put them into CurrQueue
                    news = maze.getNeighbors(curr[0][0], curr[0][1])
                    for new in news:
                        if (visitMap[new[0]][new[1]] == [[0,0], 0] or visitMap[new[0]][new[1]][1] > curr[1] + 1):
                            newPoint = [[new[0], new[1]], curr[1] + 1, abs(new[0] - end[0]) + abs(new[1] - end[1])]
                            visitMap[new[0]][new[1]] = [curr[0], newPoint[1]]
                            # print(newPoint[0], ",", newPoint[1], ',', newPoint[3])
                            k = 0
                            while (1):
                                if (k == len(CurrQueue)):
                                    break
                                if(CurrQueue[k][1] + CurrQueue[k][2] > newPoint[1] + newPoint[2]):
                                    break
                                k += 1
                            CurrQueue.insert(k, newPoint)
                    
            # print(news)
            # print(path)
                currPath = []   
                CurrP = curr[0]
                while (CurrP != startPoint):
                    currPath.insert(0, CurrP)
                    CurrP = visitMap[CurrP[0]][CurrP[1]][0]
                distances[i][j] = [len(currPath), currPath]
            else:
                currPath = []
                for k in range(len(distances[j][i][1]) - 1):
                    currPath.insert(0, distances[j][i][1][k])
                currPath.append(end)
                distances[i][j] = [len(currPath), currPath]
                
    pathStart = [[0], 0, MSTCal(allPoints, [], distances)]
    pathQueue = []
    pathQueue.append(pathStart)
    currPath = pathStart
    print(len(allPoints))
    while (len(currPath[0]) != len(allPoints)):
        print(len(currPath))
        currPath = pathQueue.pop(0)
        for i in range(len(allPoints)):
            if i not in currPath[0]:
                nextPath = []
                nextPath.extend(currPath[0])
                nextPath.append(i)
                nextDis = currPath[1] + distances[currPath[0][len(currPath[0]) - 1]][i][0]
                
                toDelete = []
                ifAdd = 1
                for prePath in pathQueue:
                    if (prePath[0][len(prePath[0]) - 1] == i and len(prePath[0]) == len(nextPath)):
                        ifEqual = 1
                        for point in prePath[0]:
                            if (not(point in nextPath)):
                                ifEqual = 0
                                break
                        if (ifEqual):
                            if (prePath[1] > nextDis):
                                toDelete.append(prePath)
                            else:
                                ifAdd = 0
                for deletePoint in toDelete:
                    pathQueue.remove(deletePoint)
                if (ifAdd):
                    pathMST = MSTCal(allPoints, currPath[0], distances)
                    k = 0
                    while (1):
                        if (k == len(pathQueue)):
                            break
                        if nextDis + pathMST < pathQueue[k][1] + pathQueue[k][2]:
                            break
                        k += 1
                    pathQueue.insert(k, [nextPath, nextDis, pathMST])
                    
        
    minPath = currPath[0]
    for i in range(len(allPoints) - 1):
        path.extend(distances[minPath[i]][minPath[i+1]][1])
    path.insert(0, ss)
    print(path)
    return path, explored

def MSTCal(points, path, distances):
    vertices = []
    edges = []
    idx = []
    if (len(points)-len(path)<2):
        return 0
    for i in range(len(points)):
        if (not(i in path)):
            vertices.append(i)
    for i in range(len(points)):
        idx.append(-1)
    for i in range(len(vertices)):
        for j in range(len(vertices) - i - 1):
            k = 0
            while (1):
                if (k == len(edges)):
                    break
                if distances[vertices[i]][vertices[i+j+1]][0] < edges[k][1][0]:
                    break
                k += 1
            edges.insert(k, [[vertices[i], vertices[i+j+1]], distances[vertices[i]][vertices[i+j+1]]])
    remains = len(idx)
    
    MST = 0
    while (remains != len(path) + 1):
        currE = edges.pop(0)
        while(root(currE[0][0], idx) == root(currE[0][1], idx)):
            currE = edges.pop(0)
        idx[root(currE[0][1], idx)] = currE[0][0]
        MST += currE[1][0]
        remains = 0
        for i in range(len(idx)):
            if idx[i] == -1:
                remains += 1
    return MST    
    
def root(point, idx):
    re = point
    while(idx[re] != -1):
        re = idx[re]
    return re
    
def dfs(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    # initialization
    possible_paths = Stack() 
    current_path = [] 
    path = [] 
    shortest_path_len = float('inf')
    start_point = maze.getStart()
    points_visited = []

    # enqueue the start poiht
    possible_paths.push([start_point])

    while(possible_paths.is_empty() == False):
        # get the current path to be explored
        current_path = possible_paths.pop()
        # get the current path length
        current_path_len = len(current_path)
        # get the current point
        current_point = current_path[-1]
        # continue if the current point has been visited
        if (current_point in points_visited): continue
        # mark the current point as visited
        points_visited.append(current_point)
        # check if the current point is the objective point
        if (maze.isObjective(current_point[0], current_point[1])):
            # check if the current path is the shortest path
            if (current_path_len < shortest_path_len):
                shortest_path_len = len(current_path)
                path = current_path
            continue
        if (current_path_len >= shortest_path_len): continue
        # explore the neighbors of the current point
        for neighbor_point in maze.getNeighbors(current_point[0], current_point[1]):
            if (neighbor_point not in points_visited):
                possible_paths.push(current_path + [neighbor_point])

    return path, len(points_visited)
