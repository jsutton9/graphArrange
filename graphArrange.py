import random
import math
import time
import Image
from multiprocessing import Process, Queue

pi = math.pi
red = 0x0000ff
green = 0x00ff00
blue = 0xff0000
hred = 0x000080
hgreen = 0x008000
hblue = 0x800000
colors = [red, red+hgreen, red+green, green, green+blue, 
        hgreen+blue, blue, blue+hred, blue+red, 
        hred+hgreen+hblue, 0]

def randomGraph(n):
    ret = [[0 for _ in xrange(n)] for _ in xrange(n)]
    for i in xrange(n):
        row = []
        for j in xrange(i+1, n):
            c = random.expovariate(1.0)
            ret[i][j] = c
            ret[j][i] = c
    return ret

def randomPositions(n, w=1.0):
    return [(w*random.random()-w/2.0, \
            w*random.random()-w/2.0) for _ in xrange(n)]

def getPartialFitness(f):
    def partialFitness(graph, positions, i, pos):
        x0, y0 = pos
        score = 0.0
        n = len(graph)
        for j in xrange(n):
            if i != j:
                x1, y1 = positions[j]
                r2 = (x1-x0)**2 + (y1-y0)**2
                score += f(r2, graph[i][j])
        return score
    return partialFitness

def getFitness(f):
    def fitness(graph, positions):
        n = len(graph)
        score = 0.0
        for i in xrange(n):
            for j in xrange(i+1, n):
                x0, y0 = positions[i]
                x1, y1 = positions[j]
                r2 = (x1-x0)**2 + (y1-y0)**2
                score += f(r2, graph[i][j])
        return score
    return fitness

def randomShift(pos, sigma):
    theta = pi*random.random()
    r = random.gauss(0, sigma)
    x = pos[0] + r*math.cos(theta)
    y = pos[1] + r*math.sin(theta)
    return (x, y)

def anneal(graph, positions, steps, ti, tf, fit):
    r = (tf/ti)**(1.0/steps)
    t = ti
    n = len(graph)
    partialFitnesses = [fit(graph, positions, i, \
        positions[i]) for i in xrange(n)]
    for _ in xrange(steps):
        i = random.randint(0, n-1)
        newPos = randomShift(positions[i], t)
        f = fit(graph, positions, i, newPos)
        if f < partialFitnesses[i]:
            positions[i] = newPos
            partialFitnesses[i] = f
        t *= r
    return positions

def multiAnneal(graph, positions, steps, ti, tf, fit):
    r = (tf/ti)**(1.0/steps)
    t = ti
    n = len(graph)
    score = fit(graph, positions)
    for _ in xrange(steps):
        newPositions = []
        for pos in positions:
            newPositions.append(randomShift(pos, t))
        newScore = fit(graph, newPositions)
        if newScore < score:
            positions = newPositions
            score = newScore
        t *= r
    return positions

def printState(graph, positions, partialFitness):
    for pos in positions:
        print pos
    print ""
    partialFitnesses = []
    for i in xrange(len(graph)):
        partialFitnesses.append(partialFitness(graph, 
            positions, i, positions[i]))
    for f in partialFitnesses:
        print f
    print ""
    print sum(partialFitnesses)/2
    print ""

def drawGraph(positions, colors):
    im = Image.new("RGB", (256, 256), "white")
    minX = positions[0][0]
    maxX = minX
    minY = positions[0][1]
    maxY = minY
    for pos in positions:
        if pos[0] < minX:
            minX = pos[0]
        if pos[0] > maxX:
            maxX = pos[0]
        if pos[1] < minY:
            minY = pos[1]
        if pos[1] > maxY:
            maxY = pos[1]
    width = maxX - minX
    height = maxY - minY
    if width < height:
        minX -= (height-width)/2
        maxX += (height-width)/2
    else:
        minY -= (width-height)/2
        maxY += (width-height)/2
    size = max(height, width)
    for i in xrange(len(positions)):
        pos = positions[i]
        x = int(250*(pos[0]-minX)/size + 3)
        y = int(250*(pos[1]-minY)/size + 3)
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                im.putpixel((x+j, y+k), colors[i])
    return im

def runTrial(graph, annealFunc, resultQueue):
    resultQueue.put(annealFunc(graph, 
        randomPositions(len(graph))))

# make graph
n = 40
'''
graph = [[1, 20, 1, 1, 1],
         [20, 1, 20, 1, 1],
         [1, 20, 1, 20, 1],
         [1, 1, 20, 1, 20],
         [1, 1, 1, 20, 1]]
graph = [[ 0, 20, 20,  1,  1,  1,  1],
         [20,  0, 20,  1,  1,  1,  1],
         [20, 20,  0,  1,  1,  1,  1],
         [ 1,  1,  1,  0, 20,  1,  1],
         [ 1,  1,  1, 20,  0, 20,  1],
         [ 1,  1,  1,  1, 20,  0, 20],
         [ 1,  1,  1,  1,  1, 20,  0]]
graph = randomGraph(20)
'''

graph = []
for i in xrange(n):
    graph.append([1]*n)
    graph[i][i-1] = 100
    graph[i][i] = 0
    graph[i][(i+1)%n] = 100

# set up fitness functions
f0 = lambda r2, edge: edge*r2 - r2**.5
f1 = lambda r2, edge: edge*2**r2 - 2**(-r2)
f2 = lambda r2, edge: edge*r2**.5 + 1.0/r2**.25
fitness0 = getFitness(f0)
fitness1 = getFitness(f1)
fitness2 = getFitness(f2)
fitnesses = [fitness0, fitness1, fitness2]

fitness = fitness0
partialFitness = getPartialFitness(f0)

# set up anneal functions
anneal0 = lambda g, p: multiAnneal(g, p, 
        20000, 1.0, .0003, fitness)
anneal1 = lambda g, p: multiAnneal(g, p,
        100000, 1.0, .0003, fitness)
annealFunc = anneal0

# run trials
trials = 6
procs = []
q = Queue()
time0 = time.time()
for _ in xrange(trials):
    p = Process(target=runTrial, 
            args=(graph, annealFunc, q))
    p.start()
    procs.append(p)
for p in procs:
    p.join()
time1 = time.time()
p = [q.get() for _ in xrange(trials)]

scores = [fitness(graph, p[i]) for i in xrange(trials)]
print "score: " + str(sum(scores)/trials)
print "time: %d:%05.2f" % (int((time1-time0)/60), \
        (time1-time0)%60)

# draw and save graphs
for i in xrange(trials):
    im = drawGraph(p[i], colors+[0]*n)
    im.save("graphs/graph"+str(i)+".bmp")
