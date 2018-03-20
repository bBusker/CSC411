import math 
def mutualInformation(py, py_x, px):
    hy = -(py * math.log(py,2)) - ((1-py) * math.log((1-py), 2))
    hyx = []
    for p in py_x:
        hyx.append(-(p * math.log(p,2)) - ((1-p) * math.log((1-p), 2)))
    return hy - (px * hyx[0]) - (1-px)*hyx[1]

print mutualInformation(909./2287, [662./1934, 247./353], 1934./2287)
print mutualInformation(662./1934, [556./1821, 96./113], 1821./1934)
