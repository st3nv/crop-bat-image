import itertools
from collections import Counter
import math
import numpy as np


pntlist = [(147, 106), (131, 126), (130, 152), (171, 104), (129, 188), (206, 105)]

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        # As in if statement
        return min(360 - ang_deg, 180-(360-ang_deg))
    else: 

        return min(ang_deg, 180-ang_deg)

line1_maxang = ()
line2_maxang = ()

angmax = -1

for pnt4 in itertools.combinations(pntlist, r=4):
    # print("All 4 comb: ", pnt4)
    for line1 in itertools.combinations(pnt4, r=2):
        cpart = Counter(line1)
        call = Counter(pnt4)
        line2 = tuple(list((call-cpart).elements()))
        # print("Line 1: ", line1)
        # print("Line 2: ", line2)
        angtmp = ang(line1, line2)
        if angtmp > angmax:
            angmax = angtmp
            line1_maxang = line1
            line2_maxang = line2
        # print("Angle between: ", angtmp)
        # print("\n")

print("Max angle: ", angmax)
print("Line 1: ", line1_maxang)
print("Line 2: ", line2_maxang)

# numbers = [1,2,3,4,5,6,7,8,9]
# A = [2,5,6,9]

# c1 = Counter(numbers)
# c2 = Counter(A)

# diff = c1 - c2
# print(list(diff.elements()))