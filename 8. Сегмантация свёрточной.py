from PIL import Image
from math import ceil, sqrt
from math import floor
import matplotlib.pyplot as plt
import scipy
import numpy as np

# https://habrahabr.ru/post/142818/
def checkByte(a):
    if a > 255:
        a = 255
    if a < 0:
        a = 0
    return a


def svertka(a, b):
    sum = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            sum += a[i][j] * b[i][j]
    return sum
#   return (a*b).sum
def median(a):
    c = []
    for i in range(len(a)):
        for j in range(len(a[0])):
            c.append(a[i][j])
    c.sort()
    return c[ceil(len(c) / 2)]
def max(a):
    c = []
    for i in range(len(a)):
        for j in range(len(a[0])):
            c.append(a[i][j])
    c.sort()
    return c[len(c) - 1]
def min(a):
    c = []
    for i in range(len(a)):
        for j in range(len(a[0])):
            c.append(a[i][j])
    c.sort()
    return c[0]


im = Image.open("D:\Programing\Projects\DeepLearningSchool\lenna.jpg")
plt.imshow(im)
plt.show()
pixels = im.load()


imFinal = im.copy()
pixels2 = imFinal.load()

filter = [
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
]


div = np.array(filter).sum()
print (div)
if div == 0:
    div = 1


for i in range(floor(len(filter)/2), im.width - floor(len(filter)/2)):
    for j in range(floor(len(filter)/2), im.height - floor(len(filter)/2)):
        matrR = []
        matrG = []
        matrB = []
        for n in range(-floor(len(filter)/2), ceil(len(filter)/2)):
            rowR = []
            rowG = []
            rowB = []
            for m in range(-floor(len(filter)/2), ceil(len(filter)/2)):
                r, g, b = pixels[i + n, j + m]
                rowR.append(r)
                rowG.append(g)
                rowB.append(b)
            matrR.append(rowR)
            matrG.append(rowG)
            matrB.append(rowB)

        r = checkByte(round(svertka(matrR, filter) / div))
        g = checkByte(round(svertka(matrG, filter) / div))
        b = checkByte(round(svertka(matrB, filter) / div))

       # r = checkByte(min(matrR))
       # g = checkByte(min(matrG))
       # b = checkByte(min(matrB))
        '''
        if r < 512:
            pixels2[i, j] = (255, 255, 255)
        else:
            pixels2[i, j] = (0, 0, 0)'''
        pixels2[i, j] = (r, g, b)
print("Hello")

plt.imshow(imFinal)
plt.show()