import csv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.transforms import Bbox

#cambiar archivo
archivo = 'BALE_d.txt'

X = []
Y = []
Z = []

with open(archivo) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        X.append(float(row[0]))
        Y.append(float(row[1]))
        Z.append(float(row[2]))

xmax, ymax, zmax = max(X), max(Y), max(Z)
xmin, ymin, zmin = min(X), min(Y), min(Z)

maxi = (xmax, ymax, zmax)
mini = (xmin, ymin, zmin)

vectors = []
for i in range(len(X)):
    vectors.append([ X[i], Y[i], Z[i] ])

vectors.sort()


ax = plt.axes(projection = '3d')
for i in vectors:
    ax.scatter3D(i[0], i[1], i[2], color = 'red')
plt.show()

def count_xrow(xinf, xsup, yinf, ysup, zinf, zsup, vec, eps):
    cuenta = 0
    while xsup <= mini[0]+length+10:
        #for a single box
        for v in vec:
            if v[0]>=xinf and v[0]<= xsup:
                if v[1]>=yinf and v[1]<= ysup:
                    if v[2]>=zinf and v[2]<= zsup:
                        cuenta += 1
                        vec.remove(v)
                        break
        #to the next box on the same x-row
        xinf = xsup
        xsup += eps
    return cuenta


length = max(xmax-xmin, ymax-ymin, zmax-zmin)

eps = length
boxsize = []
boxcount = []

def number(f,n):
    eps = length/(f**n)
    count = 0
    vecs = vectors.copy()
    xinf = mini[0]-1
    yinf = mini[1]-1
    zinf = mini[2]-1

    xsup = xinf + eps
    ysup = yinf + eps
    zsup = zinf + eps

    while zsup <= mini[2]+length+10:
        
        while ysup <= mini[1]+length+10:
            count += count_xrow(xinf, xsup, yinf, ysup, zinf, zsup, vecs, eps)
            xinf = mini[0]-1
            xsup = xinf + eps
            yinf = ysup
            ysup = yinf + eps
        yinf = mini[1]-1
        ysup = yinf + eps
        zinf = zsup
        zsup = zinf + eps

    return(count,eps)

for i in range(6,-1,-1):
    ans = number(2,i)
    boxcount.append(ans[0])
    boxsize.append(1/ans[1])

for i in range(4,-1,-1):
    ans = number(3,i)
    boxcount.append(ans[0])
    boxsize.append(1/ans[1])
for i in range(2,-1,-1):
    ans = number(5,i)
    boxcount.append(ans[0])
    boxsize.append(1/ans[1])

bc = np.array(boxcount)
bs = np.array(boxsize)

lg_bc = np.log(bc)
lg_bs = np.log(bs)


myfitting = np.polyfit(lg_bs,lg_bc, deg=1)

from sklearn.metrics import r2_score
coeff = myfitting[0]

predict = np.poly1d(coeff)
R2 = r2_score(y_values, predict(x_values))
print(R2)


b, a = np.polyfit(lg_bs,lg_bc, deg=1)

plt.scatter(lg_bs, lg_bc)
xseq = np.linspace(-10, 1, num=100)
plt.plot(xseq, a + b * xseq, color="k", lw=2.5)
plt.show()

print('Dim Fractal = '+ str(b))
