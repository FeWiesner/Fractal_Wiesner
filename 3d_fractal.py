import csv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

#cambiar archivo
network = ['HANO_','BALE_']
side =['d', 'i', 'k']

for net in network:
    for s in side:
                
        X = []
        Y = []
        Z = []

        with open(net+s+'.txt') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if net == 'HANO_':
                    a = 100
                    b = -4
                    fine = 16j
                else: 
                    a = 1
                    b = -5.5
                    fine = 90j
                X.append(float(row[0])/a)
                Y.append(float(row[1])/a)
                Z.append(float(row[2]))

        xmax, ymax, zmax = max(X), max(Y), max(Z)
        xmin, ymin, zmin = min(X), min(Y), min(Z)

        maxi = (xmax, ymax, zmax)
        mini = (xmin, ymin, zmin)

        from scipy.interpolate import griddata

        points = []
        for i in range(len(X)):
            points.append([X[i], Y[i]])
        points = np.array(points)
        Z = np.array(Z)


        grid_x, grid_y = np.mgrid[xmin:xmax:fine, ymin:ymax:fine]
        grid = griddata(points, Z, (grid_x, grid_y), method='cubic')

        grid = np.nan_to_num(grid, nan = -1)

        vectors = []
        r = int(str(fine)[:-1])
        for i in range(r):
            for j in range(r):
                if grid[i][j] != -1:
                    punto = [grid_x[i][0], grid_y[0][j], grid[i][j]]
                    vectors.append(punto)

        print(len(vectors))

        '''
        ax = plt.axes(projection = '3d')
        for i in vectors:
            ax.scatter3D(i[0], i[1], i[2], color = 'red')
        plt.show()


        plt.imshow(grid.T, extent=(xmin, xmax, ymin, ymax), origin='lower')
        plt.title('Cubic')
        plt.show()
        '''


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


        import scipy.stats

        for i in range(1,100):
            ans = number(i,1)
            boxcount.append(ans[0])
            boxsize.append(1/ans[1])
            if i > r/2:
                bc = np.array(boxcount)
                bs = np.array(boxsize)
                lg_bc = np.log(bc)
                lg_bs = np.log(bs)
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(lg_bs,lg_bc)
                if r_value > 0.99:
                    break

        fig, ax = plt.subplots()
        ax.scatter(lg_bs, lg_bc)
        ax.set_xlabel('log(1/e)')
        ax.set_ylabel('log(N(e)')
        ax.set_title('Box Dimension Regression')
        ax.text(b,1, 'log(N(e)) = '+str(round(intercept,2))+' + ' + str(round(slope,2)) + 'log(1/e)')
        ax.plot([min(lg_bs),max(lg_bs)], [intercept + slope * min(lg_bs), intercept + slope * max(lg_bs)], color="k", lw=2.5)
        plt.savefig(net+s+'.png')

        with open('results.txt', 'a', encoding = 'utf-8') as f:
            f.write(str(net+s)+'\n')
            f.write(str(r_value)+'\n')
            f.write('Dim Fractal = '+ str(slope)+ '\n\n')
