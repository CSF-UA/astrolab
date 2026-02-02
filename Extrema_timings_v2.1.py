import matplotlib.pyplot as plt
import scipy.optimize as spo
import matplotlib
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import glob



def get_data(name):
    
    # Межі зоряної величини, не чіпаєте, залишаєте, як є
    mmin = -10000
    mmax = 10000
    
    f = open(name, 'r')
    S = f.readlines()
    f.close()
    JD, mag = [], []
    for i in range(len(S)):
        a = S[i].split()
        m = float(a[1])
        if m >= mmin and m <= mmax:
            JD.append(float(a[0]))
            mag.append(m)
    JD, mag = np.array(JD), np.array(mag)
    return JD, mag



def get_intervals(filename):
    # FILE WITH INTERVALS SHOULD CONTAIN ONLY NUMBERS OF POINTS FOR START AND FINISH
    f = open(filename, 'r')
    S = f.readlines()
    f.close()
    start, finish = [], []
    for i in range(len(S)):
        a = S[i].split()
        start.append(int(a[0]))
        finish.append(int(a[1]))
    return start, finish



def JUST_DO_IT(x, y, start, finish):
    E = [] # E = moments of extrema
    T = [] # T = types of extrema
    for n in range(len(start)):
        xx = x[start[n]:finish[n]]
        yy = y[start[n]:finish[n]]
        extr, tp = approximate(xx, yy, n)
        E.append(extr)
        T.append(tp)
    return E, T



def poly(x, pp):
    # Y = PP[0] + PP[1]*X + PP[2]*X^2 + ...
    y = np.zeros(len(x))
    for i in range(len(pp)):
        y += pp[i] * x**i
    return y



def approximate(x, y, AAA):
    T0 = 'ERROR'
    tp = 'none'
    nn = range(3, 11)
    s = list(np.zeros(len(nn)))
    coef = []
    x_av = np.average(x)
    x = x - x_av  # work around the mean

    # making approximations with various n
    for n in nn:
        p0 = np.zeros(n)
        par = spo.leastsq(lambda pp: y - poly(x, pp), p0, full_output=1)[0]
        s[n - 3] = np.sum((y - poly(x, par))**2)
        coef.append(par)

    # best approximation
    N = s.index(min(s))
    par = coef[N]

    # generate array to calculate T0
    xx = np.linspace(np.min(x), np.max(x), 10000)
    yy = poly(xx, par)
    yy_list = list(yy)

    # get position of highest and lowest points
    nmin = yy_list.index(min(yy_list))
    nmax = yy_list.index(max(yy_list))

    # whatever is closer to the center
    diff_min = abs(xx[nmin])
    diff_max = abs(xx[nmax])
    if diff_min < diff_max:
        T0 = xx[nmin] + x_av
        tp = 'min'
    if diff_max < diff_min:
        T0 = xx[nmax] + x_av
        tp = 'max'

    # visualization(x, y, coef, x_av, T0, AAA)
    return T0, tp



def visualization(x, y, coef, x_av, T0, n):
    fig = plt.figure(n)
    fig.set_size_inches(12, 8)
    plt.xlabel('time, JD - 2 4570 000', fontsize=14)
    plt.ylabel('magnitude, mmag', fontsize=14)
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)

    plt.plot(x + x_av, y, '.k', markersize=5)

    for i in range(len(coef)):
        par = coef[i]
        xx = np.linspace(np.min(x), np.max(x), 10000)
        yy = poly(xx, par)
        plt.plot(xx + x_av, yy, label='n=' + str(i + 3), linewidth=2)

    if T0 != 'ERROR':
        plt.plot([T0, T0], [np.min(y), np.max(y)], '-r', linewidth=3, label='extremum')
    plt.legend(loc='best', fontsize=16)
    
    plt.savefig(str(n)+'.png', dpi=150)
    plt.close()



def write_it_down(E, T, directory):
    r = ''
    for i in range(len(T)):
        r += str(E[i]) + ' ' + str(T[i]) + '\n'
    f = open(directory + "RESULTS.txt", 'w')
    f.writelines(r)
    f.close()



# Обираємо файл з даними
data = filedialog.askopenfilename()
# Обираємо файл з інтервалами
intervals = filedialog.askopenfilename()
# Обираємо папку, куди зберігати результати
directory = filedialog.askdirectory()

x, y = get_data(data)
start, finish = get_intervals(intervals)
E, T = JUST_DO_IT(x, y, start, finish)
write_it_down(E, T, directory)