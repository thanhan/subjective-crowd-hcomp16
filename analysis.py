import matplotlib.pyplot as plt
import collections
import numpy as np
import copy
import csv
from sklearn.metrics import roc_auc_score
import pickle


def plot_ipw(data):
    workers = []
    for i in data:
        workers.append(i[0])

    c = collections.Counter(workers)
    plt.plot(sorted(c.values(), reverse = True))

    plt.xlabel('Workers')
    plt.ylabel('Labels')
    plt.xlim([0, 3241])
    return c


def get_list_workers(data):
    workers = []
    for i in data:
        workers.append(i[0])
    workers = list(set(workers))
    return workers


def get_dic_url_labels(data):
    dic = {}
    for i in data:
        video = i[2]
        if len(i[3]) < 1: continue
        l = int(i[3][0])
        if video not in dic:
            dic[video] = []
        dic[video].append(l)

    return dic

def get_dic_conf_labels(data):
    dic = {}
    for i in data:
        conf = (i[6], int(i[7]), int(i[8]), int(i[9]))
        if len(i[3]) < 1: continue
        l = int(i[3][0])
        if conf not in dic:
            dic[conf] = []
        dic[conf].append(l)

    return dic

def get_dic_conf_wl(data):
    dic = {}
    for i in data:
        conf = (i[6], int(i[7]), int(i[8]), int(i[9]))
        if len(i[3]) < 1: continue
        w = i[0]
        l = int(i[3][0])
        if conf not in dic:
            dic[conf] = ([], [])
        dic[conf][0].append(w)
        dic[conf][1].append(l)

    return dic



def get_conf(i):
    conf = (i[6], int(i[7]), int(i[8]), int(i[9]))
    return conf



def plot_labels(L, ax = plt):
    """
    histogram of labels for a video
    """
    ax.bar(np.asarray([1,2,3,4,5])-0.5, [L.count(i) for i in [1,2,3,4,5] ])




def agreement(data, dic):
    dic_w = {}
    for i in data:
        video = i[2]
        if len(i[3]) < 1: continue
        l = int(i[3][0])
        w = i[0]
        average_l = np.mean(dic[video])
        if w not in dic_w: dic_w[w] = []
        dic_w[w].append( l - average_l)

    dic_mean = {}
    for w in dic_w:
        dic_mean[w] = np.mean(dic_w[w])

    a = dic_mean.values()
    #plt.hist(a, np.arange(-3, 3, 0.5))
    #plt.xlabel('Average difference to average rating')
    #plt.ylabel('Number of workers')

    return dic_w, dic_mean

def show_var(dic, bench = 'angry_birds', ax = plt):
    """
    given dic of conf -> [items]:
    """
    list_freq = [422400, 729600, 1036800, 1497600, 1958400, 2457600]

    
    a = []
    for cores in [1,2,3,4]:
        b = []
        for freq in list_freq:
            #b.append( np.var( dic[(bench, cores, freq, 0)]))
            b.append( dic[(bench, cores, freq, 0)] )
        a.append(b)
        #print a

    ax.set_yticklabels(['0','1','2','3','4'])
    ax.set_xticklabels([''])
    ax.matshow(a, cmap=plt.cm.gray, origin = 'lower')        
    ax.set_title(bench)
    
    #ax.colorbar()
    return ax


apps =  [ 'angry_birds', 'youtube', 'gladiator', 'chrome_cnn', 
            'epic_citadel', 'facebook', 'photoshop', 'compubench_rs_particles', 
            'compubench_rs_gaussian', 'compubench_rs_julia', 'compubench_rs_facedetection', 'compubench_rs_ambiant']

 

def show_var_all(dic):
    app = [ ['angry_birds', 'youtube', 'gladiator', 'chrome_cnn'], 
            ['epic_citadel', 'facebook', 'photoshop', 'compubench_rs_particles'], 
            ['compubench_rs_gaussian', 'compubench_rs_julia', 'compubench_rs_facedetection', 'compubench_rs_ambiant'] ]

    fig, axs = plt.subplots(3,4)

    for i in range(3):
        for j in range(4):
            ax = show_var(dic, app[i][j], axs[i][j])

    #fig.colorbar(ax)


def read_list(st):
    s = st.find('[')
    e = st.find(']')
    if ((s ==-1) or (e == -1)): 
        return []
    st = st[s+1:e]
    print s, e, st
    res = []
    for x in st.split():
        res.append(float(x))

    return res


def read_res(file_name):
    f = list(open(file_name))
    res = []
    for i, line in enumerate(f):
        if line.startswith('average'):
            l = read_list(line + f[i+1])
            res.append(l)

    return res


list_prw = [0.0, 0.05, 0.10, 0.15, 0.20]
list_prl = [0.4, 0.6, 0.8, 1.0]
list_ptr = [0.25, 0.50, 0.75, 1.0]
list_plt = [0.25, 0.50, 0.75, 1.0]
list_pwk = [0.0, 0.05, 0.10, 0.15, 0.20]


def plot(res, xl, i = 0, xlabel = '', ylabel = 'MAE'):
    plt.xlim(xl[0], xl[-1])
    y = zip(*res)[0+i]
    plt.plot(xl, y, label = 'Linear Regression')
    y = zip(*res)[2+i]
    plt.plot(xl, y, label = 'New')
    y = zip(*res)[4+i]
    plt.plot(xl, y, label = 'New(no spam)')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)



bad_guys = ['ATVU7N0LN2VY3', 'A3VOMP0WOJTB4I', 'A3NWQ0535HAXUD', 'A315IS7QIVORIT', 'AFKZZEURTCR8U', 'A1R3W25JDAUQ09', 'A29X1A8AF3TUQP', 'A3DTBL9NV7SDQC', 'AUUAFG5DUF89I', 'A1NKBXOTZAI1YK', 'A2UVEVCYVC553J', 'A1KFK92GEFQ9F5', 'AQ4PB1PVI9U7V', 'A2LVPGEPC1EUOW', 'ACIHCWKHNFC7U', 'A16WLNBWXNYSGU', 'AIM6UIPPVF05M', 'ANIRTJBXLLWKY', 'A19HCBT1EH8484', 'A111JI6APXR6QV']

def read_honeypot():
    f = open('honeypot_hits.csv')
    reader = csv.DictReader(f)


def eval_detect():
    f = open('find_uw.pkl')
    (list_workers, B_score, M_score, BM_score, B_rank, M_rank, BM_rank, dic_diff) = pickle.load(f)

    y = []
    for i, w in enumerate(list_workers):
        if w in bad_guys:
            y.append(1)
        else:
            y.append(0)

    y = np.asarray(y)
    #B_score = 1 - np.asarray(B_score)
    #M_score = 1 - np.asarray(M_score)
    #BM_score = 1 - np.asarray(BM_score)
    
    
    print roc_auc_score(y, B_score, average = None), roc_auc_score(y, M_score, average = None), roc_auc_score(y, BM_score, average = None)


