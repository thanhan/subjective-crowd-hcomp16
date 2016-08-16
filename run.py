import merge
import learner
import numpy as np
import sys
import pickle
import csv

direc = 'res3/'

import os

import matplotlib.pyplot as plt

seeds = [1,2,3,4,5]


def process(res):
    """
    process the results
    """
    (list_mv, ss_baseline, ss_new, ss_var) = res
    a = zip(*list_mv)
    return list(a[0]) + list(a[1]) + [ss_baseline, ss_new, ss_var]


list_prw = [0.0, 0.05, 0.10, 0.15, 0.20] 
def exp1():
    """
    PRW
    experiment 1: varying proportion of 'spammers' in 5, 10, 20, 40 %
    each spammer spams 80% of the time
    """
    data = merge.load()
    save = []

    for prop in list_prw:
        res = []
        for seed in seeds:
            print prop, seed
            e = learner.eval(data, prw = prop, rand_seed = seed, noise = 'empirical')
            #((lr_m, lr_v), (new_m, new_v), (ns_m, ns_v), ss_baseline, ss_new) = e.eval_all()
            temp = e.eval_all()
            res.append( process(temp) )

        print 'prw = ', prop
        print 'average Res SS(baseline, new)', np.mean(res, 0)
        save.append( list(np.mean(res, 0)) )

    f = open(direc + 'exp1.pkl', 'w');
    pickle.dump(save, f)


list_prl = [0.2, 0.4, 0.6, 0.8, 1.0] 
def exp2():
    """
    PRL
    experiment 2: varying how often a 'spammer' spams in 40, 60, 80, 100 %
    assume 10% spammers
    """
    data = merge.load()
    save = []
    
    for prop in list_prl:
        res = []
        for seed in seeds:
            print prop, seed
            e = learner.eval(data, prl = prop, rand_seed = seed, noise = 'empirical')
            #((lr_m, lr_v), (new_m, new_v), (ns_m, ns_v), ss_baseline, ss_new) = e.eval_all()
            temp = e.eval_all()
            res.append( process(temp) )

        print 'prl = ', prop
        print 'average Res SS(baseline, new)', np.mean(res, 0)
        save.append( list(np.mean(res, 0)) )

    f = open(direc + 'exp2.pkl', 'w');
    pickle.dump(save, f)


list_ptr = [0.2, 0.4, 0.6, 0.8, 1.0] 
def exp3():
    """
    experiment 3: vary ptr 
    """
    data = merge.load()
    save = []
    
    for prop in list_ptr:
        res = []
        for seed in seeds:
            print prop, seed
            e = learner.eval(data, ptr = prop, rand_seed = seed, noise = 'empirical')
            #((lr_m, lr_v), (new_m, new_v), (ns_m, ns_v), ss_baseline, ss_new) = e.eval_all()
            temp = e.eval_all()
            res.append( process(temp) )

        print 'ptr = ', prop
        print 'average Res SS(baseline, new)', np.mean(res, 0)
        save.append( list(np.mean(res, 0)) )

    f = open(direc + 'exp3.pkl', 'w');
    pickle.dump(save, f)


list_plt =  [0.2, 0.4, 0.6, 0.8, 1.0]
def exp4():
    """
    experiment 4: vary plt
    """
    data = merge.load()
    save = []
    
    for prop in list_plt:
        res = []
        for seed in seeds:
            print prop, seed
            e = learner.eval(data, plt = prop, rand_seed = seed, noise = 'empirical')
            #((lr_m, lr_v), (new_m, new_v), (ns_m, ns_v), ss_baseline, ss_new) = e.eval_all()
            temp = e.eval_all()
            res.append( process(temp) )

        print 'plt = ', prop
        print 'average Res SS(baseline, new)', np.mean(res, 0)
        save.append( list(np.mean(res, 0)) )

    f = open(direc + 'exp4.pkl', 'w');
    pickle.dump(save, f)



list_pwk = [0.0, 0.05, 0.10, 0.15, 0.20]
def exp5():
    """
    experiment 5: vary pwk
    """
    data = merge.load()
    save = []
    
    for prop in list_pwk:
        res = []
        for seed in seeds:
            print prop, seed
            e = learner.eval(data, pwk = prop, rand_seed = seed, noise = 'empirical')
            #((lr_m, lr_v), (new_m, new_v), (ns_m, ns_v), ss_baseline, ss_new) = e.eval_all()
            temp = e.eval_all()
            res.append( process(temp) )

        print 'pwk = ', prop
        print 'average Res SS(baseline, new)', np.mean(res, 0)
        save.append( list(np.mean(res, 0)) )

    f = open(direc + 'exp5.pkl', 'w');
    pickle.dump(save, f)



def tune(x):
    x = int(x)
    print x
    data = merge.load()
    e = learner.eval(data, ptr = 0.80, rand_seed = 0, noise = 'empirical')
    print 'ptr = 0.80'
    list_can = [0.7, 0.8, 0.9, 0.95, 0.99]
    h = list_can[x]
    M = learner.model(e.train_data)
    M.init_em(h)
    for i in range(3):
        M.em(1)
        print h, e.eval_val(M)
 
    #e = learner.eval(data, prl = 0.4, rand_seed = 0, noise = 'empirical')
    #print 'prl =  0.4'
    #for h in [0.2, 0.1, 0.05, 0.025, 0.01, 0.005]:
    #    M = learner.model_stoch(e.train_data)
    #    M.init_em(h)
    #    M.em(3)
    #    res = e.eval_val(M)
    #    print h, res
 


def run_all(x, param = 'prw'):
    x = int(x)
    print x
    data = merge.load()
    p = x / 5
    s = x % 5

    if param == 'prw':
        e = learner.eval(data, ptrain = 0.8, pval = 0.0, prw = list_prw[p], rand_seed = s, noise = 'empirical')
        print 'prw = ', list_prw[p], 'seed = ', s
    elif param == 'prl':
        e = learner.eval(data, ptrain = 0.8, pval = 0.0, prl = list_prl[p], rand_seed = s, noise = 'empirical')
        print 'prl = ', list_prl[p], 'seed = ', s
    elif param == 'ptr':
        e = learner.eval(data, ptrain = 0.8, pval = 0.0, ptr = list_ptr[p], rand_seed = s, noise = 'empirical')
        print 'ptr = ', list_ptr[p], 'seed = ', s
    elif param == 'plt':
        e = learner.eval(data, ptrain = 0.8, pval = 0.0, plt = list_plt[p], rand_seed = s, noise = 'empirical')
        print 'plt = ', list_plt[p], 'seed = ', s
    elif param == 'pwk':
        e = learner.eval(data, ptrain = 0.8, pval = 0.0, pwk = list_pwk[p], rand_seed = s, noise = 'empirical')
        print 'pwk = ', list_pwk[p], 'seed = ', s

    res = e.eval_all()
    print res

    #f = open('condor1/pwk/' + str(x) + '.pkl', 'w');
    #pickle.dump(res, f)

#new_ptr = [0.05, 0.1, 0.2, 0.5, 1.0]

#new_plt = [0.1, 0.2, 0.3, 0.4, 0.5]

new_ptr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
new_plt = [0.25, 0.5, 1.0]

def exp7(x):
    """
    No spammers inserted
    """
    x = int(x)
    print x
    data = merge.load()
    s = x % 5
    x = x / 5

    p = x
    t = 2
    

    e = learner.eval(data, ptrain = 0.8, pval = 0.0, prw = 0.0, ptr = new_ptr[p], plt = new_plt[t], rand_seed = s, noise = 'empirical')
    print 'ptr = ', new_ptr[p], 'plt = ', new_plt[t], 'seed = ', s
    
    res = e.eval_all()
    print res
 


def exp7_old(x):
    """
    No spammers inserted
    """
    x = int(x)
    print x
    data = merge.load()
    p = (x / 5) % 5
    s = x % 5
    t = x / 25

    e = learner.eval(data, ptrain = 0.8, pval = 0.0, prw = 0.0, ptr = new_ptr[p], plt = new_plt[t], rand_seed = s, noise = 'empirical')
    print 'ptr = ', new_ptr[p], 'plt = ', new_plt[t], 'seed = ', s
    
    res = e.eval_all()
    print res
    #f = open('condor4/pwk/' + str(x) + '.pkl', 'w');
    #pickle.dump(res, f)




def get_rank(workers, scores):
    """
    spam score -> rank
    """
    a = zip(scores, workers)
    b = sorted(a)
    dic = {}
    i = 0
    for s, w in b:
        dic[w] = i
        i+= 1
    return dic

def find_uw():
    """
    find unreliable workers    
    """
    data = merge.load()
    e = learner.eval(data, ptrain = 1.0, pval = 0.0, prw = 0.0, rand_seed = 0, noise = 'empirical')
    M = learner.model(e.train_data)
    M.init_em(0.8)
    M.em()

    BM = learner.model_var(e.train_data)
    BM.init_em()
    BM.em()
    #BM.e_step()


    B = learner.baseline_spam(e.train_data)

    list_workers = e.train_workers
    B_score  =  B.spam_score(list_workers)
    M_score  =  M.spam_score(list_workers)
    BM_score = BM.spam_score(list_workers)

    B_rank  = get_rank(list_workers, B_score)
    M_rank  = get_rank(list_workers, M_score)
    BM_rank = get_rank(list_workers, BM_score)

    for i, w in enumerate(list_workers):
        print i, w, B_score[i], M_score[i], BM_score[i], B_rank[w], M_rank[w], BM_rank[w]

    dic_diff = {}
    for w in list_workers:
        dic_diff[w] = abs(B_rank[w] - M_rank[w]) + abs(B_rank[w] - BM_rank[w]) + abs(M_rank[w] - BM_rank[w])


    print dic_diff
    f = open('find_uw.pkl', 'w')
    pickle.dump( (list_workers, B_score, M_score, BM_score, B_rank, M_rank, BM_rank, dic_diff) , f)
    f.close()


def write_csv():
    f = open('find_uw.pkl')
    (list_workers, B_score, M_score, BM_score, B_rank, M_rank, BM_rank, dic_diff) = pickle.load(f)
    f.close()

    f = open('workers.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['ID', 'Baseline', 'New', 'B-New', 'Baseline rank', 'New rank', 'B-New rank', 'Disagree'])

    for i, w in enumerate(list_workers):
        entry = [w, B_score[i], M_score[i], BM_score[i], B_rank[w], M_rank[w], BM_rank[w], dic_diff[w] ]
        writer.writerow(entry)

def main(exp_num):
    if exp_num==0:
        #tune(sys.argv[2])
        run_all(sys.argv[2], 'pwk')
    elif exp_num==1:
        exp1()
    elif exp_num==2:
        exp2()
    elif exp_num==3:
        exp3()
    elif exp_num==4:
        exp4()
    elif exp_num==5:
        exp5()
    elif exp_num==6:
        find_uw()
    elif exp_num==7: # exp with no spammers inserted, in a sparse condition
        exp7(sys.argv[2])

if  __name__ =='__main__':
    main(int(sys.argv[1]))



def plot(filename, x, xlabel):
    f = open(filename)
    save = pickle.load(f)
    f.close()

    lab = ['LR', 'new99', 'new9', 'no spam', 'LR+spam', 'bay91', 'bay191']
    mk = ['x','+','.','s','o']

    for i in range(5):
        y = []
        for j in range(5):
            y.append(save[j][i])
        plt.plot(x, y, label = lab[i], marker = mk[i])

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel('MAE')


def read_plot(param = 'prw', xl = [0.0, 0.05, 0.10, 0.15, 0.20], s2 = 1, xl0 = 0, xl1 = 0.27 ):
    """
    s2 = start location for AUC plot
    """
    direc = 'condor3/' + param + '/'
    res = []
    for i in range(25):
        f = open(direc + 'out.' + str(i))
        l = list(f)
        exec( 'a = ' + l[5] )
        res.append(process(a))
    a = []
    a.append( list(np.mean(res[0:5], 0)))
    a.append( list(np.mean(res[5:10], 0)))
    a.append( list(np.mean(res[10:15], 0)))
    a.append( list(np.mean(res[15:20], 0)))
    a.append( list(np.mean(res[20:25], 0)))

    f, axarr = plt.subplots(3, sharex=True)
    a = zip(*a)
    axarr[0].plot(xl, a[0], label = 'LR2', marker = '.', markersize = 20, linewidth = 3)
    axarr[0].plot(xl, a[1], label = 'NEW',marker = 'v', markersize = 15, linewidth = 3)
    axarr[0].plot(xl, a[2], label = 'B-NEW',marker = '^', markersize = 15, linewidth = 3)
    axarr[0].set_xlim(xl0, xl1)

    axarr[0].set_ylabel('MAE-mean')
    axarr[0].legend(loc = 'right', ncol = 1)
    
    axarr[1].plot(xl, a[3], label = 'LR2', marker = '.', markersize = 20, linewidth = 3)
    axarr[1].plot(xl, a[4], label = 'NEW',marker = 'v', markersize = 15, linewidth = 3)
    axarr[1].plot(xl, a[5], label = 'B-NEW',marker = '^', markersize = 15, linewidth = 3)

    axarr[1].set_ylabel('MAE-var')
    axarr[1].legend(loc = 'right', ncol = 1)
    
    axarr[2].plot(xl[s2:], a[6][s2:], label = 'AD', marker = '.', markersize = 20, linewidth = 3)
    axarr[2].plot(xl[s2:], a[7][s2:], label = 'NEW',marker = 'v', markersize = 15, linewidth = 3)
    axarr[2].plot(xl[s2:], a[8][s2:], label = 'B-NEW',marker = '^', markersize = 15, linewidth = 3)

    axarr[2].set_ylabel('AUC')
    #axarr[2].legend(loc = 'upper left', ncol = 3, bbox_to_anchor=(0., 1.02, 1., .102))
    axarr[2].legend(loc = 'right', ncol = 1) 
    #plt.xlabel()
   

def plot_all():
    read_plot('prw', list_prw, 1, 0, 0.27)
    plt.savefig('prw.png')

    read_plot('prl', list_prl, 0, 0.2, 1.27)
    plt.savefig('prl.png')

    read_plot('ptr', list_ptr, 0, 0.2, 1.27)
    plt.savefig('ptr.png')

    read_plot('plt', list_plt, 0, 0.2, 1.27)
    plt.savefig('plt.png')

    read_plot('pwk', list_pwk, 0, 0, 0.27)
    plt.savefig('pwk.png')

 
def read_exp7(xl0 = 0.1 , xl1 = 1.0):
    """
    
    """
    x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    direc = 'condor4/' + '/'
    res = []
    for i in range(50):
        f = open(direc + 'out.' + str(i))
        l = list(f)
        exec( 'a = ' + l[5] )
        res.append(process(a))
    #return res

    a = []
    for i in range(0, 50, 5):
        a.append( list(np.mean(res[i:i+5], 0)) )
    
    
    f, axarr = plt.subplots(2, sharex=True)
    a = zip(*a)
    axarr[0].plot(x, a[0], label = 'LR', marker = '.', markersize = 20, linewidth = 3)
    axarr[0].plot(x, a[1], label = 'NEW',marker = 'v', markersize = 15, linewidth = 3)
    axarr[0].set_xlim(xl0, xl1)

    axarr[0].set_ylabel('MAE-mean')
    axarr[0].legend(loc = 'upper right', ncol = 1)
    
    axarr[1].plot(x, a[3], label = 'LR', marker = '.', markersize = 20, linewidth = 3)
    axarr[1].plot(x, a[4], label = 'NEW',marker = 'v', markersize = 15, linewidth = 3)

    axarr[1].set_ylabel('MAE-var')
    axarr[1].legend(loc = 'upper right', ncol = 1)
    
 
