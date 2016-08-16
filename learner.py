import analysis
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
import sklearn
#from sklearn.kernel_ridge import KernelRidge
from sklearn import svm
from sklearn import gaussian_process
from scipy.special import digamma

import scipy.stats
from sklearn.metrics import roc_auc_score
import copy
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt


list_freq = [422400, 729600, 1036800, 1497600, 1958400, 2457600]

def preprocess(dic, app = 'angry_birds'):
    """
    given dic: conf -> labels
    return the design matrix, each row = (features, target)
    """



    a = np.zeros((24, 3))
    i = 0
    for cores in [1,2,3,4]:
        for freq in list_freq:
            m = np.mean(dic[app, cores, freq, 0])
            a[i,:] = np.asarray([cores, freq, m])
            i+= 1

    return a

def learn(a):
    data = a[:,0:-1]
    target = a[:,-1]
    LR = linear_model.LinearRegression()

    scores = cross_validation.cross_val_score(LR, data, target, cv=4, scoring = 'mean_absolute_error')

    print 'LR:', -scores.mean()

    LR_poly2 = Pipeline([('poly', PolynomialFeatures(degree=3)),
                      ('linear', linear_model.LinearRegression())])

    scores = cross_validation.cross_val_score(LR_poly2, data, target, cv=4, scoring = 'mean_absolute_error')
    print 'LR_poly3:', -scores.mean()


    Ridge = linear_model.Ridge (alpha = 0)
    scores = cross_validation.cross_val_score(Ridge, data, target, cv=4, scoring = 'mean_absolute_error')
    print 'Ridge:', -scores.mean()


    KR = KernelRidge()
    scores = cross_validation.cross_val_score(KR, data, target, cv=4, scoring = 'mean_absolute_error')
    print 'KR:', -scores.mean()


    SVR = clf = svm.SVR()
    scores = cross_validation.cross_val_score(SVR, data, target, cv=4, scoring = 'mean_absolute_error')
    print 'SVR:', -scores.mean()


    GP = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
    scores = cross_validation.cross_val_score(GP, data, target, cv=4, scoring = 'mean_absolute_error')
    print 'GP:', -scores.mean()


app = [ 'angry_birds', 'youtube', 'gladiator', 'chrome_cnn',
        'epic_citadel', 'facebook', 'photoshop', 'compubench_rs_particles',
        'compubench_rs_gaussian', 'compubench_rs_julia', 'compubench_rs_facedetection', 'compubench_rs_ambiant']

gpu = [200, 320, 389, 462.4, 578]

def create_features(conf):
    f = [0]*12 + [0,0,0,1]
    for i, a in enumerate(app):
        if a == conf[0]:
            f[i] = 1
    f[12] = conf[1] * 1.0 / 4
    f[13] = conf[2] * 1.0 / 2457600
    f[14] = gpu[conf[3]] *1.0 / 578
    f[15] = 1.0
    return f


class model:
    def __init__(self, data, lambda_w = 0, lambda_v = 0, A = 0, B = 0):
        """
        data is an array of [wid, question, url, rating, time, ip, application, cpu cores, cpu freq, gpu index]
        e.g:
        ['A2XXXXXXXXXXX', 'Question Random Assignment - Text 18',
        'www.youtube.com/embed/ZqD83lS8exs?wmode=transparent',
        '1 - Very Dissatisfied', 'Thu Jul 31 02:56:58 GMT 2014',
        '11.11.1.111', 'compubench_rs_particles', '3', '2457600', '0']
        
        lambda: weight for regularization of W and V
        A, B: weights (prior) on theta
        organize:
            - F: a features matrix,  row i = list of features for item i
            - L: crowd label, elem i = (list of W, list of L)            
        """ 
        self.data = data
        self.dic_conf_wl = analysis.get_dic_conf_wl(data)        
        n = len(self.dic_conf_wl)
        self.list_conf = self.dic_conf_wl.keys()
        self.F = []
        self.empi_mean = []
        self.empi_var  = []
        for conf in self.list_conf:
            f = create_features(conf)
            self.F.append(f)
            labels = self.dic_conf_wl[conf][1]
            self.empi_mean.append( np.mean(labels) )
            self.empi_var.append ( np.var(labels) )

        self.F = np.asarray(self.F)

        self.L = []
        for conf in self.list_conf:
            labels = self.dic_conf_wl[conf]
            self.L.append(labels)

        self.n = len(self.L)    # number of items
        self.m = len(self.F[0]) # number of features

        # build dic_w_il
        self.dic_w_il = {}
        for i in range(self.n):
            workers, labels = self.L[i]
            for w, l in zip(workers, labels):
                if w not in self.dic_w_il: self.dic_w_il[w] = []
                self.dic_w_il[w].append( (i,l))

        self.ep = 1e-100
        self.lambda_w = lambda_w
        self.lambda_v = lambda_v
        self.A = A
        self.B = B



    def get_mean(self, i):
        return self.F[i].dot(self.w)

    def get_std(self, i):
        return np.exp( self.F[i].dot(self.v) )
        

    def get_var(self, i):
        return pow( self.get_std(i), 2)

    def spam_dist(self, l):
        """
        distribution of labels from spammers
        """
        return scipy.stats.norm(3,self.s).pdf(l)

    def e_step(self):
        """
        evaluate posterior over Z
        """
        self.pt = []
        for i in range(self.n):
            self.pt.append([])
            workers, labels = self.L[i]
            for w, l in zip(workers, labels):
                p1 = scipy.stats.norm.pdf(l, loc = self.get_mean(i), scale = self.get_std(i) ) * self.theta[w]
                p0 = self.spam_dist(l) * (1-self.theta[w])
                p = p1 *1.0/ (p0 + p1)
                self.pt[i].append(p)
        

   
    def expected_ll(self, w, v):
        """
        return expected log likelihood
        """
        res = 0
        for i in range(self.n):
            workers, labels = self.L[i]
            for worker, l, pt1 in zip(workers, labels, self.pt[i]):
                pt0 = 1 - pt1
                #theta = self.theta[worker]
                ll0 = np.log( self.spam_dist(l) ) # + np.log(1-theta)
                mean = self.F[i].dot(w)
                std =  np.exp(self.F[i].dot(v))
                if std < self.ep: std = self.ep
                ll1 = scipy.stats.norm.logpdf(l, loc = mean, scale = std )# + np.log(theta)
 
                res += pt0*ll0 + pt1*ll1

        #regularization
        for i in range(self.m-1):
            res -= self.lambda_w * w[i]*w[i] + self.lambda_v * v[i]*v[i]
                
        return res


    def grad_expected_ll(self, w, v):
        gw = np.zeros( (self.m,) )
        gv = np.zeros( (self.m,) )

        for i in range(self.n):
            workers, labels = self.L[i]
            for worker, l, pt1 in zip(workers, labels, self.pt[i]):
                
                wtc = self.F[i].dot(w)
                sigma = np.exp(self.F[i].dot(v))
                if sigma < self.ep: sigma = self.ep
                update_w = pt1*(l-wtc)/pow(sigma,2)*self.F[i]
                gw += update_w

                update_v = pt1*(-self.F[i] + pow(l-wtc,2)/pow(sigma,2)*self.F[i])
                gv += update_v
        
        for i in range(self.m-1):
            gw[i] -= 2 * self.lambda_w * w[i]
            gv[i] -= 2 * self.lambda_v * v[i]

        return np.hstack( (gw, gv) )

       
    def check_grad(self, ep = 0.0000001, check_range = None):
        if check_range==None: check_range = range(self.m)
        w = np.random.rand(self.m) - 0.5
        v = np.random.rand(self.m) - 0.5

        a = self.expected_ll(w, v)

        fw = np.zeros(self.m)
        fv = np.zeros(self.m)

        for i in check_range:
            x = np.zeros(self.m)
            x[i] = ep
            fw[i] = (self.expected_ll(w + x, v) - a ) / ep
            fv[i] = (self.expected_ll(w, v + x) - a ) / ep
        
        print w
        print v

        print 'calculated grad = ', zip(range(self.m*2), self.grad_expected_ll(w, v))
        print 'finite diff grad = ',  zip(range(self.m*2), np.hstack((fw, fv)))

    def m_step_theta(self):
        """
        set theta_j to max expected ll
        """
        for w in self.dic_w_il: self.theta[w] = 0
        for i in range(self.n):
            workers, labels = self.L[i]
            for w, l, pt1 in zip(workers, labels, self.pt[i]):
                self.theta[w] += pt1

        for w in self.dic_w_il:
            num = len(self.dic_w_il[w])
            r = self.theta[w] * 1.0 / num            
            self.theta[w] = r
            #if (r < 0.85 ):
            #    self.theta[w] = r # no regularize
            #else:
            #    # regularize
            #    self.theta[w] = (self.theta[w] * 1.0 + self.A) / ( len( self.dic_w_il[w]) + self.A + self.B)
        
        # set self.s = sd of spam dist
        s = 0
        sw = 0
        for i in range(self.n):
            workers, labels = self.L[i]
            for w, l, pt1 in zip(workers, labels, self.pt[i]):
                s += pow(l - 3, 2)*(1-self.theta[w])
                sw += 1- self.theta[w]
        
        if sw > 0:
            self.s = pow(s*1.0/sw, 0.5)

    def m_step_wv(self, update_v = True):
        """
        maximize w and v
        """
        m = self.m
        f  = lambda x: -self.expected_ll(x[:m], x[m:])
        fp = lambda x: -self.grad_expected_ll(x[:m], x[m:])
        x0 = np.hstack( (self.w, self.v) )

        #opt_method = 'Nelder-Mead'
        opt_method = 'BFGS'
        res = scipy.optimize.minimize(f, x0, method=opt_method, jac=fp)
        #print res

        self.w = res.x[:m]
        if update_v:
            self.v = res.x[m:]


    def m_step(self, update_v = True):
        """
        maximize expected ll of w, v, theta
        """
        self.m_step_theta()
        self.m_step_wv(update_v)


        
    def init_wv(self):
        """
        init the params w and v 
        using the results of linear regression on empirical
        """
        self.lr = sklearn.linear_model.LinearRegression(fit_intercept = False)
        self.lr.fit(self.F, self.empi_mean)
        self.w = self.lr.coef_

        self.lr.fit(self.F, np.log( pow(np.asarray(self.empi_var), 0.5)) )
        self.v = self.lr.coef_


    def init_em(self, h_theta = 0.8):
        """
        init w, v, theta 
        """
        self.s = 2 # sd of spam distribution

        self.w = np.zeros((self.m,))
        self.v = np.zeros((self.m,))
        self.theta = {}
       
        dic_ul = analysis.get_dic_url_labels(self.data)
        dic_w, dic_mean = analysis.agreement(self.data, dic_ul)

        for w in self.dic_w_il:
            #self.theta[w] = 1 - h_theta*abs(dic_mean[w])
            self.theta[w] = h_theta

        self.init_wv()
            
        #self.pt = []
        #for i in range(self.n):
        #    self.pt.append([])
        #    workers, labels = self.L[i]
        #    for w, l in zip(workers, labels):
        #        self.pt[i].append(0.99)


       

    def em(self, w_it = 3, v_it = 1):
        """

        """
        # iterate
        for it in range(w_it):
            self.e_step()
            update_v = it < v_it
            self.m_step(update_v)

    
    def get_var_f(self, f):
        """
        return variance for a feature vector
        """
        return pow ( np.exp ( f.dot(self.v) ), 2)
 

    def predict(self, list_conf):
        """
        predict mean and var of new conf
        """
        res_mean = []
        res_var = []
        for conf in list_conf:
            f = np.asarray( create_features(conf) )
            mean = f.dot(self.w)
            var  = self.get_var_f(f)
            res_mean.append(mean)
            res_var.append(var)

        return (res_mean, res_var)

    def spam_score(self, workers):
        """
        return prob of the worker being a spammer
        """
        res = []
        for w in workers:
            res.append(1 - self.theta[w])
        return res

    def get_dic(self, w, v):
        """
        return dics of conf to mean and var
        using prediction by w and v
        """
        dic_mean = {}
        dic_var = {}
        for i in range(self.n):
            conf = tuple(self.list_conf[i])
            f = self.F[i]
            mean = f.dot(w)
            var = pow( np.exp(f.dot(v)), 2) 
            dic_mean[conf] = mean
            dic_var[conf] = var

        return (dic_mean, dic_var)


class LR:
    """
    baseline: linear regression
    """

    def __init__(self, data, hetero = False):
        self.dic_conf_wl = analysis.get_dic_conf_wl(data)        
        n = len(self.dic_conf_wl)
        list_conf = self.dic_conf_wl.keys()
        self.F = []
        self.empi_mean = []
        self.empi_var  = []
        for conf in list_conf:
            f = create_features(conf)
            self.F.append(f)
            labels = self.dic_conf_wl[conf][1]
            self.empi_mean.append( np.mean(labels) )
            self.empi_var.append ( np.var(labels) )

        self.lr_mean = sklearn.linear_model.LinearRegression(fit_intercept = False)
        self.lr_mean.fit(self.F, self.empi_mean)

        self.const_var = np.sum((self.lr_mean.predict(self.F) - self.empi_mean)**2) *1.0/ (n-2)

        self.lr_var = sklearn.linear_model.LinearRegression(fit_intercept = False)
        
        #self.lr_var.fit(self.F,  self.empi_var )
        #self.lr_var.fit(self.F, np.log( pow(np.asarray(self.empi_var), 0.5)))

        self.hetero = hetero
        
    
    def predict(self, list_conf):
        """
        predict mean and var of new conf
        """
        self.tF = []
        for conf in list_conf:
            f = create_features(conf)
            self.tF.append(f)

        res_mean = self.lr_mean.predict(self.tF)
        
        if self.hetero:
            res_var = self.lr_var.predict(self.tF)
            #res_var = pow( np.exp(self.lr_var.predict(self.tF)), 2)
        else:
            res_var = [self.const_var] * len(list_conf)

        return (res_mean, res_var)

    




class baseline_spam(model):
    """
    baselines for spam detection
    """
    def __init__(self, data):
        model.__init__(self, data)
        #get spam score
        self.ss = {}
        for w in self.dic_w_il:
            self.ss[w] = 0
            for i, l in self.dic_w_il[w]:
                # difference between label and average label
                self.ss[w] += np.abs( l - np.mean(self.L[i][1]) )
            self.ss[w] = self.ss[w] * 1.0 / len(self.dic_w_il[w])
                
        #normalize:
        max_score = max(self.ss.values())
        for w in self.ss:
            self.ss[w] = self.ss[w] * 1.0 / max_score



    def spam_score(self, workers):
        res = []
        for w in workers:
            res.append(self.ss[w])
        return res


empirical_spam = [0.13, 0.25, 0.22, 0.27, 0.14]

def plot_empi_spam():
    fig, ax = plt.subplots()
    ax.bar(np.asarray([1,2,3,4,5]) - 0.5, empirical_spam)
    ax.set_xlabel('Rating')
    ax.set_ylabel('Proportion')
    ax.set_xticks(np.asarray([1,2,3,4,5]))

class eval():
    """
    evaluate
    """
    def __init__(self, data, ptrain = 0.6, pval = 0.2, prw = 0.1, prl = 0.8, ptr = 1.0, plt = 1.0, pwk = 0.0, rand_seed = 1234, noise = 'empirical', bad_guys = []): 
        """
        ptrain = train set
        pval   = validation set
        prw = proportion of random workers (spammers)
        prl = proportion of random labels (how often a random worker gives a random label)
        ptr = proportion of train conf
        plt = proportion of labels for each conf in the train set
        pwk = proportion of workers to be removed (remove the ones with high diff)
        """        
        self.data = copy.deepcopy(data)
        self.pwk = pwk
        self.del_wk()
        self.dic_conf_wl = analysis.get_dic_conf_wl(self.data)        
        self.list_conf = self.dic_conf_wl.keys()
        #self.rs = np.random.RandomState(1)
        #self.rs.shuffle(self.list_conf)
        self.rs = np.random.RandomState(rand_seed)
        self.rs.shuffle(self.list_conf)
 
        self.n = len(self.list_conf)
        self.n_train = int(ptrain * self.n) # number of total train conf
        self.n_given = int(self.n_train * ptr) # number of train conf given to method

        self.train_conf = self.list_conf[:self.n_given]

        self.n_val = int(pval * self.n) # number of total validation conf
        self.val_conf = self.list_conf[self.n_train:self.n_train+self.n_val]

        self.test_conf  = self.list_conf[self.n_train+self.n_val:]

        # get gold L for test
        self.gold_mean = []; self.gold_var = []; self.gold_num = []
        for conf in self.test_conf:
            labs = self.dic_conf_wl[conf][1]
            workers = self.dic_conf_wl[conf][0]
            labels = []
            for l, w in zip(labs, workers):
                if w not in bad_guys:
                    labels.append(l)
            self.gold_mean.append( np.mean(labels) )
            self.gold_var.append ( np.var(labels) )
            self.gold_num.append( len(labels) )
        
        # also get gold L for train
        self.train_mean = []; self.train_var = []; self.train_num = []
        for conf in self.train_conf:
            labels = self.dic_conf_wl[conf][1]
            self.train_mean.append( np.mean(labels) )
            self.train_var.append ( np.var(labels) )
            self.train_num.append( len(labels) )
        
        # also get gold L for valildataion
        self.val_mean = []; self.val_var = []; self.val_num = []
        for conf in self.val_conf:
            labels = self.dic_conf_wl[conf][1]
            self.val_mean.append( np.mean(labels) )
            self.val_var.append ( np.var(labels) )
            self.val_num.append( len(labels) )
 
        self.plt = plt
        self.get_train_data()

        #inject noise
        train_workers = analysis.get_list_workers(self.train_data)
        self.rs.shuffle(train_workers)
        self.n_random_workers = int(prw * len(train_workers))
        self.random_workers = train_workers[:self.n_random_workers]
        self.train_workers = train_workers

        self.noise = noise
        self.prl = prl
        self.inject_noise()



    def rand_rating(self):
        if self.noise == 'uniform':
            return self.rs.randint(1,6)
        elif self.noise == 'empirical':
            return np.nonzero(self.rs.multinomial(1, empirical_spam))[0][0]+1
        else:
            raise "unknown noise"

    def inject_noise(self):
        for i in range(len(self.train_data)):
            w = self.train_data[i][0]
            if w in self.random_workers:
                if np.random.uniform() < self.prl:
                    self.train_data[i][3] = str(self.rand_rating())


    def get_train_data(self):
        self.train_data = []
        dic_conf_num = {}# conf-> number of crowd labels this conf got
        
        for d in self.data:
            conf = analysis.get_conf(d)
            if conf in self.train_conf:
                if conf not in dic_conf_num: dic_conf_num[conf] = 0
                if dic_conf_num[conf] > self.plt * len(self.dic_conf_wl[conf][0]): continue
                dic_conf_num[conf] += 1
                self.train_data.append(d)
 

    def del_wk(self):
        """
        remove workers with high deviation
        remove a proportion of self.pwk workers
        """
        if self.pwk == 0.0: return
        dic_ul = analysis.get_dic_url_labels(self.data)
        dic_w, dic_mean = analysis.agreement(self.data, dic_ul)
        self.workers = sorted(dic_mean.items(), key = lambda i : abs(i[1]), reverse = False)
        nwk = len(self.workers)
        keep_workers = list( zip(*self.workers[:int((1-self.pwk) * nwk)])[0])  # list of workers to keep
        new_data = []
        for i in self.data:
            if i[0] in keep_workers:
                new_data.append(i)
        self.data = new_data


    
    def get_mae(self, a, b):
        if ( len(a) != len(b) ) : raise "len not equal"
        res = 0
        for x,y in zip(a,b):
            res += np.abs(x-y)
        res = res * 1.0 / len(a)
        return res

    def eval(self, model):
        """
        model has beeen trained
        model has a predict method
        """
        res_mean, res_var = model.predict(self.test_conf)        
        mae_mean = self.get_mae(res_mean, self.gold_mean)
        mae_var  = self.get_mae(res_var, self.gold_var)
        #print "correlation: ", pearsonr(res_var, self.gold_var)        
        return [mae_mean, mae_var]

    def eval_val(self, model):
        """
        model has beeen trained
        model has a predict method
        evaluate on validation data
        """
        res_mean, res_var = model.predict(self.val_conf)        
        mae_mean = self.get_mae(res_mean, self.val_mean)
        mae_var  = self.get_mae(res_var, self.val_var)
        #print "correlation: ", pearsonr(res_var, self.gold_var)        
        return [mae_mean, mae_var]


    def print_val(self, model):
        res_mean, res_var = model.predict(self.val_conf)
        mae_var  = self.get_mae(res_var, self.val_var)
        s = 0
        for i,j in zip(res_var, self.val_var):
            print i, j, i - j
            s += (i-j)
        print "s = ", s

    def eval_all(self, em_it = 3):
        """
        evaluate
        """
        # LR
        #lr = LR(self.train_data, hetero = True)
        lr = LR(self.train_data, hetero = False)
        eval_lr = self.eval(lr)
        
        # nospam model
        #ns = model_nospam(self.train_data)
        #ns.init_em()
        #ns.em(em_it)
        #eval_ns = self.eval(ns)
 
        # model
        #new99 = model(self.train_data)
        #new99.init_em(0.99)
        #new99.em(em_it)
        #eval_new99 = self.eval(new99)

        new8 = model(self.train_data)
        #new8.init_em(0.99)
        new8.init_em(1)
        new8.em(1,1)
        eval_new8 = self.eval(new8)
 
        # fix bias model
        #fb  = model_fixbias(self.train_data)
        #fb.init_em()
        #fb.em(em_it)
        #eval_fb = self.eval(fb)

        # variational model
        var82 = model_var(self.train_data, 9.9, 0.1)
        var82.init_em()
        #var82.e_step()
        var82.em(1,1)
        eval_var82 = self.eval(var82)

        #var191 = model_var(self.train_data, 19, 1)
        #var191.init_em()
        #var191.em(em_it)
        #eval_var191 = self.eval(var191)

        # spamer score

        ss_baseline = self.detect_spammer(baseline_spam(self.train_data))
        ss_new = self.detect_spammer(new8)
        ss_var82 = self.detect_spammer(var82)
 
        print "linear reg/baseline:", eval_lr, ss_baseline
        #print "no spam model:", eval_ns
        print "new model:", eval_new8, ss_new
        #print "new model(fixbias)", eval_fb
        print "var model", eval_var82, ss_var82

        #return ([eval_lr, eval_new99, eval_new9, eval_ns, eval_fb, eval_var91, eval_var191], ss_baseline, ss_new)
        #return ([eval_lr, eval_new8], ss_baseline, ss_new)
        return ([eval_lr, eval_new8, eval_var82], ss_baseline, ss_new, ss_var82)
 

    def detect_spammer(self, model):
        """
        return AUC of the model in detecting the spammers.
        model has a method spam_score(list_workers) that return the prob of being spammer
        """
        # in self.train_workers, the first n_random_workers
        if self.n_random_workers == 0:
            return -1
        score = model.spam_score(self.train_workers)
        y = [1] * self.n_random_workers + [0] * (len(self.train_workers) - self.n_random_workers)
        return roc_auc_score(y, score)





class model_constvar(model):
    """
    same model with constant variance
    """
    def __init__(self, data):
        model.__init__(self,data)
        self.std = 1

    def get_std(self, i):
        return self.std


    def expected_ll(self, w):
        """
        return expected log likelihood
        """
        res = 0
        for i in range(self.n):
            workers, labels = self.L[i]
            for worker, l, pt1 in zip(workers, labels, self.pt[i]):
                pt0 = 1 - pt1
                theta = self.theta[worker]
                ll0 = np.log(self.spam_dist(l)) + np.log(1-theta)
                mean = self.F[i].dot(w)
                std =  self.std
                if std < self.ep: std = self.ep
                ll1 = scipy.stats.norm.logpdf(l, loc = mean, scale = std ) + np.log(theta)
 
                res += pt0*ll0 + pt1*ll1
                
        return res


    def grad_expected_ll(self, w):
        gw = np.zeros( (self.m,) )

        for i in range(self.n):
            workers, labels = self.L[i]
            for worker, l, pt1 in zip(workers, labels, self.pt[i]):
                
                wtc = self.F[i].dot(w)
                sigma = self.std
                if sigma < self.ep: sigma = self.ep
                update_w = pt1*(l-wtc)/pow(sigma,2)*self.F[i]
                gw += update_w
        

        return gw



    def m_step_var(self):
        s1 = 0
        s2 = 0
        for i in range(self.n):
            workers, labels = self.L[i]
            for worker, l, pt1 in zip(workers, labels, self.pt[i]):
                wtx = self.F[i].dot(self.w)
                s1 += pt1*pow(l-wtx,2)
                s2 += pt1
                
        self.var =  s1*1.0/s2



    def m_step(self):
        """
        maximize theta, W and var
        """
        self.m_step_theta()

        
        
        # maximize W
        m = self.m
        f  = lambda x: -self.expected_ll(x)
        fp = lambda x: -self.grad_expected_ll(x)
        x0 = self.w

        #opt_method = 'Nelder-Mead'
        opt_method = 'BFGS'
        res = scipy.optimize.minimize(f, x0, method=opt_method, jac=fp)
        print res
        self.w = res.x

        # maximize var
        self.m_step_var()


class model_var(model):
    """
    theta is hidden variable
    inference by meanfield variational
    """

    def __init__(self, data, A = 8.0, B = 2.0):
        model.__init__(self, data)
        self.A = A
        self.B = B


    def init_em(self):
        """
        init params: w and v
        init variational params alpha, beta, gamma
        """
        self.s = 2
        model.init_wv(self)
        self.alpha = {}
        self.beta = {}
        for w in self.dic_w_il:
            self.alpha[w] = self.A
            self.beta[w] = self.B
        
        self.gamma = []
        for i in range(self.n):
            workers, labels = self.L[i]
            self.gamma.append([])
            for w, l in zip(workers, labels):
                self.gamma[i].append( self.A*1.0/(self.A+self.B) )


    def update(self, a, b):
        self.n_update += 1
        self.change += np.abs(a-b)

    def e_step(self, max_it = 10):
       
        for it in range(max_it):
            self.change = 0; self.n_update = 0
            # update q(Z)
            for i in range(self.n):
                workers, labels = self.L[i]
                for w, l, j in zip(workers, labels, range(len(workers))):
                    alpha = self.alpha[w]
                    beta  = self.beta[w]
                    z0 = (self.spam_dist(l)) * np.exp( digamma(beta) - digamma(alpha+beta) )
                    z1 =  scipy.stats.norm.pdf(l, loc = self.get_mean(i), scale = self.get_std(i) ) * np.exp( digamma(alpha) - digamma(alpha+beta) ) 
                    g =  z1*1.0 / (z0+z1)
                    self.update(self.gamma[i][j], g)
                    self.gamma[i][j] = g
         
            # update q(theta)
            new_alpha = {}
            new_beta = {}
            for w in self.dic_w_il:
                new_alpha[w] = self.A
                new_beta[w] = self.B
                
            for i in range(self.n):
                workers, labels = self.L[i]
                for w, l, g in zip(workers, labels, self.gamma[i]):
                    new_alpha[w] += (1-g)
                    new_beta[w] += g

            for w in self.dic_w_il:
                self.update(self.alpha[w], new_alpha[w])
                self.alpha[w] = new_alpha[w]
                self.update(self.beta[w], new_beta[w])
                self.beta[w] = new_beta[w]
            
            #
            avg_change = self.change * 1.0/self.n_update
            if avg_change < 0.01:break


    def m_step(self, update_v = True):
        self.pt = self.gamma
        model.m_step_wv(self, update_v)

    def spam_score(self, workers):
        """
        return prob of being a spammer
        """
        res = []
        for w in workers:
            a = self.alpha[w]- self.A; b = self.beta[w] - self.B
            res.append( a * 1.0/(a+b) )
        return res

        
class model_nospam(model):
    def __init__(self, data):
        model.__init__(self, data)

    def e_step(self):
        self.pt = []
        for i in range(self.n):
            self.pt.append([])
            workers, labels = self.L[i]
            for w, l in zip(workers, labels):
                self.pt[i].append(1.0)



class test:
    def __init__(self, data, n_train = 1):
        self.data = copy.deepcopy(data)
        self.reduce()
        self.dic_conf_wl = analysis.get_dic_conf_wl(self.data)        
        self.list_conf = self.dic_conf_wl.keys()
        #self.rs = np.random.RandomState(rand_seed)
        #self.rs.shuffle(self.list_conf)

        self.n = len(self.list_conf)
        self.n_train = n_train

        self.train_conf = self.list_conf[:self.n_train]
        self.test_conf  = self.list_conf[self.n_train:]

        # get gold L for test
        self.gold_mean = []; self.gold_var = []; self.gold_num = []
        for conf in self.test_conf:
            labels = self.dic_conf_wl[conf][1]
            self.gold_mean.append( np.mean(labels) )
            self.gold_var.append ( np.var(labels) )
            self.gold_num.append( len(labels) )
        # also get gold L for train
        self.train_mean = []; self.train_var = []; self.train_num = []
        for conf in self.train_conf:
            labels = self.dic_conf_wl[conf][1]
            self.train_mean.append( np.mean(labels) )
            self.train_var.append ( np.var(labels) )
            self.train_num.append( len(labels) )

        self.get_train_data()

    
    def reduce(self):
        """
        del label so that each conf has ~ same number of L
        """
        dic_conf_wl = analysis.get_dic_conf_wl(self.data)
        dic_conf_num = {}
        for conf in dic_conf_wl.keys():
            labels = dic_conf_wl[conf][1]
            dic_conf_num[conf] = len(labels)

        new_data = []
        for d in self.data:
            conf = analysis.get_conf(d)
            if dic_conf_num[conf] > 50:
                dic_conf_num[conf] -= 1
            else:
                new_data.append(d)

        self.data = new_data


    def get_train_data(self):
        self.train_data = []
        for d in self.data:
            conf = analysis.get_conf(d)
            if conf in self.train_conf:
                self.train_data.append(d)
    
    def run(self, model, n_it = 1):
        self.M = model(self.train_data)
        self.M.init_em()
        x = self.M.predict(self.train_conf)
        self.M.em(n_it)
        x1 = self.M.predict(self.train_conf)
        self.print_res(self.train_var, x, x1)
    
    def print_res(self, gold, x, x1):
        sum_x = 0
        sum_x1 = 0
        for i in range(len(gold)):
            #print i, gold[i], x[1][i], x1[1][i]
            sum_x  += x[1][i] - gold[i]
            sum_x1 += x1[1][i] - gold[i]

        e0 = eval([])
        print 'mae x = ' , e0.get_mae(gold, x[1])
        print 'mae x1 = ', e0.get_mae(gold, x1[1])
        print 'sum x = ', sum_x, '  sum x1 = ', sum_x1
 

class model_fixbias(model):
    def m_step_wv(self):
        self.wm = []
        self.wv = []
        for i in range(self.n):
            workers, labels = self.L[i]
            m = len(labels)
            s = 0
            sw = 0
            for w, l, pt1 in zip(workers, labels, self.pt[i]):
                  s += pt1 * l
                  sw += pt1
            wmean = s * 1.0 / sw
            self.wm.append(wmean)
            s = 0
            for w, l, pt1 in zip(workers, labels, self.pt[i]):
                s += pt1 * pow(l - wmean, 2)
            wvar = s * 1.0 / sw
            self.wv.append(wvar)
  
        self.lr = sklearn.linear_model.LinearRegression(fit_intercept = False)
        self.lr.fit(self.F, self.wm)
        self.w = self.lr.coef_

        self.lr.fit(self.F, np.log( pow(np.asarray(self.wv), 0.5)) )
        self.v = self.lr.coef_

     
        
 
class model_vf(model_var, model_fixbias):
    def m_step(self):
        self.pt = self.gamma
        model_fixbias.m_step_wv(self)



class model_stoch(model):
    """
    using stochastic gradient descent to optimize params
    """
    def __init__(self, data, lr_w = 0.001, lr_v = 0.001):
        model.__init__(self, data)
        self.lr_w = lr_w
        self.lr_v = lr_v


    def m_step_wv(self):
        """
        maximize w and v using SGD
        """
        for it in range(50):
          for i in range(self.n):
            gw = np.zeros( (self.m,) )
            gv = np.zeros( (self.m,) )
            workers, labels = self.L[i]
            for worker, l, pt1 in zip(workers, labels, self.pt[i]):
                
                wtc = self.F[i].dot(self.w)
                sigma = np.exp(self.F[i].dot(self.v))
                if sigma < self.ep: sigma = self.ep
                update_w = pt1*(l-wtc)/pow(sigma,2)*self.F[i]
                gw += update_w

                update_v = pt1*(-self.F[i] + pow(l-wtc,2)/pow(sigma,2)*self.F[i])
                gv += update_v
        
            self.w = self.w + self.lr_w * gw
            self.v = self.v + self.lr_v * gv
        #for i in range(self.m-1):
        #    gw[i] -= 2 * self.lambda_w * w[i]
        #    gv[i] -= 2 * self.lambda_v * v[i]
 


class model_school(model):
    def __init__(self, data):
        model.__init__(self, data)

    
    def get_mean(self, i, k):
        return self.F[i].dot(self.w[k])

    def get_std(self, i, k):
        return np.exp(self.F[i].dot(self.v[k]))

    def get_var(self, i, k):
        return pow(self.get_std(i,k), 2)

    def e_step(self):
        """
        evaluate posterior over Z
        """
        self.pt = []
        for i in range(self.n):
            self.pt.append([])
            workers, labels = self.L[i]
            for w, l in zip(workers, labels):
                p1 = scipy.stats.norm.pdf(l, loc = self.get_mean(i, 1), scale = self.get_std(i, 1) ) * self.theta[w]
                p0 = scipy.stats.norm.pdf(l, loc = self.get_mean(i, 0), scale = self.get_std(i, 0) )  * (1-self.theta[w])
                p = p1 *1.0/ (p0 + p1)
                self.pt[i].append(p)
      

    def expected_ll(self, x):
        """
        return expected log likelihood
        """
        l = len(x)/2
        w = x[:l].reshape((2, self.m))
        v = x[l:].reshape((2, self.m))
        res = 0
        for i in range(self.n):
            workers, labels = self.L[i]
            for worker, l, pt1 in zip(workers, labels, self.pt[i]):
                pt0 = 1 - pt1
                #theta = self.theta[worker]
                ll = [0,0]
                for k in [0,1]:
                    mean = self.F[i].dot(w[k])
                    std =  np.exp(self.F[i].dot(v[k]))
                    if std < self.ep: std = self.ep
                    ll[k] = scipy.stats.norm.logpdf(l, loc = mean, scale = std )# + np.log(theta)
 
                res += pt0*ll[0] + pt1*ll[1]

        #regularization
        #for i in range(self.m-1):
        #    res -= self.lambda_w * w[i]*w[i] + self.lambda_v * v[i]*v[i]
                
        return res



    def grad_expected_ll(self, x):
        l = len(x)/2
        w = x[:l].reshape((2, self.m))
        v = x[l:].reshape((2, self.m))

        gw = np.zeros( (2, self.m) )
        gv = np.zeros( (2, self.m) )

        for k in [0,1]:
          for i in range(self.n):
            workers, labels = self.L[i]
            for worker, l, pt1 in zip(workers, labels, self.pt[i]):
                
                wtc = self.F[i].dot(w[k])
                sigma = np.exp(self.F[i].dot(v[k]))
                if sigma < self.ep: sigma = self.ep
                pt = pt1 if (k==1) else 1 - pt1
                update_w = pt*(l-wtc)/pow(sigma,2)*self.F[i]
                gw[k] += update_w[k]

                update_v = pt*(-self.F[i] + pow(l-wtc,2)/pow(sigma,2)*self.F[i])
                gv[k] += update_v[k]
        
        # regularization
        #for i in range(self.m-1):
        #    gw[i] -= 2 * self.lambda_w * w[i]
        #    gv[i] -= 2 * self.lambda_v * v[i]

        return np.hstack( (gw[0,:], gw[1,:], gv[0,:], gv[1,:]) )
  


    def m_step_wv(self):
        """
        maximize w and v
        """
        m = self.m
        f  = lambda x: -self.expected_ll(x)
        fp = lambda x: -self.grad_expected_ll(x)
        x0 = np.hstack( (self.w.reshape((2*m,)), self.v.reshape((2*m,))) )

        #opt_method = 'Nelder-Mead'
        opt_method = 'BFGS'
        res = scipy.optimize.minimize(f, x0, method=opt_method, jac=fp)
        #print res
        x = res.x
        l = len(x)/2
        self.w = x[:l].reshape((2, self.m))
        self.v = x[l:].reshape((2, self.m))

 
    def m_step(self):
        """
        maximize expected ll of w, v, theta
        """
        self.m_step_theta()
        self.m_step_wv()

    def init_wv(self):
        """
        init the params w and v 
        using the results of linear regression on empirical
        """
        self.lr = sklearn.linear_model.LinearRegression(fit_intercept = False)
        self.lr.fit(self.F, self.empi_mean)
        self.w[0] = self.lr.coef_ + self.rs.normal(0, 0.1, self.m) 
        self.w[1] = self.lr.coef_ + self.rs.normal(0, 0.1, self.m) 
 

        self.lr.fit(self.F, np.log( pow(np.asarray(self.empi_var), 0.5)) )

        self.v[0] = self.lr.coef_ + self.rs.normal(0, 0.1, self.m)
        self.v[1] = self.lr.coef_ + self.rs.normal(0, 0.1, self.m)




    def init_em(self, rseed = 1):
        """
        init w, v, theta 
        """
        self.rs = np.random.RandomState(rseed)
        self.w = np.zeros((2, self.m))
        self.v = np.zeros((2, self.m))
        self.theta = {}
        
        for w in self.dic_w_il:
            self.theta[w] = self.rs.rand()

        self.init_wv()
      
    def em(self, n_it = 3):
        """

        """
        # iterate
        for it in range(n_it):
            self.e_step()
            self.m_step()


