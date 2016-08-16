import csv
import pytz
import dateutil.parser
import os.path
import pickle
import numpy as np



def read_url_map():
    f = 'mobilecpu-mturk-data/youtube_url_mappings.csv'
    reader = open(f)
    reader.readline()
    reader = csv.reader(reader, delimiter=',', quotechar='"')
    dic = {}
    for line in reader:
        dic[line[0]] = (line[1], line[2], line[3], line[4])
    return dic

def read_qmap(f):
    
    dic = {}
    reader = open(f)
    for line in reader:
        if line.startswith('Question'):
            q = line.strip()
        elif line.find('www') > 0:
            st = line.find('www')
            en = line.find('transparent') + len('transparent')
            url = line[st:en]
            dic[q] = url
            if url == '': print 'empty url', line

    #print "read qmap returns ", f, dic
    return dic


def readFiles(mturkFiles = [], surveyFiles = [], qmapFile = None):

    mturk = []
    survey = []
    qmap = []

    for f in mturkFiles:
        csvfile = open(f, 'r')
        csvfile.readline()
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        temp = list(reader)
        mturk = mturk + temp

    for f in surveyFiles:
        csvfile = open(f, 'rb')
        csvfile.readline()
        csvfile.readline()
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        temp = list(reader)
        survey = survey + temp

    qmap = read_qmap(qmapFile)

    return (mturk, survey, qmap)

mturk_name  = 'mobilecpu-mturk-data/mturk_android_batch_{}_{}.csv'
survey_name = 'mobilecpu-mturk-data/survey_android_batch_{}.csv'
qmap_name = 'mobilecpu-mturk-data/question_mappings_android_batch_{}.txt'
batch_list = range(1,21) + range(30, 41) + [40] + range(101, 108)
#batch_list = [1, 2, 3]

def input():
    url_map = read_url_map()

    res = []
    for batch in batch_list:
        survey_file = survey_name.format(batch)
        qmap_file   = qmap_name.format(batch)
        if os.path.exists(survey_file) and os.path.exists(qmap_file):
            mturk_files = []
            for trial in [1,2]:
                mturk_file = mturk_name.format(batch, trial)
                if os.path.exists(mturk_file):
                    mturk_files.append(mturk_file)
            
            print 'batch', batch
            (mturk, survey, qmap) = readFiles(mturk_files, [survey_file], qmap_file)
            merged = merge(mturk, survey, qmap, url_map)
            res = res + merged

    return res

#mturk:
#workerid = 15
#random word = 27
#accept time = 17
#submit time = 18

#survey
#question = 9
#response = 10
#random word = 11
#start date= 2
#end   date = 3
#ip         = 4

def merge(mturk, survey, qmap, url_map):
    print qmap
    res = []
    dic_ip = {}
    #first pass: learn ip of workers

    for i,m in enumerate(mturk):
        match = []
        for s in survey:
            if m[27] == s[11]:
                match.append((m[15], s[4]))
        if len(match) == 1:
            dic_ip[m[15]] = [match[0][1]]

    for i, m in enumerate(mturk):
        match = None
        for s in survey:
            if m[27] == s[11] and s[10] != '':
                m_accept = dateutil.parser.parse(m[17]).replace(tzinfo = pytz.utc)
                m_submit = dateutil.parser.parse(m[18]).replace(tzinfo = pytz.utc)
                s_start  = dateutil.parser.parse(s[2]) .replace(tzinfo = pytz.utc)
                s_end    = dateutil.parser.parse(s[3]) .replace(tzinfo = pytz.utc)
            
                #if abs((m_accept - s_start).total_seconds()) < 600 and abs((s_end - m_submit).total_seconds())  < 600:
                x = abs((m_accept - s_start).total_seconds()) + abs((s_end - m_submit).total_seconds())
                if m[15] in dic_ip and s[4] in dic_ip[m[15]]: x = 0
                if match == None or x < match[0]:
                    video = qmap[s[9]]
                    conf = url_map[video]
                    match = (x, [m[15], s[9], video, s[10], m[17], s[4], conf[0], conf[1], conf[2], conf[3]])
        
        if match == None:
            #print "no match ", i, m 
            pass
        else:
            vec = match[1]
            res.append(vec)
            if vec[0] not in dic_ip: dic_ip[vec[0]] = []
            dic_ip[vec[0]].append(vec[5])

    return res



def print_res(res):
    f = open('merged_data.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['id', 'question', 'url', 'rating', 'accept time', 'ip', 'benchmark', 'cores', 'cpu', 'gpu'])
    for x in res:
        writer.writerow(x)

    f.close()


def load():
    f = open('data.pkl')
    d = pickle.load(f)
    # shuffle data:
    rs = np.random.RandomState(1)
    rs.shuffle(d)
    f.close()
    return d
