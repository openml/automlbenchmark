import argparse
import numpy as np
import  operator
import os, fnmatch
import sys
import glob
import json

#command line to get avg and std from result for a particular task: python get_res_avg.py  -t blood-transfusion
#command line to get avg and std from result for all tasks that has results: python get_res_avg.py
parser = argparse.ArgumentParser()
# parser.add_argument('dataset', type=str,
#                     help='The predictions file to load and compute the scores for.')
parser.add_argument('-t', '--task', metavar='task_id', nargs='*', default=None,
                    help="The specific task name (as defined in the benchmark file) to run. "
                         "If not provided, then all tasks from the benchmark will be run.")
args = parser.parse_args()

if not args.task:
    res_dir = './results/'
else:
    res_dir = './results/' + str(args.task)
for r, d, f in os.walk(res_dir):
    # print (r,d,f)
    if 'backup' in r: continue
    for file in f:
        if 'results.csv' in file:
            res_dic = {} #keys are algnames
            filename = os.path.join(r, file)
            run_add_id = 0
            last_fold = -1
            with open(filename) as file_:
                for line in file_:
                    info = line.split(',')
                    # print(len(info), info)
                    try:
                        fold = int(info[4])
                        if fold!=last_fold+1:
                            if last_fold==9:
                                run_add_id +=1
                            if fold!=0:
                                last_fold=-1
                                continue
                        last_fold=fold
                        # id,task,framework,fold,result,mode,version,params,tag,utc,duration,models,seed,info,_, __ = info
                        # run_sig = str(run_add_id) + ',' + id + ',' + task + ',' +framework +',' + mode +',' + version +',' + params + ',' + tag
                        run_sig = str(run_add_id) + ',' + ','.join(info[:4])
                        if fold==0:
                            res_dic[run_sig] = []
                        result = float(info[5])

                        try:
                            for temp in range(len(info)):
                                if info[temp].count(':') ==2 and '{' not in info[temp]:
                                    duration = float(info[temp +1])
                        except:
                            duration = 0

                        # if info[9].count(':')>1:
                        #     duration = float(info[10])
                        # elif info[10].count(':')>1:
                        #     duration = float(info[11])
                        # else:
                        #     duration = 0
                        # print(info[9], info[10], info[-6])
                        res_dic[run_sig].append([result,duration])
                    except:
                        last_fold = -1
            res_mean_dic = {}
            res_std_dic = {}
            for key, results in res_dic.items():
                if len(results)!=10: continue
                results = list(zip(*results))
                res_list = np.array(results[0])
                duration_list = np.array(results[1])
                try:
                    res_mean_dic[key] = (np.mean(res_list),np.mean(duration_list))
                    res_std_dic[key] = np.std(res_list)
                except:
                    # res_mean_dic[key] = ''
                    # res_std_dic[key] = ''
                    pass
            keys = list(res_mean_dic.keys())
            keys.sort(key=lambda k: k.split(',')[2])
            avg_filename = filename.split('.csv')[0] + '_avg.csv'
            with open(avg_filename, 'w') as file_:
                header = 'run_add_id,id,task,framework,result_mean,result_std,duration\n'
                file_.write(header)
                for key in keys:
                    print(key)
                    file_.write(str(key) + ',' + str(res_mean_dic[key][0]) + ',' +  str(res_std_dic[key])+','+str(res_mean_dic[key][1])  + '\n')
        elif r.endswith('/predictions') and file.endswith('.csv') and 'leaderboard' not in file \
            or file.endswith('.pkl'):
            # os.remove(os.path.join(r,file))
            pass
        elif file=='results.json':
            lines = []
            warned = False
            with open(os.path.join(r,file)) as f:
                for line in f:
                    data = json.loads(line)
                    if data[3] is None: 
                        if not warned:
                            print(r,line)
                            warned=True
                        continue
                    info = data[3]['info']
                    train_time = float(info['train_time'])
                    if train_time>0:
                        lines.append(line)
            with open(os.path.join(r,file),'w') as f:
                f.writelines(lines)
        elif file=='configs.json':
            # os.remove(os.path.join(r,file))
            pass
