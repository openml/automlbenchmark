import numpy as np
import glob
from collections import namedtuple

def compare_with_benchmark_scaled(framework_list,task_tunedRF_dic, task_list, framework_dic_scaled):
    worse_than_tunedRF_num, eq_worse_than_CP_num = {}, {}
    for framework in framework_list:
        if framework not in worse_than_tunedRF_num:
            worse_than_tunedRF_num[framework] = 0
            eq_worse_than_CP_num[framework] = 0

        for task in task_tunedRF_dic.keys():
            if task in task_list and task in framework_dic_scaled[framework]:
                if task in framework_dic_scaled[framework]:
                    if float(framework_dic_scaled[framework][task]) < 1.0:
                        worse_than_tunedRF_num[framework] +=1
                    if float(framework_dic_scaled[framework][task]) <= 0.0:
                        eq_worse_than_CP_num[framework] +=1
    return worse_than_tunedRF_num, eq_worse_than_CP_num

def get_ratio_dic(worse_than_tunedRF_num,eq_worse_than_CP_num, l):
    worse_than_tunedRF_ratio = {}
    eq_worse_than_CP_ratio = {}
    for k,v in worse_than_tunedRF_num.items():
        worse_than_tunedRF_ratio[k] = float(v)/l
        eq_worse_than_CP_ratio[k] = eq_worse_than_CP_num[k]/l
    return worse_than_tunedRF_ratio, eq_worse_than_CP_ratio 

def convert_loss_into_score(loss, is_multi_class = False):
    if not is_multi_class:
        score = 1.0 - loss
    else:
        score = -loss
    return score
    
def get_tunedRF_constPredictor(file_dir):
    files = glob.glob(file_dir + '/*/result_summary.csv')
    # print('files', files)
    tunedRF_dic = {}
    constPredictor_dic = {}
    type_dic = {}
    for f in files:
        lines = [line.rstrip('\n') for line in open(f)]
        data = lines[0].split(',')
        tunedRF_index = data.index('tunedrandomforest')
        constPredictor_index = data.index('constantpredictor')
        task_type_index = data.index('type')
        for l in lines[1:]:
            data = l.split(',')
            task = str(data[1]).lower()
            # fix inconsistant name 
            if task == 'bank-marketing':
                task = 'bank_marketing'
            if task == 'guiellermo':
                task = 'guillermo'
            tunedRF_dic[task] = float(data[tunedRF_index])
            constPredictor_dic[task] = float(data[constPredictor_index])
            type_dic[task] = str(data[task_type_index])
    return tunedRF_dic, constPredictor_dic, type_dic

def get_data_size(file_name):
    data_size_dic = {}
    print(file_name)
    data_file = open(file_name, 'r')
    for line in data_file.readlines()[1:]:
        data_info = line.split(',')
        data_name, data_size = str(data_info[0]), int(data_info[4]) 
        data_name = data_name.lower()
        data_size_dic[data_name] = data_size
    return data_size_dic

def get_res_mean_std(task_dic, framework_dic, task, framework, res_mean, res_std):
    if task not in task_dic:
        task_dic[task] = {}
    if framework not in task_dic[task]:
        task_dic[task][framework] = (res_mean, res_std)
    else:
        old_new_mean = [task_dic[task][framework][0], res_mean]
        old_new_std = [task_dic[task][framework][1], res_std]
        better_index = np.asarray(old_new_mean).argmax()
        task_dic[task][framework] = (old_new_mean[better_index], old_new_std[better_index])
        
    if framework not in framework_dic:
        framework_dic[framework] ={}
    if task not in framework_dic[framework]:
        framework_dic[framework][task] = (res_mean, res_std)
    else:
        old_new_mean = [framework_dic[framework][task][0], res_mean]
        old_new_std = [framework_dic[framework][task][1], res_std]
        better_index = np.asarray(old_new_mean).argmax()
        framework_dic[framework][task] = (old_new_mean[better_index], old_new_std[better_index])
    return task_dic, framework_dic


def get_res_mean_std_all(framework_dic_all, task, framework, res_mean, res_std):
    if framework not in framework_dic_all:
        framework_dic_all[framework] ={}
    if task not in framework_dic_all[framework]:
        framework_dic_all[framework][task] = []
    framework_dic_all[framework][task].append((res_mean, res_std))
    return framework_dic_all

def get_task_all_org_score_list(framework_dic_all, task):
    res_mean_all_list = []
    res_std_all_list = []
    for framework in framework_dic_all.keys():
        if task in framework_dic_all[framework].keys():
            l = len(framework_dic_all[framework][task])
            res_mean = [float(framework_dic_all[framework][task][i][0]) for i in range(l)]  
            res_std = [float(framework_dic_all[framework][task][i][1]) for i in range(l)]  
            res_mean_all_list = res_mean_all_list + res_mean
            res_std_all_list = res_std_all_list + res_std
    return res_mean_all_list, res_std_all_list




def get_task_min_max(task_tunedRF_dic, best_bench_dic, task_constPredictor_dic, task, normalize_type, res_mean_all_list):
    if len(res_mean_all_list)!=0:
        if task in task_tunedRF_dic.keys():
            if 'RF' in normalize_type:
                max_score = task_tunedRF_dic[task]
            else:
                max_score = best_bench_dic[task]
            min_score = task_constPredictor_dic[task]
        else:
            # max_score = max(res_mean_all_list)
            # min_score = min(res_mean_all_list)
            max_score =1.0
            min_score = 0.0 
    return max_score, min_score

def get_task_org_score_list(framework_dic, task ):
    res_mean_list = []
    res_std_list = []
    for framework in  framework_dic.keys():
        if task in framework_dic[framework].keys():
            res_mean = float(framework_dic[framework][task][0])
            res_std = float(framework_dic[framework][task][1])
            res_mean_list.append(res_mean )
            res_std_list.append(res_std)
    return res_mean_list, res_std_list



def get_task_min_max_dic(task_list, framework_dic_all, task_tunedRF_dic, best_bench_dic, task_constPredictor_dic,  normalize_type):
    task_max,task_min = {}, {}
    for task in task_list:
        res_mean_all_list, res_std_all_list = get_task_all_org_score_list(framework_dic_all, task)
        # res_mean_list, res_std_list = get_task_org_score_list(framework_dic, task )
        task_max_score, task_min_score = get_task_min_max(task_tunedRF_dic, best_bench_dic, task_constPredictor_dic,  task, normalize_type, res_mean_all_list)
        task_max[task], task_min[task] = task_max_score, task_min_score
    return task_max, task_min

def get_best_Predictor(file_dir):
    files = glob.glob(file_dir + '/*/result_summary.csv')
    # print('files', files)
    best_dic = {}
    best_name_dic = {}
    for f in files:
        lines = [line.rstrip('\n') for line in open(f)]
        for l in lines[1:]:
            data = l.split(',')
            type_, task = str(data[0]).lower(), str(data[1]).lower()
            if type_ == 'binary':
                best_func = max
            else:
                best_func = min
            constPredictor_res = float(data[2].strip('()'))
            performance_list = []
            for i in range(2,9):
                try:
                    res = float(data[i].strip('()'))
                except:
                    res = constPredictor_res
                performance_list.append(res)
            best_res = best_func(performance_list)
            best_res_name = np.argmax(np.array(performance_list))
            if task not in best_dic:
                best_dic[task] = best_res
                best_name_dic[task] = best_res_name
            else:
                if type_ == 'binary':
                    best_dic[task] = best_func(best_dic[task], best_res)
                else:
                    best_dic[task] = best_func(best_dic[task], best_res)
    return best_dic

def scale_score(org_score, org_score_std, max_score, min_score):
    max_min_diff = float(max_score - min_score)
    if max_min_diff !=0:
        scaled_score = (org_score - min_score)/max_min_diff
        scaled_std = org_score_std/max_min_diff
    else:
        scaled_score = 0
        scaled_std = org_score_std/float(org_score)
    return scaled_score, scaled_std

def remove_nonplot_tasks(task_list, plot_task_list):
    remove_task_list = []
    for task in task_list:
        task = task.lower()
        if task not in plot_task_list:
            remove_task_list.append(task)
    for t in remove_task_list:
        if t in task_list:
            task_list.remove(t)
    return task_list


