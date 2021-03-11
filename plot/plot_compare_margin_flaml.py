import argparse
import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from plot_res_flaml import get_tunedRF_constPredictor, get_data_size, \
    get_res_mean_std, get_best_Predictor, remove_nonplot_tasks, scale_score, \
    get_res_mean_std_all, get_task_min_max_dic
from plot_radar_flaml import get_framework_alias
from util import data_stat_file

SMALL_LARGE_threshold = 200000

def main():
    task_tunedRF_dic, task_constPredictor_dic, task_type_dic = get_tunedRF_constPredictor('../results/benchmark_results/')
    best_bench_dic = get_best_Predictor('../results/benchmark_results/')                 
    data_size_dic = get_data_size(data_stat_file)
    
    multi_class_list = []

    dataset_list_reg = []
    dataset_list_class_binary = []
    dataset_list_class_multi = []

    dataset_list_small = []

    #data_size_dic includes all the datasets we tested
    for data_name, data_size in data_size_dic.items(): 
        if data_name in task_type_dic.keys():
            if task_type_dic[data_name] == 'multiclass':
                multi_class_list.append(data_name)
        else:
            task_type_dic[data_name] = 'regression'

        if data_size < SMALL_LARGE_threshold:
            dataset_list_small.append(data_name)
        else:
            if task_type_dic[data_name] == 'regression':
                dataset_list_reg.append(data_name)
            elif task_type_dic[data_name] == 'binary':
                dataset_list_class_binary.append(data_name)
            else:
                dataset_list_class_multi.append(data_name)

    multi_class_list = [t.lower() for t in multi_class_list]
    dataset_list_reg = [t.lower() for t in dataset_list_reg]
    dataset_list_class_binary = [t.lower() for t in dataset_list_class_binary]
    dataset_list_class_multi = [t.lower() for t in dataset_list_class_multi]
    dataset_list_small = [t.lower() for t in dataset_list_small]
    # print(len(multi_class_list), len(dataset_list_class_binary))
    dataset_list_dic = {}
    dataset_list_dic['large_reg'] = [i.lower() for i in dataset_list_reg] 
    dataset_list_dic['large_class_binary'] = [i.lower() for i in dataset_list_class_binary]  
    dataset_list_dic['large_class_multi'] = [i.lower() for i in dataset_list_class_multi]  
    dataset_list_dic['small'] = [i.lower() for i in dataset_list_small]  

    dataset_list_dic['all_class'] = list(task_tunedRF_dic.keys())
    dataset_list_dic['all_reg'] = list(set(data_size_dic.keys()) - set(task_tunedRF_dic.keys()))
    dataset_list_dic['all'] = list(data_size_dic.keys())
    worse_than_tunedRF_num = {}
    total_classification_num = len(task_tunedRF_dic)
    #get tunedRF and constantPredictor results from benchmark results
    
    
    for dataset_name in multi_class_list:
        task_tunedRF_dic[dataset_name] = -task_tunedRF_dic[dataset_name] 
        task_constPredictor_dic[dataset_name] = -task_constPredictor_dic[dataset_name]

    parser = argparse.ArgumentParser()
    parser.add_argument('-t1', '--time_1', metavar='t1', type = float, default=None,
                        help="time_budget_1")
    parser.add_argument('-f1', '--framework_1', metavar='framework_id_1',  default=None,
                        help="The specific framework name (as defined in the benchmark file) to run. "
                            "If not provided, then all tasks from the benchmark will be run.")
    parser.add_argument('-flist', '--flist', dest='flist', nargs='*' , 
        default= None, help="The framework list") #'BOHB',
    parser.add_argument('-t2', '--time_2', metavar='time_2', type = float, default=None,
                        help="time_budget_1")
    parser.add_argument('-f2', '--framework_2', metavar='framework_id_2', default=None,
                        help="The specific framework name (as defined in the benchmark file) to run. "
                            "If not provided, then all tasks from the benchmark will be run.")

    parser.add_argument('-e', '--estimator', metavar='estimator', type = str, default=None,
                        help="The specific estimator name.")
    parser.add_argument('-d', '--dataset_type', metavar='dataset_type',  default=None,
                        help="The specific dataset name (as defined in the benchmark file) to run. ")
    parser.add_argument('-tolerance_ratio', '--tolerance_ratio', metavar='tolerance_ratio',  default=0.0,
                        help="tolerance_ratio when comparing two methods")
    parser.add_argument('-file', '--file_address', metavar='file_address',  default=None,
                        help="result file address. ")
    parser.add_argument('-n', '--normalize_type', metavar='normalize_type',  default='tunedRF',
                        help="The specific normalization type to use. ")
    parser.add_argument('-show_margin', dest='show_margin', action='store_true')
    parser.add_argument('-show_score', dest='show_score', action='store_true')
    parser.add_argument('-intersect', dest='intersect', action='store_true')

    args = parser.parse_args()

    normalize_type = str(args.normalize_type)
    filename = str(args.file_address)

    lines = [line.rstrip('\n') for line in open(filename)]
    tolerance_ratio = float(args.tolerance_ratio)
    markder_list = ['o', 's', 'x', '>', '<', 'h', 'p', 'x', 'v', '^', '1', '2', '>', '<', 'o', 's', 'x', 'v', 'h', 's']

    if args.show_margin:
        plot_dataset_type = str(args.dataset_type)
        framework_1 = args.framework_1
        time_1 = int(args.time_1)
        time_2 = int(args.time_2)
        task_dic = {}
        framework_dic = {}

        task_dic_2 = {}
        framework_dic_2 = {}

        task_dic_all = {}
        framework_dic_all = {}

        for line in lines[1:]:
            info = line.split(',')
            if len(info) ==1:
                info = line.split(' ')
            try:
                _, openmlid, task, framework, res_mean, res_std, duration = info
            except:
                continue
            framework = framework.replace('_from_log', '')
            if 'TPOT' in framework and 'xgboost' in framework:
                framework = 'TPOT_xgboost'
            task= task.lower()
            task = task.replace('_full', '')
            if task not in best_bench_dic.keys() and normalize_type == 'best':
                continue
            if task in multi_class_list:
                res_mean = -float(res_mean)
            try:
                duration = float(duration)
                framework_dic_all = get_res_mean_std_all(framework_dic_all, task, framework, res_mean, res_std)
                # task_dic_all, framework_dic_all = get_res_mean_std(task_dic_all, framework_dic_all, task, framework, res_mean, res_std)
                if float(duration) <= 1.3*float(args.time_1):
                    # print('duration', duration, 1.3*float(args.time))
                    task_dic, framework_dic = get_res_mean_std(task_dic, framework_dic, task, framework, res_mean, res_std)

                if float(duration) <= 1.3*float(args.time_2):
                    task_dic_2, framework_dic_2 = get_res_mean_std(task_dic_2, framework_dic_2, task, framework, res_mean, res_std)
            except:
                pass
        task_list = sorted(list(task_dic.keys()))
        plot_dataset_type = str(args.dataset_type)
        task_list = remove_nonplot_tasks(task_list, dataset_list_dic[plot_dataset_type])
        for t in dataset_list_dic[plot_dataset_type]:
            if t not in task_list:
                print('missing task', t)

        framework_2 = args.framework_2
        better_equ_num = 0
        better_num = 0
        worse_num = 0
        better_margin_list_dic = {'regression': [], 'binary': [], 'multiclass': []}
        worse_marin_list_dic = {'regression': [], 'binary': [], 'multiclass': []}
        better_margin_list = []
        worse_marin_list = []
        total_num = 0
        worse_than_task_list = []

        #get normalized score
        task_max, task_min = get_task_min_max_dic(task_list, framework_dic_all, task_tunedRF_dic, best_bench_dic, task_constPredictor_dic,  normalize_type)

        framework_dic_scaled = {}
        framework_dic_2_scaled = {}
        for task in task_list:
            mean_max = task_max[task]
            mean_min = task_min[task]

            if task in framework_dic[framework_1].keys():
                # res_mean_1 = float(framework_dic[framework_1][task][0])
                # res_std_1 = float(framework_dic[framework_1][task][1])
                org_score, org_score_std = float(framework_dic[framework_1][task][0]), float(framework_dic[framework_1][task][1])
                max_score, min_score = mean_max, mean_min
                res_mean_1, res_std_1 = scale_score(org_score, org_score_std, max_score, min_score)
            else:
                if args.intersect:
                    continue
                res_mean_1, res_std_1 = 0,0
            if task in framework_dic_2[framework_2].keys():
                    # print(framework_dic[framework][task])
                    # res_mean_2 = float(framework_dic_2[framework_2][task][0])
                    # res_std_2 = float(framework_dic_2[framework_2][task][1])
                    # print(task, res_mean_1, res_mean_2)

                    org_score, org_score_std = float(framework_dic_2[framework_2][task][0]), float(framework_dic_2[framework_2][task][1])
                    res_mean_2, res_std_2 = scale_score(org_score, org_score_std, mean_max, mean_min)
            else:
                if args.intersect: continue
                res_mean_2,res_std_2 = 0,0
            total_num +=1
            if res_mean_1 >= (1-tolerance_ratio)*res_mean_2:
                better_equ_num +=1

            margin = abs(res_mean_1 - res_mean_2)
            # margin_ratio = res_mean_1/float(res_mean_2)
            # margin = margin_ratio-1.0
            task_type = task_type_dic[task]
            if res_mean_1 > (1 + tolerance_ratio)*res_mean_2:
                # print('better', )
                better_num +=1
                better_margin_list.append(margin)
                better_margin_list_dic[task_type].append(margin)
            elif res_mean_1 < (1-tolerance_ratio)*res_mean_2:
                worse_num +=1
                worse_marin_list.append(margin)
                worse_than_task_list.append(task)
                worse_marin_list_dic[task_type].append(margin)
        winning_message = 'No winning'
        if len(better_margin_list)!=0:
            median = statistics.median(better_margin_list)
            min_ = min(better_margin_list)
            max_ = max(better_margin_list)
            m_25 = np.percentile(better_margin_list, 25)
            m_75 = np.percentile(better_margin_list, 75)  
            winning_message =   str("+%.2f" % median) + ' & ' + str( "+%.2f" % m_75) + ' & ' + str("+%.2f" % max_)  
            # print('Wining!!!!!',   "%.2f" % median,  "%.2f" % m_75,  "%.2f" % max_)
            # print('Wining!!!!!',   "%.2f" % m_25,  "%.2f" % median,  "%.2f" % m_75)

        lossing_message = '* & * & *'
        if len(worse_marin_list) !=0:
            lose_median = statistics.median(worse_marin_list)
            lose_min = min(worse_marin_list)
            lose_max = max(worse_marin_list)
            lose_m_25 = np.percentile(worse_marin_list, 25)
            lose_m_75 = np.percentile(worse_marin_list, 75) 
            lossing_message =  str("-%.2f" % lose_median) + ' & ' + str( "-%.2f" % lose_m_75) + ' & ' + str("-%.2f" % lose_max)  
            # print('Lossing!!!', "%.2f" % lose_median, "%.2f" % lose_m_75,  "%.2f" % lose_max)
            # print(worse_marin_list)
            # print('Lossing!!!',  "%.2f" % lose_m_25, "%.2f" % lose_median, "%.2f" % lose_m_75)
        win_lose_margin_message = winning_message + lossing_message
        # print(win_lose_margin_message)

        total_num = float(total_num)
        avg_better_margin_dic = {}
        avg_worse_margin_dic ={}
        for k, v in better_margin_list_dic.items():
            if len(v)!=0:
                # print(v)
                avg_better_margin_dic[k] = np.mean(np.array(v))
                std = np.std(np.array(v))
                # print(k, avg_better_margin_dic[k], std)
            else:
                avg_better_margin_dic[k] = 'NO'
        for k, v in worse_marin_list_dic.items():
            if len(v)!=0:
                avg_worse_margin_dic[k] = np.mean(np.array(v))
            else:
                avg_worse_margin_dic[k] = 'NO'
        better_eq_ratio, better_ratio, worse_ratio =  "{0:.0f}\%".format(100*better_equ_num/total_num), "{0:.0f}\%".format(100*better_num/total_num), "{0:.0f}\%".format(100*worse_num/total_num)
        # print( 'better or equal:', better_equ_num/total_num)
        # print( 'better:', better_num/total_num)
        # print( 'worse:', worse_num/total_num)

        framework_2 = get_framework_alias(framework_2)
        compare_margin_message = ' & '+str(framework_2.ljust(30)) + ' & '  +   str(better_ratio) + ' & ' + winning_message + ' & ' + str(worse_ratio) + ' & ' + lossing_message+ ' & ' + better_eq_ratio + r' \\'
        print(compare_margin_message)
        #print('-'*100)
    elif args.show_score:
        plot_dataset_type = str(args.dataset_type)
        for t in [60,600,3600]:
            alias = str(t)         
            # for framework_1 in ['flaml','lightautoml', 'lightautoml-reallynoblend']:
            for framework_1 in ['flaml','flaml_old', ]:
                task_dic = {}
                framework_dic = {}
                task_dic_all = {}
                framework_dic_all = {}
                for line in lines[1:]:
                    info = line.split(',')
                    if len(info) ==1:
                        info = line.split(' ')
                    try:
                        _, openmlid, task, framework, res_mean, res_std, duration = info
                    except:
                        continue

                    if len(info) == 7:
                        _, openmlid, task, framework, res_mean, res_std, duration = info
                    elif len(info) == 8:
                        _, openmlid, task, framework, _, res_mean, res_std, duration = info
                    print('framework', framework)
                    framework = framework.replace('_from_log', '')
                    if 'TPOT' in framework and 'xgboost' in framework:
                        framework = 'TPOT_xgboost'
                    task= task.lower()
                    task = task.replace('_full', '')
                    if task not in best_bench_dic.keys() and normalize_type == 'best':
                        continue
                    if task in multi_class_list:
                        res_mean = -float(res_mean)
                    try:
                        duration = float(duration)
                        framework_dic_all = get_res_mean_std_all(framework_dic_all, task, framework, res_mean, res_std)
                        # task_dic_all, framework_dic_all = get_res_mean_std(task_dic_all, framework_dic_all, task, framework, res_mean, res_std)
                        if float(duration) <= 1.3*float(t):
                            task_dic, framework_dic = get_res_mean_std(task_dic, framework_dic, task, framework, res_mean, res_std)

                    except:
                        pass
                task_list = sorted(list(task_dic.keys()))
                task_list = remove_nonplot_tasks(task_list, dataset_list_dic[plot_dataset_type])

                task_max, task_min = get_task_min_max_dic(task_list, framework_dic_all, task_tunedRF_dic, best_bench_dic, task_constPredictor_dic,  normalize_type)
                s,total_num = 0,0
                slist = []
                vlist = []
                framework_dic_scaled = {}
                if framework_1 in framework_dic:
                    for task in task_list:
                        mean_max = task_max[task]
                        mean_min = task_min[task]
                        total_num +=1
                        if task in framework_dic[framework_1].keys():
                            # res_mean_1 = float(framework_dic[framework_1][task][0])
                            # res_std_1 = float(framework_dic[framework_1][task][1])
                            org_score, org_score_std = float(framework_dic[framework_1][task][0]), float(framework_dic[framework_1][task][1])
                            max_score, min_score = mean_max, mean_min
                            res_mean_1, res_std_1 = scale_score(org_score, org_score_std, max_score, min_score)
                        else:
                            res_mean_1, res_std_1 = 0,0
                        s+=res_mean_1
                        slist.append(res_mean_1)
                        if res_std_1>0:
                            vlist.append(res_std_1)
                if total_num==0: total_num=1
                alias += ' & {0:.2f}'.format(np.mean(slist))
                #alias += r'\pm {0:.2f}'.format(np.mean(vlist))

                #alias += r' & {0:.2f}\pm {1:.2f}'.format(np.mean(slist),np.std(slist))
            print(alias+r'\\')
            
    else:
        compared_budget = [(60,60),(600,600),(3600,3600),(60,600),(600,3600),(60,3600)]
        labels = []
        framework_1 = args.framework_1
        flist = args.flist
        margin = [[[] for method in flist] for t in compared_budget]
        for i, framework_2 in enumerate(flist):
            plot_dataset_type = str(args.dataset_type) if framework_2!='autosklearn_xgboost' else 'all_reg'
            alias = get_framework_alias(framework_2)    
            labels.append(alias)  
            for j, t in enumerate(compared_budget):
                time_1,time_2 = t
                task_dic = {}
                framework_dic = {}

                task_dic_2 = {}
                framework_dic_2 = {}

                task_dic_all = {}
                framework_dic_all = {}
                for line in lines[1:]:
                    info = line.split(',')
                    if len(info) ==1:
                        info = line.split(' ')
                    # try:
                    #     _, openmlid, task, framework, _, res_mean, res_std, duration = info
                    # except:
                    #     continue
                    if len(info) == 7:
                        _, openmlid, task, framework, res_mean, res_std, duration = info
                    elif len(info) == 8:
                        _, openmlid, task, framework, _, res_mean, res_std, duration = info
                    print('framework', framework)
                    framework = framework.replace('_from_log', '')
                    task= task.lower()
                    task = task.replace('_full', '')
                    if task not in best_bench_dic.keys() and normalize_type == 'best':
                        continue
                    if task in multi_class_list:
                        res_mean = -float(res_mean)
                    try:
                        duration = float(duration)
                        framework_dic_all = get_res_mean_std_all(framework_dic_all, task, framework, res_mean, res_std)
                        if float(duration) <= 1.3*float(time_1):
                            task_dic, framework_dic = get_res_mean_std(task_dic, framework_dic, task, framework, res_mean, res_std)

                        if float(duration) <= 1.3*float(time_2):
                            task_dic_2, framework_dic_2 = get_res_mean_std(task_dic_2, framework_dic_2, task, framework, res_mean, res_std)
                    except:
                        pass
                # print(list(framework_dic.keys()))
                if not framework_2 in framework_dic_2: continue
                if not framework_1 in framework_dic: continue
                task_list = sorted(list(task_dic.keys()))
                task_list = remove_nonplot_tasks(task_list, dataset_list_dic[plot_dataset_type])
                for t in dataset_list_dic[plot_dataset_type]:
                    if t not in task_list:
                        print('missing task', t)

                better_equ_num = 0
                better_num = 0
                worse_num = 0
                total_num = 0
                worse_than_task_list = []

                task_max, task_min = get_task_min_max_dic(task_list, framework_dic_all, task_tunedRF_dic, best_bench_dic, task_constPredictor_dic,  normalize_type)

                framework_dic_scaled = {}
                framework_dic_2_scaled = {}
                for task in task_list:
                    mean_max = task_max[task]
                    mean_min = task_min[task]
                    total_num +=1
                    if task in framework_dic[framework_1].keys():
                        org_score, org_score_std = float(framework_dic[framework_1][task][0]), float(framework_dic[framework_1][task][1])
                        max_score, min_score = mean_max, mean_min
                        res_mean_1, res_std_1 = scale_score(org_score, org_score_std, max_score, min_score)
                    else:
                        res_mean_1, res_std_1 = 0,0
                    if task in framework_dic_2[framework_2].keys():
                            org_score, org_score_std = float(framework_dic_2[framework_2][task][0]), float(framework_dic_2[framework_2][task][1])
                            res_mean_2, res_std_2 = scale_score(org_score, org_score_std, mean_max, mean_min)
                    else:
                        res_mean_2,res_std_2 = 0,0

                    if res_mean_1 >= (1-tolerance_ratio)*res_mean_2:
                                better_equ_num +=1
                    margin[j][i].append(res_mean_1-res_mean_2)
                    if res_mean_1-res_mean_2<-0.05:
                        print(task, compared_budget[j], framework_1,
                            framework_2, res_mean_1-res_mean_2)
                better_eq_ratio, better_ratio, worse_ratio =  "{0:.0f}\%".format(100*better_equ_num/total_num), "{0:.0f}\%".format(100*better_num/total_num), "{0:.0f}\%".format(100*worse_num/total_num)
                alias += ' & '+better_eq_ratio
            print(alias+r'\\')
        matplotlib.rcParams.update({'font.size': 18})
        title = ['60s vs. 60s', '600s vs. 600s', '3600s vs. 3600s', '60s vs. 600s', '600s vs. 3600s', '60s vs. 3600s']
        num_methods = len(labels)
        print(labels)
        labels = flist[::-1]
        for j,t in enumerate(compared_budget):
            if j and j!=3 and num_methods>3:
                fig,ax = plt.subplots(1,1,figsize=(5,3))
                fig.subplots_adjust(left=0.05, right=0.95, top=0.9)
                ax.boxplot(margin[j][::-1], labels = ['']*num_methods,
                    vert=False)
            else:
                fig,ax = plt.subplots(1,1,figsize=(6.5,3))
                fig.subplots_adjust(left=0.25, right=0.95, top=0.9)
                ax.boxplot(margin[j][::-1], labels = labels, vert=False)
            plt.plot([0]*(num_methods+1), [i+0.5 for i in range(num_methods+1)], markersize=18, label = 'FLAML',
                linewidth=3, linestyle = 'dashed')
            for label in ax.get_yticklabels():
                label.set_fontsize('large')
            plt.title(title[j])
            plt.xlim(-.21,2.5)
            fig.savefig(f'./plots/flaml/drilldown_{t}.pdf')
            try:
                print(title[j], max(max(x) for x in margin[j] if x),
                min(min(x) for x in margin[j] if x))
            except:
                pass

if __name__ == '__main__':
    main()
