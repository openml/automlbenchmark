import argparse
import numpy as np
import ast 
import json
import matplotlib
import matplotlib.pyplot as plt
from plot_res_flaml import get_framework_alias
from plot_res import convert_loss_into_score, get_tunedRF_constPredictor
import os.path

def get_const_predict_loss(dataset, file_dir, convert_to_score = False):
    dataset=dataset.lower()
    is_multi_class = False
    try:
        _, constPredictor_dic, task_type_dic = get_tunedRF_constPredictor(file_dir)
        if dataset in constPredictor_dic:
            task_type = task_type_dic[dataset]
            if 'multi' in task_type:
                is_multi_class = True
            if is_multi_class:
                const_predict_loss = constPredictor_dic[dataset] 
            else:
                const_predict_loss = 1.0 - constPredictor_dic[dataset] 
        else:
            const_predict_loss = 1.0  #float('+inf')
    except:
        const_predict_loss = 1.0
    if convert_to_score:
        score = convert_loss_into_score(const_predict_loss, is_multi_class)
        return score, is_multi_class
    return const_predict_loss

def get_tunedRF_predict_loss(dataset, file_dir, convert_to_score = False):
    dataset=dataset.lower()
    is_multi_class = False
    try:
        tunedRF_losss_dic, __, task_type_dic = get_tunedRF_constPredictor(file_dir)

        if dataset in tunedRF_losss_dic:
            task_type = task_type_dic[dataset]
            if 'multi' in task_type:
                is_multi_class = True
            if is_multi_class:
                tunedRF_predict_loss = tunedRF_losss_dic[dataset]
            else:
                tunedRF_predict_loss = 1.0 - tunedRF_losss_dic[dataset]  #result_summary is reporting scores (1.0-loss) for binary classification
        else:
            tunedRF_predict_loss = 0.0 #float('+inf')
    except:
        tunedRF_predict_loss = 0.0
    if convert_to_score:
        score = convert_loss_into_score(tunedRF_predict_loss, is_multi_class)
        # print('score',score)
        return score, is_multi_class
    return tunedRF_predict_loss

def retrain_from_log(method_name, filename, time_budget):
    method_name = method_name.lower()
    file_exist = os.path.exists(filename)
    #print(filename, file_exist)
    if not file_exist:
        from_log_method = [], []
    else:
        if 'flaml' in method_name:
            from_log_method = retrain_from_log_FLAML(filename, time_budget)   
    
    return from_log_method

def retrain_from_log_FLAML(filename, time_budget):
    best_config = None
    best_val_loss = float('+inf')
    training_duration = 0.0

    training_time_list = []
    best_error_list = []
    best_config_list = []
    with open(filename) as file_:
        for line in file_:
            data = line.split('\t')
            if data[0][0]=='b': continue
            time_used = float(data[4])
            training_duration = time_used
            val_loss = float(data[3])
            config = ast.literal_eval(data[5])
            if time_used <= time_budget:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_config = config
                    training_time_list.append(training_duration)
                    best_error_list.append(best_val_loss)
                    best_config_list.append(best_config)
            else:
                training_time_list.append(time_budget)
                best_error_list.append(best_val_loss)
                best_config_list.append(best_config)
                break
    return training_time_list, best_error_list

def any_second_performance(training_time_list, best_error_list, time_budget, const_predict_loss):
    time_budget = int(time_budget)
    training_time_list = [0] + training_time_list
    # best_error_list = [best_error_list[0]] + best_error_list
    best_error_list = [min(const_predict_loss, i) for i in  best_error_list]
    best_error_list = [const_predict_loss] + best_error_list
    #print(training_time_list[:5],best_error_list[0:5])
    
    index = 1
    training_time_list_second = []
    best_error_list_second = []
    #print(training_time_list)
    pre_time = 0 
    for t in range(1, len(training_time_list) +1):
        #pre_time = int(training_time_list[index-1])
        #if pre_time>=time_budget: break
        if t == len(training_time_list):
            cur_time = time_budget
        else:
            cur_time = int(training_time_list[index])+1
            if cur_time>time_budget:
                cur_time=time_budget
        pre_error = float(best_error_list[index-1])
        training_time_list_second = training_time_list_second + list(range(pre_time, cur_time))
        best_error_list_second = best_error_list_second + [pre_error]*len(range(pre_time, cur_time))
        index +=1
        pre_time=cur_time
        if cur_time==time_budget: break
    #print(len(training_time_list_second))
    if len(training_time_list_second) !=0:
        if training_time_list_second[-1] < time_budget-1:
            missing_list=  range(int(training_time_list_second[-1] +1), time_budget)
            training_time_list_second = training_time_list_second + list(missing_list)
            best_error_list_second = best_error_list_second + [best_error_list_second[-1]]*len(missing_list)
    # print(training_time_list_second[:2],best_error_list_second[:2])
    training_time_list_second[0]=0.01
    return training_time_list_second, best_error_list_second

def start_from_init(training_time_list, best_error_list, first_time, first_error):
    l = len(training_time_list)
    l2 = len(best_error_list)
    #print(l,len(best_error_list))
    for i in range(0,l):
        training_time_list[i]+=first_time
        if i<l2 and best_error_list[i]>first_error:
            best_error_list[i]=first_error
    training_time_list.insert(0,first_time)
    best_error_list.insert(0,first_error)
    #print(training_time_list[:3],training_time_list[-2:])
    return training_time_list,best_error_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-time', '--time', metavar='time', type = float, default=None,
                        help="time_budget")
    parser.add_argument('-f', '--fold_num', metavar='fold_num', nargs='*' , default= range(10), help="The fold number")
    parser.add_argument('-m', '--method_list', dest='namelist', nargs='*' , default= [], help="The method list")
    parser.add_argument('-l', '--legend_loc', dest='loc',default='lower left')
    parser.add_argument('-d', '--dataset', metavar='dataset', nargs='*', default=None,
                    help="The specific dataset name.")
    parser.add_argument('-file', '--file_address', metavar='file_address',  default=None,
                        help="result file address. ")
    args = parser.parse_args()
    loc=args.loc.replace('_',' ')
    filename_address = str(args.file_address)
    time_budget = float(args.time)
    name_list = args.namelist
    datasets = args.dataset
            
    fold_num_list = args.fold_num
    linestyles = ['dotted','dashed','dashdot',(0, (1, 1))]
    markers = ['o','v','s','*','^']
    matplotlib.rcParams.update({'font.size': 18})
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    colors=['blue','black','green','red']
    for dataset in datasets:
        print(dataset)
        fig,ax = plt.subplots(1,1,figsize=(8,8))
        fig.subplots_adjust(left=0.25, right=0.95, top=0.95)
        first_time = []
        first_error = []
        min_loss=-np.inf
        names = name_list[-1]
        log_suffix = '.log'
        for fold_num in fold_num_list:
            fold_num = str(fold_num)  
            log_file_name = names + dataset + '_' + fold_num  + log_suffix
            filename = filename_address + log_file_name
            training_time_list, best_error_list = retrain_from_log(names, filename, time_budget)
            print(best_error_list)
            first_time.append(training_time_list[0] if len(training_time_list)>0 else 0)
            first_error.append(best_error_list[0] if len(best_error_list)>0 else np.inf)
        #print(first_time)
        min_loss=first_error[0]
        const_predict_loss = get_const_predict_loss(dataset, file_dir = '../results/benchmark_results/')
        #print(const_predict_loss)
        print(min_loss)
        max_loss = -np.Inf
        for j,names in enumerate(name_list):
            v = []
            #print(names)
            if 'flaml' in names.lower():
                log_suffix = '.log'
            else:
                log_suffix = ''
            index_ = 0
            starting_time_list = []
            end_time_list = []
            max_loss_list = []
            min_loss_list = []
            training_time_list_second_list = []
            best_error_list_second_list =[]
            min_time = time_budget
            for i,fold_num in enumerate(fold_num_list):
                #print(fold_num)
                fold_num = str(fold_num)  
                log_file_name = names + dataset + '_' + fold_num  + log_suffix
                filename = filename_address + log_file_name
                training_time_list, best_error_list = retrain_from_log(names, filename, time_budget)
                v.append(list(zip(training_time_list, [min(x,const_predict_loss) for x in best_error_list])))
                if not len(training_time_list):
                    v[-1].append((0, const_predict_loss))
                    v[-1].append((time_budget, const_predict_loss))
                else:
                    if training_time_list[-1]<time_budget:
                        v[-1].append((time_budget, min(best_error_list[-1],const_predict_loss)))
                    min_time = min(min_time, training_time_list[0])
                training_time_list_second, best_error_list_second = any_second_performance(training_time_list, best_error_list, time_budget, const_predict_loss )
                if len(training_time_list_second) == time_budget:
                    training_time_list_second_list.append(training_time_list_second)
                    best_error_list_second_list.append(best_error_list_second)
                index_ +=1
            for run in v:
                run.insert(0,(min_time/2,const_predict_loss))
                # print(run)
            ids = np.array([0 for run in v]) # the current index for each run
            x = np.array([run[0][0] for run in v]) # the current x for each run
            currentx = x.max()
            for i, run in enumerate(v):
                while x[i]<currentx and ids[i]<len(run)-1 and run[ids[i]+1
                    ][0]<currentx: 
                    ids[i]+=1
                    x[i]=run[ids[i]][0]            
            y = np.array([run[ids[i]][1] for i, run in enumerate(v)])
            xlist, reglist, lblist, ublist = [0], [const_predict_loss], [
                const_predict_loss], [const_predict_loss]
            while True:
                maxx = x.max()
                xlist.append(maxx)
                reglist.append(y.mean())
                lblist.append(y.min())
                ublist.append(y.max())
                nextx = np.Inf # the minimal next point
                for i, run in enumerate(v):
                    if ids[i]<len(run)-1 and run[ids[i]+1][0]<nextx:
                        nextx = run[ids[i]+1][0]
                if np.isinf(nextx): break
                for i, run in enumerate(v):
                    if ids[i]<len(run)-1 and run[ids[i]+1][0]==nextx:
                        ids[i]+=1
                        x[i]=nextx
                        y[i]=run[ids[i]][1]
            
            best_error_list_second_list = np.array(best_error_list_second_list)
            avg_error = best_error_list_second_list.mean(axis = 0)
            std_error = best_error_list_second_list.std(axis = 0)
            # max_error = best_error_list_second_list.max(axis = 0)
            min_error = best_error_list_second_list.min(axis = 0)
            # if min(min_error)<min_loss:
            #     print(names,best_error_list_second_list[:,-1])
            min_loss=min(min_loss,min(min_error))
            min_loss_list.append(avg_error[-1])
            max_loss_list.append(avg_error[0])
            #print(avg_error[0:5], min_error[0:5], std_error[0:5])
            framework_alias  = get_framework_alias(names)
            # plt.plot(training_time_list, best_error_list, marker = markder_list[index_] , label = framework_alias)
            # print(len(best_error_list_second), len(training_time_list_second))
            #print(training_time_list_second_list[0][:2],best_error_list_second_list[0][:2])
            if 'flaml_' == names.lower():
                plt.plot(xlist, reglist, linewidth=4, label = framework_alias, color='black')
            else:
                plt.plot(xlist, reglist, linewidth=5, label = framework_alias, linestyle=linestyles[j], marker=markers[j], markevery=(j*5+3,2791), markersize=18, color=colors[j])
            # plt.fill_between(training_time_list_second, avg_error- std_error, avg_error + std_error, alpha=0.4)
            plt.fill_between(xlist, lblist, ublist, alpha=0.4)
            if reglist[1]>max_loss: max_loss = reglist[1]

        plt.xlabel('Wall clock time [s]', fontsize=20)
        plt.ylabel('error', fontsize=20)
        plt.xscale('log')
        plt.yscale('log') 
        # plt.xlim(xmin=1)
        #plt.yscale('log')   
        #plt.ticklabel_format(axis='y',style='plain')
        if loc!='None':
            plt.legend(loc =loc, prop={'size': 16}, ncol = 1)
        # plt.ylim([0.12, 1.0])
        # min_time = min(starting_time_list)
        # print(starting_time_list, min_time )
        #max_loss = max(max_loss_list)
        xmin = None
        if len(first_error)>0:
            max_loss = min(const_predict_loss, max(first_error))
            if ax.get_ylim()[1]>max_loss: plt.ylim(ymax=max_loss)
            # plt.ylim(ymax=const_predict_loss)
            if xmin: plt.xlim(xmin=xmin)
        min_avg_loss = min(min_loss_list)  
        #min_loss = min(min_loss, -0.001)
        if ax.get_ylim()[0]<min_loss:
            plt.ylim(ymin=min_loss)
        #print(ax.get_ylim()[0])
        if ax.get_ylim()[0]<0.01:
            plt.ylim(ymin=ax.get_ylim()[0]-0.005)
            
        plt.xlim(xmax=time_budget)
        fig_name = 'LC_' + dataset
        if len(fold_num_list)==1:
            fig_name+= '_' + fold_num
        plt.savefig( './plots/flaml/' + fig_name + '.pdf')
        plt.close()
 
if __name__ == '__main__':
    main()
