import argparse
import operator
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from plot_res import get_tunedRF_constPredictor, get_data_size, get_res_mean_std, \
    get_best_Predictor, remove_nonplot_tasks, scale_score, get_res_mean_std_all, \
        get_task_min_max_dic, compare_with_benchmark_scaled, get_ratio_dic

SMALL_LARGE_threshold = 200000

def get_framework_alias(framework_name):
    if 'bohb' in framework_name.lower():
        framework_alias = 'HpBand.'
    elif 'tpot' in framework_name.lower():
        #else:
            framework_alias = 'TPOT'
    elif 'autosklearn' in framework_name.lower():
        framework_alias = 'Auto-sk.'
    elif 'h2o' in framework_name.lower():
        framework_alias = 'H2O'
    elif 'autogluon' in framework_name.lower():
        framework_alias = 'AutoGluon'
    else:
        framework_alias = framework_name.replace('ns','fulldata').rstrip('_')
    
    return framework_alias

def get_winning_num_dic_bak(task_dic, framework_list):
    print(framework_list)
    winning_num_dic = {}
    winning_ratio_dic = {}
    for task, task_res in task_dic.items():
        compared_task_res = {}
        for f, res in task_res.items():
            if f in framework_list:
                compared_task_res[f] = res
        winning_f = max(compared_task_res.items(), key=operator.itemgetter(1))[0]
        if winning_f not in winning_num_dic:
            winning_num_dic[winning_f] = 0
        winning_num_dic[winning_f] +=1
    task_num = len(task_dic)
    for f, num in winning_num_dic.items():
        winning_ratio_dic[f] = num/float(task_num)

    return winning_num_dic, winning_ratio_dic

def get_winning_num_dic(task_list, framework_dic_scaled, framework_list, tol):
    print(framework_list)
    winning_num_dic = {}
    winning_ratio_dic = {}
    for f in framework_list:
        if f not in winning_num_dic.keys():
            winning_num_dic[f] = 0
    for task in task_list:
        task_res_list = []
        task_winners = []
        for f in framework_list:
            if f in framework_dic_scaled.keys():
                if task in framework_dic_scaled[f].keys():
                    task_res_list.append(framework_dic_scaled[f][task])
        winner_index = np.argmax(task_res_list)
        winner_name = framework_list[winner_index]
        winner_res = framework_dic_scaled[winner_name][task]
        for i in range(len(task_res_list)):
            res = task_res_list[i]
            framework_name = framework_list[i]
            if res >= (1-tol)*winner_res:
                task_winners.append(framework_name)
        for f in task_winners:
            winning_num_dic[f] +=1
    task_num = len(task_list)
    for f in framework_list:
        if f not in winning_num_dic.keys():
            winning_num_dic[f] = 0
    for f, num in winning_num_dic.items():
        winning_ratio_dic[f] = num/float(task_num)

    return winning_num_dic, winning_ratio_dic

def attach_ordinal(num):
    """helper function to add ordinal string to integers

    1 -> 1%
    56 -> 56%
    """
    return str(num) + '%'

def format_score(scr,  test_meta):
    """
    Build up the score labels for the right Y-axis by first
    appending a carriage return to each string and then tacking on
    the appropriate meta information (i.e., 'laps' vs 'seconds'). We
    want the labels centered on the ticks, so if there is no meta
    info (like for pushups) then don't add the carriage return to
    the string
    """
    md = test_meta
    if md:
        return '{0}\n{1}'.format(scr, md)
    else:
        return scr

def plot_winner_results(scores, testNames, time_budget):
    #  create the figure
    if time_budget == 600:
        fig, ax1 = plt.subplots(figsize=(8.5, 6))
        fig.subplots_adjust(left=0.05, right=0.8)
    elif time_budget == 3600:
        fig, ax1 = plt.subplots(figsize=(9, 6))
        fig.subplots_adjust(left=0.05, right=0.75)
    else:
        fig, ax1 = plt.subplots(figsize=(10.5, 6))
        fig.subplots_adjust(left=0.23, right=0.85)
    fig.canvas.set_window_title('winners')
    pos = np.arange(len(testNames))
    if time_budget == 60:
        rects = ax1.barh(pos, [scores[k].percentile for k in testNames],
                        align='center',
                        height=0.5,
                        tick_label=testNames)
        for label in ax1.get_yticklabels():
            label.set_fontsize('xx-large')
    else:
        rects = ax1.barh(pos, [scores[k].percentile for k in testNames],
                        align='center',
                        height=0.5, tick_label=['' for x in testNames])

    title_name = str(int(time_budget)) + 's'
    ax1.set_title(title_name, fontsize='xx-large')
    ax1.set_xlim([0, 100])
    # ax1.xaxis.set_major_locator(MaxNLocator(11))
    ax1.xaxis.grid(True, linestyle='--', which='major',
                color='grey', alpha=.25)

    # Plot a solid vertical gridline to highlight the median position
    ax1.axvline(50, color='grey', alpha=0.25)

    # Set the right-hand Y-axis ticks and labels
    ax2 = ax1.twinx()

    # scoreLabels = [format_score(scores[k].score, k) for k in testNames]
    scoreLabels = [format_score(scores[k].score, '') for k in testNames]

    # set the tick locations
    ax2.set_yticks(pos)
    # make sure that the limits are set equally on both yaxis so the
    # ticks line up
    ax2.set_ylim(ax1.get_ylim())

    # set the tick labels
    ax2.set_yticklabels(scoreLabels,fontsize='xx-large')
    if time_budget==3600:
        ax2.set_ylabel('Scaled Scores',fontsize='xx-large')

    rect_labels = []
    # Lastly, write in the ranking inside each bar to aid in interpretation
    for rect in rects:
        # Rectangle widths are already integer-valued but are floating
        # type, so it helps to remove the trailing decimal point and 0 by
        # converting width to int type
        width = int(rect.get_width())

        rankStr = attach_ordinal(width)
        # The bars aren't wide enough to print the ranking inside
        if width < 40:
            # Shift the text to the right side of the right edge
            xloc = 5
            # Black against white background
            clr = 'black'
            align = 'left'
        else:
            # Shift the text to the left side of the right edge
            xloc = -5
            # White on magenta
            clr = 'white'
            align = 'right'

        # Center the text vertically in the bar
        yloc = rect.get_y() + rect.get_height() / 2
        label = ax1.annotate(rankStr, xy=(width, yloc), xytext=(xloc, 0),
                            textcoords="offset points",
                            ha=align, va='center',
                            color=clr, weight='bold', clip_on=True, size='xx-large')
        rect_labels.append(label)
    #plt.rcParams.update({'font.size': 24})

    # return all of the artists created
    return {'fig': fig,
            'ax': ax1,
            'ax_right': ax2,
            'bars': rects,
            'perc_labels': rect_labels}

def plot_winners(winning_ratio_dic, framework_dic_scaled, time_budget, framework_list_alias):
    np.random.seed(42)
    Score = namedtuple('Score', ['score', 'percentile'])
    # GLOBAL CONSTANTS
    testNames = framework_list_alias
    testNames = testNames[::-1]
    scores = {}
    figure_name = 'FLAML_winners_' + str(int(time_budget))
    for f, winning_per in winning_ratio_dic.items():
        alias_name =get_framework_alias(f)
        mean_res = np.mean(list(framework_dic_scaled[f].values()))
        scores[alias_name] = Score("%.2f" % mean_res, winning_per*100)
    plot_winner_results(scores, testNames, time_budget)
    plt.savefig( './plots/flaml/' + figure_name + '.pdf')

def main():
    task_tunedRF_dic, task_constPredictor_dic, task_type_dic = get_tunedRF_constPredictor('../results/benchmark_results/')
    best_bench_dic = get_best_Predictor('../results/benchmark_results/')                 
    data_size_dic = get_data_size('../results/data_stat.csv')
    
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

    dataset_list_dic = {}
    dataset_list_dic['large_reg'] = [i.lower() for i in dataset_list_reg] 
    dataset_list_dic['large_class_binary'] = [i.lower() for i in dataset_list_class_binary]  
    dataset_list_dic['large_class_multi'] = [i.lower() for i in dataset_list_class_multi]  
    dataset_list_dic['small'] = [i.lower() for i in dataset_list_small]  

    dataset_list_dic['all_class'] = list(task_tunedRF_dic.keys())
    dataset_list_dic['all_reg'] = list(set(data_size_dic.keys()) - set(task_tunedRF_dic.keys()))
    dataset_list_dic['all'] = list(dataset_list_dic['all_reg'] + dataset_list_dic['all_class'])
    #get tunedRF and constantPredictor results from benchmark results
    
    for dataset_name in multi_class_list:
        task_tunedRF_dic[dataset_name] = -task_tunedRF_dic[dataset_name] 
        task_constPredictor_dic[dataset_name] = -task_constPredictor_dic[dataset_name]

    parser = argparse.ArgumentParser()
    parser.add_argument('-time', '--time', metavar='time', type = float, default=None,
                        help="time_budget")
    parser.add_argument('-f', '--frameworks', metavar='framework_id', nargs='*', default=None,
                        help="The specific framework name (as defined in the benchmark file) to run. "
                            "If not provided, then all tasks from the benchmark will be run.")
    parser.add_argument('-e', '--estimator', metavar='estimator', type = str, default=None,
                        help="The specific estimator name.")
    parser.add_argument('-d', '--dataset_type', metavar='dataset_type',  default=None,
                        help="The specific dataset name (as defined in the benchmark file) to run. ")
    parser.add_argument('-file', '--file_address', metavar='file_address',  default=None,
                        help="result file address. ")
    parser.add_argument('-n', '--normalize_type', metavar='normalize_type',  default='tunedRF',
                        help="The specific normalization type to use. ")
    args = parser.parse_args()

    plot_dataset_type = str(args.dataset_type)
    normalize_type = str(args.normalize_type)

    filename = str(args.file_address)

    lines = [line.rstrip('\n') for line in open(filename)]
    task_dic = {}
    framework_dic = {}

    task_dic_all = {}
    framework_dic_all = {}


    for line in lines[1:]:
        info = line.split(',')
        if len(info) ==1:
            info = line.split(' ')
        try:
            _, openmlid, task, framework, _, res_mean, res_std, duration = info
        except:
            continue
        framework = framework.replace('_from_log', '')
        
        task= task.lower()

        if task not in best_bench_dic.keys() and normalize_type == 'best':
            continue
        if task in multi_class_list:
            res_mean = -float(res_mean)
        try:
            duration = float(duration)
            framework_dic_all = get_res_mean_std_all(framework_dic_all, task, framework, res_mean, res_std)
            if float(duration) <= 1.3*float(args.time):
                # print('duration', duration, 1.3*float(args.time))
                task_dic, framework_dic = get_res_mean_std(task_dic, framework_dic, task, framework, res_mean, res_std)
        except:
            pass

    markder_list = ['o', 's', 'x', '>', '<', 'h', 'p', 'x', 'v', '^', '1', '2', '>', '<', 'o', 's', 'x', 'v', 'h', 's']
    task_list = sorted(list(task_dic.keys()))
    plot_dataset_type = str(args.dataset_type)
    task_list = remove_nonplot_tasks(task_list, dataset_list_dic[plot_dataset_type])
    print(len(task_list),len(multi_class_list))

    index_ = 0
    org_framework_list = args.frameworks
    #revise TPOT name
    framework_list = []
    framework_dic_scaled ={}
    for f in org_framework_list:
        name = f
        framework_list.append(name)
    all_possible_time_budget = [60,600,3600]
    time_budget = float(args.time)
    task_max, task_min = get_task_min_max_dic(task_list, framework_dic_all, task_tunedRF_dic, best_bench_dic, task_constPredictor_dic,  normalize_type)
    
    task_list_short = [task[:7] for task in task_list]
    #plt.figure(figsize=(20,10))
    N = len(task_list)
    aliases = []
    for framework in framework_list:
        res_mean_scaled_list = []
        res_std_scaled_list = []
        for task in task_list:
        # for task in task_min.keys():
            mean_max = task_max[task]
            mean_min = task_min[task]
            if framework in framework_dic.keys():
                if task in framework_dic[framework].keys():
                    org_score, org_score_std = float(framework_dic[framework][task][0]), float(framework_dic[framework][task][1])
                    max_score, min_score = mean_max, mean_min
                    res_mean_scaled, res_std_scaled = scale_score(org_score, org_score_std, max_score, min_score)
                else:
                    res_mean_scaled = 0.0
                    res_std_scaled = 0.0
            else:
                res_mean_scaled = 0.0
                res_std_scaled = 0.0
            res_mean_scaled_list.append(res_mean_scaled )
            res_std_scaled_list.append(res_std_scaled)
            if framework not in framework_dic_scaled:
                framework_dic_scaled[framework] = {}
            framework_dic_scaled[framework][task] = res_mean_scaled

        framework_alias = str(get_framework_alias(framework))
        aliases.append(framework_alias)
    worse_than_tunedRF_num, eq_worse_than_CP_num = compare_with_benchmark_scaled(framework_list,task_tunedRF_dic, task_list, framework_dic_scaled)
    worse_than_tunedRF_ratio, eq_worse_than_CP_ratio = get_ratio_dic(worse_than_tunedRF_num, eq_worse_than_CP_num, float(len(task_tunedRF_dic)))
    print('worse than RF ratio:', worse_than_tunedRF_ratio,len(task_tunedRF_dic))
    print('eq_worse_than_CP_ratio:', eq_worse_than_CP_ratio,len(task_tunedRF_dic))
    # plt.show()

    #Plot winning_num_dic
    # winning_num_dic, winning_ratio_dic = get_winning_num_dic(task_dic, framework_list)
    winning_num_dic, winning_ratio_dic = get_winning_num_dic(task_list, framework_dic_scaled, framework_list, tol = 0.001)

    # winning_num_dic_dic = {time_budget: winning_num_dic}
    plot_winners(winning_ratio_dic, framework_dic_scaled, time_budget, aliases)
    print(winning_num_dic)
    print(winning_ratio_dic)
 
if __name__ == '__main__':
    main()
