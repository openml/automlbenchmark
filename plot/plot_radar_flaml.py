import argparse
import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from plot_res import  get_res_mean_std_all, get_task_min_max_dic

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    def draw_poly_patch(self):
        # rotate theta such that the first axis is at the top
        verts = unit_poly_verts(theta + np.pi / 2)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def __init__(self, *args, **kwargs):
            super(RadarAxes, self).__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels,**kwargs):
            self.set_thetagrids(np.degrees(theta), labels, **kwargs)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta + np.pi / 2)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts

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

def get_framework_alias(framework_name):
    if 'bohb' in framework_name.lower():
        framework_alias = 'HpBandSter'
    elif 'tpot' in framework_name.lower():
        framework_alias = 'TPOT'
    elif 'autosklearn' in framework_name.lower():
        framework_alias = 'Auto-sklearn'
    elif 'h2o' in framework_name.lower():
        framework_alias = 'H2OAutoML'
    elif 'autogluon' in framework_name.lower():
        framework_alias = 'AutoGluon'
    else:
        framework_alias = framework_name

    return framework_alias

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

        if task_type_dic[data_name] == 'regression':
            dataset_list_reg.append(data_name)
        elif task_type_dic[data_name] == 'binary':
            dataset_list_class_binary.append(data_name)
        else:
            dataset_list_class_multi.append(data_name)

    # print(multi_class_list)
    # multi_class_list = ['Covertype', 'Dionis', 'Jannis', 'Fashion-MNIST', 'connect-4', 'Helena', 'volkert', 'shuttle', 'jungle_chess_2pcs_raw_endgame_complete', 'dilbert', 'Robert', 'vehicle', 'cnae-9','car','mfeat-factors', 'segment','fabert']
    # # #dataset names
    # # dataset_list_reg = [ 'poker', 'bng_pbc', 'bng_echomonths', 'mv', 'pol', 'house_8L', 'house_16H', 'houses', 'fried', '2dplanes', 'bng_lowbwt', 'bng_pharynx', 'bng_breastTumor', 'bng_pwLinear']
    # # dataset_list_class_binary = ['adult', 'Airlines', 'Albert', 'Amazon_employee_access', 'APSFailure', 'higgs', 'KDDCup09_appetency', 'MiniBooNE','numerai28.6', 'bank_marketing', 'nomao', 'riccardo', 'guillermo' ]
    # # dataset_list_class_multi = [ 'Covertype', 'Dionis', 'Jannis', 'Fashion-MNIST', 'connect-4', 'Helena', 'volkert', 'shuttle', 'jungle_chess_2pcs_raw_endgame_complete', 'dilbert', 'Robert']
    # # dataset_list_small = [ 'credit-g','jasmine','Australian','blood-transfusion','kc1','kr-vs-kp','sylvine','phoneme','christine', 'vehicle', 'cnae-9','car','mfeat-factors', 'segment','fabert'] 

    multi_class_list = [t.lower() for t in multi_class_list]
    dataset_list_reg = [t.lower() for t in dataset_list_reg]
    dataset_list_class_binary = [t.lower() for t in dataset_list_class_binary]
    dataset_list_class_multi = [t.lower() for t in dataset_list_class_multi]
    dataset_list_small = [t.lower() for t in dataset_list_small]

    dataset_list_dic = {}
    dataset_list_dic['large_reg'] = [i.lower() for i in dataset_list_reg] 
    dataset_list_dic['all_bin'] = [i.lower() for i in dataset_list_class_binary]  
    dataset_list_dic['all_multi'] = [i.lower() for i in dataset_list_class_multi]  
    dataset_list_dic['small'] = [i.lower() for i in dataset_list_small]  

    dataset_list_dic['all_class'] = list(task_tunedRF_dic.keys())
    dataset_list_dic['all_reg'] = list(set(data_size_dic.keys()) - set(task_tunedRF_dic.keys()))

    worse_than_tunedRF_num = {}
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
    times = [60,600,3600]
    framework_list = args.frameworks

    normalize_type = args.normalize_type
    filename = str(args.file_address)

    lines = [line.rstrip('\n') for line in open(filename)]
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
        task= task.lower()
        task = task.replace('_full', '')

        if task not in best_bench_dic.keys() and normalize_type == 'best':
            continue
        if task in multi_class_list:
            res_mean = -float(res_mean)
        try:
            framework_dic_all  = get_res_mean_std_all(framework_dic_all, task, framework, res_mean, res_std)
            task_dic_all, __ = get_res_mean_std(task_dic_all, framework_dic_all, task, framework, res_mean, res_std)
        except:
            pass
    task_list = sorted(list(task_dic_all.keys()),key=lambda x:data_size_dic[x])
    task_list = remove_nonplot_tasks(task_list, dataset_list_dic[plot_dataset_type])
    task_list_short = [task[:7] for task in task_list]
    N = len(task_list)
    if  'reg' in plot_dataset_type:
        ylim=[-0.1,1]
    elif 'small' in plot_dataset_type:
        ylim=[0.0, 1.4]
    elif 'multi' in plot_dataset_type:
        ylim=[-0.5,1.5]
    else:
        ylim=[-0.5,1.5]

    theta = radar_factory(N, frame='polygon')
    first = 'all_bin'
    if plot_dataset_type == first:
        fig, axes = plt.subplots(figsize=(19, 5.6), nrows=1, ncols=3,subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=0.5, hspace=0.0, top=0.85, bottom=0.0)
    else:
        fig, axes = plt.subplots(figsize=(17, 4), nrows=1, ncols=3,subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(wspace=0.5, hspace=0.0, top=0.85, bottom=0.08)
    markder_list = ['o', 's', 'x', '>', '<', 'h', 'p', 'x', 'v', '^', '1', '2', '>', '<', 'o', 's', 'x', 'v', 'h', 's']
    for i,ax in enumerate(axes):
        ax.set_rgrids([0,1],weight='bold',size='xx-large')
        ax.set_ylim(ylim)
        time = times[i]
        if plot_dataset_type==first:
            ax.set_title(str(time)+'s', weight='bold', size=20, position=(0.5, 1.1),
                         horizontalalignment='center', verticalalignment='center')
        task_dic = {}
        framework_dic = {}

        for line in lines[1:]:
            info = line.split(',')
            if len(info) ==1:
                info = line.split(' ')
            try:
                _, openmlid, task, framework, res_mean, res_std, duration = info
            except:
                continue
            framework = framework.replace('_from_log', '')
            framework = framework.replace('_cv', '')
            task= task.lower()
            task = task.replace('_full', '')

            if task not in best_bench_dic.keys() and normalize_type == 'best':
                continue
            if task in multi_class_list:
                res_mean = -float(res_mean)
            try:
                duration = float(duration)
                if float(duration) <= 1.3*float(time):
                    # print('duration', duration, 1.3*float(args.time))
                    task_dic, framework_dic = get_res_mean_std(task_dic, framework_dic, task, framework, res_mean, res_std)
            except:
                pass
        index_ = 0
        task_max, task_min = get_task_min_max_dic(task_list, framework_dic_all, task_tunedRF_dic, best_bench_dic, task_constPredictor_dic,  normalize_type)

        aliases = []
        linestyles = ['dotted','dashed','dashdot',(0, (1, 1)),(0, (5, 1))]
        for j,framework in enumerate(framework_list):
            framework = framework.replace('_cv', '')
            res_mean_scaled_list = []
            res_std_scaled_list = []

            if framework not in worse_than_tunedRF_num:
                worse_than_tunedRF_num[framework] = 0

            for task in task_tunedRF_dic.keys():
                if task in task_list:
                    
                    if framework in framework_dic and task in framework_dic[framework]:
                        if float(framework_dic[framework][task][0]) < task_tunedRF_dic[task]:
                            worse_than_tunedRF_num[framework] +=1
                    else:
                        worse_than_tunedRF_num[framework] +=1

            for task in task_list:
            # for task in task_min.keys():
                mean_max = task_max[task]
                mean_min = task_min[task]
                if framework in framework_dic:
                    if task in framework_dic[framework].keys():
                        if (mean_max - mean_min) !=0:
                            res_mean_scaled =  (float(framework_dic[framework][task][0]) - mean_min)/(mean_max - mean_min)
                            res_std_scaled = ( float(framework_dic[framework][task][1]) )/(mean_max - mean_min)
                        else:
                            res_mean_scaled = 0.0
                            res_std_scaled = ( float(framework_dic[framework][task][1]))/(float(framework_dic[framework][task][0]) )
                    else:
                        res_mean_scaled = 0.0
                        res_std_scaled = 0.0
                else:
                    res_mean_scaled = 0.0
                    res_std_scaled = 0.0
                res_mean_scaled_list.append(res_mean_scaled )
                res_std_scaled_list.append(res_std_scaled)

            framework_alias = str(get_framework_alias(framework))
            aliases.append(framework_alias)
            if framework_alias=='FLAML':
                ax.plot(theta, res_mean_scaled_list, color='k',linewidth=4,linestyle='solid')
                ax.fill(theta, res_mean_scaled_list, alpha=0.25, color='k')
            else:
                ax.plot(theta, res_mean_scaled_list,linewidth=4,linestyle=linestyles[j])
                ax.fill(theta, res_mean_scaled_list, alpha=0.25)

            ax.set_varlabels(task_list_short, size='large')
            index_ +=1
    if plot_dataset_type==first:
        ax=axes[0]
        legend = ax.legend(aliases, fontsize=20, ncol = 12, loc=(-0.2, 1.2),labelspacing=0.1,
                       )

    #ax.legend(framework_list, loc ='upper right', prop={'size': 12 if len(framework_list)>4 else 18}, ncol = 4)
    figure_name =  'compare_' + str(normalize_type) +  '_' + str(args.estimator) + '_'+ str(plot_dataset_type) 


    matplotlib.rcParams.update({'font.size': 25})
    plt.savefig( './plots/flaml/' + figure_name + '.pdf')
    print('worse than RF:', worse_than_tunedRF_num, len(dataset_list_dic[plot_dataset_type]))
    #plt.show()
 
if __name__ == '__main__':
    main()
