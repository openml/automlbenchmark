# Make directories
mkdir plots 
mkdir plots/flaml 

python plot_res_flaml.py -file ../results/results_avg.csv  -f  flaml flaml_old lightautoml  -time 60 -e all  -d all
python plot_res_flaml.py -file ../results/results_avg.csv  -f  flaml flaml_old lightautoml  -time 600 -e all  -d all
python plot_res_flaml.py -file ../results/results_avg.csv  -f  flaml flaml_old lightautoml  -time 3600 -e all  -d all

python plot_compare_margin_flaml.py -file ../results/results_avg.csv -d all -tolerance_ratio 0.001 -f1 'flaml'  -flist 'flaml_old' 'lightautoml'