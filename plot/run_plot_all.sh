
# Make directories
mkdir plots 
mkdir plots/flaml 

# #Plot learning curve 
# python plot_learning_curve_flaml.py -file ../results/log/  -time 3600 -m FLAML_roundrobin_ FLAML_ns_ FLAML_cv_ FLAML_ -d bng_pbc -l None
# python plot_learning_curve_flaml.py -file ../results/log/  -time 3600 -m FLAML_roundrobin_ FLAML_ns_ FLAML_cv_ FLAML_ -l lower_left -d Albert
# python plot_learning_curve_flaml.py -file ../results/log/  -time 3600 -m FLAML_roundrobin_ FLAML_ns_ FLAML_cv_ FLAML_ -l None -d Covertype

# # plot radar
# python plot_radar_flaml.py -file ../results/results_avg.csv  -f  autosklearn  BOHB H2OAutoML TPOT FLAML -e all  -d all_reg
# python plot_radar_flaml.py -file ../results/results_avg.csv  -f  autosklearn  BOHB H2OAutoML TPOT FLAML -e all  -d all_bin
# python plot_radar_flaml.py -file ../results/results_avg.csv  -f  autosklearn  BOHB H2OAutoML TPOT FLAML -e all  -d all_multi

# # plot winners
# python plot_res_flaml.py -file ../results/results_avg.csv  -f  autosklearn  BOHB H2OAutoML TPOT  FLAML -time 60 -e all  -d all
# python plot_res_flaml.py -file ../results/results_avg.csv  -f  autosklearn  BOHB H2OAutoML TPOT  FLAML -time 600 -e all  -d all
# python plot_res_flaml.py -file ../results/results_avg.csv  -f  autosklearn  BOHB H2OAutoML TPOT AutoGluon-best flaml-log  -time 3600 -e all  -d all

# #FLAML compare better or equal 
# python plot_compare_margin_flaml.py -file ../results/results_avg.csv -d all -tolerance_ratio 0.001
# python plot_res_flaml.py -file ../results/results_avg.csv  -f AutoGluon_best flaml-log  -time 3600 -e all  -d all



