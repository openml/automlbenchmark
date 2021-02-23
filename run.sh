# python3 -m venv ./venv
source venv/bin/activate
# # remember to call `deactivate` once you're done using the application
# # bash flaml_1c1m.sh
# sleep 600s
# bash flaml_1c10m.sh
# sleep 6000s
# bash flaml_1c1h.sh


# screen -Sdm riccardo_flaml python runbenchmark.py flaml_old all 10m1c -t riccardo -f 0
# sleep 10s
# screen -Sdm riccardo_flaml python runbenchmark.py flaml_old all 10m1c -t riccardo -f 1
# sleep 10s
# screen -Sdm riccardo_flaml python runbenchmark.py flaml_old all 10m1c -t riccardo -f 2
# sleep 10s
# screen -Sdm riccardo_flaml python runbenchmark.py flaml_old all 10m1c -t riccardo -f 3
# sleep 10s
# screen -Sdm riccardo_flaml python runbenchmark.py flaml_old all 10m1c -t riccardo -f 4
# sleep 10s
# screen -Sdm riccardo_flaml python runbenchmark.py flaml_old all 10m1c -t riccardo -f 5
# sleep 10s
# screen -Sdm riccardo_flaml python runbenchmark.py flaml_old all 10m1c -t riccardo -f 6
# sleep 10s
# screen -Sdm riccardo_flaml python runbenchmark.py flaml_old all 10m1c -t riccardo -f 7
# sleep 10s
# screen -Sdm riccardo_flaml python runbenchmark.py flaml_old all 10m1c -t riccardo -f 8
# sleep 10s
# screen -Sdm riccardo_flaml python runbenchmark.py flaml_old all 10m1c -t riccardo -f 9
# sleep 10s

# screen -Sdm riccardo_autosklearn2 python runbenchmark.py autosklearn2  all 10m1c -t riccardo -f 0
# sleep 10s
# screen -Sdm riccardo_autosklearn2 python runbenchmark.py autosklearn2  all 10m1c -t riccardo -f 1
# sleep 10s
# screen -Sdm riccardo_autosklearn2 python runbenchmark.py autosklearn2  all 10m1c -t riccardo -f 2
# sleep 10s
# screen -Sdm riccardo_autosklearn2 python runbenchmark.py autosklearn2  all 10m1c -t riccardo -f 3
# sleep 10s
# screen -Sdm riccardo_autosklearn2 python runbenchmark.py autosklearn2  all 10m1c -t riccardo -f 4
# sleep 10s
# screen -Sdm riccardo_autosklearn2 python runbenchmark.py autosklearn2  all 10m1c -t riccardo -f 5
# sleep 10s
# screen -Sdm riccardo_autosklearn2 python runbenchmark.py autosklearn2  all 10m1c -t riccardo -f 6
# sleep 10s
# screen -Sdm riccardo_autosklearn2 python runbenchmark.py autosklearn2  all 10m1c -t riccardo -f 7
# sleep 10s
# screen -Sdm riccardo_autosklearn2 python runbenchmark.py autosklearn2  all 10m1c -t riccardo -f 8
# sleep 10s
# screen -Sdm riccardo_autosklearn2 python runbenchmark.py autosklearn2  all 10m1c -t riccardo -f 9
# sleep 10s

bash lightaml_1c1m.sh
sleep 1200s
bash lightaml_1c10m.sh
sleep 12000s
bash lightaml_1c1h.sh