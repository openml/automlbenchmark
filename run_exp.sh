# python3 -m venv ./venv
source venv/bin/activate
# remember to call `deactivate` once you're done using the application

#1hh
bash flaml_1c1m.sh
sleep 600s
bash tpot_1c1m.sh
sleep 600s
bash autogluon_1c1m.sh
sleep 600s
bash autosklearn_1c1m.sh
sleep 600s
bash h2o_1c1m.sh
sleep 600s

#8h
bash flaml_1c10m.sh
sleep 6000s
bash tpot_1c10m.sh
sleep 6000s
bash autosklearn_1c10m.sh
sleep 6000s
bash autogluon_1c10m.sh
sleep 6000s
bash h2o_1c10m.sh
sleep 6000s

#55h
bash flaml_1c1h.sh
sleep 40000s
bash tpot_1c1h.sh
sleep 40000s
bash autogluon_1c1h.sh
sleep 40000s
bash autosklearn_1c1h.sh
sleep 40000s
bash h2o_1c1h.sh

