cd ~/matwings/marker

source activate marker

nohup python server.py --port 8013 --worker 4 --cpu 8 > run_$(date +%y%m%d).log 2>&1 &
