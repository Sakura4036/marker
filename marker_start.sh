cd ~/matwings/marker

source activate marker
CUDA_VISIBLE_DEVICES=1
nohup python server.py --port 8013 --workers 16 > logs/run_$(date +%y%m%d).log 2>&1 &
