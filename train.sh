# debug training
CUDA_VISIBLE_DEVICES=4 accelerate launch --config_file acc_configs/gpu1.yaml main.py big_debug --workspace workspace_debug_debug
# # training (use slurm for multi-nodes training)
# WORKSPACE=workspace-nonsense-get-pointcloud
# accelerate launch --config_file acc_configs/gpu8.yaml main.py big --workspace $WORKSPACE
