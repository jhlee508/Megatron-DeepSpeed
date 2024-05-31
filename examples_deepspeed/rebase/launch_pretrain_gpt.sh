#/!bin/bash

export LD_PRELOAD=$HOME/samccl-ws/samccl/build/libsccl.so
export LD_LIBRARY_PATH=$HOME/samccl-ws/samccl/build:$LD_LIBRARY_PATH

# export NCCL_DEBUG=WARN
# export NCCL_DEBUG_SUBSYS=ALL


### NVIDIA training
# salloc  \
#         -p xccl -w v05,v06 --gres=gpu:4 --exclusive \
#   srun  \
#         --ntasks-per-node 1 \
#         --export=ALL,NNODES=2,NPERNODES=4,MASTER_NODE=v05 \
#         bash ds_pretrain_gpt_125M_nvidia.sh \


### AMD training
# salloc  \
#         -p xccl -w em13,em14 --gres=gpu:8 --exclusive \
#   srun  \
#         --ntasks-per-node 1 \
#         --export=ALL,NNODES=1,NPERNODES=8,MASTER_NODE=em13 \
#         bash ds_pretrain_gpt_125M_amd.sh


### Heterogeneous-GPU training
salloc  \
        -p xccl -w v05,v06 --gres=gpu:4 --exclusive \
      : -p xccl -w em13 --gres=gpu:8 --exclusive \
  srun  \
        --ntasks-per-node 1 \
        --export=ALL,NNODES=3,NPERNODES=4,MASTER_NODE=v05 \
        bash ds_pretrain_gpt_125M_nvidia.sh \
      : --ntasks-per-node 1 \
        --export=ALL,NNODES=3,NPERNODES=8,MASTER_NODE=v05 \
        bash ds_pretrain_gpt_125M_amd.sh