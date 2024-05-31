# Megatron-DeepSpeed-SCCL

Megatron-DeepSpeed setup for testing SCCL. 

## Setup

### Conda Environment

- NVIDIA (CUDA)

    ```bash
    # PyTorch (v2.3)
    pip3 install torch torchvision torchaudio

    # DeepSpeed
    DS_BUILD_OPS=1 DS_BUILD_SPARSE_ATTN=0 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_FP_QUANTIZER=0 pip3 install deepspeed
    ds_report # for compatibility check

    # Apex (master)
    git clone https://github.com/NVIDIA/apex
    mv apex apex-cuda
    cd apex-cuda
    export TORCH_CUDA_ARCH_LIST=7.0
    pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

    # FlashAttention (Unavailable on V00)
    ```

- AMD (ROCM)

    ```bash
    # PyTorch (v2.3)
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

    # DeepSpeed
    export PYTORCH_ROCM_ARCH=gfx908
    DS_BUILD_OPS=1 DS_BUILD_SPARSE_ATTN=0 DS_BUILD_EVOFORMER_ATTN=0 pip3 install deepspeed
    ds_report # for compatibility check

    # Apex (master)
    git clone https://github.com/ROCmSoftwarePlatform/apex.git
    mv apex apex-rocm
    cd apex-rocm
    export HCC_AMDGPU_TARGET=gfx908
    export PYTORCH_ROCM_ARCH=gfx908
    export GPU_ARCHS=gfx908
    pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

    # FlashAttention (optional) 
    # Add '--use-flash-attn' flag in the script.
    git clone --recursive https://github.com/ROCmSoftwarePlatform/flash-attention.git
    cd flash-attention
    py_version=$(python -V | grep -oP '(?<=[.])\w+(?=[.])')
    patch $HOME/anaconda3/envs/<env_name>/lib/python3.${py_version}/site-packages/torch/utils/hipify/hipify_python.py hipify_patch.patch
    python setup.py install
    ```


### Megatron-DeepSpeed

```bash
# Clone Megatron-DeepSpeed (Latest, 7eb36a1)
git clone https://github.com/microsoft/Megatron-DeepSpeed
cd Megatron-DeepSpeed
```

#### Requirements
```
pip install -r requirements.txt
```

#### Prepare Data
```bash
bash dataset/download_vocab.sh
bash dataset/download_oscar.sh 
```

#### Preprocess Data
```bash
python3 tools/preprocess_data.py \
    --input data/oscar-en-10k.jsonl \
    --output-prefix data/mega-ds-gpt2-oscar \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file data/gpt2-merges.txt \
    --vocab-file data/gpt2-vocab.json \
    --append-eod \
    --workers 8
```

#### Pretraining GPT
```bash
cd Megatron-DeepSpeed/example_deepspeed/rebase

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
```



