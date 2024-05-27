This repo is forked from RWKV-infctx-trainer

# RWKV-5.2 ,6.0 Infctx LoRA Experiment Project RWKV-infctx-trainer-LoRA

I added the LoRA training function to the original repository. 
## 2024.05.28 New Release on RWKV-v6-4bit folder
   - Added a 4-bit quantization mode to Infctx-LoRA, enabling training of 14B models on Single 24GB GPU
   - Now supports outputs for Emb, Head, and LoRA checkpoints only, freeing you from the cumbersome process of DeepSpeed conversion.
   - Enabled CUDA Kernel for faster training.
   - Added Output,Gate LoRA.
   - for merging use this script.
```sh
python merge_lora.py <Alpha> <Base Model> <LORA Checkpoint> <Output FileName>
```
## 2024.02.13 Update
   - Added Gate and Head Layer Training if enabled LoRA(r,k,v) + Full Resolution(Gate+Head)
## 2024.02.11 Update
   - RWKV v6.0 test support. (without cuda kernel, 1.6b and 3b test)
## 2023.01.16 Update
   - Rocm5.6 HIP Ver Added.(tested on 2 x MI100)








This is really experiment repo.


Somehow I feel like I'm training well, but I'm not sure.


The basic commands follow those of RWKV-infctx-trainer

Examples of training commands can be found in 01-test.sh, so please make changes as needed.

I'd be happy if I could improve it, whether it's a fork or a pull request.

RWKV5-World 7b can be trained 16384ctx on Single 24GB GPU

This repo works
-  1.Add LoRA Layer to Receptance,Key,Value each att and ffn.
-  2.Training with DeepSpeed
-  3.out deepspeed checkpoint with LoRA layer.
-  4.Convert to Pytorch Checkpoint
-  5.Extract LoRA Layer and Merge to original Checkpoint
-  6.Maybe LoRA merged and working??
 

Training Steps
- 0.Edit Configs.


 if change lora_r and alpha
 edit like this on config yaml.
 
```sh
lora_r: 8 
lora_alpha: 16
lora_dropout: 0.01
```
- 1.train
```sh
python3 lightning_trainer.py fit -c config.yaml
```
- 2.merge
```sh
 python export_checkpoint_lora.py --checkpoint_dir %DEEPSPEED_CHECKPOINT_DIR% --output_file %LoRA_MERGED_Checkpoint_Dir% --base_model model/RWKV-5-World-3B-v2-20231113-ctx4096.pth --lora_alpha 16 --r 1 --k 1 --v 1
```

i tested it on RTX4090 Cuda12.2


# And Thanks to:
RWKV-LM @BlinkDL
RWKV-LM-LoRA @Blealtan
RWKV-infctx-trainer @ RWKV
@SmerkyG


# License
same with RWKV-LM and RWKV-LM-LoRA and RWKV-infctx-trainer

Apache 2.0


@ 2024 OpenMOSE
