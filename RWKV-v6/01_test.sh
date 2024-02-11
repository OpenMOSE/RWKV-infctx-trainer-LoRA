

# JSONL -> trainデータとvalidation用データセットが自動分割(0.01)される
# JSONL make dataset train and validation
python3 preload_datapath.py config.yaml

# Train
python3 lightning_trainer.py fit -c config.yaml
#

#たぶんLoRAMergeしてくれるはず
#LoRA Merge Command
#python export_checkpoint_lora.py --checkpoint_dir checkpoint/test-v12.ckpt/ --output_file testoutput.pth --base_model model/RWKV-5-World-3B-v2-20231113-ctx4096.pth --lora_alpha 16

