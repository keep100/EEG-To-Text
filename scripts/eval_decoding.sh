python3 eval_decoding_old.py \
    --checkpoint_path ./checkpoints/decoding/best/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_2.7e-06_unique_sent_5_15_21_20.pt \
    --config_path ./config/decoding/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_2.7e-06_unique_sent.json \
    -cuda cuda:6
