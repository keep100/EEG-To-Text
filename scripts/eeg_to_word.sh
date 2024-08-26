python eeg_to_text.py \
    --num_epochs 15 \
    --batch_size 128 \
    --lr 1e-5 \
    --phase train \
    --checkpoint_path ./checkpoints/word-level/best/b32_epoch50_lr1e-05_8-23-10-27.pt \
    --cuda cuda:0