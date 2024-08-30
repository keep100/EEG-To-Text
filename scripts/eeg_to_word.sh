python eeg_to_text.py \
    --num_epochs 15 \
    --batch_size 32 \
    --lr 1e-5 \
    --phase train \
    --checkpoint_path ./checkpoints/word-level/best/b128_epoch15_lr1e-05_8-29-15-33.pt \
    --cuda cuda:0