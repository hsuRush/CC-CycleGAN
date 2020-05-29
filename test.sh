
CUDA_VISIBLE_DEVICES=3 python test_cyclegan.py \
    --model resnet18_2222_16 \
    --batch 1 \
    --iteration 0 \
    --exp_dir exp2_no_ctc \
    --resume_epoch 200 \
    --set train \
    --direction A2B 
