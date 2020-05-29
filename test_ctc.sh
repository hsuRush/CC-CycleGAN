
CUDA_VISIBLE_DEVICES=2 python test_cyclegan.py \
    --model resnet18_2222_16 \
    --batch 1 \
    --iteration 0 \
    --ctc_condition \
    --exp_dir exp1 \
    --resume_epoch 245 \
    --set train \
    --direction A2B  
