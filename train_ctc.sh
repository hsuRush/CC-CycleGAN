
CUDA_VISIBLE_DEVICES=3 python train_cyclegan.py \
    --model resnet18_2222_16 \
    --epoch 200 \
    --batch 8 \
    --ctc_weights ./ctc_pretrain_weights/best-model.h5 \
    --ctc_resume \
    --ctc_condition \
    --exp_dir exp2  

# train with ctc condiiton
