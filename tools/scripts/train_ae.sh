#!/bin/bash
BS=${1-8}

export OMP_NUM_THREADS=1
python -W ignore tools/main_ae.py \
    datasets=coco \
    debug=$DEBUG \
    base.wandb=False \
    base.train_kwargs.batch_size=$BS \
    base.train_kwargs.accumulate=1 \
    base.train_kwargs.train_num_steps=90000 \
    base.train_kwargs.fp16=True \
    base.vae_model_kwargs.num_mid_blocks=0 \
    base.vae_model_kwargs.num_upscalers=2 \
    base.optimizer_name='adamw' \
    base.optimizer_kwargs.lr=1e-4 \
    base.optimizer_kwargs.weight_decay=0.05 \
    base.transformation_kwargs.type=crop_resize_pil \
    base.transformation_kwargs.size=512 \
    base.eval_kwargs.mask_th=0.8 \
    base.train_kwargs.prob_inpainting=0.0 \
    base.vae_model_kwargs.parametrization=gaussian \
