#!/bin/bash
BS=${1-32}

# verify the pretrained model (at 'pretrained/ldmseg.pt')

export OMP_NUM_THREADS=1
python -W ignore tools/main_ldm.py \
    datasets=coco \
    base.train_kwargs.batch_size=$BS \
    base.train_kwargs.weight_dtype=float16 \
    base.vae_model_kwargs.scaling_factor=0.18215 \
    base.transformation_kwargs.type=crop_resize_pil \
    base.transformation_kwargs.size=512 \
    base.eval_kwargs.count_th=512 \
    base.sampling_kwargs.num_inference_steps=50 \
    base.train_kwargs.self_condition=True \
    base.model_kwargs.cond_channels=4 \
    base.load_path='pretrained/ldmseg.pt' \
    base.eval_only=True \
