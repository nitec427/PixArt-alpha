_base_ = ['../PixArt_xl2_internal.py']

image_size = 1024

# model setting
model = 'PixArtMS_XL_2'     # model for multi-scale training
fp32_attention = True
load_from = None
vae_pretrained = "output/pretrained_models/sd-vae-ft-ema"
window_block_indexes = []
window_size=0
use_rel_pos=False
aspect_ratio_type = 'ASPECT_RATIO_1024'         # base aspect ratio [ASPECT_RATIO_512 or ASPECT_RATIO_256]
multi_scale = True     # if use multiscale dataset model training
lewei_scale = 2.0

# training setting
num_workers=10
train_batch_size = 12   # max 14 for PixArt-xL/2 when grad_checkpoint
num_epochs = 10 # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=1000)
save_model_epochs=1
save_model_steps=2000

log_interval = 20
eval_sampling_steps = 200
work_dir = 'output/debug'
