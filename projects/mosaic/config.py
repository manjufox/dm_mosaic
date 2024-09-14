from dataclasses import dataclass
from pathlib import Path
import shutil
from functools import cache

# @cache
@dataclass
class TrainingConfig:
    #Project
    project_name = Path(__file__).parent.name
    project_dir = Path(__file__).parent
    save_base_dir = Path("T:/")
    default_root_dir = save_base_dir / "project" / project_name
    output_dir = default_root_dir / "result" 
    cache_dir = save_base_dir / "dataset"
    ckpt_dir = default_root_dir / "ckpt"
    ckpt_path = ckpt_dir / f"{project_name}_loss.ckpt"
   # shutil.rmtree(output_dir, ignore_errors=True)
    for dir in [output_dir, cache_dir, default_root_dir,ckpt_dir]:
        if not dir.exists():
            dir.mkdir(parents=True)
    overwrite_output_dir = True

    #Dataset
    # dataset_name = "timbrooks/instructpix2pix-clip-filtered"
    dataset_name = "bigdata-pw/Flickr"
    data_files = None
    # dataset_name = "parquet"
    # data_files = save_root_dir / "dataset" / "Flickr"
    max_length = 968
    url_collumn_name = "url_c"
    split = "train"
    shuffle = True
    streaming = True
    num_workers = 0
    seed = 42
    buffer_size = 10000


    #Hyperparameter
    device = "cuda:0"
    
    debug = False
    
    if debug:
        num_steps = 100
        num_warmup_steps = 0
        batch_size = 2
        num_epochs = 100
        learning_rate = 1e-3
        min_lr = 1e-8
        profiler = "simple"
    else:
        num_steps = 100
        num_warmup_steps = 0
        batch_size = 2
        num_epochs = 100
        learning_rate = 1e-3
        min_lr = 1e-8
        profiler = "simple"
    take_num = num_steps
    max_steps = num_steps

    #Model
    #Encoder
    # encoder_name = "mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k"
    # encoder_name = "vit_betwixt_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k"
    encoder_name = "eva02_base_patch14_448.mim_in22k_ft_in1k"
    encoder_pretrained = True
    encoder_in_shape = (3,448,448)
    encoder_out_shape = (768,32,32)
    num_classes = 1

    #Diffusers
    num_train_timesteps = 1
    diffusers_encoder_name = "caformer_m36.sail_in1k_384" #"timm-gernet_s"

    #decoder
    decoder_in_shape = encoder_out_shape
    decoder_out_shape = encoder_in_shape

    image_size = encoder_in_shape[1]

    huggingface_hub_token = "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"



