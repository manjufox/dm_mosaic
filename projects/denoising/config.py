#%%
from dataclasses import dataclass
from pathlib import Path
import shutil
from functools import cache
import platform

# @cache
@dataclass
class TrainingConfig:
    dataset_name = "bigdata-pw/Flickr"
    data_files = None
    
    #Project
    pf = platform.system()
    match pf:
        case "Windows":
            save_base_dir = Path("T:/")
        case "Linux":
            save_base_dir = Path("/mnt/d")
            # save_base_dir = Path(__file__).parent.parent.parent/"result"
    project_name = Path(__file__).parent.name
    project_dir = Path(__file__).parent
    default_root_dir = save_base_dir / "project" / project_name
    output_dir = default_root_dir / "result" 
    log_dir = default_root_dir / "log"
    cache_dir = save_base_dir / "dataset" / dataset_name.split("/")[-1] / "cache"
    ckpt_dir = default_root_dir / "ckpt"
    ckpt_path = ckpt_dir / f"{project_name}_loss.ckpt"
   # shutil.rmtree(output_dir, ignore_errors=True)
    for dir in [output_dir, cache_dir, default_root_dir,ckpt_dir]:
        if not dir.exists():
            dir.mkdir(parents=True)
    overwrite_output_dir = True

    #Dataset
    # dataset_name = "timbrooks/instructpix2pix-clip-filtered"
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
    debug = True
    
    if debug:
        image_size = 128
        num_warmup_steps = 0
        batch_size = 2
        num_steps = 100 
        num_epochs = 1000
        max_lr = 0.001
        learning_rate = 1e-4
        min_lr = 1e-7
        profiler = "simple"
        save_image_each_n_steps = 10
        evaluate_each_n_steps = 50
        check_point_each_n_steps = 500
    else:
        image_size = 256
        num_warmup_steps = 0
        batch_size = 5
        num_steps = 1000
        num_epochs = 1000
        max_lr = 1e-3
        learning_rate = 1e-4
        min_lr = 1e-10
        profiler = "simple"
        save_image_each_n_steps = 10
        evaluate_each_n_steps = 10
        check_point_each_n_steps = 500

    #Model
    #Encoder
    # encoder_name = "mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k"
    # encoder_name = "vit_betwixt_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k"
    # encoder_name = "eva02_base_patch14_448.mim_in22k_ft_in1k"
    # encoder_pretrained = True
    # feature_shape =[(batch_size,768,32,32),
    #                 (batch_size,768,32,32),
    #                 (batch_size,768,32,32),
    # ]
    # encoder_in_shape = (batch_size,3,448,448)
    # encoder_out_shape = (batch_size,768,32,32)
    # num_classes = 1

    # #Diffusers
    num_train_timesteps = 100

    # #decoder
    # decoder_in_shape = encoder_out_shape
    # decoder_out_shape = encoder_in_shape

    # huggingface_hub_token = "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"

    #Trainer




# %%
if __name__ == "__main__":
    conf = TrainingConfig()
    print(conf)
# %%
    conf.output_dir.exists()
# %%
