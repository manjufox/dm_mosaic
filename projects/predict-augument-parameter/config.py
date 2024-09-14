from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingConfig:
    #Project
    project_name = "predict-augument-parameter"
    project_dir = Path(__file__).parent
    save_base_dir = Path("T:/")
    default_root_dir = save_base_dir / "project" / project_name
    output_dir = default_root_dir / "result" 
    cache_dir = save_base_dir / "dataset"
    ckpt_dir = default_root_dir / "ckpt"
    ckpt_path = ckpt_dir / "best.ckpt"
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
    url_collumn_name = "url_c"
    split = "train"
    shuffle = True
    streaming = True
    num_workers = 0
    seed = 42
    buffer_size = 10000

    #Image
    channel_size = 3
    input_image_size = 448
    output_image_size = 448

    #Hyperparameter
    device = "cuda:0"
    
    debug = True
    
    if debug:
        num_steps = 1e-50
        num_warmup_steps = 0
        batch_size = 5
        num_epochs = 10
        learning_rate = 1e-5
        profiler = "advanced"
    else:
        num_steps = 1e-50
        num_warmup_steps = 10
        batch_size = 6
        num_epochs = 10
        learning_rate = 1e-5
        profiler = None
    take_num = num_steps
    max_steps = num_steps

    #Model
    encoder_name = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"
    encoder_pretrained = True
    encoder_in_size = (batch_size,3,448,448)
    encoder_out_size = (batch_size,1025,768)
    num_classes = 1

