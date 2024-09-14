from dataclasses import dataclass
from pathlib import Path

from transformers import TrainingArguments
import trl

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


@dataclass
class TrainingConfig:
    channel_size = 3
    image_size = 224
    output_image_size = 224

    debug = False

    if debug:
        num_steps = 100
        num_warmup_steps = 0
        batch_size = 2
        num_epochs = num_steps
        learning_rate = 1e-3
        profiler = "advanced"
    else:
        num_steps = 1e-5
        num_warmup_steps = 10
        batch_size = 6
        num_epochs = num_steps
        learning_rate = 1e-5
        profiler = None

    take_num = num_steps
    max_steps = num_steps

    device = "cuda:0"
    is_parallel = False

    # dataset_name = "timbrooks/instructpix2pix-clip-filtered"
    dataset_name = "parquet"
    data_dir = r"X:\dataset\NSFW\data"
    split = "train"  # "train[:100]"#None#"train"#"train[:20]"
    streaming = True
    num_proc = None
    num_workers = 8
    seed = 42
    buffer_size = 10000

    encoder_name = "caformer_b36.sail_in22k_ft_in1k_384"
    encoder_pretrained = False

    project_dir = Path(__file__).parent
    save_dir = Path(r"X:")
    output_dir = save_dir / dataset_name / "result"
    dataset_cache_dir = save_dir / "dataset" / "cache" / dataset_name
    default_root_dir = save_dir / dataset_name

    for dir in [output_dir, dataset_cache_dir, default_root_dir]:
        if not dir.exists():
            dir.mkdir(parents=True)

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    hub_private_repo = True
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook


# config = TrainingConfig()

# training_args = TrainingArguments(
#     do_train=True,
#     do_eval=True,
#     output_dir=str(config.output_dir),

#     per_device_train_batch_size=config.batch_size,
#     per_device_eval_batch_size=config.batch_size,
#     num_train_epochs=50,
#     gradient_accumulation_steps=1,
#     fp16=True,
#     weight_decay=0.05,
#     eval_strategy="steps",
#     eval_steps=100,
#     save_strategy="steps",
#     save_steps=100,
#     max_steps=10000,
#     label_names = "image",
#     load_best_model_at_end=False,
#     push_to_hub=config.push_to_hub,
#     overwrite_output_dir=config.overwrite_output_dir,
#     seed=config.seed,
#     optim="adamw_torch",
#     run_name="adamw_torch",
#     gradient_checkpointing=False,
#     logging_strategy="steps",
#     logging_steps=1,
#     logging_dir=str(config.output_dir / "logs"),
#     # torch_compile = True,
#     remove_unused_columns=False,
#     auto_find_batch_size=True,
# )
