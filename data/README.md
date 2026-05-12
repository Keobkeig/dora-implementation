# Dataset Sources

This project does not commit raw datasets. Most datasets are downloaded automatically by the training scripts into the local Hugging Face cache. The Cornell Grasp dataset is the only dataset expected to be extracted under this top-level `data/` directory.

## Hugging Face cached datasets

These are downloaded on first use by the scripts and stored in the local Hugging Face cache, usually `~/.cache/huggingface/datasets` unless `HF_HOME` or `HF_DATASETS_CACHE` is set.

| Experiment | Dataset source | Hugging Face link | Loader | Script |
|------------|----------------|-------------------|--------|--------|
| GLUE NLP tasks | `glue` configs `sst2`, `mrpc`, `rte` | https://huggingface.co/datasets/nyu-mll/glue | `datasets.load_dataset("glue", task)` | `code/scripts/train_glue.py` |
| Speech Commands audio | `google/speech_commands`, config `v0.02` | https://huggingface.co/datasets/google/speech_commands | `datasets.load_dataset("google/speech_commands", "v0.02", trust_remote_code=True)` | `code/scripts/train_speech_commands.py` |
| Push-T VLA | `lerobot/pusht` | https://huggingface.co/datasets/lerobot/pusht | `LeRobotDataset("lerobot/pusht")` | `code/scripts/train_vla.py` |

The model weights and processors used by these experiments are also downloaded on first use into the Hugging Face cache by `transformers.from_pretrained`.

## Cornell Grasp

Cornell Grasp is loaded from `data/cornell_grasps/` because the training script expects image and grasp annotation files on disk. Use the Kaggle mirror `oneoneliu/cornell-grasp`: https://www.kaggle.com/datasets/oneoneliu/cornell-grasp.

From `code/`, download with the Kaggle API:

```bash
uv run scripts/download_cornell_grasp.py
```

Or pass a manually downloaded archive:

```bash
uv run scripts/download_cornell_grasp.py --zip_path /path/to/cornell-grasp.zip
```

For grasp experiments, run:

```bash
bash scripts/run_grasp_experiments.sh --data_dir ../data/cornell_grasps
```
