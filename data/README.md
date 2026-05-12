# Dataset Sources

This project does not commit raw datasets. Most datasets are downloaded automatically by the training scripts into the local Hugging Face cache. The Cornell Grasp dataset is the only dataset expected to be extracted under this top-level `data/` directory.

## Hugging Face cached datasets

These are downloaded on first use by the scripts and stored in the local Hugging Face cache, usually `~/.cache/huggingface/datasets` unless `HF_HOME` or `HF_DATASETS_CACHE` is set.

| Experiment | Dataset source | Loader | Script |
|------------|----------------|--------|--------|
| GLUE NLP tasks | `glue` configs `sst2`, `mrpc`, `rte` | `datasets.load_dataset("glue", task)` | `code/scripts/train_glue.py` |
| Speech Commands audio | `google/speech_commands`, config `v0.02` | `datasets.load_dataset("google/speech_commands", "v0.02", trust_remote_code=True)` | `code/scripts/train_speech_commands.py` |
| Push-T VLA | `lerobot/pusht` | `LeRobotDataset("lerobot/pusht")` | `code/scripts/train_vla.py` |

The model weights and processors used by these experiments are also downloaded on first use into the Hugging Face cache by `transformers.from_pretrained`.

## Cornell Grasp

Cornell Grasp is loaded from a local directory because the training script expects image and grasp annotation files on disk.

Default local path:

```text
data/cornell_grasps/
```

Preferred project command from `code/`:

```bash
uv run scripts/download_cornell_grasp.py
```

That script downloads the Kaggle mirror:

```text
oneoneliu/cornell-grasp
https://www.kaggle.com/datasets/oneoneliu/cornell-grasp
```

Kaggle API use requires a Kaggle account and an API token at `~/.kaggle/kaggle.json`. If the zip is downloaded manually from Kaggle, pass it explicitly:

```bash
uv run scripts/download_cornell_grasp.py --zip_path /path/to/cornell-grasp.zip
```

The loader also documents the original Cornell archive source:

```text
http://pr.cs.cornell.edu/grasping/rect_data/data.tar.gz
```

After extraction, run Cornell Grasp experiments from `code/` with:

```bash
bash scripts/run_grasp_experiments.sh --data_dir ../data/cornell_grasps
```
