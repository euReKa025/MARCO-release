# MARCO

**Multi-Round Agentic Reinforcement for Conditional Molecular Optimization**

Paper: <!-- Add the paper URL here. -->

MARCO trains molecular editors to propose a molecule, receive property and
similarity feedback, and revise the proposal over multiple rounds. The training
pipeline starts with per-subtask supervised fine-tuning (SFT), exports a
Hugging Face checkpoint, and optimizes a multi-round policy with group-relative
reinforcement learning using [verl](https://github.com/verl-project/verl).

The model follows an instruction and returns an edited molecule inside
`<SMILES>...</SMILES>`. Property predictors and RDKit evaluate each proposal;
MARCO combines molecule quality and revision trends into a trajectory reward.

## Contents

- [Installation](#installation)
- [Property predictors](#property-predictors)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Repository layout](#repository-layout)

## Installation

Training requires Linux, NVIDIA GPUs, and a CUDA-compatible PyTorch/SGLang
installation. The source environment used Python 3.12 and CUDA 12.8. CPU tests
and offline metric computation can also run on macOS.

```bash
git clone https://github.com/euReKa025/MARCO-release.git
cd MARCO-release
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data,test]'
```

For training, prepare the pinned verl checkout and its compatibility patch:

```bash
bash scripts/setup_verl.sh
python -m pip install -r requirements-training.txt
python -m pip install -e ../verl
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip check
```

`requirements-training.txt` records core versions observed in the source
environment, rather than a complete platform lockfile. Install matching CUDA
wheels and compiler tools on the target machine. The setup script preserves
existing checkouts and rejects a different verl revision. For HF-only inference,
install `python -m pip install -e '.[inference]'`; verl and SGLang are unnecessary.

Run the CPU unit tests:

```bash
python -m pytest -q
```

The verl integration test module is skipped when verl is not installed.

## Property predictors

MARCO uses ADMET-AI for ADMET properties, a DRD2 classifier, and RDKit-based QED
and penalized logP. Install predictors separately from the training environment:

```bash
conda create -n mumo python=3.10 -y
conda run -n mumo python -m pip install -r requirements-predictors.txt
git clone https://github.com/tmlr-group/RePO.git ../RePO
cp ../RePO/multiprop_utils/clf_py36.pkl third_party/repo_predictors/multiprop_utils/
cp ../RePO/multiprop_utils/fpscores.pkl.gz third_party/repo_predictors/multiprop_utils/
```

The predictor assets and ADMET-AI weights are downloaded separately; obtain
them from the upstream providers if they are absent from your RePO checkout.

From the MARCO environment, start the local services:

```bash
export CONDA_MUMO_ENV=mumo
export PREDICTOR_BIND_HOST=127.0.0.1
eval "$(bash scripts/verl/start_predictors.sh)"
bash scripts/verl/test_predictors.sh
```

Default endpoints are `http://127.0.0.1:10086/predict/` (ADMET) and
`http://127.0.0.1:10087/predict/` (DRD2). Set `MUMO_PYTHON` to an absolute Python
path when using a virtual environment instead of Conda. Existing services can
be supplied through `MARCO_ADMET_API` and `MARCO_DRD2_API` with
`START_PREDICTORS=0` for training or `AUTO_START_SERVERS=false` for inference.

## Data preparation

The three-objective tasks use [MuMOInstruct](https://huggingface.co/datasets/NingLab/MuMOInstruct)
in the [RePO](https://github.com/tmlr-group/RePO) raw-data layout:

| Task | Property setting |
| --- | --- |
| BDP | `bbbp+drd2+plogp` |
| BDQ | `bbbp+drd2+qed` |
| BPQ | `bbbp+plogp+qed` |

```bash
python -m marco.data.build_canonical_mumo_jsonl \
  --repo-data-root ../RePO/data --output-dir data/canonical
```

SFT uses reasoning examples aligned to the original instructions and target
molecules. To generate these examples, configure a compatible teacher endpoint:

```bash
export SFT_TEACHER_BASE_URL='https://your-teacher-endpoint'
export SFT_TEACHER_MODEL='your-teacher-model'
# Set SFT_TEACHER_API_KEY in the environment when required by your endpoint.
python -m marco.data.build_sft_think_data \
  --canonical-dir data/canonical --output-dir data/sft_think --splits train
bash scripts/build_repo_source_data.sh
```

Alternatively, supply existing raw teacher examples at `SFT_THINK_DIR`.
Teacher outputs are not bundled; regenerating them does not reproduce the exact
historical SFT corpus. See the builders' `--help` for input options.
When the raw data has no validation split, the canonical builder aliases
`test_seen` as `val`; this is not an independent held-out validation set.

With the predictors running, recompute source properties for consistent online
training rewards and build the training parquet files:

```bash
python -m marco.data.recompute_source_properties \
  --input-root data/canonical --output-root data/canonical_source_recomputed
bash scripts/build_recomputed_source_data.sh
```

Keep `data/canonical` for standalone benchmark evaluation. Training files are
written under `data/{sft,rlhf,grpo_single_turn}_recomputed/by_subtask/`.

## Training

The following commands illustrate the BDP pipeline. Set `TASK` to another
property setting to train a separate model. GPU count is configurable; choose
batch sizes and tensor parallelism for the target hardware.

```bash
export TASK='bbbp+drd2+plogp'
export N_GPUS_PER_NODE=2
```

### 1. Per-subtask SFT

```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
TRAIN_PARQUET="data/sft_recomputed/by_subtask/$TASK/train.parquet" \
OUTPUT_DIR="outputs/sft/$TASK" \
TRAIN_BATCH_SIZE=32 MICRO_BATCH_SIZE_PER_GPU=2 \
TOTAL_EPOCHS=2 OPTIM_LR=5e-6 \
bash scripts/verl/run_marco_sft_qwen.sh
```

### 2. Export to Hugging Face

```bash
CHECKPOINT_ROOT="outputs/sft/$TASK" \
OUTPUT_HF_DIR="outputs/sft_hf/$TASK" \
bash scripts/verl/export_marco_sft_to_hf.sh
```

The exporter selects the latest numeric step. Use `CHECKPOINT_DIR` to select a
specific checkpoint. The same exporter supports FSDP actor checkpoints from RL.

### 3. MARCO from SFT

```bash
MODEL_PATH="$(pwd)/outputs/sft_hf/$TASK" \
TRAIN_PARQUET="data/rlhf_recomputed/by_subtask/$TASK/train.parquet" \
VAL_PARQUET="data/rlhf_recomputed/by_subtask/$TASK/val.parquet" \
OUTPUT_DIR="outputs/marco/$TASK" \
TRAIN_BATCH_SIZE=32 ACTOR_PPO_MINI_BATCH_SIZE=32 \
MAX_TURNS=5 MAX_RESPONSE_LENGTH=1024 \
ROLLOUT_N=4 ACTOR_LR=6e-7 USE_KL_LOSS=true KL_LOSS_COEF=0.03 \
TOTAL_EPOCHS=1 TRAINER_VAL_BEFORE_TRAIN=false TRAINER_TEST_FREQ=-1 \
bash scripts/verl/run_marco_rl_ord_qwen.sh
```

These are runnable starting settings, not a claim that one configuration
reproduces every paper cell. Training uses `think_answer` prompts and five
turns by default. See the launcher and YAML configuration for reward weights,
rollout sampling, checkpoint frequency, and distributed settings.

`scripts/verl/run_grpo_baseline_qwen.sh` provides a single-turn verl baseline;
use `data/grpo_single_turn_recomputed/by_subtask/$TASK/` as its data source.
The original RePO method is implemented in the upstream RePO repository.

## Evaluation

Export the selected RL checkpoint to HF, then run the same standalone evaluator
for all methods. `MAX_TURNS=1` is Same-1 and `MAX_TURNS=5` is Same-5; this controls
the evaluation budget independently of the training horizon.

```bash
CHECKPOINT_ROOT="outputs/marco/$TASK" \
OUTPUT_HF_DIR="outputs/marco_hf/$TASK" \
bash scripts/verl/export_marco_sft_to_hf.sh

MODEL_PATH="$(pwd)/outputs/marco_hf/$TASK" \
PROPERTY_SETTING="$TASK" SEEN_SETTING=seen \
MAX_TURNS=1 MAX_NEW_TOKENS=1024 DO_SAMPLE=false \
OUTPUT_DIR="outputs/eval/$TASK/seen/same1" OUTPUT_NAME=trajectories.json \
bash scripts/run_multi_turn_inf_noargs.sh

INPUT_JSON="outputs/eval/$TASK/seen/same1/trajectories.json" \
PROPERTY_SETTING="$TASK" SEEN_SETTING=seen \
OUTPUT_FOLDER="outputs/eval/$TASK/seen/same1/metrics" \
bash scripts/run_multi_turn_evaluate_noargs.sh
```

Repeat with `SEEN_SETTING=unseen` and distinct output paths. For Same-5 use
`MAX_TURNS=5` and a separate `same5` directory.

The evaluator writes per-example CSVs and aggregate metrics. Report:

- **SR**: property-success rate over all examples.
- **Sim**: mean similarity over valid selected candidates.
- **SR x Sim**: the product of these aggregate values.
- **SWS**: mean per-example success-weighted similarity with canonical
  exact-copy exclusion; it requires source and candidate SMILES.

Compute the complete scalar summary from a detailed CSV:

```bash
python -m marco.eval.sws \
  --input-csv "outputs/eval/$TASK/seen/same1/metrics/detailed_results_IND_seen_${TASK}marco.csv" \
  --expected-n 500 --output-json "outputs/eval/$TASK/seen/same1/scores.json"
```

Paper results use standalone HF evaluation. Training-time validation is a
checkpoint-screening signal. Keep data splits, predictor versions, decoding
parameters, checkpoint selection, and turn budgets with each reported result.

## Repository layout

```text
marco/
  data/          Canonical data, SFT alignment, and parquet builders
  env/           Molecule validation, similarity, and predictor client
  prompts/       Output parsing and feedback construction
  reward/        Property and validity reward primitives
  rollout/       Post-hoc trajectory reward settlement
  verl/          Configurations, agent loop, interactions, and reward bridge
  eval/          Standalone HF inference and metrics
  tests/         CPU unit tests
scripts/         Portable training, export, predictor, and evaluation entrypoints
patches/         Compatibility patch for the pinned verl revision
third_party/    Predictor adapters (weights obtained separately)
```

## Acknowledgements

This implementation builds on [verl](https://github.com/verl-project/verl),
[RePO](https://github.com/tmlr-group/RePO),
[MuMOInstruct](https://huggingface.co/datasets/NingLab/MuMOInstruct),
[ADMET-AI](https://github.com/swansonk14/admet_ai), and
[RDKit](https://www.rdkit.org/). See [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

Citation information will be added with the paper link.
