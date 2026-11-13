#!/usr/bin/env python3
"""Per-mouse block-level nested-CV learning curves for eight two-unit RNNs.

Run from any directory, with the repository's Python dependencies installed:
    python code/models/session_learning_curve.py --output /path/to/results
    python code/models/session_learning_curve.py --dry-run
    python code/models/session_learning_curve.py --smoke-test --output /tmp/ws20-smoke

Defaults match training_with_nested_cv.ipynb: 10 weight seeds (1..10),
5 sparsity penalties, 3 energy penalties, AdamW lr=.01,
10,000 epochs maximum, and patience 100. Every session count and model runs
all 10 outer folds and all 9 inner folds. Like nested_cv.run_outer_fold,
the inner model with the lowest validation NLL is evaluated on the outer
fold; there is no extra refit or averaging of hyperparameters across folds.

Use the most recent 2/4/6/8/10 datetime-named sessions for each mouse. Defaults
use datasets.SEQUENCE_LENGTH and training.BATCH_SIZE. Shuffle nonoverlapping blocks
into folds, NOT sessions: sessions can occur in both train and test. Each
block consumes sequence_length + 1 source trials to produce sequence_length
input/next-choice target pairs, as in AB_Dataset; session remainders are discarded. Counts with
fewer than 10 blocks are skipped. Held-out NLL excludes forced choices
and regularization. Plot the equal-weight mean of 10 outer-fold NLLs;
also save the trial-weighted pooled NLL and fold standard deviation.

Independent inner fits across models and outer folds run in CPU processes,
with one thread each by default. Mice run sequentially through the same pool size.
All seed/penalty configurations remain vectorized within each fit.

Writes the same model pickle, info JSON, and trial table as submit_jobs ->
nested_cv -> save_data, below data/rnns/nested_session_learning_curve/
<mouse>/sessions_N/<model_type>/<constraint>/outer_fold_N/inner_fold_M/.
The local loop schedules independent fits with joblib, without SLURM.
Inner checkpoints permit automatic resume with identical settings and data.
--smoke-test retains all models and folds but uses two seeds, one penalty
pair, and two epochs: its results are only a pipeline check.
"""

from __future__ import annotations

import argparse
from copy import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime

# Tiny recurrent fits use independent CPU processes; avoid GPU contention.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HIP_VISIBLE_DEVICES"] = "-1"

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed, parallel_config, effective_n_jobs
from torch.utils.data import DataLoader, Dataset, Subset

from NM_TinyRNN.code.models import datasets as ds
from NM_TinyRNN.code.models.rnns import TinyRNN
from NM_TinyRNN.code.models.training import Trainer, BATCH_SIZE, EARLY_STOP
from NM_TinyRNN.code.models.save_data import save_inner_fold_results

N_FOLDS = 10
MODELS = list(itertools.product(["monoGRU", "GRU"], ["tanh", "relu"], [False, True]))


class BlockDataset(Dataset):
    """Subset of complete source blocks, with explicit Trainer split injection."""

    def __init__(self, source, block_ids):
        self.sequence_length = source.sequence_length
        self.inputs = source.inputs[block_ids]
        self.targets = source.targets[block_ids]
        self.mask = source.forced_choice_mask[block_ids]
        self.forced_choice_mask = self.mask
        # Save only selected sessions, retaining remainder trials as unused.
        names = source.subject_df.loc[source.subject_df.sequence_block_idx.isin(block_ids),
                                      'session_folder_name'].unique()
        self.subject_df = source.subject_df.loc[
            source.subject_df.session_folder_name.isin(names)].copy().reset_index(drop=True)
        mapping = {block: i for i, block in enumerate(block_ids)}
        self.subject_df['sequence_block_idx'] = self.subject_df.sequence_block_idx.map(mapping)
        self.splits = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index], self.mask[index]

    def _session_split(self, **_kwargs):
        if self.splits is None:
            raise RuntimeError("Explicit block splits have not been assigned")
        return self.splits


def block_folds(n_blocks, seed):
    if n_blocks < N_FOLDS:
        return []
    return [x.tolist() for x in np.array_split(
        np.random.default_rng(seed).permutation(n_blocks), N_FOLDS
    )]


def fold_splits(folds, outer, inner):
    if outer == inner:
        raise ValueError("Outer and inner fold must differ")
    return {
        "train": [b for i, fold in enumerate(folds) if i not in (outer, inner) for b in fold],
        "val": folds[inner],
        "eval": folds[outer],
    }


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def held_out_nll(model, dataset, indices, batch_size):
    """Sum natural-log loss over free-choice targets before taking the mean."""
    device = next(model.parameters()).device
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y, mask in DataLoader(Subset(dataset, indices), batch_size=batch_size):
            logits, _ = model(x.to(device))
            free = mask.to(device) == 0
            n = int(free.sum())
            if n:
                loss = torch.nn.functional.cross_entropy(logits[free], y.to(device)[free], reduction="sum")
                total += float(loss)
                count += n
    if count == 0 or not math.isfinite(total):
        raise ValueError("Held-out fold has no free-choice targets or nonfinite NLL")
    return total / count, total, count


def train_inner(model, dataset, folds, outer, inner, directory, trainer_kwargs, threads,
                resume_checkpoint=None):
    """Train locally and use the same saver called by nested_cv._run_inner_fold."""
    torch.set_num_threads(threads)
    from functools import partial
    from NM_TinyRNN.code.models import training
    from tqdm import tqdm
    training.tqdm = partial(tqdm, disable=True)
    model_id = model.get_model_id()
    checkpoint = directory / '.checkpoints' / model_id / f"outer_{outer+1:02d}_inner_{inner+1:02d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Canonical HPC inner folders are 0..8, skipping the outer fold.
    inner_idx = inner if inner < outer else inner - 1
    fold_dir = directory / f'outer_fold_{outer+1}' / f'inner_fold_{inner_idx}'
    marker = fold_dir / f'{model_id}_complete.json'
    trainer = Trainer(**trainer_kwargs)
    local_dataset = copy(dataset)
    local_dataset.splits = fold_splits(folds, outer, inner)
    cached = checkpoint if checkpoint.exists() else resume_checkpoint
    if cached is not None and cached.exists():
        result = torch.load(cached, map_location='cpu', weights_only=True)
    else:
        state, config, _ = trainer.fit(model, local_dataset)
        val_nll = float(trainer._last_best_val_loss)
        if not math.isfinite(val_nll) or not all(torch.isfinite(v).all() for v in state.values()):
            raise RuntimeError(f'Nonfinite training result at outer={outer+1}, inner={inner+1}')
        result = {'state_dict': {k: v.detach().cpu() for k, v in state.items()},
                  'config': list(config), 'val_nll': val_nll, 'inner_fold': inner+1}
    artifacts = [fold_dir / f'{model_id}_{suffix}' for suffix in
                 ('model.pickle', 'info.json', 'trials_data.htsv')]
    if not marker.exists() or not all(p.exists() for p in artifacts):
        save_inner_fold_results(
            result=dict(result, val_loss=result['val_nll']), base_model=model,
            dataset=local_dataset, splits=local_dataset.splits, trainer=trainer,
            save_path=directory, outer_loop_number=outer+1, inner_fold_idx=inner_idx,
            outer_eval_blocks=folds[outer])
        write_json(marker, {'model_id': model_id, 'outer_fold': outer+1,
                            'inner_fold_idx': inner_idx, 'source_inner_fold': inner+1})
    if not checkpoint.exists():
        save_checkpoint(checkpoint, result)
    print(f'Completed {model_id} outer={outer+1} inner={inner_idx} val_nll={result["val_nll"]:.6f}', flush=True)
    return result


def summarize(rows, output, smoke):
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "fold_results.csv", index=False)
    summaries = []
    for (name, n_sessions), group in frame.groupby(["model", "n_sessions"], sort=False):
        if len(group) != N_FOLDS:
            continue
        summaries.append({"model": name, "n_sessions": int(n_sessions),
                          "mean_held_out_nll": group.held_out_nll.mean(),
                          "fold_sd": group.held_out_nll.std(ddof=1),
                          "pooled_nll": group.nll_sum.sum() / group.n_free.sum(),
                          "n_free": int(group.n_free.sum()), "n_folds": len(group)})
    summary = pd.DataFrame(summaries)
    if summary.empty:
        return
    summary.to_csv(output / "summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, group in summary.groupby("model", sort=False):
        group = group.sort_values("n_sessions")
        ax.plot(group.n_sessions, group.mean_held_out_nll, marker="o", label=name)
    ax.set(xlabel="Number of sessions (most recent sessions)",
           ylabel="Mean held-out NLL (nats / free-choice target)",
           title=f"{output.name}: 10-fold nested block CV" + (" — SMOKE TEST ONLY" if smoke else ""))
    ax.set_xticks(sorted(summary.n_sessions.unique()))
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "held_out_nll.png", dpi=180)
    fig.savefig(output / "held_out_nll.pdf")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", nargs="+", default=["WS20"], help="Mouse IDs; default: WS20")
    parser.add_argument("--sessions", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument("--output", type=Path, default=REPO / "data" / "rnns" / "nested_session_learning_curve")
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffled block partitions")
    parser.add_argument("--max-epochs", type=int, default=10000)
    parser.add_argument("--early-stop", type=int, default=EARLY_STOP)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=ds.SEQUENCE_LENGTH,
                        help="Predictions per block (default: datasets.SEQUENCE_LENGTH)")
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--sparsity-lambdas", nargs="+", type=float, default=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    parser.add_argument("--energy-lambdas", nargs="+", type=float, default=[1e-1, 1e-2, 1e-3])
    parser.add_argument("--hebbian-lambdas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--covariance-lambdas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--jobs", type=int, default=-1, help="Concurrent independent fits; -1 uses available CPUs across models and all folds")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch/BLAS threads per worker (default: 1)")
    parser.add_argument("--resume-from", type=Path, help="Import compatible checkpoints from the previous runner")
    parser.add_argument("--dry-run", action="store_true", help="Save/print split plan without training")
    parser.add_argument("--smoke-test", action="store_true", help="All folds/models, two seeds, one penalty pair, two epochs")
    args = parser.parse_args()
    if any(x < 1 for x in args.sessions) or min(args.sequence_length, args.max_epochs, args.early_stop, args.batch_size, args.threads) < 1:
        parser.error("Sequence length, session counts, epochs, patience, batch size and threads must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive")
    penalties = args.sparsity_lambdas + args.energy_lambdas + args.hebbian_lambdas + args.covariance_lambdas
    if any(not math.isfinite(x) or x < 0 for x in penalties):
        parser.error("Regularization penalties must be finite and nonnegative")
    if args.jobs == 0:
        parser.error("--jobs cannot be zero")
    args.sessions = sorted(set(args.sessions))
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    # Absolute repository-supported data root; works independently of cwd.
    ds.DATA_PATH = REPO / "data"
    subject = args.subjects[0]
    args.output = args.output / subject
    subject_path = ds.DATA_PATH / "AB_behaviour" / subject
    sessions = sorted((p for p in subject_path.iterdir() if p.is_dir() and (p / "trials.htsv").is_file()),
                      key=lambda p: datetime.strptime(p.name, "%Y-%m-%d-%H%M%S"))
    if max(args.sessions) > len(sessions):
        raise ValueError(f"Requested {max(args.sessions)} sessions; {subject} has {len(sessions)}")
    source = ds.AB_Dataset(subject_path, sequence_length=args.sequence_length,
                           input_encoding="unipolar", input_forced_choice=True, device=torch.device("cpu"))
    trainer_kwargs = dict(weight_seeds=args.weight_seeds,
                          sparsity_lambdas=args.sparsity_lambdas,
                          energy_lambdas=args.energy_lambdas,
                          hebbian_lambdas=args.hebbian_lambdas, covariance_lambdas=args.covariance_lambdas,
                          learning_rate=args.learning_rate, batch_size=args.batch_size,
                          max_epochs=args.max_epochs, early_stop=args.early_stop)
    if args.smoke_test:
        trainer_kwargs.update(weight_seeds=[1, 2], sparsity_lambdas=[1e-3],
                              energy_lambdas=[1e-2], max_epochs=2)
    plan = []
    block_map = source.subject_df.dropna(subset=["sequence_block_idx"]).groupby("session_folder_name").sequence_block_idx.unique()
    for n in args.sessions:
        names = [f"{subject}/{p.name}" for p in sessions[-n:]]
        ids = sorted(int(b) for name in names for b in block_map.get(name, []))
        plan.append({"n_sessions": n, "sessions": names, "source_block_ids": ids,
                     "n_blocks": len(ids), "folds": block_folds(len(ids), args.seed),
                     "status": "included" if len(ids) >= N_FOLDS else "skipped: fewer than 10 blocks"})
    file_paths = [Path(__file__), REPO / "code/models/rnns.py",
                  REPO / "code/models/training.py", REPO / "code/models/datasets.py",
                  REPO / "code/models/save_data.py"]
    file_paths += [p / "trials.htsv" for p in sessions[-max(args.sessions):]]
    manifest = {"subject": subject, "sequence_length": args.sequence_length, "outer_folds": N_FOLDS,
                "inner_folds": N_FOLDS - 1, "models": MODELS, "trainer": trainer_kwargs,
                "seed": args.seed, "smoke_test": args.smoke_test, "plan": plan,
                "versions": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__},
                "device": "cuda" if torch.cuda.is_available() else "cpu", "threads": args.threads,
                "sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in file_paths}}
    # Normalize tuples to JSON lists before comparing a resumed run.
    manifest = json.loads(json.dumps(manifest))
    if args.resume_from is not None:
        previous = json.loads((args.resume_from / 'run.json').read_text())
        for key in ('subject', 'sequence_length', 'outer_folds', 'inner_folds',
                    'models', 'trainer', 'seed', 'smoke_test', 'plan', 'versions', 'device'):
            if previous[key] != manifest[key]:
                raise ValueError(f'Previous run differs in {key}; cannot import checkpoints')
        for relative, digest in previous['sha256'].items():
            # These two intentional default edits are overridden explicitly here.
            current = (REPO / relative).read_bytes()
            if relative == 'code/models/session_learning_curve.py':
                continue
            if relative == 'code/models/datasets.py':
                current = current.replace(b'SEQUENCE_LENGTH = 32', b'SEQUENCE_LENGTH = 64')
            if relative == 'code/models/training.py':
                current = current.replace(b'batch_size: int = 8', b'batch_size: int = 128')
            if hashlib.sha256(current).hexdigest() != digest:
                raise ValueError(f'Previous run differs in {relative}; cannot import checkpoints')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "run.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Output contains a different run configuration/data/code; use a new --output directory")
    write_json(manifest_path, manifest)
    included = [p for p in plan if p["folds"]]
    for p in plan:
        print(f"{p['n_sessions']} sessions: {p['n_blocks']} blocks — {p['status']}", flush=True)
    n_configs = math.prod(len(trainer_kwargs[key]) for key in (
        "weight_seeds", "sparsity_lambdas", "energy_lambdas", "hebbian_lambdas", "covariance_lambdas"))
    print(f"{len(included) * len(MODELS) * 90} inner fits, each with {n_configs} vectorized configurations", flush=True)
    if args.dry_run or not included:
        return
    jobs = 1 if torch.cuda.is_available() else effective_n_jobs(args.jobs)
    print(f"Running {jobs} inner-fold workers, {args.threads} threads each", flush=True)
    with parallel_config(backend="loky", inner_max_num_threads=args.threads):
        with Parallel(n_jobs=jobs, return_as="generator_unordered", pre_dispatch=jobs) as pool:
            run_training(included, source, args, trainer_kwargs, pool)


def run_training(included, source, args, trainer_kwargs, pool):
    rows = []
    pending = {}
    tasks = []
    for p in included:
        dataset = BlockDataset(source, p["source_block_ids"])
        for kind, activation, bias in MODELS:
            name = f"{kind}_{activation}_bias{'on' if bias else 'off'}"
            directory = (args.output / f"sessions_{p['n_sessions']:02d}" / kind /
                         ("energy" if activation == "relu" else "sparsity"))
            directory.mkdir(parents=True, exist_ok=True)
            model = TinyRNN(rnn_type=kind, hidden_size=2, nonlinearity=activation,
                            input_size=source.inputs.shape[-1], input_encoding="unipolar",
                            input_forced_choice=True, init_decoder=False, decoder_bias=bias)
            for outer in range(N_FOLDS):
                result_path = directory / f"outer_fold_{outer+1}" / f"{model.get_model_id()}_outer_result.json"
                key = (p['n_sessions'], name, outer)
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    rows.append(dict(result, model=name, n_sessions=p['n_sessions'], n_blocks=p['n_blocks']))
                    continue
                pending[key] = [[], model, dataset, p, result_path]
                for inner in range(N_FOLDS):
                    if inner != outer:
                        tasks.append(delayed(tagged_inner)(key, model, dataset, p['folds'], outer,
                                     inner, directory, trainer_kwargs, args.threads,
                                     (args.resume_from / f"sessions_{p['n_sessions']:02d}" / name /
                                      f"outer_{outer+1:02d}_inner_{inner+1:02d}.pt")
                                     if args.resume_from else None))
    for key, result in pool(tasks):
        results, model, dataset, p, result_path = pending[key]
        results.append(result)
        if len(results) != N_FOLDS - 1:
            continue
        best = min(results, key=lambda r: (r['val_nll'], r['inner_fold']))
        model.load_state_dict(best['state_dict'])
        nll, loss_sum, n_free = held_out_nll(model, dataset, p['folds'][key[2]], args.batch_size)
        row = dict(outer_fold=key[2]+1, held_out_nll=nll, nll_sum=loss_sum, n_free=n_free,
                   selected_inner_fold=best['inner_fold'], selected_val_nll=best['val_nll'],
                   winning_config=best['config'])
        write_json(result_path, row)
        rows.append(dict(row, model=key[1], n_sessions=p['n_sessions'], n_blocks=p['n_blocks']))
        # Write numerical progress after each completed outer fold.
        pd.DataFrame(rows).to_csv(args.output / 'fold_results.csv', index=False)
        del pending[key]
    summarize(rows, args.output, args.smoke_test)
    print(f"Saved results in {args.output.resolve()}", flush=True)


def tagged_inner(key, *args):
    return key, train_inner(*args)


if __name__ == "__main__":
    # Keep each mouse's output and resume manifest independent.
    import subprocess
    cli = parse_args()
    mice = cli.subjects or sorted(p.name for p in (REPO / 'data/AB_behaviour').iterdir()
                                  if p.is_dir() and p.name.startswith(('WS', 'MA')))
    if len(mice) == 1:
        # main parses argv again, so explicitly supply the discovered subject.
        if not cli.subjects:
            sys.argv.extend(['--subjects', mice[0]])
        main()
    else:
        original = sys.argv[1:]
        # Strip subject/output arguments; each child gets its own values.
        for flag in ('--subjects', '--output'):
            if flag in original:
                start = original.index(flag)
                end = start + 1
                while end < len(original) and not original[end].startswith('--'):
                    end += 1
                del original[start:end]
        for mouse in mice:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), *original,
                            '--subjects', mouse, '--output', str(cli.output)], check=True)
