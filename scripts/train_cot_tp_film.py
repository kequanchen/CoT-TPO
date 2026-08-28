from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def convert_mat_value(x):
    if hasattr(x, "_fieldnames"):
        return {field: convert_mat_value(getattr(x, field)) for field in x._fieldnames}
    if isinstance(x, np.ndarray) and x.dtype == object:
        return [convert_mat_value(item) for item in x]
    return x


def load_mat_samples(mat_path: Path, key: str | None = None) -> list:
    mat = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    if key is not None:
        if key not in mat:
            data_keys = [name for name in mat.keys() if not name.startswith("__")]
            raise KeyError(f"Key {key!r} not found in {mat_path}. Available keys={data_keys}")
        raw_data = mat[key]
    else:
        data_keys = [name for name in mat.keys() if not name.startswith("__")]
        if not data_keys:
            raise KeyError(f"No data variable found in {mat_path}. Keys={list(mat.keys())}")
        raw_data = mat[data_keys[0]]

    data = convert_mat_value(raw_data)
    if isinstance(data, dict):
        return [data]
    return data


def safe_array(value):
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def neighbor_present(neighbor_hist: np.ndarray, x_col: int, y_col: int) -> bool:
    if neighbor_hist is None or neighbor_hist.size == 0:
        return False
    return bool(np.any(np.abs(neighbor_hist[:, [x_col, y_col]]) > 1e-6))


def build_history(sample: dict, args) -> np.ndarray:
    hist_len = args.hist_len
    status = int(sample.get("lane_status", 0))
    ctx = sample["ctx"]

    ego_full = safe_array(ctx["ego"])
    ego_hist = ego_full[:hist_len, :]
    if ego_hist.shape[0] != hist_len:
        raise ValueError("ego history length mismatch")

    ego_x = ego_hist[:, args.ego_x_col]
    ego_y = ego_hist[:, args.ego_y_col]
    ego_vx = ego_hist[:, args.ego_vx_col]
    ego_vy = ego_hist[:, args.ego_vy_col]

    x0 = ego_x[0]
    y0 = ego_y[0]
    ego_feat = np.stack([ego_x - x0, ego_y - y0, ego_vx, ego_vy], axis=1).astype(np.float32)

    if status == 2:
        neighbor_keys = ["phys_ol", "phys_of", "phys_off"]
    else:
        neighbor_keys = ["phys_tl", "phys_tf", "phys_tff"]

    neighbor_feats = []
    for key in neighbor_keys:
        neighbor = safe_array(ctx.get(key, None))
        if neighbor is None:
            neighbor_feats.append(np.zeros((hist_len, 4), dtype=np.float32))
            continue

        neighbor_hist = neighbor[:hist_len, :]
        if neighbor_hist.shape[0] != hist_len:
            padded = np.zeros((hist_len, neighbor_hist.shape[1]), dtype=np.float32)
            n_keep = min(hist_len, neighbor_hist.shape[0])
            padded[:n_keep] = neighbor_hist[:n_keep]
            neighbor_hist = padded

        if not neighbor_present(neighbor_hist, args.neighbor_x_col, args.neighbor_y_col):
            neighbor_feat = np.zeros((hist_len, 4), dtype=np.float32)
        else:
            dx = neighbor_hist[:, args.neighbor_x_col] - ego_x
            dy = neighbor_hist[:, args.neighbor_y_col] - ego_y
            vx = neighbor_hist[:, args.neighbor_vx_col]
            vy = neighbor_hist[:, args.neighbor_vy_col]
            neighbor_feat = np.stack([dx, dy, vx, vy], axis=1).astype(np.float32)

        neighbor_feats.append(neighbor_feat)

    features = np.stack([ego_feat] + neighbor_feats, axis=1).astype(np.float32)
    return features.reshape(-1)


def build_target(sample: dict, args) -> np.ndarray:
    ctx = sample["ctx"]
    ego_full = safe_array(ctx["ego"])
    ego_hist = ego_full[: args.hist_len, :]
    x0 = float(ego_hist[0, args.ego_x_col])
    y0 = float(ego_hist[0, args.ego_y_col])

    future = safe_array(sample["y_future"])[: args.pred_len, :]
    if future.shape[0] != args.pred_len:
        raise ValueError("future trajectory length mismatch")

    target = future.copy()
    target[:, 0] -= x0
    target[:, 1] -= y0
    return target.reshape(-1).astype(np.float32)


def load_llm_vectors(vector_dir: Path) -> tuple[dict[int, np.ndarray], int]:
    ids_path = vector_dir / "ids.npy"
    vectors_path = vector_dir / "c.npy"
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing {ids_path}")
    if not vectors_path.exists():
        raise FileNotFoundError(f"Missing {vectors_path}")

    ids = np.asarray(np.load(str(ids_path)), dtype=np.int64).reshape(-1)
    vectors = np.load(str(vectors_path)).astype(np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"Strategy vectors must be a 2D array, got shape={vectors.shape}")
    if len(ids) != vectors.shape[0]:
        raise ValueError(f"Vector id mismatch in {vector_dir}: ids={len(ids)}, rows={vectors.shape[0]}")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"Strategy-vector IDs must be unique under {vector_dir}")
    return {int(ids[i]): vectors[i] for i in range(len(ids))}, int(vectors.shape[1])


def require_vector_coverage(
    id_to_vector: dict[int, np.ndarray], sample_count: int, split_name: str
) -> None:
    """Require one exact, zero-based vector ID for every MAT row in use."""

    missing = sorted(set(range(sample_count)) - set(id_to_vector))
    if missing:
        preview = ", ".join(str(value) for value in missing[:10])
        raise ValueError(
            f"Missing {split_name} strategy vectors for zero-based MAT indices: {preview}"
        )


class LLMJointDataset(Dataset):
    def __init__(
        self,
        samples: list,
        id_to_vector: dict[int, np.ndarray],
        llm_dim: int,
        args,
        start_id_fallback: int = 0,
    ):
        self.samples = samples
        self.id_to_vector = id_to_vector
        self.llm_dim = llm_dim
        self.args = args
        self.start_id_fallback = start_id_fallback

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = self.start_id_fallback + idx
        hist = build_history(sample, self.args)
        target = build_target(sample, self.args)
        llm_vector = self.id_to_vector.get(sample_id)
        if llm_vector is None:
            raise KeyError(f"No strategy vector found for zero-based MAT index {sample_id}")
        return (
            torch.tensor(hist, dtype=torch.float32),
            torch.tensor(llm_vector, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(sample_id, dtype=torch.int32),
        )


def finite_indices(dataset: Dataset) -> tuple[list[int], list[tuple[int, int]]]:
    good = []
    bad = []
    for idx in range(len(dataset)):
        hist, llm_vector, target, sample_id = dataset[idx]
        is_valid = torch.isfinite(hist).all() and torch.isfinite(llm_vector).all() and torch.isfinite(target).all()
        if is_valid:
            good.append(idx)
        else:
            bad.append((idx, int(sample_id)))
    return good, bad


def compute_mean_std(dataset: Dataset, indices: list[int], dim_hist: int, dim_target: int, eps: float):
    n = 0
    sum_hist = torch.zeros(dim_hist, dtype=torch.float64)
    sum2_hist = torch.zeros(dim_hist, dtype=torch.float64)
    sum_target = torch.zeros(dim_target, dtype=torch.float64)
    sum2_target = torch.zeros(dim_target, dtype=torch.float64)

    for idx in indices:
        hist, _, target, _ = dataset[idx]
        hist64 = hist.double()
        target64 = target.double()
        sum_hist += hist64
        sum2_hist += hist64 * hist64
        sum_target += target64
        sum2_target += target64 * target64
        n += 1

    mean_hist = (sum_hist / n).float()
    var_hist = (sum2_hist / n - mean_hist.double().pow(2)).clamp_min(eps).float()
    std_hist = torch.sqrt(var_hist).clamp_min(eps)

    mean_target = (sum_target / n).float()
    var_target = (sum2_target / n - mean_target.double().pow(2)).clamp_min(eps).float()
    std_target = torch.sqrt(var_target).clamp_min(eps)

    return mean_hist, std_hist, mean_target, std_target


class NormalizedWrapper(Dataset):
    def __init__(self, base_dataset: Dataset, hist_mean, hist_std, target_mean, target_std):
        self.base_dataset = base_dataset
        self.hist_mean = hist_mean
        self.hist_std = hist_std
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        hist, llm_vector, target, sample_id = self.base_dataset[idx]
        hist_norm = (hist - self.hist_mean) / self.hist_std
        target_norm = (target - self.target_mean) / self.target_std
        return hist_norm, llm_vector, target_norm, sample_id


@torch.no_grad()
def denormalize_target(value: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor):
    return value * target_std.to(value.device) + target_mean.to(value.device)


class StrategyStudent(nn.Module):
    def __init__(self, in_dim=160, hid_dim=128, out_dim=31):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CVAE(nn.Module):
    def __init__(self, cond_dim, output_dim, latent_dim, hidden_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(cond_dim + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(self, cond, target):
        hidden = self.encoder(torch.cat([cond, target], dim=1))
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, -10.0, 10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, cond, z):
        return self.decoder(torch.cat([cond, z], dim=1))

    def forward(self, cond, target=None):
        if target is not None:
            mu, logvar = self.encode(cond, target)
            z = self.reparameterize(mu, logvar)
        else:
            batch_size = cond.size(0)
            z = torch.randn(batch_size, self.latent_dim, device=cond.device)
            mu, logvar = None, None
        recon = self.decode(cond, z)
        return recon, mu, logvar


class ResidualFiLMTransformerLayer(nn.Module):
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, gamma, beta):
        hidden = self.norm1(x)
        hidden = gamma * hidden + beta
        attn_out, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        x = x + self.dropout(attn_out)

        hidden = self.norm2(x)
        hidden = gamma * hidden + beta
        return x + self.dropout(self.ffn(hidden))


class StrategyFiLMResidualTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        strategy_dim,
        pred_len,
        d_model,
        nhead,
        num_layers,
        gamma_scale,
        beta_scale,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.num_layers = num_layers
        self.d_model = d_model
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, pred_len, d_model))
        self.strategy_to_film = nn.Sequential(
            nn.Linear(strategy_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_layers * 2 * d_model),
        )
        self.layers = nn.ModuleList(
            [ResidualFiLMTransformerLayer(d_model=d_model, nhead=nhead) for _ in range(num_layers)]
        )
        self.out = nn.Linear(d_model, 2)

    def forward(self, feature, strategy):
        batch_size = feature.size(0)
        hidden = self.input_proj(feature).unsqueeze(1).repeat(1, self.pred_len, 1)
        hidden = hidden + self.pos_emb

        film = self.strategy_to_film(strategy).view(batch_size, self.num_layers, 2, self.d_model)
        gamma_raw = film[:, :, 0, :].unsqueeze(2)
        beta_raw = film[:, :, 1, :].unsqueeze(2)
        gamma = 1.0 + self.gamma_scale * torch.tanh(gamma_raw)
        beta = self.beta_scale * torch.tanh(beta_raw)

        for layer_idx, layer in enumerate(self.layers):
            hidden = layer(hidden, gamma[:, layer_idx], beta[:, layer_idx])

        delta = self.out(hidden).reshape(batch_size, -1)
        film_strength = (gamma - 1.0).abs().mean() + beta.abs().mean()
        return delta, film_strength


class JointLLMCVAETransformerResidual(nn.Module):
    def __init__(self, llm_dim, output_dim, args):
        super().__init__()
        self.output_dim = output_dim
        self.cond_dim = args.hist_dim
        self.cvae = CVAE(
            cond_dim=self.cond_dim,
            output_dim=output_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
        )
        self.fuse_dim_in = args.hist_dim + output_dim
        self.transformer = StrategyFiLMResidualTransformer(
            input_dim=self.fuse_dim_in,
            strategy_dim=llm_dim,
            pred_len=args.pred_len,
            d_model=args.transformer_dim,
            nhead=args.transformer_heads,
            num_layers=args.transformer_layers,
            gamma_scale=args.film_gamma_scale,
            beta_scale=args.film_beta_scale,
        )

    def sample_candidate(self, cond_for_cvae, num_samples):
        samples = []
        for _ in range(num_samples):
            sample, _, _ = self.cvae(cond_for_cvae, target=None)
            samples.append(sample.unsqueeze(0))
        return torch.cat(samples, dim=0).mean(dim=0)

    def forward(self, hist_norm, strategy_raw, num_samples, strategy_dropout_p=0.0):
        if self.training and strategy_dropout_p > 0:
            keep_mask = (torch.rand(strategy_raw.size(0), 1, device=strategy_raw.device) > strategy_dropout_p).float()
            strategy_raw = strategy_raw * keep_mask

        cond_for_cvae = hist_norm
        candidate = self.sample_candidate(cond_for_cvae, num_samples=num_samples)
        fused = torch.cat([hist_norm, candidate], dim=1)
        delta, film_strength = self.transformer(fused, strategy_raw)
        final = candidate + delta
        return final, candidate, delta, cond_for_cvae, film_strength


def kl_div(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def ade_fde_from_flat(pred_flat: torch.Tensor, target_flat: torch.Tensor, pred_len: int):
    batch_size = pred_flat.size(0)
    pred = pred_flat.view(batch_size, pred_len, 2)
    target = target_flat.view(batch_size, pred_len, 2)
    l2 = torch.norm(pred - target, dim=-1)
    return l2.mean(), l2[:, -1].mean()


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def evaluate_trajectory_model(
    model: nn.Module,
    student: nn.Module | None,
    data_loader: DataLoader,
    device: torch.device,
    mse: nn.Module,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    args,
) -> dict[str, float]:
    model.eval()
    if student is not None:
        student.eval()

    total_mse = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_film_strength = 0.0
    total_seen = 0

    for hist_norm, llm_offline, target_norm, _ in data_loader:
        hist_norm = hist_norm.to(device)
        llm_offline = llm_offline.to(device)
        target_norm = target_norm.to(device)

        strategy_in = student(hist_norm) if student is not None else llm_offline
        target = denormalize_target(target_norm, target_mean, target_std)

        repeat_mse = 0.0
        repeat_ade = 0.0
        repeat_fde = 0.0
        repeat_film_strength = 0.0
        for _ in range(args.eval_repeats):
            final_norm, _, _, _, film_strength = model(
                hist_norm,
                strategy_in,
                num_samples=args.num_samples,
                strategy_dropout_p=0.0,
            )
            mse_loss = mse(final_norm, target_norm)
            final = denormalize_target(final_norm, target_mean, target_std)
            ade_det, fde_det = ade_fde_from_flat(final, target, pred_len=args.pred_len)
            repeat_mse += float(mse_loss)
            repeat_ade += float(ade_det)
            repeat_fde += float(fde_det)
            repeat_film_strength += float(film_strength.detach())

        repeat_mse /= args.eval_repeats
        repeat_ade /= args.eval_repeats
        repeat_fde /= args.eval_repeats
        repeat_film_strength /= args.eval_repeats

        batch_size = hist_norm.size(0)
        total_seen += batch_size
        total_mse += repeat_mse * batch_size
        total_ade += repeat_ade * batch_size
        total_fde += repeat_fde * batch_size
        total_film_strength += repeat_film_strength * batch_size

    if total_seen == 0:
        raise ValueError("Evaluation set contains no finite samples")

    return {
        "mse_norm": total_mse / total_seen,
        "ade_det": total_ade / total_seen,
        "fde_det": total_fde / total_seen,
        "film_strength": total_film_strength / total_seen,
        "num_samples": total_seen,
    }


def build_optimizer(model: nn.Module, args):
    film_params = []
    base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "strategy_to_film" in name:
            film_params.append(param)
        else:
            base_params.append(param)

    groups = [
        {"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": film_params, "lr": args.film_lr, "weight_decay": args.film_weight_decay},
    ]
    return optim.Adam(groups)


def load_student_checkpoint(student: nn.Module, ckpt_path: Path, device) -> None:
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    for key in ["state_dict", "model", "student"]:
        if isinstance(checkpoint, dict) and key in checkpoint and isinstance(checkpoint[key], dict):
            student.load_state_dict(checkpoint[key], strict=True)
            print(f"Loaded student weights from key='{key}'")
            return
    if isinstance(checkpoint, nn.Module):
        student.load_state_dict(checkpoint.state_dict(), strict=True)
        print("Loaded student weights from module checkpoint")
        return
    if isinstance(checkpoint, dict):
        student.load_state_dict(checkpoint, strict=True)
        print("Loaded student weights from state dict")
        return
    raise ValueError(f"Unsupported student checkpoint format: {ckpt_path}")


def freeze_student(student: nn.Module) -> None:
    """Keep the validation-selected distilled Student fixed downstream."""

    for param in student.parameters():
        param.requires_grad_(False)
    student.eval()


def select_samples(samples: list, max_samples: int | None, split_name: str) -> list:
    if max_samples is None:
        return samples
    if max_samples <= 0:
        return samples
    if len(samples) < max_samples:
        raise ValueError(f"{split_name} set has {len(samples)} samples, but {max_samples} were requested")
    return samples[:max_samples]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CoT-TP FiLM trajectory prediction model.")
    parser.add_argument("--train-mat", type=Path, default=Path("data/train_dataset1to1.mat"))
    parser.add_argument("--val-mat", type=Path, default=Path("data/validation_dataset1to1.mat"))
    parser.add_argument("--test-mat", type=Path, default=Path("data/test_dataset1to1.mat"))
    parser.add_argument("--train-key", type=str, default="train_data")
    parser.add_argument("--val-key", type=str, default="validation_data")
    parser.add_argument("--test-key", type=str, default="test_data")
    parser.add_argument("--train-vector-dir", type=Path, default=Path("doc/traininput"))
    parser.add_argument("--val-vector-dir", type=Path, default=Path("doc/validationinput"))
    parser.add_argument("--test-vector-dir", type=Path, default=Path("doc/testinput"))
    parser.add_argument("--student-ckpt", type=Path, default=Path("checkpoints/strategy_student_distill.pth"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cot_tp_film"))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hist-len", type=int, default=10)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--hist-dim", type=int, default=160)
    parser.add_argument("--ego-x-col", type=int, default=3)
    parser.add_argument("--ego-y-col", type=int, default=4)
    parser.add_argument("--ego-vy-col", type=int, default=16)
    parser.add_argument("--ego-vx-col", type=int, default=17)
    parser.add_argument("--neighbor-x-col", type=int, default=2)
    parser.add_argument("--neighbor-y-col", type=int, default=3)
    parser.add_argument("--neighbor-vx-col", type=int, default=4)
    parser.add_argument("--neighbor-vy-col", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--film-lr", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--film-weight-decay", type=float, default=0.0)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--transformer-dim", type=int, default=128)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--alpha-recon", type=float, default=0.3)
    parser.add_argument("--beta-kl", type=float, default=0.001)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help=(
            "Number of stochastic CVAE candidate trajectories averaged to form "
            "each baseline trajectory in Eq. (9)"
        ),
    )
    parser.add_argument(
        "--eval-repeats",
        type=int,
        default=20,
        help=(
            "Number of independent stochastic predictions used to compute the "
            "mean evaluation metrics"
        ),
    )
    parser.add_argument("--strategy-dropout-p", type=float, default=0.0)
    parser.add_argument("--film-gamma-scale", type=float, default=0.25)
    parser.add_argument("--film-beta-scale", type=float, default=0.10)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-student", dest="use_student", action="store_false")
    parser.set_defaults(use_student=True)
    args = parser.parse_args()

    args.train_mat = resolve_path(args.train_mat)
    args.val_mat = resolve_path(args.val_mat)
    args.test_mat = resolve_path(args.test_mat)
    args.train_vector_dir = resolve_path(args.train_vector_dir)
    args.val_vector_dir = resolve_path(args.val_vector_dir)
    args.test_vector_dir = resolve_path(args.test_vector_dir)
    args.student_ckpt = resolve_path(args.student_ckpt)
    args.out_dir = resolve_path(args.out_dir)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.eval_repeats <= 0:
        raise ValueError("--eval-repeats must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if len({args.train_mat, args.val_mat, args.test_mat}) != 3:
        raise ValueError("Training, validation, and test MAT paths must be distinct")
    if len({args.train_vector_dir, args.val_vector_dir, args.test_vector_dir}) != 3:
        raise ValueError("Training, validation, and test vector directories must be distinct")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dim = args.pred_len * 2

    if not args.train_mat.exists():
        raise FileNotFoundError(f"Train MAT not found: {args.train_mat}")
    if not args.val_mat.exists():
        raise FileNotFoundError(f"Validation MAT not found: {args.val_mat}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = args.out_dir / "scaler_llm_cvae.npz"

    print("Loading MATLAB training samples")
    all_train_samples = load_mat_samples(args.train_mat, args.train_key)
    train_samples = select_samples(all_train_samples, args.max_train_samples, "train")
    print(f"Training samples: available={len(all_train_samples)} used={len(train_samples)}")

    print("Loading MATLAB validation samples")
    all_val_samples = load_mat_samples(args.val_mat, args.val_key)
    val_samples = select_samples(all_val_samples, args.max_val_samples, "validation")
    print(f"Validation samples: available={len(all_val_samples)} used={len(val_samples)}")

    print("Loading strategy vectors")
    id_to_train_vector, llm_dim_train = load_llm_vectors(args.train_vector_dir)
    id_to_val_vector, llm_dim_val = load_llm_vectors(args.val_vector_dir)
    if llm_dim_train != llm_dim_val:
        raise ValueError(f"Strategy vector dimension mismatch: train={llm_dim_train}, val={llm_dim_val}")
    llm_dim = llm_dim_train
    require_vector_coverage(id_to_train_vector, len(train_samples), "training")
    require_vector_coverage(id_to_val_vector, len(val_samples), "validation")

    train_raw = LLMJointDataset(train_samples, id_to_train_vector, llm_dim, args)
    val_raw = LLMJointDataset(val_samples, id_to_val_vector, llm_dim, args)

    good_train, bad_train = finite_indices(train_raw)
    good_val, bad_val = finite_indices(val_raw)
    print(f"Finite training samples: good={len(good_train)} bad={len(bad_train)}")
    print(f"Finite validation samples: good={len(good_val)} bad={len(bad_val)}")
    if len(good_train) == 0 or len(good_val) == 0:
        raise ValueError("No finite samples are available for training or validation")

    hist_mean, hist_std, target_mean, target_std = compute_mean_std(
        train_raw,
        good_train,
        dim_hist=args.hist_dim,
        dim_target=output_dim,
        eps=args.eps,
    )
    np.savez(
        str(scaler_path),
        hist_mean=hist_mean.numpy(),
        hist_std=hist_std.numpy(),
        target_mean=target_mean.numpy(),
        target_std=target_std.numpy(),
    )

    train_dataset = NormalizedWrapper(train_raw, hist_mean, hist_std, target_mean, target_std)
    val_dataset = NormalizedWrapper(val_raw, hist_mean, hist_std, target_mean, target_std)

    train_loader = DataLoader(
        Subset(train_dataset, good_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(val_dataset, good_val),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    student = None
    if args.use_student:
        student = StrategyStudent(in_dim=args.hist_dim, hid_dim=128, out_dim=llm_dim).to(device)
        load_student_checkpoint(student, args.student_ckpt, device)
        freeze_student(student)

    model = JointLLMCVAETransformerResidual(llm_dim=llm_dim, output_dim=output_dim, args=args).to(device)
    optimizer = build_optimizer(model, args)
    mse = nn.MSELoss()

    run_config_path = args.out_dir / "run_config.json"
    metrics_csv_path = args.out_dir / "metrics_history.csv"
    metrics_jsonl_path = args.out_dir / "metrics_history.jsonl"
    best_metrics_path = args.out_dir / "best_metrics.json"
    final_test_metrics_path = args.out_dir / "final_test_metrics.json"
    final_summary_path = args.out_dir / "final_summary.json"

    for path in [
        metrics_csv_path,
        metrics_jsonl_path,
        best_metrics_path,
        final_test_metrics_path,
        final_summary_path,
        args.out_dir / "metrics_history.json",
    ]:
        if path.exists():
            path.unlink()

    run_config = vars(args).copy()
    run_config.update(
        {
            "train_samples_available": len(all_train_samples),
            "train_samples_used": len(train_samples),
            "val_samples_available": len(all_val_samples),
            "val_samples_used": len(val_samples),
            "finite_train_samples": len(good_train),
            "finite_val_samples": len(good_val),
            "llm_dim": llm_dim,
            "output_dim": output_dim,
            "scaler_path": str(scaler_path),
            "selection_split": "validation",
            "selection_metric": "ADE",
            "student_frozen_during_trajectory_training": bool(args.use_student),
        }
    )
    for key, value in list(run_config.items()):
        if isinstance(value, Path):
            run_config[key] = str(value)
    write_json(run_config_path, run_config)

    metric_fields = [
        "epoch",
        "phase",
        "train_total_loss",
        "train_pred_mse",
        "train_recon_mse",
        "train_kl",
        "train_film_strength",
        "val_mse_norm",
        "val_ade_det",
        "val_fde_det",
        "val_film_strength",
        "best_val_ade_so_far",
        "is_best",
    ]

    best_val_metric = float("inf")
    best_record = None
    best_path = args.out_dir / "best_llm_cvae_residual_det.pth"
    metrics_history = []

    for epoch in range(1, args.epochs + 1):
        if args.use_student:
            student.eval()
        model.train()

        train_loss = 0.0
        train_pred_mse = 0.0
        train_recon_mse = 0.0
        train_kl = 0.0
        train_film_sum = 0.0
        n_train = 0

        for hist_norm, llm_offline, target_norm, _ in train_loader:
            hist_norm = hist_norm.to(device)
            llm_offline = llm_offline.to(device)
            target_norm = target_norm.to(device)

            if args.use_student:
                with torch.no_grad():
                    strategy_in = student(hist_norm)
            else:
                strategy_in = llm_offline

            optimizer.zero_grad()
            final_norm, _, _, cond_for_cvae, film_strength = model(
                hist_norm,
                strategy_in,
                num_samples=args.num_samples,
                strategy_dropout_p=args.strategy_dropout_p,
            )
            loss_pred = mse(final_norm, target_norm)
            recon_norm, mu, logvar = model.cvae(cond_for_cvae, target=target_norm)
            loss_recon = mse(recon_norm, target_norm)
            loss_kl = kl_div(mu, logvar)
            loss = loss_pred + args.alpha_recon * loss_recon + args.beta_kl * loss_kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = hist_norm.size(0)
            n_train += batch_size
            train_loss += loss.item() * batch_size
            train_pred_mse += float(loss_pred.detach()) * batch_size
            train_recon_mse += float(loss_recon.detach()) * batch_size
            train_kl += float(loss_kl.detach()) * batch_size
            train_film_sum += float(film_strength.detach()) * batch_size

        train_loss /= n_train
        train_pred_mse /= n_train
        train_recon_mse /= n_train
        train_kl /= n_train
        train_film_strength = train_film_sum / n_train

        val_metrics = evaluate_trajectory_model(
            model,
            student,
            val_loader,
            device,
            mse,
            target_mean,
            target_std,
            args,
        )
        val_mse = val_metrics["mse_norm"]
        val_ade_det = val_metrics["ade_det"]
        val_fde_det = val_metrics["fde_det"]
        val_film_strength = val_metrics["film_strength"]

        phase_flag = "frozen_student" if args.use_student else "teacher_vector"
        print(
            f"Epoch {epoch:03d} [{phase_flag}] | "
            f"Train: total={train_loss:.6f} predMSE={train_pred_mse:.6f} "
            f"reconMSE={train_recon_mse:.6f} kl={train_kl:.6f} "
            f"film_strength={train_film_strength:.3f} | "
            f"Validation: MSE(norm)={val_mse:.6f} ADE={val_ade_det:.6f} "
            f"FDE={val_fde_det:.6f} film_strength={val_film_strength:.3f}"
        )

        is_best = val_ade_det < best_val_metric
        record = {
            "epoch": epoch,
            "phase": phase_flag,
            "train_total_loss": float(train_loss),
            "train_pred_mse": float(train_pred_mse),
            "train_recon_mse": float(train_recon_mse),
            "train_kl": float(train_kl),
            "train_film_strength": float(train_film_strength),
            "val_mse_norm": float(val_mse),
            "val_ade_det": float(val_ade_det),
            "val_fde_det": float(val_fde_det),
            "val_film_strength": float(val_film_strength),
            "best_val_ade_so_far": float(min(best_val_metric, val_ade_det)),
            "is_best": bool(is_best),
        }
        metrics_history.append(record)

        write_header = not metrics_csv_path.exists()
        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metric_fields)
            if write_header:
                writer.writeheader()
            writer.writerow(record)
        with open(metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if epoch % 5 == 0:
            checkpoint_path = args.out_dir / f"ckpt_epoch{epoch:03d}.pth"
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "llm_dim": llm_dim,
                "output_dim": output_dim,
                "selection_split": "validation",
                "metric_ADE_det": val_ade_det,
                "metric_FDE_det": val_fde_det,
                "val_mse_norm": val_mse,
                "film_strength_val": val_film_strength,
                "scaler_path": str(scaler_path),
                "use_student": args.use_student,
                "student_frozen": bool(args.use_student),
                "phase": phase_flag,
            }
            if args.use_student:
                checkpoint["student"] = student.state_dict()
            torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_val_metric = val_ade_det
            best_record = record
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "llm_dim": llm_dim,
                "output_dim": output_dim,
                "selection_split": "validation",
                "best_metric_ADE_det": best_val_metric,
                "best_val_ADE_det": best_val_metric,
                "val_FDE_det": val_fde_det,
                "val_mse_norm": val_mse,
                "film_strength_val": val_film_strength,
                "scaler_path": str(scaler_path),
                "use_student": args.use_student,
                "student_frozen": bool(args.use_student),
                "phase": phase_flag,
            }
            if args.use_student:
                checkpoint["student"] = student.state_dict()
            torch.save(checkpoint, best_path)
            write_json(
                best_metrics_path,
                {
                    "best_epoch": epoch,
                    "selection_split": "validation",
                    "best_val_ade_det": float(best_val_metric),
                    "best_val_fde_det": float(val_fde_det),
                    "best_val_mse_norm": float(val_mse),
                    "best_film_strength_val": float(val_film_strength),
                    "best_model_path": str(best_path),
                    "record": best_record,
                },
            )

    if best_record is None or not best_path.exists():
        raise RuntimeError("Training finished without a validation-selected checkpoint")

    best_checkpoint = torch.load(str(best_path), map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"], strict=True)
    if args.use_student:
        if "student" not in best_checkpoint:
            raise KeyError("Validation-selected checkpoint does not contain student weights")
        student.load_state_dict(best_checkpoint["student"], strict=True)
        freeze_student(student)

    print("Loading the test set for one final evaluation")
    if not args.test_mat.exists():
        raise FileNotFoundError(f"Test MAT not found: {args.test_mat}")
    all_test_samples = load_mat_samples(args.test_mat, args.test_key)
    test_samples = select_samples(all_test_samples, args.max_test_samples, "test")
    print(f"Test samples: available={len(all_test_samples)} used={len(test_samples)}")

    id_to_test_vector, llm_dim_test = load_llm_vectors(args.test_vector_dir)
    if llm_dim_test != llm_dim:
        raise ValueError(f"Strategy vector dimension mismatch: train={llm_dim}, test={llm_dim_test}")
    require_vector_coverage(id_to_test_vector, len(test_samples), "test")

    test_raw = LLMJointDataset(test_samples, id_to_test_vector, llm_dim, args)
    good_test, bad_test = finite_indices(test_raw)
    print(f"Finite test samples: good={len(good_test)} bad={len(bad_test)}")
    if len(good_test) == 0:
        raise ValueError("No finite samples are available for final test evaluation")

    test_dataset = NormalizedWrapper(test_raw, hist_mean, hist_std, target_mean, target_std)
    test_loader = DataLoader(
        Subset(test_dataset, good_test),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    set_seed(args.seed)
    test_metrics = evaluate_trajectory_model(
        model,
        student,
        test_loader,
        device,
        mse,
        target_mean,
        target_std,
        args,
    )
    final_test_payload = {
        "checkpoint": str(best_path),
        "selected_epoch": best_record["epoch"],
        "selection_split": "validation",
        "selection_metric": "validation_ADE",
        "evaluation_seed": args.seed,
        "test_evaluations": 1,
        "test_samples_available": len(all_test_samples),
        "test_samples_used": len(test_samples),
        "finite_test_samples": len(good_test),
        "bad_test_samples": len(bad_test),
        "test_mse_norm": float(test_metrics["mse_norm"]),
        "test_ade_det": float(test_metrics["ade_det"]),
        "test_fde_det": float(test_metrics["fde_det"]),
        "test_film_strength": float(test_metrics["film_strength"]),
    }
    write_json(final_test_metrics_path, final_test_payload)

    run_config.update(
        {
            "test_samples_available": len(all_test_samples),
            "test_samples_used": len(test_samples),
            "finite_test_samples": len(good_test),
            "bad_test_samples": len(bad_test),
            "test_evaluations": 1,
        }
    )
    write_json(run_config_path, run_config)

    write_json(
        final_summary_path,
        {
            "num_epochs": args.epochs,
            "best_record": best_record,
            "last_record": metrics_history[-1] if metrics_history else None,
            "final_test_metrics": final_test_payload,
            "metrics_csv": str(metrics_csv_path),
            "metrics_jsonl": str(metrics_jsonl_path),
            "best_metrics": str(best_metrics_path),
            "best_model_path": str(best_path),
            "final_test_metrics_path": str(final_test_metrics_path),
        },
    )
    write_json(args.out_dir / "metrics_history.json", {"records": metrics_history})
    print(f"Metrics saved to {metrics_csv_path}")
    print(
        f"Final test | MSE(norm)={test_metrics['mse_norm']:.6f} "
        f"ADE={test_metrics['ade_det']:.6f} FDE={test_metrics['fde_det']:.6f} "
        f"film_strength={test_metrics['film_strength']:.3f}"
    )
    print(f"Final test metrics saved to {final_test_metrics_path}")
    print(f"Final summary saved to {final_summary_path}")


if __name__ == "__main__":
    main()
