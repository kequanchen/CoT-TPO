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
    if key is not None and key in mat:
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

    ids = np.load(str(ids_path)).astype(np.int32)
    vectors = np.load(str(vectors_path)).astype(np.float32)
    if len(ids) != vectors.shape[0]:
        raise ValueError(f"Vector id mismatch in {vector_dir}: ids={len(ids)}, rows={vectors.shape[0]}")
    return {int(ids[i]): vectors[i] for i in range(len(ids))}, int(vectors.shape[1])


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
        sample_id = int(sample.get("sample_id", self.start_id_fallback + idx))
        hist = build_history(sample, self.args)
        target = build_target(sample, self.args)
        llm_vector = self.id_to_vector.get(sample_id)
        if llm_vector is None:
            llm_vector = np.zeros((self.llm_dim,), dtype=np.float32)
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


def build_optimizer(model: nn.Module, args, student: nn.Module | None = None, phase2: bool = False):
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
    if phase2 and student is not None:
        student_params = [param for param in student.parameters() if param.requires_grad]
        if student_params:
            groups.append(
                {
                    "params": student_params,
                    "lr": args.student_lr,
                    "weight_decay": args.student_weight_decay,
                }
            )
    return optim.Adam(groups)


def load_student_checkpoint(student: nn.Module, ckpt_path: Path, device, required: bool) -> bool:
    if ckpt_path is None or not ckpt_path.exists():
        if required:
            raise FileNotFoundError(f"Student checkpoint not found: {ckpt_path}")
        print(f"[Warning] Student checkpoint not found: {ckpt_path}. The student is randomly initialized.")
        return False

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    for key in ["state_dict", "model", "student"]:
        if isinstance(checkpoint, dict) and key in checkpoint and isinstance(checkpoint[key], dict):
            student.load_state_dict(checkpoint[key], strict=True)
            print(f"Loaded student weights from key='{key}'")
            return True
    if isinstance(checkpoint, nn.Module):
        student.load_state_dict(checkpoint.state_dict(), strict=True)
        print("Loaded student weights from module checkpoint")
        return True
    if isinstance(checkpoint, dict):
        student.load_state_dict(checkpoint, strict=True)
        print("Loaded student weights from state dict")
        return True
    raise ValueError(f"Unsupported student checkpoint format: {ckpt_path}")


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
    parser.add_argument("--test-mat", type=Path, default=Path("data/test_dataset1to1.mat"))
    parser.add_argument("--train-vector-dir", type=Path, default=Path("doc/traininput"))
    parser.add_argument("--test-vector-dir", type=Path, default=Path("doc/testinput"))
    parser.add_argument("--student-ckpt", type=Path, default=Path("checkpoints/strategy_student_distill.pth"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cot_tp_film"))
    parser.add_argument("--max-train-samples", type=int, default=None)
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
    parser.add_argument("--phase1-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--film-lr", type=float, default=1.5e-4)
    parser.add_argument("--student-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--film-weight-decay", type=float, default=0.0)
    parser.add_argument("--student-weight-decay", type=float, default=0.0)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--transformer-dim", type=int, default=128)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--alpha-recon", type=float, default=0.3)
    parser.add_argument("--beta-kl", type=float, default=0.001)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--strategy-dropout-p", type=float, default=0.0)
    parser.add_argument("--film-gamma-scale", type=float, default=0.25)
    parser.add_argument("--film-beta-scale", type=float, default=0.10)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-student", dest="use_student", action="store_false")
    parser.add_argument("--require-student-ckpt", action="store_true")
    parser.set_defaults(use_student=True)
    args = parser.parse_args()

    args.train_mat = resolve_path(args.train_mat)
    args.test_mat = resolve_path(args.test_mat)
    args.train_vector_dir = resolve_path(args.train_vector_dir)
    args.test_vector_dir = resolve_path(args.test_vector_dir)
    args.student_ckpt = resolve_path(args.student_ckpt)
    args.out_dir = resolve_path(args.out_dir)
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dim = args.pred_len * 2

    if not args.train_mat.exists():
        raise FileNotFoundError(f"Train MAT not found: {args.train_mat}")
    if not args.test_mat.exists():
        raise FileNotFoundError(f"Test MAT not found: {args.test_mat}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = args.out_dir / "scaler_llm_cvae.npz"

    print("Loading MATLAB training samples")
    all_train_samples = load_mat_samples(args.train_mat, "train_data")
    train_samples = select_samples(all_train_samples, args.max_train_samples, "train")
    print(f"Training samples: available={len(all_train_samples)} used={len(train_samples)}")

    print("Loading MATLAB test samples")
    all_test_samples = load_mat_samples(args.test_mat, "test_data")
    test_samples = select_samples(all_test_samples, args.max_test_samples, "test")
    print(f"Test samples: available={len(all_test_samples)} used={len(test_samples)}")

    print("Loading strategy vectors")
    id_to_train_vector, llm_dim_train = load_llm_vectors(args.train_vector_dir)
    id_to_test_vector, llm_dim_test = load_llm_vectors(args.test_vector_dir)
    if llm_dim_train != llm_dim_test:
        raise ValueError(f"Strategy vector dimension mismatch: train={llm_dim_train}, test={llm_dim_test}")
    llm_dim = llm_dim_train

    train_raw = LLMJointDataset(train_samples, id_to_train_vector, llm_dim, args)
    test_raw = LLMJointDataset(test_samples, id_to_test_vector, llm_dim, args)

    good_train, bad_train = finite_indices(train_raw)
    good_test, bad_test = finite_indices(test_raw)
    print(f"Finite training samples: good={len(good_train)} bad={len(bad_train)}")
    print(f"Finite test samples: good={len(good_test)} bad={len(bad_test)}")
    if len(good_train) == 0 or len(good_test) == 0:
        raise ValueError("No finite samples are available for training or evaluation")

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
    test_dataset = NormalizedWrapper(test_raw, hist_mean, hist_std, target_mean, target_std)

    train_loader = DataLoader(
        Subset(train_dataset, good_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        Subset(test_dataset, good_test),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    student = None
    if args.use_student:
        student = StrategyStudent(in_dim=args.hist_dim, hid_dim=128, out_dim=llm_dim).to(device)
        load_student_checkpoint(student, args.student_ckpt, device, args.require_student_ckpt)
        for param in student.parameters():
            param.requires_grad_(False)
        student.eval()

    model = JointLLMCVAETransformerResidual(llm_dim=llm_dim, output_dim=output_dim, args=args).to(device)
    optimizer = build_optimizer(model, args, phase2=False)
    mse = nn.MSELoss()

    run_config_path = args.out_dir / "run_config.json"
    metrics_csv_path = args.out_dir / "metrics_history.csv"
    metrics_jsonl_path = args.out_dir / "metrics_history.jsonl"
    best_metrics_path = args.out_dir / "best_metrics.json"
    final_summary_path = args.out_dir / "final_summary.json"

    for path in [metrics_csv_path, metrics_jsonl_path, best_metrics_path, final_summary_path, args.out_dir / "metrics_history.json"]:
        if path.exists():
            path.unlink()

    run_config = vars(args).copy()
    run_config.update(
        {
            "train_samples_available": len(all_train_samples),
            "train_samples_used": len(train_samples),
            "test_samples_available": len(all_test_samples),
            "test_samples_used": len(test_samples),
            "finite_train_samples": len(good_train),
            "finite_test_samples": len(good_test),
            "llm_dim": llm_dim,
            "output_dim": output_dim,
            "scaler_path": str(scaler_path),
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
        "test_mse_norm",
        "test_ade_det",
        "test_fde_det",
        "test_film_strength",
        "best_ade_so_far",
        "is_best",
    ]

    best_metric = float("inf")
    best_record = None
    metrics_history = []

    for epoch in range(1, args.epochs + 1):
        if args.use_student and epoch == args.phase1_epochs + 1:
            print("Entering phase 2: unfreezing StrategyStudent")
            for param in student.parameters():
                param.requires_grad_(True)
            student.train()
            optimizer = build_optimizer(model, args, student=student, phase2=True)

        if args.use_student and epoch <= args.phase1_epochs:
            student.eval()
        elif args.use_student:
            student.train()
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
                if epoch <= args.phase1_epochs:
                    with torch.no_grad():
                        strategy_in = student(hist_norm)
                else:
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
            if args.use_student and epoch > args.phase1_epochs:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
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

        model.eval()
        if args.use_student:
            student.eval()

        test_mse = 0.0
        test_ade_det = 0.0
        test_fde_det = 0.0
        test_film_sum = 0.0
        n_test = 0

        with torch.no_grad():
            for hist_norm, llm_offline, target_norm, _ in test_loader:
                hist_norm = hist_norm.to(device)
                llm_offline = llm_offline.to(device)
                target_norm = target_norm.to(device)

                if args.use_student:
                    strategy_in = student(hist_norm)
                else:
                    strategy_in = llm_offline

                final_norm, _, _, _, film_strength = model(
                    hist_norm,
                    strategy_in,
                    num_samples=args.num_samples,
                    strategy_dropout_p=0.0,
                )
                mse_loss = mse(final_norm, target_norm)
                final = denormalize_target(final_norm, target_mean, target_std)
                target = denormalize_target(target_norm, target_mean, target_std)
                ade_det, fde_det = ade_fde_from_flat(final, target, pred_len=args.pred_len)

                batch_size = hist_norm.size(0)
                n_test += batch_size
                test_mse += mse_loss.item() * batch_size
                test_ade_det += float(ade_det) * batch_size
                test_fde_det += float(fde_det) * batch_size
                test_film_sum += float(film_strength.detach()) * batch_size

        test_mse /= n_test
        test_ade_det /= n_test
        test_fde_det /= n_test
        test_film_strength = test_film_sum / n_test

        phase_flag = "phase1" if epoch <= args.phase1_epochs else "phase2"
        print(
            f"Epoch {epoch:03d} [{phase_flag}] | "
            f"Train: total={train_loss:.6f} predMSE={train_pred_mse:.6f} "
            f"reconMSE={train_recon_mse:.6f} kl={train_kl:.6f} "
            f"film_strength={train_film_strength:.3f} | "
            f"Test: MSE(norm)={test_mse:.6f} ADE={test_ade_det:.6f} "
            f"FDE={test_fde_det:.6f} film_strength={test_film_strength:.3f}"
        )

        is_best = test_ade_det < best_metric
        record = {
            "epoch": epoch,
            "phase": phase_flag,
            "train_total_loss": float(train_loss),
            "train_pred_mse": float(train_pred_mse),
            "train_recon_mse": float(train_recon_mse),
            "train_kl": float(train_kl),
            "train_film_strength": float(train_film_strength),
            "test_mse_norm": float(test_mse),
            "test_ade_det": float(test_ade_det),
            "test_fde_det": float(test_fde_det),
            "test_film_strength": float(test_film_strength),
            "best_ade_so_far": float(min(best_metric, test_ade_det)),
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
                "metric_ADE_det": test_ade_det,
                "metric_FDE_det": test_fde_det,
                "film_strength_test": test_film_strength,
                "scaler_path": str(scaler_path),
                "use_student": args.use_student,
                "phase": phase_flag,
            }
            if args.use_student:
                checkpoint["student"] = student.state_dict()
            torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_metric = test_ade_det
            best_record = record
            best_path = args.out_dir / "best_llm_cvae_residual_det.pth"
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "llm_dim": llm_dim,
                "output_dim": output_dim,
                "best_metric_ADE_det": best_metric,
                "metric_FDE_det": test_fde_det,
                "film_strength_test": test_film_strength,
                "scaler_path": str(scaler_path),
                "use_student": args.use_student,
                "phase": phase_flag,
            }
            if args.use_student:
                checkpoint["student"] = student.state_dict()
            torch.save(checkpoint, best_path)
            write_json(
                best_metrics_path,
                {
                    "best_epoch": epoch,
                    "best_ade_det": float(best_metric),
                    "best_fde_det": float(test_fde_det),
                    "best_test_mse_norm": float(test_mse),
                    "best_film_strength_test": float(test_film_strength),
                    "best_model_path": str(best_path),
                    "record": best_record,
                },
            )

    write_json(
        final_summary_path,
        {
            "num_epochs": args.epochs,
            "best_record": best_record,
            "last_record": metrics_history[-1] if metrics_history else None,
            "metrics_csv": str(metrics_csv_path),
            "metrics_jsonl": str(metrics_jsonl_path),
            "best_metrics": str(best_metrics_path),
            "best_model_path": str(args.out_dir / "best_llm_cvae_residual_det.pth"),
        },
    )
    write_json(args.out_dir / "metrics_history.json", {"records": metrics_history})
    print(f"Metrics saved to {metrics_csv_path}")
    print(f"Final summary saved to {final_summary_path}")


if __name__ == "__main__":
    main()
