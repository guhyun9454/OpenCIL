"""
main_ood_vil.py

OOD-VIL (VIL scenario)에서 BER(Bi-directional Energy Regularization)를 실험하기 위한 단일 실행 스크립트입니다.

- 데이터 로더: continual_datasets/build_incremental_scenario.py 의 build_continual_dataloader
- 평가/로깅 로직: integration/oodvil code.md 의 evaluate_till_now/evaluate_ood 흐름을 동일하게 반영
- Backbone: timm vit_base_patch16_224
- Replay Buffer: 최대 용량을 Bytes 단위 하이퍼파라미터로 지정 (tensor.element_size() * tensor.nelement() 합으로 계산)
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset

import timm

from continual_datasets.build_incremental_scenario import build_continual_dataloader
from continual_datasets.dataset_utils import RandomSampleWrapper, get_ood_dataset, set_data_config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_total_bytes(t: torch.Tensor) -> int:
    # 요구사항: Total Bytes = tensor.element_size() * tensor.nelement()
    return int(t.element_size() * t.nelement())


def parse_bytes(x: str) -> int:
    """
    e.g. "1048576", "512MB", "2gb", "64kb"
    """
    if isinstance(x, int):
        return int(x)
    s = str(x).strip().lower().replace("_", "")
    if s.isdigit():
        return int(s)

    units = {
        "b": 1,
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
    }
    for unit, mul in units.items():
        if s.endswith(unit) and len(s) > len(unit):
            num = float(s[: -len(unit)])
            return int(num * mul)
    raise argparse.ArgumentTypeError(f"Invalid bytes value: {x} (examples: 1048576, 512MB, 2GB)")


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """Top-k accuracy. Returns list of accuracies(%) as tensors."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class BalancedByteReplayBuffer:
    """
    Bytes 단위로 관리되는 task-balanced replay buffer.

    - 전체 max_bytes를 지금까지 본 task 수로 균등 분배(quota)하고,
      각 task가 quota를 넘지 않게 샘플을 보관합니다.
    - 바이트 계산은 (tensor.element_size() * tensor.nelement())의 합으로 수행합니다.
    """

    def __init__(self, max_bytes: int, device: str | torch.device = "cpu"):
        self.max_bytes = int(max_bytes)
        self.device = torch.device(device)

        self._storage: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self._task_bytes: Dict[int, int] = {}
        self._total_bytes: int = 0

    def __len__(self) -> int:
        return sum(len(v) for v in self._storage.values())

    @property
    def total_bytes(self) -> int:
        return int(self._total_bytes)

    def _sample_bytes(self, x: torch.Tensor, y: torch.Tensor) -> int:
        return tensor_total_bytes(x) + tensor_total_bytes(y)

    def _trim_task_to_quota(self, task_id: int, quota: int) -> None:
        if task_id not in self._storage:
            return

        samples = self._storage[task_id]
        cur_bytes = self._task_bytes.get(task_id, 0)

        if quota <= 0:
            # drop all
            self._storage[task_id] = []
            self._task_bytes[task_id] = 0
            self._total_bytes -= cur_bytes
            return

        while samples and cur_bytes > quota:
            idx = random.randrange(len(samples))
            x, y = samples.pop(idx)
            b = self._sample_bytes(x, y)
            cur_bytes -= b
            self._total_bytes -= b

        self._task_bytes[task_id] = cur_bytes

    def rebalance(self, num_tasks_seen: int) -> None:
        if self.max_bytes <= 0:
            self._storage.clear()
            self._task_bytes.clear()
            self._total_bytes = 0
            return

        num_tasks_seen = max(int(num_tasks_seen), 1)
        quota = self.max_bytes // num_tasks_seen

        for tid in list(self._storage.keys()):
            self._trim_task_to_quota(tid, quota)

    @torch.no_grad()
    def add_from_loader(
        self,
        data_loader: torch.utils.data.DataLoader,
        task_id: int,
        num_tasks_seen: int,
        max_batches: Optional[int] = None,
    ) -> None:
        if self.max_bytes <= 0:
            return

        self.rebalance(num_tasks_seen)
        quota = self.max_bytes // max(int(num_tasks_seen), 1)

        if task_id not in self._storage:
            self._storage[task_id] = []
            self._task_bytes[task_id] = 0

        cur_bytes = self._task_bytes[task_id]
        if cur_bytes >= quota:
            return

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = inputs.cpu()
            targets = targets.cpu()

            for i in range(inputs.size(0)):
                x = inputs[i].contiguous()
                y = targets[i]
                if not torch.is_tensor(y):
                    y = torch.tensor(y, dtype=torch.long)
                y = y.view(1).contiguous()

                b = self._sample_bytes(x, y)
                if b > quota:
                    continue
                if cur_bytes + b > quota:
                    break

                self._storage[task_id].append((x, y))
                cur_bytes += b
                self._total_bytes += b

            if cur_bytes >= quota:
                break

        self._task_bytes[task_id] = cur_bytes

    def sample(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self) == 0:
            raise ValueError("Replay buffer is empty – cannot sample.")

        all_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for tid in sorted(self._storage.keys()):
            all_samples.extend(self._storage[tid])

        n = min(int(n_samples), len(all_samples))
        idx = torch.randperm(len(all_samples))[:n].tolist()

        xs = [all_samples[i][0] for i in idx]
        ys = [all_samples[i][1] for i in idx]  # each is shape [1]

        x = torch.stack(xs, dim=0).to(self.device)
        y = torch.cat(ys, dim=0).to(self.device)
        return x, y


class ViTBER(nn.Module):
    """
    - backbone: timm vit_base_patch16_224 (pretrained)
    - head: ID 분류용(기본) head
    - aux_head: BER(OOD 탐지용 extra classifier) head
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,  # backbone(x) -> feature vector
        )
        self.embed_dim = getattr(self.backbone, "num_features", 768)
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.aux_head = nn.Linear(self.embed_dim, num_classes)
        self.num_classes = int(num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # (B, C)
        if feats.dim() == 3:
            feats = feats[:, 0]
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.head(feats)

    def forward_aux(self, x: torch.Tensor, detach_backbone: bool = False) -> torch.Tensor:
        feats = self.forward_features(x)
        if detach_backbone:
            feats = feats.detach()
        return self.aux_head(feats)

    @torch.no_grad()
    def sync_aux_from_head(self) -> None:
        self.aux_head.weight.copy_(self.head.weight)
        self.aux_head.bias.copy_(self.head.bias)


def energy(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """E(x) = -tau * logsumexp(logits / tau). (shape: [B])"""
    return -float(tau) * torch.logsumexp(logits / float(tau), dim=1)


def id_score_from_logits(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """
    OOD 평가에서 positive=ID(=1)로 두므로,
    ID일수록 score가 커지도록 score = -E(x) = tau * logsumexp(logits / tau) 사용.
    """
    return float(tau) * torch.logsumexp(logits / float(tau), dim=1)


def save_accuracy_heatmap(matrix: np.ndarray, task_id: int, args: argparse.Namespace) -> str:
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"accuracy_heatmap_task_{task_id+1:03d}.png")

    plt.figure(figsize=(max(6, matrix.shape[1] * 0.7), max(5, matrix.shape[0] * 0.7)))
    mask = np.isnan(matrix)
    sns.heatmap(
        matrix,
        mask=mask,
        annot=not args.fast_plot,
        fmt=".1f",
        cmap="viridis",
        vmin=0,
        vmax=100,
        cbar=True,
        square=True,
    )
    plt.title(f"Accuracy Heatmap (till task {task_id+1})")
    plt.xlabel("Task")
    plt.ylabel("Task")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def save_anomaly_histogram(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    args: argparse.Namespace,
    suffix: str = "",
    task_id: Optional[int] = None,
) -> str:
    import matplotlib.pyplot as plt

    os.makedirs(args.output_dir, exist_ok=True)
    tid = -1 if task_id is None else int(task_id)
    tag = f"_task_{tid:03d}" if tid >= 0 else ""
    suf = f"_{suffix}" if suffix else ""
    out_path = os.path.join(args.output_dir, f"anomaly_hist{suf}{tag}.png")

    plt.figure(figsize=(7, 4))
    plt.hist(id_scores, bins=50, alpha=0.6, label="ID", density=True)
    plt.hist(ood_scores, bins=50, alpha=0.6, label="OOD", density=True)
    plt.legend()
    plt.title("Score distribution (higher = more ID)")
    plt.xlabel("score")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


SUPPORTED_METHODS = ["BER", "ENERGY", "MSP", "MAXLOGIT"]


@torch.no_grad()
def compute_ood_scores(
    method: str,
    model: ViTBER,
    id_loader: torch.utils.data.DataLoader,
    ood_loader: torch.utils.data.DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    method = method.upper()

    def _scores(loader: torch.utils.data.DataLoader, use_aux: bool) -> torch.Tensor:
        scores: List[torch.Tensor] = []
        for batch_idx, (inputs, _) in enumerate(loader):
            if args.develop and batch_idx > 20:
                break
            inputs = inputs.to(device)

            if method == "BER":
                logits = model.forward_aux(inputs, detach_backbone=True)
                s = id_score_from_logits(logits, args.tau)
            elif method == "ENERGY":
                logits = model(inputs)
                s = id_score_from_logits(logits, args.tau)
            elif method == "MSP":
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                s = probs.max(dim=1).values
            elif method == "MAXLOGIT":
                logits = model(inputs)
                s = logits.max(dim=1).values
            else:
                raise ValueError(f"Unsupported OOD method: {method}")

            scores.append(s.detach().cpu())
        return torch.cat(scores, dim=0)

    id_scores = _scores(id_loader, use_aux=(method == "BER"))
    ood_scores = _scores(ood_loader, use_aux=(method == "BER"))
    return id_scores, ood_scores


class OODVILBERExperiment:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)

        self.model = ViTBER(num_classes=args.num_classes, pretrained=not args.no_pretrained).to(self.device)

        # Replay buffer is used as M_t (old task memory)
        self.replay = BalancedByteReplayBuffer(max_bytes=args.replay_buffer_bytes, device=self.device)

        # base(ID) optimizer: backbone + head (aux_head 제외)
        base_params = list(self.model.backbone.parameters()) + list(self.model.head.parameters())
        self.base_optimizer = torch.optim.AdamW(base_params, lr=args.base_lr, weight_decay=args.weight_decay)
        self.base_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self.base_optimizer, T_max=args.epochs) if args.use_scheduler else None
        )

        # BER optimizer: aux_head only
        self.ber_optimizer = torch.optim.SGD(
            self.model.aux_head.parameters(),
            lr=args.ber_lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
        self.ber_scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self.ber_optimizer, T_max=args.ber_epochs) if args.use_scheduler else None
        )

    def _set_train_mode(self, *, train_backbone: bool, train_head: bool, train_aux: bool) -> None:
        for p in self.model.backbone.parameters():
            p.requires_grad = train_backbone
        for p in self.model.head.parameters():
            p.requires_grad = train_head
        for p in self.model.aux_head.parameters():
            p.requires_grad = train_aux

        # module mode (dropout/bn)
        self.model.backbone.train(train_backbone)
        self.model.head.train(train_head)
        self.model.aux_head.train(train_aux)

    def train_one_epoch_base(
        self,
        model: ViTBER,
        criterion: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        args: argparse.Namespace,
    ) -> Tuple[float, float]:
        self._set_train_mode(train_backbone=True, train_head=True, train_aux=False)
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if args.develop and batch_idx > 20:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            # replay (old tasks)
            if args.replay_batch_size > 0 and len(self.replay) > 0:
                n_rep = min(args.replay_batch_size, len(self.replay), inputs.size(0))
                rep_x, rep_y = self.replay.sample(n_rep)
                inputs_all = torch.cat([inputs, rep_x], dim=0)
                targets_all = torch.cat([targets, rep_y], dim=0)
            else:
                inputs_all, targets_all = inputs, targets

            logits = model(inputs_all)
            loss = criterion(logits, targets_all)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            acc1 = accuracy(logits, targets_all, topk=(1,))[0].item()
            bs = inputs_all.size(0)

            total_loss += loss.item() * bs
            total_acc += acc1 * bs
            total_samples += bs

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_acc / max(total_samples, 1)
        return float(avg_loss), float(avg_acc)

    def train_one_epoch_ber(
        self,
        model: ViTBER,
        criterion: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        args: argparse.Namespace,
    ) -> Tuple[float, float]:
        """
        BER 학습: backbone + head 동결, aux_head만 학습.
        Loss = CE + alpha * (NTER + OTER)
        """
        self._set_train_mode(train_backbone=False, train_head=False, train_aux=True)
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        beta_dist = torch.distributions.beta.Beta(args.ber_beta, args.ber_beta)

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if args.develop and batch_idx > 20:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            # === CE term on (new + replay) ===
            if args.replay_batch_size > 0 and len(self.replay) > 0:
                n_rep = min(args.replay_batch_size, len(self.replay), inputs.size(0))
                rep_x, rep_y = self.replay.sample(n_rep)
                inputs_all = torch.cat([inputs, rep_x], dim=0)
                targets_all = torch.cat([targets, rep_y], dim=0)
            else:
                inputs_all, targets_all = inputs, targets

            # backbone is frozen -> detach to avoid grad/graph
            aux_logits = model.forward_aux(inputs_all, detach_backbone=True)
            loss_clf = criterion(aux_logits, targets_all)

            # === NTER (pseudo-OOD from new batch) ===
            beta = float(beta_dist.sample(()).item())
            idx = torch.randperm(targets.size(0), device=targets.device)
            # try a few times to reduce same-class pairing
            for _ in range(5):
                if torch.any(targets == targets[idx]):
                    idx = torch.randperm(targets.size(0), device=targets.device)
                else:
                    break
            dummy_inputs = beta * inputs + (1.0 - beta) * inputs[idx]
            dummy_logits = model.forward_aux(dummy_inputs, detach_backbone=True)

            e_in = energy(model.forward_aux(inputs, detach_backbone=True), args.tau)
            e_out = energy(dummy_logits, args.tau)

            loss_n = torch.pow(F.relu(e_in - args.p_in), 2).mean() + torch.pow(F.relu(args.p_out - e_out), 2).mean()

            # === OTER (mix new with old memory) ===
            if len(self.replay) > 0 and args.oter_lambda is not None:
                n_mix = min(inputs.size(0), len(self.replay), args.replay_batch_size if args.replay_batch_size > 0 else inputs.size(0))
                rep_x2, _ = self.replay.sample(n_mix)
                lam = float(args.oter_lambda)
                mixed_old = lam * inputs[:n_mix] + (1.0 - lam) * rep_x2[:n_mix]
                e_mixed = energy(model.forward_aux(mixed_old, detach_backbone=True), args.tau)
                loss_o = torch.pow(F.relu(e_mixed - args.p_in), 2).mean()
            else:
                loss_o = torch.tensor(0.0, device=device)

            loss = loss_clf + float(args.alpha) * (loss_n + loss_o)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            acc1 = accuracy(aux_logits, targets_all, topk=(1,))[0].item()
            bs = inputs_all.size(0)
            total_loss += loss.item() * bs
            total_acc += acc1 * bs
            total_samples += bs

        avg_loss = total_loss / max(total_samples, 1)
        avg_acc = total_acc / max(total_samples, 1)
        return float(avg_loss), float(avg_acc)

    def train_and_evaluate(
        self,
        model: ViTBER,
        criterion: nn.Module,
        data_loader: Sequence[Dict[str, torch.utils.data.DataLoader]],
        device: torch.device,
        class_mask: Optional[List[List[int]]],
        args: argparse.Namespace,
    ) -> None:
        """
        integration/oodvil code.md의 흐름과 동일하게:
        - 각 task 학습 후 evaluate_till_now로 누적 평가
        - (옵션) OOD 평가 + wandb 로깅
        """
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            print(f"{f'Training on Task {task_id+1}/{args.num_tasks}':=^60}")
            train_start = time.time()

            # ------------------------------
            # (A) Base(ID) incremental train
            # ------------------------------
            for epoch in range(args.epochs):
                epoch_start = time.time()
                epoch_avg_loss, epoch_avg_acc = self.train_one_epoch_base(
                    model, criterion, data_loader[task_id]["train"], self.base_optimizer, device, epoch, args
                )
                epoch_duration = time.time() - epoch_start
                print(
                    f"[BASE] Epoch [{epoch+1}/{args.epochs}] Completed in {str(datetime.timedelta(seconds=int(epoch_duration)))}: "
                    f"Avg Loss = {epoch_avg_loss:.4f}, Avg Acc@1 = {epoch_avg_acc:.2f}"
                )
                if self.base_scheduler is not None:
                    self.base_scheduler.step(epoch)

            # ---------------------------------------
            # (B) BER train (freeze base, train aux)
            # ---------------------------------------
            model.sync_aux_from_head()
            for epoch in range(args.ber_epochs):
                epoch_start = time.time()
                epoch_avg_loss, epoch_avg_acc = self.train_one_epoch_ber(
                    model, criterion, data_loader[task_id]["train"], self.ber_optimizer, device, epoch, args
                )
                epoch_duration = time.time() - epoch_start
                print(
                    f"[BER ] Epoch [{epoch+1}/{args.ber_epochs}] Completed in {str(datetime.timedelta(seconds=int(epoch_duration)))}: "
                    f"Avg Loss = {epoch_avg_loss:.4f}, Avg Acc@1 = {epoch_avg_acc:.2f}"
                )
                if self.ber_scheduler is not None:
                    self.ber_scheduler.step(epoch)

            train_duration = time.time() - train_start
            print(f"Task {task_id+1} training completed in {str(datetime.timedelta(seconds=int(train_duration)))}")

            print(f'{f"Testing on Task {task_id+1}/{args.num_tasks}":=^60}')
            eval_start = time.time()
            self.evaluate_till_now(model, data_loader, device, task_id, class_mask, acc_matrix, args)
            eval_duration = time.time() - eval_start
            print(f"Task {task_id+1} evaluation completed in {str(datetime.timedelta(seconds=int(eval_duration)))}")

            if args.ood_dataset:
                print(f"{f'OOD Evaluation':=^60}")
                ood_start = time.time()
                all_id_datasets = torch.utils.data.ConcatDataset([data_loader[t]["val"].dataset for t in range(task_id + 1)])
                ood_dataset = args._ood_dataset_obj  # set in main
                self.evaluate_ood(model, all_id_datasets, ood_dataset, device, args, task_id)
                ood_duration = time.time() - ood_start
                print(f"OOD evaluation after Task {task_id+1} completed in {str(datetime.timedelta(seconds=int(ood_duration)))}")

            # ------------------------------
            # Update replay buffer AFTER eval
            # ------------------------------
            self.replay.add_from_loader(
                data_loader[task_id]["train"],
                task_id=task_id,
                num_tasks_seen=task_id + 1,
                max_batches=args.buffer_update_batches,
            )
            if args.verbose:
                print(f"[Replay] total_bytes={self.replay.total_bytes} bytes | n_samples={len(self.replay)}")

    def evaluate_till_now(
        self,
        model: ViTBER,
        data_loader: Sequence[Dict[str, torch.utils.data.DataLoader]],
        device: torch.device,
        task_id: int,
        class_mask: Optional[List[List[int]]],
        acc_matrix: np.ndarray,
        args: argparse.Namespace,
    ) -> Dict[str, float]:
        for t in range(task_id + 1):
            acc_matrix[t, task_id] = self.evaluate_task(model, data_loader[t]["val"], device, t, class_mask, args)

        A_i = [np.mean(acc_matrix[: i + 1, i]) for i in range(task_id + 1)]
        A_last = A_i[-1]
        A_avg = np.mean(A_i)

        result_str = "[Average accuracy till task{}] A_last: {:.2f} A_avg: {:.2f}".format(task_id + 1, A_last, A_avg)

        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
            result_str += " Forgetting: {:.4f}".format(forgetting)
        else:
            forgetting = 0

        if args.wandb:
            import wandb

            wandb.log({"A_last (↑)": A_last, "A_avg (↑)": A_avg, "Forgetting (↓)": forgetting, "TASK": task_id})

        print(result_str)
        if args.verbose or args.wandb:
            sub_matrix = acc_matrix[: task_id + 1, : task_id + 1]
            result = np.where(np.triu(np.ones_like(sub_matrix, dtype=bool)), sub_matrix, np.nan)
            heatmap_path = save_accuracy_heatmap(result, task_id, args)
            if args.wandb:
                import wandb

                wandb.log({"Accuracy Heatmap": wandb.Image(heatmap_path)})

        return {"Acc@1": float(A_last)}

    def evaluate_task(
        self,
        model: ViTBER,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        task_id: int,
        class_mask: Optional[List[List[int]]],
        args: argparse.Namespace,
    ) -> float:
        criterion = torch.nn.CrossEntropyLoss().to(device)
        model.eval()
        total_acc = 0.0
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if args.develop and batch_idx > 20:
                    break

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                acc1 = accuracy(outputs, targets, topk=(1,))[0]
                batch_size = inputs.size(0)

                total_acc += acc1.item() * batch_size
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                if batch_idx % args.print_freq == 0:
                    running_avg_loss = total_loss / total_samples
                    running_avg_acc = total_acc / total_samples
                    print(
                        f"Task {task_id+1}, Batch [{batch_idx}/{len(data_loader)}]: "
                        f"Running Avg Loss = {running_avg_loss:.2f}, Running Avg Acc@1 = {running_avg_acc:.2f}"
                    )

        avg_acc = total_acc / max(total_samples, 1)
        avg_loss = total_loss / max(total_samples, 1)
        print(f"Task {task_id+1}: Final Avg Loss = {avg_loss:.2f} | Final Avg Acc@1 = {avg_acc:.2f}")
        return float(avg_acc)

    def evaluate_ood(
        self,
        model: ViTBER,
        id_datasets: torch.utils.data.Dataset,
        ood_dataset: torch.utils.data.Dataset,
        device: torch.device,
        args: argparse.Namespace,
        task_id: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        model.eval()

        ood_method = args.ood_method.upper()

        id_size, ood_size = len(id_datasets), len(ood_dataset)
        min_size = min(id_size, ood_size)
        if args.develop:
            min_size = min(min_size, 1000)
        if args.ood_develop:
            min_size = min_size if args.ood_develop is None else min(min_size, int(args.ood_develop))
        if args.verbose:
            print(f"ID dataset size: {id_size}, OOD dataset size: {ood_size}. Using {min_size} samples each for evaluation.")

        id_dataset_aligned = RandomSampleWrapper(id_datasets, min_size, args.seed) if id_size > min_size else id_datasets
        ood_dataset_aligned = RandomSampleWrapper(ood_dataset, min_size, args.seed) if ood_size > min_size else ood_dataset

        id_loader = torch.utils.data.DataLoader(
            id_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

        if ood_method == "ALL":
            methods = SUPPORTED_METHODS
        else:
            methods = [m.strip().upper() for m in ood_method.split(",")]
            unsupported = [m for m in methods if m not in SUPPORTED_METHODS]
            if unsupported:
                raise ValueError(f"지원되지 않는 OOD 메소드: {unsupported}. 지원되는 메소드: {SUPPORTED_METHODS}")

        from sklearn import metrics

        results: Dict[str, Dict[str, float]] = {}

        for method in methods:
            id_scores, ood_scores = compute_ood_scores(method, model, id_loader, ood_loader, device, args)

            if args.verbose or args.wandb:
                hist_path = save_anomaly_histogram(id_scores.numpy(), ood_scores.numpy(), args, suffix=method.lower(), task_id=task_id)
                if args.wandb:
                    import wandb

                    wandb.log({f"Anomaly Histogram TASK {task_id}": wandb.Image(hist_path)})

            binary_labels = np.concatenate([np.ones(id_scores.shape[0]), np.zeros(ood_scores.shape[0])])
            all_scores = np.concatenate([id_scores.numpy(), ood_scores.numpy()])

            fpr, tpr, _ = metrics.roc_curve(binary_labels, all_scores, drop_intermediate=False)
            auroc = metrics.auc(fpr, tpr)
            idx_tpr95 = np.abs(tpr - 0.95).argmin()
            fpr_at_tpr95 = fpr[idx_tpr95]

            print(f"[{method}]: AUROC {auroc * 100:.2f}% | FPR@TPR95 {fpr_at_tpr95 * 100:.2f}%")
            if args.wandb:
                import wandb

                wandb.log({f"{method}_AUROC (↑)": auroc * 100, f"{method}_FPR@TPR95 (↓)": fpr_at_tpr95 * 100, "TASK": task_id})

            results[method] = {"auroc": float(auroc), "fpr_at_tpr95": float(fpr_at_tpr95)}

        return results


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run BER on OOD-VIL (VIL scenario).")

    # data / scenario
    parser.add_argument("--dataset", type=str, default="iDigits", choices=["iDigits", "DomainNet", "CORe50", "CLEAR"])
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--IL_mode", type=str, default="vil", choices=["cil", "dil", "vil", "joint"])
    parser.add_argument("--num_tasks", type=int, default=20, help="VIL에서는 num_domains로 나누어 떨어져야 합니다.")
    parser.add_argument("--shuffle", action="store_true", help="class split 시 클래스 순서를 섞습니다.")

    # training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--ber_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--develop", action="store_true", help="빠른 디버그(배치/샘플 제한)")

    # optimizer
    parser.add_argument("--base_lr", type=float, default=5e-5)
    parser.add_argument("--ber_lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--use_scheduler", action="store_true")

    # BER hyperparams
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--p_in", type=float, default=-10.0)
    parser.add_argument("--p_out", type=float, default=-5.0)
    parser.add_argument("--ber_beta", type=float, default=1.0, help="Beta(ber_beta, ber_beta) for NTER mixup.")
    parser.add_argument("--oter_lambda", type=float, default=0.002, help="OTER mixup coefficient lambda.")

    # replay buffer (bytes)
    parser.add_argument("--replay_buffer_bytes", type=parse_bytes, default=0, help="예: 1048576, 512MB, 2GB")
    parser.add_argument("--replay_batch_size", type=int, default=64)
    parser.add_argument("--buffer_update_batches", type=int, default=50)

    # OOD evaluation
    parser.add_argument("--ood_dataset", type=str, default=None, help="예: Imagenet_R, TinyImagenet, CIFAR100 등")
    parser.add_argument("--ood_method", type=str, default="BER", help=f"ALL 또는 {SUPPORTED_METHODS} 조합(쉼표 구분)")
    parser.add_argument("--ood_develop", type=int, default=None, help="OOD 평가 샘플 수 제한")

    # logging / output
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/oodvil_ber")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fast_plot", action="store_true", help="heatmap에 숫자 annotation 생략(속도↑)")
    parser.add_argument("--no_pretrained", action="store_true", help="timm backbone pretrained weight 사용 안 함")

    return parser


def main() -> None:
    args = build_argparser().parse_args()
    # continual_datasets.build_continual_dataloader 내부에서 args.verbose 를 강제로 False로 설정하므로,
    # 사용자가 준 verbose 플래그는 따로 보존 후 복구합니다.
    _user_verbose = bool(args.verbose)
    args = set_data_config(args)
    seed_everything(args.seed)

    # output dir
    os.makedirs(args.output_dir, exist_ok=True)

    data_loader, class_mask, domain_list = build_continual_dataloader(args)
    args.verbose = _user_verbose
    if args.ood_dataset:
        args._ood_dataset_obj = get_ood_dataset(args.ood_dataset, args)

    print(args)

    # wandb init (integration/oodvil code.md와 동일 로직)
    args.wandb = False
    if args.wandb_run and args.wandb_project:
        import wandb

        args.wandb = True
        wandb_config = {k: v for k, v in vars(args).items() if not str(k).startswith("_")}
        wandb.init(entity="OODVIL", project=args.wandb_project, name=args.wandb_run, config=wandb_config)
        wandb.config.update({"username": getpass.getuser()})

    exp = OODVILBERExperiment(args)
    criterion = torch.nn.CrossEntropyLoss().to(exp.device)

    exp.train_and_evaluate(exp.model, criterion, data_loader, exp.device, class_mask, args)


if __name__ == "__main__":
    main()

