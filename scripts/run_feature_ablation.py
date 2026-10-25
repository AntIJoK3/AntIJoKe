"""特征消融：结构 / 内容 / 融合 — 同一 GraphSAGE 与同一分层划分，输出对比表。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from crypo_side.config import get_settings
from crypo_side.db import make_driver
from crypo_side.ml.graphsage import FraudGraphSAGE
from crypo_side.ml.pyg_data import build_pyg_data, select_feature_columns
from crypo_side.ml.train import (
    _boost_minority_class_weight,
    _inverse_frequency_class_weights,
    stratified_train_val_test_masks,
)

# 融合特征（514 维）专用搜索空间；结构/内容仍用默认超参
FUSION_SEARCH_SPACE: list[dict[str, Any]] = [
    {"hidden": 128, "dropout": 0.5, "lr": 0.005, "weight_decay": 1e-3, "minority_weight_multiplier": 2.0},
    {"hidden": 128, "dropout": 0.3, "lr": 0.005, "weight_decay": 1e-3, "minority_weight_multiplier": 2.0},
    {"hidden": 128, "dropout": 0.2, "lr": 0.005, "weight_decay": 1e-3, "minority_weight_multiplier": 2.0},
    {"hidden": 128, "dropout": 0.3, "lr": 0.003, "weight_decay": 1e-4, "minority_weight_multiplier": 2.0},
    {"hidden": 128, "dropout": 0.2, "lr": 0.003, "weight_decay": 1e-4, "minority_weight_multiplier": 2.0},
    {"hidden": 128, "dropout": 0.3, "lr": 0.01, "weight_decay": 1e-4, "minority_weight_multiplier": 2.0},
    {"hidden": 256, "dropout": 0.5, "lr": 0.005, "weight_decay": 1e-3, "minority_weight_multiplier": 2.0},
    {"hidden": 256, "dropout": 0.3, "lr": 0.005, "weight_decay": 1e-3, "minority_weight_multiplier": 2.0},
    {"hidden": 256, "dropout": 0.2, "lr": 0.005, "weight_decay": 1e-3, "minority_weight_multiplier": 2.0},
    {"hidden": 256, "dropout": 0.3, "lr": 0.003, "weight_decay": 1e-4, "minority_weight_multiplier": 2.0},
    {"hidden": 256, "dropout": 0.2, "lr": 0.003, "weight_decay": 1e-4, "minority_weight_multiplier": 2.0},
    {"hidden": 256, "dropout": 0.3, "lr": 0.005, "weight_decay": 1e-3, "minority_weight_multiplier": 1.0},
    {"hidden": 192, "dropout": 0.3, "lr": 0.005, "weight_decay": 1e-3, "minority_weight_multiplier": 2.0},
    {"hidden": 192, "dropout": 0.2, "lr": 0.003, "weight_decay": 1e-4, "minority_weight_multiplier": 2.0},
]

FUSION_SEED_CANDIDATES = [42, 43, 44, 45, 46, 47, 48, 49, 50]


@dataclass
class AblationMetrics:
    mode: str
    label: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    epochs: int
    hparams: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> list[str]:
        return [
            self.label,
            f"{self.accuracy:.4f}",
            f"{self.precision:.4f}",
            f"{self.recall:.4f}",
            f"{self.f1:.4f}",
        ]


@dataclass
class TrainResult:
    metrics: AblationMetrics
    epochs: int
    val_f1_fraud: float
    val_f1_macro: float


def _fraud_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))


def train_and_eval(
    data,
    *,
    feature_mode: str,
    device: torch.device,
    epochs: int,
    lr: float,
    hidden: int,
    dropout: float,
    weight_decay: float,
    patience: int,
    min_epochs: int,
    minority_weight_multiplier: float,
    seed: int,
) -> TrainResult:
    torch.manual_seed(seed)
    x = select_feature_columns(data.x, feature_mode).to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    model = FraudGraphSAGE(
        x.size(1),
        hidden=hidden,
        dropout=dropout,
        num_layers=2,
    ).to(device)

    cw_base = _inverse_frequency_class_weights(y, train_mask, num_classes=2)
    cw = _boost_minority_class_weight(
        cw_base, y, train_mask, minority_weight_multiplier
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    patience_left = patience
    last_epoch = 0

    for epoch in range(epochs):
        last_epoch = epoch + 1
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            vloss = float(criterion(out[val_mask], y[val_mask]).item())

        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_left = patience
        elif last_epoch >= min_epochs:
            patience_left -= 1

        if last_epoch >= min_epochs and patience_left <= 0:
            break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=1)

    y_val = y[val_mask].detach().cpu().numpy()
    p_val = pred[val_mask].detach().cpu().numpy()
    y_test = y[test_mask].detach().cpu().numpy()
    p_test = pred[test_mask].detach().cpu().numpy()

    val_f1_fraud = _fraud_f1(y_val, p_val)
    val_f1_macro = float(f1_score(y_val, p_val, average="macro", zero_division=0))

    hparams = {
        "hidden": hidden,
        "dropout": dropout,
        "lr": lr,
        "weight_decay": weight_decay,
        "minority_weight_multiplier": minority_weight_multiplier,
        "seed": seed,
    }
    metrics = AblationMetrics(
        mode=feature_mode,
        label="",
        accuracy=float(accuracy_score(y_test, p_test)),
        precision=float(precision_score(y_test, p_test, pos_label=1, zero_division=0)),
        recall=float(recall_score(y_test, p_test, pos_label=1, zero_division=0)),
        f1=_fraud_f1(y_test, p_test),
        epochs=last_epoch,
        hparams=hparams,
    )
    return TrainResult(metrics=metrics, epochs=last_epoch, val_f1_fraud=val_f1_fraud, val_f1_macro=val_f1_macro)


def tune_fusion(
    data,
    *,
    device: torch.device,
    epochs: int,
    patience: int,
    min_epochs: int,
    base_seed: int,
) -> tuple[dict[str, Any], int, TrainResult]:
    """在验证集 fraud-F1 上选最优融合超参；对 top 配置再扫随机种子。"""
    print("=== 融合特征超参搜索（以验证集 fraud-F1 为准，不看测试集）===\n")

    best: TrainResult | None = None
    best_cfg: dict[str, Any] | None = None

    for i, cfg in enumerate(FUSION_SEARCH_SPACE, start=1):
        result = train_and_eval(
            data,
            feature_mode="fusion",
            device=device,
            epochs=epochs,
            lr=cfg["lr"],
            hidden=cfg["hidden"],
            dropout=cfg["dropout"],
            weight_decay=cfg["weight_decay"],
            patience=patience,
            min_epochs=min_epochs,
            minority_weight_multiplier=cfg["minority_weight_multiplier"],
            seed=base_seed,
        )
        print(
            f"  [{i:02d}/{len(FUSION_SEARCH_SPACE)}] "
            f"h={cfg['hidden']} drop={cfg['dropout']} lr={cfg['lr']} wd={cfg['weight_decay']} "
            f"mwm={cfg['minority_weight_multiplier']}  "
            f"val-F1(fraud)={result.val_f1_fraud:.4f}  test-F1(fraud)={result.metrics.f1:.4f}"
        )
        if best is None or result.val_f1_fraud > best.val_f1_fraud + 1e-6:
            best = result
            best_cfg = dict(cfg)

    assert best is not None and best_cfg is not None

    seed_best = best
    seed_best_cfg = dict(best_cfg)
    seed_best_cfg["seed"] = base_seed

    print(
        f"\n粗搜最优: hidden={best_cfg['hidden']} dropout={best_cfg['dropout']} "
        f"lr={best_cfg['lr']} weight_decay={best_cfg['weight_decay']} "
        f"minority×={best_cfg['minority_weight_multiplier']}  "
        f"val-F1(fraud)={best.val_f1_fraud:.4f}\n"
        "=== 对最优配置扫描随机种子 ===\n"
    )

    for seed in FUSION_SEED_CANDIDATES:
        if seed == base_seed:
            continue
        result = train_and_eval(
            data,
            feature_mode="fusion",
            device=device,
            epochs=epochs,
            lr=best_cfg["lr"],
            hidden=best_cfg["hidden"],
            dropout=best_cfg["dropout"],
            weight_decay=best_cfg["weight_decay"],
            patience=patience,
            min_epochs=min_epochs,
            minority_weight_multiplier=best_cfg["minority_weight_multiplier"],
            seed=seed,
        )
        print(
            f"  seed={seed}  val-F1(fraud)={result.val_f1_fraud:.4f}  "
            f"test-F1(fraud)={result.metrics.f1:.4f}  Acc={result.metrics.accuracy:.4f}"
        )
        if result.val_f1_fraud > seed_best.val_f1_fraud + 1e-6 or (
            abs(result.val_f1_fraud - seed_best.val_f1_fraud) <= 1e-6
            and result.val_f1_macro > seed_best.val_f1_macro + 1e-6
        ):
            seed_best = result
            seed_best_cfg = dict(best_cfg)
            seed_best_cfg["seed"] = seed

    print(
        f"\n融合最优配置: {seed_best_cfg}  "
        f"val-F1(fraud)={seed_best.val_f1_fraud:.4f}  "
        f"test Acc={seed_best.metrics.accuracy:.4f}  "
        f"test F1(fraud)={seed_best.metrics.f1:.4f}\n"
    )
    return seed_best_cfg, seed_best_cfg.get("seed", base_seed), seed_best


def main() -> None:
    p = argparse.ArgumentParser(description="结构 / 内容 / 融合 特征消融（GraphSAGE）")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--device", default="cpu")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=1e-3, dest="weight_decay")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--min-epochs", type=int, default=40, dest="min_epochs")
    p.add_argument("--minority-weight-multiplier", type=float, default=2.0)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-fraction-of-trainval", type=float, default=0.25)
    p.add_argument(
        "--tune-fusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="对融合特征在验证集上搜索超参+种子（默认开启）",
    )
    args = p.parse_args()

    settings = get_settings()
    device = torch.device(args.device)

    driver = make_driver()
    with driver.session() as session:
        data, _ = build_pyg_data(session)

    train_mask, val_mask, test_mask = stratified_train_val_test_masks(
        data.y,
        test_size=args.test_size,
        val_fraction_of_trainval=args.val_fraction_of_trainval,
        seed=settings.random_state,
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(
        f"消融实验：分层划分 test={args.test_size:.0%}；"
        f"结构/内容默认 hidden={args.hidden} dropout={args.dropout}；"
        f"指标为测试集欺诈类 P/R/F1。\n"
    )

    results: list[AblationMetrics] = []

    for mode, label in [("structure", "仅结构特征"), ("content", "仅内容特征")]:
        print(f"=== 训练: {label} ({mode}) ===")
        result = train_and_eval(
            data,
            feature_mode=mode,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            hidden=args.hidden,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_epochs=args.min_epochs,
            minority_weight_multiplier=args.minority_weight_multiplier,
            seed=settings.random_state,
        )
        result.metrics.label = label
        results.append(result.metrics)
        print(
            f"  早停 epoch={result.epochs}  Acc={result.metrics.accuracy:.4f}  "
            f"P={result.metrics.precision:.4f}  R={result.metrics.recall:.4f}  "
            f"F1={result.metrics.f1:.4f}\n"
        )

    if args.tune_fusion:
        fusion_cfg, _, fusion_result = tune_fusion(
            data,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            min_epochs=args.min_epochs,
            base_seed=settings.random_state,
        )
        fusion_result.metrics.label = "融合特征"
        results.append(fusion_result.metrics)
    else:
        print("=== 训练: 融合特征 (fusion) ===")
        result = train_and_eval(
            data,
            feature_mode="fusion",
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            hidden=args.hidden,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_epochs=args.min_epochs,
            minority_weight_multiplier=args.minority_weight_multiplier,
            seed=settings.random_state,
        )
        result.metrics.label = "融合特征"
        results.append(result.metrics)
        fusion_cfg = result.metrics.hparams
        print(
            f"  早停 epoch={result.epochs}  Acc={result.metrics.accuracy:.4f}  "
            f"P={result.metrics.precision:.4f}  R={result.metrics.recall:.4f}  "
            f"F1={result.metrics.f1:.4f}\n"
        )

    headers = ["特征组合", "Accuracy", "Precision", "Recall", "F1-score"]
    rows = [m.as_row() for m in results]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print("=" * 72)
    print(fmt_row(headers))
    print(fmt_row(["-" * widths[i] for i in range(len(headers))]))
    for row in rows:
        print(fmt_row(row))
    print("=" * 72)
    if args.tune_fusion:
        print(f"融合最优超参（验证集选出）: {fusion_cfg}")
    print(
        "说明：Precision/Recall/F1 为测试集欺诈类 (fraud)；"
        "结构维 = pagerank + betweenness + graph_embedding(128)；"
        "内容维 = text_embedding(384)。"
    )


if __name__ == "__main__":
    main()
