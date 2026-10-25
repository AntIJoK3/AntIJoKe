from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from crypo_side.config import get_settings
from crypo_side.db import make_driver
from crypo_side.ml.graphsage import FraudGraphSAGE
from crypo_side.ml.pyg_data import build_pyg_data


def masks(n: int, train_ratio: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(n * train_ratio)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + max(1, (n - n_train) // 2)]
    test_idx = perm[n_train + len(val_idx) :]
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def stratified_train_val_test_masks(
    y: torch.Tensor,
    *,
    test_size: float,
    val_fraction_of_trainval: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按标签分层：先划出 test_size 为测试集，再在剩余 (1-test_size) 内划出验证集用于早停。

    训练 / 验证 / 测试 三个子集中 **normal(0) 与 fraud(1) 占比与全图一致**（sklearn stratify）。
    梯度更新仅使用 train_mask；与「训练+验证=80%、测试=20%」一致时，取 test_size=0.2，
    并在 80% 内用 val_fraction_of_trainval 切出验证（例如 0.25 → 总节点约 60% 训练 / 20% 验证 / 20% 测试）。
    """
    n = y.size(0)
    y_np = y.detach().cpu().numpy()
    idx = np.arange(n, dtype=np.int64)

    idx_tv, idx_te = train_test_split(
        idx,
        test_size=test_size,
        stratify=y_np,
        random_state=seed,
    )
    y_tv = y_np[idx_tv]
    idx_tr, idx_va = train_test_split(
        idx_tv,
        test_size=val_fraction_of_trainval,
        stratify=y_tv,
        random_state=seed + 1,
    )

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[torch.from_numpy(idx_tr)] = True
    val_mask[torch.from_numpy(idx_va)] = True
    test_mask[torch.from_numpy(idx_te)] = True
    return train_mask, val_mask, test_mask


def _print_label_split(name: str, y: torch.Tensor, m: torch.Tensor) -> None:
    yy = y.detach().cpu().numpy().ravel()
    mm = m.detach().cpu().numpy().astype(bool)
    if not mm.any():
        print(f"  {name}: (空)")
        return
    yy = yy[mm]
    n0 = int((yy == 0).sum())
    n1 = int((yy == 1).sum())
    tot = n0 + n1
    print(
        f"  {name}: n={tot}  normal(benign)={n0} ({n0 / tot:.1%})  "
        f"fraud(phishing等)={n1} ({n1 / tot:.1%})"
    )


def _inverse_frequency_class_weights(y: torch.Tensor, mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    y_sub = y[mask]
    counts = torch.bincount(y_sub, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    return y_sub.numel() / (num_classes * counts)


def _boost_minority_class_weight(
    cw: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, multiplier: float
) -> torch.Tensor:
    """在逆频率权重上，对训练集中样本更少的一类再乘以 multiplier（>1 更重视少数类）。"""
    if multiplier == 1.0:
        return cw
    y_sub = y[mask]
    counts = torch.bincount(y_sub, minlength=cw.numel()).float()
    minority = int(torch.argmin(counts))
    out = cw.clone()
    out[minority] *= multiplier
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="训练 GraphSAGE（图卷积）欺诈检测模型")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--device", default="cpu")
    p.add_argument("--hidden", type=int, default=256, help="GraphSAGE 隐藏层宽度（融合特征推荐 256）")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout（融合 514 维特征推荐 0.2）")
    p.add_argument("--weight-decay", type=float, default=1e-3, dest="weight_decay")
    p.add_argument("--patience", type=int, default=30, help="验证损失早停耐心轮数（--no-early-stop 时无效）")
    p.add_argument(
        "--min-epochs",
        type=int,
        default=40,
        dest="min_epochs",
        help="至少训练轮数后再允许早停（--no-early-stop 时无效）",
    )
    p.add_argument(
        "--no-early-stop",
        action="store_true",
        help="关闭早停，始终训练满 --epochs 轮；结束后仍载入验证集 loss 最优的权重做测试与写入 graphsage.pt",
    )
    p.add_argument("--no-class-weight", action="store_true", help="关闭 CrossEntropy 类别权重（均匀）")
    p.add_argument(
        "--minority-weight-multiplier",
        type=float,
        default=2.0,
        help=(
            "在「逆频率」CrossEntropy 类别权重基础上，对训练集里样本更少的一类再乘该系数。"
            "1.0=仅逆频率；2.0≈少数类错误在 loss 中再放大一倍（常见为 fraud）。默认 2.0"
        ),
    )
    p.add_argument(
        "--split-mode",
        choices=("stratified", "random"),
        default="stratified",
        help="stratified=按标签分层划分（默认）；random=原随机 60/20/20 风格",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="stratified 时测试集占比（默认 0.2 即 8:2 中的 2）",
    )
    p.add_argument(
        "--val-fraction-of-trainval",
        type=float,
        default=0.25,
        help="stratified 时在「非测试」子集内划给验证集的比例（默认 0.25 → 总节点约 60%% 训练 / 20%% 验证 / 20%% 测试）",
    )
    args = p.parse_args()

    settings = get_settings()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "当前环境无法使用 CUDA（常见原因：pip 安装的是 CPU 版 PyTorch）。\n"
            "请改用 --device cpu，或从 https://pytorch.org 选择与显卡驱动匹配的 CUDA 版 PyTorch 重装。"
        )

    driver = make_driver()
    with driver.session() as session:
        data, _ = build_pyg_data(session)

    if args.split_mode == "stratified":
        try:
            train_mask, val_mask, test_mask = stratified_train_val_test_masks(
                data.y,
                test_size=args.test_size,
                val_fraction_of_trainval=args.val_fraction_of_trainval,
                seed=settings.random_state,
            )
        except ValueError as e:
            raise SystemExit(
                "分层划分失败（每类样本过少时 sklearn 无法分层）。可改用 --split-mode random，或增加数据。\n"
                f"原因: {e}"
            ) from e
        print(
            "数据划分: stratified — 测试集占比 "
            f"{args.test_size:.0%}；在剩余样本内验证集占该子集的 {args.val_fraction_of_trainval:.0%}。"
            "各集内 normal/fraud 比例与全图一致（benign→normal，phishing 等→fraud）。"
        )
    else:
        train_mask, val_mask, test_mask = masks(data.num_nodes, 0.6, settings.random_state)
        print("数据划分: random（60%% 训练 / 约 20%% 验证 / 约 20%% 测试），非分层。")

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data = data.to(device)

    print("各子集标签分布（用于核对分层比例）:")
    _print_label_split("全图", data.y, torch.ones(data.num_nodes, dtype=torch.bool))
    _print_label_split("train(梯度)", data.y, train_mask.to(device))
    _print_label_split("val(早停)", data.y, val_mask.to(device))
    _print_label_split("test(最终指标)", data.y, test_mask.to(device))

    print(
        "主模型: GraphSAGE — PyTorch Geometric 的 SAGEConv，"
        "在每条边上对邻居特征做聚合，属于图卷积网络（GNN）的一种。"
    )

    model = FraudGraphSAGE(
        data.x.size(1),
        hidden=args.hidden,
        dropout=args.dropout,
        num_layers=2,
    ).to(device)

    if args.no_class_weight:
        criterion = nn.CrossEntropyLoss()
        print("损失函数: CrossEntropyLoss（无类别权重）")
    else:
        cw_base = _inverse_frequency_class_weights(data.y, data.train_mask, num_classes=2)
        cw = _boost_minority_class_weight(
            cw_base, data.y, data.train_mask, args.minority_weight_multiplier
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        print("损失函数: CrossEntropyLoss(weight=...) — 先按训练集逆频率，再对少数类 ×", args.minority_weight_multiplier)
        print("  逆频率基准:", cw_base.detach().cpu().tolist())
        print("  最终权重:", cw.detach().cpu().tolist())

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.no_early_stop:
        print(f"早停: 关闭 — 将跑满 {args.epochs} 轮（结束后仍用验证集最优权重评测与保存）")
    else:
        print(f"早停: 开启 — patience={args.patience}, min_epochs={args.min_epochs}")

    best_val = float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            vloss = float(criterion(out[data.val_mask], data.y[data.val_mask]).item())

        improved = vloss < best_val - 1e-6
        if improved:
            best_val = vloss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            if not args.no_early_stop and epoch + 1 >= args.min_epochs:
                patience_left -= 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"epoch {epoch+1} train_loss={loss.item():.4f} val_loss={vloss:.4f}")

        if (
            not args.no_early_stop
            and epoch + 1 >= args.min_epochs
            and patience_left <= 0
        ):
            print(f"早停于 epoch {epoch+1}（验证损失 {args.patience} 轮未改善）")
            break

    if args.no_early_stop:
        print(f"已跑满 {args.epochs} 轮（未早停）")

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)

        tm = data.test_mask
        y_t = data.y[tm].detach().cpu().numpy()
        p_t = pred[tm].detach().cpu().numpy()

        acc = (pred[tm] == data.y[tm]).float().mean().item()
        print("GraphSAGE 测试准确率:", round(acc, 4))
        print("分类报告 (test):\n", classification_report(y_t, p_t, digits=4, zero_division=0))

    settings.gnn_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "in_channels": data.x.size(1),
            "hparams": {
                "hidden": args.hidden,
                "num_layers": 2,
                "dropout": args.dropout,
                "minority_weight_multiplier": args.minority_weight_multiplier,
                "no_class_weight": args.no_class_weight,
                "no_early_stop": args.no_early_stop,
            },
        },
        settings.gnn_checkpoint,
    )
    print("已保存:", settings.gnn_checkpoint)


if __name__ == "__main__":
    main()
