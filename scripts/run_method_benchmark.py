"""方法对比：RF / XGBoost / Node2Vec+XGB / GCN / GraphSAGE — 同一分层划分。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from node2vec import Node2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from crypo_side.config import get_settings
from crypo_side.db import make_driver
from crypo_side.ml.gcn import FraudGCN
from crypo_side.ml.graphsage import FraudGraphSAGE
from crypo_side.ml.pyg_data import build_pyg_data
from crypo_side.ml.train import (
    _boost_minority_class_weight,
    _inverse_frequency_class_weights,
    stratified_train_val_test_masks,
)

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit("请先安装 xgboost: pip install xgboost") from e


@dataclass
class MethodMetrics:
    method: str
    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_row(self) -> list[str]:
        return [
            self.method,
            f"{self.accuracy:.4f}",
            f"{self.precision:.4f}",
            f"{self.recall:.4f}",
            f"{self.f1:.4f}",
        ]


def _fraud_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MethodMetrics:
    return MethodMetrics(
        method="",
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    )


def _masks_to_numpy(
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
):
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy().ravel()
    tr = train_mask.detach().cpu().numpy().astype(bool)
    va = val_mask.detach().cpu().numpy().astype(bool)
    te = test_mask.detach().cpu().numpy().astype(bool)
    return x_np, y_np, tr, va, te


def eval_random_forest(x_np, y_np, tr, te) -> MethodMetrics:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(x_np[tr], y_np[tr])
    pred = clf.predict(x_np[te])
    m = _fraud_metrics(y_np[te], pred)
    m.method = "随机森林（仅节点特征）"
    return m


def eval_xgboost(x_np, y_np, tr, te) -> MethodMetrics:
    n_pos = max(int((y_np[tr] == 1).sum()), 1)
    n_neg = max(int((y_np[tr] == 0).sum()), 1)
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=n_neg / n_pos,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(x_np[tr], y_np[tr], verbose=False)
    pred = clf.predict(x_np[te])
    m = _fraud_metrics(y_np[te], pred)
    m.method = "XGBoost（仅节点特征）"
    return m


def train_node2vec(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    embedding_dim: int = 128,
    walk_length: int = 20,
    num_walks: int = 10,
    seed: int = 42,
) -> np.ndarray:
    if edge_index.numel() == 0:
        return np.zeros((num_nodes, embedding_dim), dtype=np.float32)

    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    ei = edge_index.detach().cpu().numpy()
    for u, v in zip(ei[0], ei[1]):
        g.add_edge(int(u), int(v))

    n2v = Node2Vec(
        g,
        dimensions=embedding_dim,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=1,
        seed=seed,
        quiet=True,
    )
    model = n2v.fit(window=10, min_count=1, batch_words=4)
    emb = np.zeros((num_nodes, embedding_dim), dtype=np.float32)
    for i in range(num_nodes):
        key = str(i)
        if key in model.wv:
            emb[i] = model.wv[key]
    return emb


def eval_node2vec_xgb(
    edge_index: torch.Tensor,
    num_nodes: int,
    y_np: np.ndarray,
    tr,
    te,
    *,
    device: torch.device,
) -> MethodMetrics:
    emb = train_node2vec(edge_index, num_nodes, seed=42)
    n_pos = max(int((y_np[tr] == 1).sum()), 1)
    n_neg = max(int((y_np[tr] == 0).sum()), 1)
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=n_neg / n_pos,
        objective="binary:logistic",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(emb[tr], y_np[tr], verbose=False)
    pred = clf.predict(emb[te])
    m = _fraud_metrics(y_np[te], pred)
    m.method = "Node2Vec + XGBoost"
    return m


def train_gnn(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_epochs: int,
    minority_weight_multiplier: float,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    cw_base = _inverse_frequency_class_weights(y, train_mask, num_classes=2)
    cw = _boost_minority_class_weight(
        cw_base, y, train_mask, minority_weight_multiplier
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    patience_left = patience

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vloss = float(criterion(out[val_mask], y[val_mask]).item())

        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_left = patience
        elif epoch + 1 >= min_epochs:
            patience_left -= 1

        if epoch + 1 >= min_epochs and patience_left <= 0:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


def eval_gnn(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    *,
    method_name: str,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_epochs: int,
    minority_weight_multiplier: float,
    seed: int,
) -> MethodMetrics:
    model = train_gnn(
        model,
        x,
        edge_index,
        y,
        train_mask,
        val_mask,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        min_epochs=min_epochs,
        minority_weight_multiplier=minority_weight_multiplier,
        seed=seed,
    )
    model.eval()
    with torch.no_grad():
        pred = model(x.to(device), edge_index.to(device)).argmax(dim=1)
    y_test = y[test_mask].detach().cpu().numpy()
    p_test = pred[test_mask].detach().cpu().numpy()
    m = _fraud_metrics(y_test, p_test)
    m.method = method_name
    return m


def main() -> None:
    p = argparse.ArgumentParser(description="欺诈检测方法对比实验")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--min-epochs", type=int, default=40, dest="min_epochs")
    p.add_argument("--node2vec-epochs", type=int, default=50)
    p.add_argument("--graphsage-seed", type=int, default=48, help="GraphSAGE 随机种子（融合调优后默认 48）")
    args = p.parse_args()

    settings = get_settings()
    device = torch.device(args.device)

    driver = make_driver()
    with driver.session() as session:
        data, _ = build_pyg_data(session)

    train_mask, val_mask, test_mask = stratified_train_val_test_masks(
        data.y,
        test_size=0.2,
        val_fraction_of_trainval=0.25,
        seed=settings.random_state,
    )

    x_np, y_np, tr, va, te = _masks_to_numpy(data.x, data.y, train_mask, val_mask, test_mask)

    print("方法对比：节点特征 = feat_vector（514 维融合特征）；GNN 使用同构图边。\n")

    results: list[MethodMetrics] = []

    print("1/5 随机森林 …")
    results.append(eval_random_forest(x_np, y_np, tr, te))

    print("2/5 XGBoost …")
    results.append(eval_xgboost(x_np, y_np, tr, te))

    print("3/5 Node2Vec + XGBoost …")
    results.append(
        eval_node2vec_xgb(
            data.edge_index,
            data.num_nodes,
            y_np,
            tr,
            te,
            device=device,
        )
    )

    gnn_kw = dict(
        device=device,
        epochs=args.epochs,
        lr=0.005,
        weight_decay=1e-3,
        patience=args.patience,
        min_epochs=args.min_epochs,
        minority_weight_multiplier=2.0,
        seed=settings.random_state,
    )

    print("4/5 GCN …")
    gcn = FraudGCN(data.x.size(1), hidden=256, dropout=0.2, num_layers=2)
    results.append(
        eval_gnn(
            gcn,
            data.x,
            data.edge_index,
            data.y,
            train_mask,
            val_mask,
            test_mask,
            method_name="GCN",
            **gnn_kw,
        )
    )

    print("5/5 GraphSAGE (SAGEConv) …")
    sage = FraudGraphSAGE(data.x.size(1), hidden=256, dropout=0.2, num_layers=2)
    sage_kw = dict(gnn_kw)
    sage_kw["seed"] = args.graphsage_seed
    results.append(
        eval_gnn(
            sage,
            data.x,
            data.edge_index,
            data.y,
            train_mask,
            val_mask,
            test_mask,
            method_name="GraphSAGE (SAGEConv)",
            **sage_kw,
        )
    )

    headers = ["方法", "Accuracy", "Precision", "Recall", "F1-score"]
    rows = [m.as_row() for m in results]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    print("\n" + "=" * 80)
    print(fmt_row(headers))
    print(fmt_row(["-" * widths[i] for i in range(len(headers))]))
    for row in rows:
        print(fmt_row(row))
    print("=" * 80)
    print("说明：Precision/Recall/F1 为测试集欺诈类 (fraud=1)；划分与 train.py 分层一致。")


if __name__ == "__main__":
    main()
