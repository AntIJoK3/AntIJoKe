from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pydantic import BaseModel, ConfigDict, Field

from crypo_side.api.alerts import list_alerts
from crypo_side.api.infer import export_subgraph
from crypo_side.api.predict_pipeline import run_predict_async
from crypo_side.config import get_settings
from crypo_side.db import make_driver

app = FastAPI(
    title="欺诈网站图检测原型",
    version="0.1.0",
    description=(
        "**如何得到预测结果**：打开 **POST /predict** → 点击 **Try it out** → "
        "在 Request body 填入 JSON（至少含 `url`）→ **Execute** → 在页面下方的 **Responses** 里查看返回体，"
        "字段 **`fraud_probability`** 为 0~1 的欺诈概率（越大越可疑），**`mode`** 为 `gnn` 或 `heuristic`。"
        " 本页只是接口文档，不会自动显示上一次预测结果。"
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_static = Path(__file__).resolve().parents[2] / "static"
if _static.exists():
    app.mount("/ui", StaticFiles(directory=str(_static), html=True), name="ui")


class PredictIn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"url": "https://www.wikipedia.org", "label": None},
                {"url": "http://suspicious.example.com", "label": None},
            ]
        }
    )

    url: str = Field(..., description="待检测 URL（必填）")
    label: str | None = Field(
        None,
        description="可选：写入 Neo4j 的标签 fraud 或 normal；纯检测可不填",
        examples=[None, "fraud", "normal"],
    )


class AlertEvidenceOut(BaseModel):
    shared_ips: list[str] = Field(default_factory=list, description="与其他站点共享的 IP")
    shared_ns: list[str] = Field(default_factory=list, description="与其他站点共享的 NS")
    louvain_community: int | None = Field(default=None, description="Louvain 社区编号")
    community_fraud_count: int = Field(description="同社区内已知 fraud 站点数（不含自身）")
    fraud_neighbor_count: int = Field(description="1-hop 同构边上 fraud 邻居数")


class AlertOut(BaseModel):
    alert_id: str
    url: str
    fraud_probability: float
    threshold: float
    triggered_at: str
    mode: str
    evidence: AlertEvidenceOut


class PredictOut(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://example.com",
                "fraud_probability": 0.82,
                "gang_hint": "louvain:3,fraud_neighbors:1",
                "mode": "gnn",
                "alert": {
                    "alert_id": "550e8400-e29b-41d4-a716-446655440000",
                    "url": "https://example.com",
                    "fraud_probability": 0.82,
                    "threshold": 0.7,
                    "triggered_at": "2026-05-26T10:00:00+00:00",
                    "mode": "gnn",
                    "evidence": {
                        "shared_ips": ["1.2.3.4"],
                        "shared_ns": ["ns1.example.net"],
                        "louvain_community": 243,
                        "community_fraud_count": 5,
                        "fraud_neighbor_count": 1,
                    },
                },
            }
        }
    )

    url: str = Field(description="归一化后的 URL")
    fraud_probability: float = Field(description="欺诈概率 0~1，越大越可疑")
    gang_hint: str | None = Field(description="Louvain 社区等与黑产邻居的提示，可能为 null")
    mode: str = Field(description="gnn=已加载 GraphSAGE 权重；heuristic=无权重时的粗分")
    alert: AlertOut | None = Field(
        default=None,
        description="仅当 fraud_probability ≥ 阈值（默认 0.7，见 FRAUD_ALERT_THRESHOLD）时返回预警记录",
    )
    timing_ms: dict[str, float] | None = Field(
        default=None,
        description="仅基准测试请求头 X-Profile-Latency: 1 时返回各环节毫秒耗时",
    )


@app.get("/", tags=["说明"])
def index():
    """浏览器根路径：说明如何拿到预测 JSON。"""
    return {
        "message": "欺诈站点图检测 API 已运行",
        "swagger_使用步骤": [
            "1. 打开 /docs",
            "2. 找到 POST /predict → Try it out",
            "3. Request body 填写 {\"url\": \"https://你要测的站\"}（label 可省略）",
            "4. 点 Execute",
            "5. 在下方 Responses → Response body 里查看 fraud_probability、mode、gang_hint",
        ],
        "links": {
            "web_ui": "/ui/",
            "docs": "/docs",
            "health": "/health",
            "graph": "/graph?domain=域名",
            "alerts": "/alerts",
        },
    }


def _neo4j_ping() -> tuple[bool, str | None]:
    try:
        driver = make_driver()
        try:
            driver.verify_connectivity()
            return True, None
        finally:
            driver.close()
    except (ServiceUnavailable, OSError) as e:
        return False, str(e)
    except Neo4jError as e:
        return False, str(e)


@app.get("/health", tags=["系统"])
def health():
    """检查 API 与 Neo4j 是否可用。"""
    neo4j_ok, neo4j_err = _neo4j_ping()
    return {
        "ok": neo4j_ok,
        "neo4j": neo4j_ok,
        "neo4j_uri": get_settings().neo4j_uri,
        "neo4j_error": neo4j_err,
    }


@app.post(
    "/predict",
    response_model=PredictOut,
    tags=["检测"],
    summary="预测单条 URL 的欺诈概率",
    response_description="在 Swagger 中需展开 Responses 查看 JSON；关键字段为 fraud_probability。",
)
async def predict(body: PredictIn, request: Request, response: Response):
    profile = request.headers.get("x-profile-latency") == "1"
    try:
        payload, timings = await run_predict_async(body.url, body.label, profile=profile)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    if profile and timings:
        response.headers["X-Timing-Enrich-Ms"] = f"{timings.get('enrich', 0):.2f}"
        response.headers["X-Timing-E2E-Ms"] = f"{timings.get('e2e', 0):.2f}"
    return PredictOut(**payload, timing_ms=timings)


@app.get(
    "/graph",
    tags=["图数据"],
    summary="导出以域名、URL 或节点 ID 为中心的一跳子图 JSON",
)
def graph(
    domain: str | None = Query(None, description="域名或 URL 片段"),
    url: str | None = Query(None, description="Website 节点的完整 url 属性"),
    center_id: str | None = Query(None, description="Neo4j elementId，用于基础设施节点钻取"),
    max_edges: int = Query(120, ge=10, le=500, description="返回关系数上限"),
):
    if not any([domain, url, center_id]):
        raise HTTPException(
            status_code=400,
            detail="需提供 query 参数 domain、url 或 center_id 之一",
        )
    try:
        return export_subgraph(
            domain=domain,
            url=url,
            center_id=center_id,
            max_edges=max_edges,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get(
    "/alerts",
    tags=["预警"],
    summary="查询历史欺诈预警记录",
)
def alerts_history(
    limit: int = Query(50, ge=1, le=500, description="返回条数上限"),
    min_probability: float | None = Query(
        None, ge=0.0, le=1.0, description="最低欺诈概率过滤（仅查询历史）"
    ),
    since: str | None = Query(
        None,
        description="ISO8601 起始时间，如 2026-05-01T00:00:00+00:00",
    ),
):
    try:
        items = list_alerts(
            limit=limit,
            min_probability=min_probability,
            since_iso=since,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {
        "threshold": get_settings().fraud_alert_threshold,
        "count": len(items),
        "items": items,
    }


def create_app() -> FastAPI:
    return app
