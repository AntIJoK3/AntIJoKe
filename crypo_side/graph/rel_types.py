"""Website—Website 同构边类型（用于 PyG / 回退特征 / 推理子图）。

默认仅含图中一定存在的类型；补全 WHOIS/NS 并跑 `02_project_homogeneous.cypher` 后，
可在 `.env` 设置 `GRAPH_WEBSITE_HOMOG_EDGES=SHARED_IP|CONTENT_SIMILAR|SHARED_EMAIL|SHARED_NS`。
"""

from crypo_side.config import get_settings


def website_homog_edge_types() -> str:
    t = get_settings().graph_website_homog_edges.strip()
    return t if t else "SHARED_IP|CONTENT_SIMILAR"
