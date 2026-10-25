// GDS：删除旧投影（若存在）后重建内存图
// 注意：native projection 要求此处列出的「每种关系类型」在库中至少存在一条实例。
// DeepURL / 仅 DNS 数据通常只有 HAS_IP、SHARED_IP、CONTENT_SIMILAR；若已采集 WHOIS/NS 等，
// 可改为使用 03_gds_project_full.cypher 或在本文件中追加对应关系键。
CALL gds.graph.drop('fraud-projection', false) YIELD graphName;

CALL gds.graph.project(
  'fraud-projection',
  ['Website', 'IP'],
  {
    HAS_IP: {orientation: 'UNDIRECTED'},
    SHARED_IP: {orientation: 'UNDIRECTED'},
    CONTENT_SIMILAR: {orientation: 'UNDIRECTED'}
  }
) YIELD graphName, nodeCount, relationshipCount;
