// 完整异构投影（适用于已存在下列关系类型的图；若某类型在库中不存在，gds.graph.project 会失败）
CALL gds.graph.drop('fraud-projection', false) YIELD graphName;

CALL gds.graph.project(
  'fraud-projection',
  ['Website', 'IP', 'RegistrantEmail', 'AnalyticsID', 'NameServer', 'SSLCert'],
  {
    HAS_IP: {orientation: 'UNDIRECTED'},
    USES_NS: {orientation: 'UNDIRECTED'},
    REGISTERED_BY: {orientation: 'UNDIRECTED'},
    HAS_ANALYTICS: {orientation: 'UNDIRECTED'},
    HAS_CERT: {orientation: 'UNDIRECTED'},
    CONTENT_SIMILAR: {orientation: 'UNDIRECTED'},
    SHARED_IP: {orientation: 'UNDIRECTED'},
    SHARED_EMAIL: {orientation: 'UNDIRECTED'},
    SHARED_NS: {orientation: 'UNDIRECTED'}
  }
) YIELD graphName, nodeCount, relationshipCount;
