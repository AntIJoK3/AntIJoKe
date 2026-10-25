// 预警节点约束（可选，Neo4j 5+）
CREATE CONSTRAINT alert_id_unique IF NOT EXISTS
FOR (a:Alert) REQUIRE a.alert_id IS UNIQUE;

// 示例：查询最近预警
// MATCH (a:Alert) RETURN a ORDER BY a.triggered_at DESC LIMIT 20;
