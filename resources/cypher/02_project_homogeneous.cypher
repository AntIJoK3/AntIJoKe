// 同构投影：Website — Website（共享基础设施）；用 elementId 去重无向边对
MATCH (w1:Website)-[:HAS_IP]->(ip:IP)<-[:HAS_IP]-(w2:Website)
WHERE elementId(w1) < elementId(w2)
MERGE (w1)-[r:SHARED_IP]->(w2)
ON CREATE SET r.weight = 1.0
ON MATCH SET r.weight = 1.0;

MATCH (w1:Website)-[:REGISTERED_BY]->(e:RegistrantEmail)<-[:REGISTERED_BY]-(w2:Website)
WHERE elementId(w1) < elementId(w2)
MERGE (w1)-[:SHARED_EMAIL]->(w2);

MATCH (w1:Website)-[:USES_NS]->(n:NameServer)<-[:USES_NS]-(w2:Website)
WHERE elementId(w1) < elementId(w2)
MERGE (w1)-[:SHARED_NS]->(w2);
