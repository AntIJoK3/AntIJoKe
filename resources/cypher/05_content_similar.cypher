// 基于向量索引建立语义相似边（阈值可调）
MATCH (w:Website)
WHERE w.text_embedding IS NOT NULL
WITH w
CALL db.index.vector.queryNodes('content_vec_idx', 12, w.text_embedding)
YIELD node AS other, score
WHERE score > 0.88 AND other <> w
MERGE (w)-[r:CONTENT_SIMILAR]->(other)
ON CREATE SET r.similarity = score
ON MATCH SET r.similarity = score;
