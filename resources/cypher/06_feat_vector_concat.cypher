// 拼接 feat_vector（需已存在 pagerank、betweenness、graph_embedding、text_embedding）
MATCH (w:Website)
WHERE w.text_embedding IS NOT NULL
  AND w.graph_embedding IS NOT NULL
  AND w.pagerank IS NOT NULL
  AND w.betweenness IS NOT NULL
SET w.feat_vector = [toFloat(w.pagerank), toFloat(w.betweenness)] + w.graph_embedding + w.text_embedding;
