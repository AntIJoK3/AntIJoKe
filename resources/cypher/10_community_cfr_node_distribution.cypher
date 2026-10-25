// 不同社区欺诈浓度（CFR）区间内的节点分布
// CFR = 社区内 fraud 节点数 / (fraud + normal) 节点数
// 前置：04_gds_features 已写入 w.louvain_community
// 仅统计 label ∈ {fraud, normal} 且已分配 Louvain 社区的 Website

MATCH (w:Website)
WHERE w.label IN ['fraud', 'normal']
  AND w.louvain_community IS NOT NULL
WITH w.louvain_community AS comm, w.label AS label
WITH comm,
     count(*) AS node_cnt,
     sum(CASE WHEN label = 'fraud' THEN 1 ELSE 0 END) AS fraud_cnt
WITH comm, node_cnt, fraud_cnt,
     toFloat(fraud_cnt) / node_cnt AS cfr
WITH comm, node_cnt, fraud_cnt, cfr,
     CASE
       WHEN fraud_cnt = 0 THEN 'CFR = 0.0（纯正常）'
       WHEN cfr > 0.0 AND cfr < 0.3 THEN '0.0 < CFR < 0.3'
       WHEN cfr >= 0.3 AND cfr < 0.7 THEN '0.3 ≤ CFR < 0.7'
       WHEN cfr >= 0.7 THEN 'CFR ≥ 0.7（高欺诈浓度）'
     END AS cfr_bin,
     CASE
       WHEN fraud_cnt = 0 THEN 1
       WHEN cfr > 0.0 AND cfr < 0.3 THEN 2
       WHEN cfr >= 0.3 AND cfr < 0.7 THEN 3
       WHEN cfr >= 0.7 THEN 4
     END AS bin_order
WITH cfr_bin, bin_order, comm, node_cnt
WITH cfr_bin, bin_order,
     count(DISTINCT comm) AS community_count,
     sum(node_cnt) AS node_count
ORDER BY bin_order
WITH collect({
       cfr_bin: cfr_bin,
       bin_order: bin_order,
       community_count: community_count,
       node_count: node_count
     }) AS bins
WITH bins, reduce(t = 0, b IN bins | t + b.node_count) AS total_nodes
UNWIND range(0, size(bins) - 1) AS idx
WITH bins, total_nodes, idx, bins[idx] AS b,
     reduce(cum = 0, j IN range(0, idx) | cum + bins[j].node_count) AS cumulative_nodes
RETURN b.cfr_bin AS `CFR 区间`,
       b.community_count AS `社区数量`,
       b.node_count AS `包含节点数`,
       round(toFloat(b.node_count) / total_nodes, 4) AS `区间节点占比`,
       round(toFloat(cumulative_nodes) / total_nodes, 4) AS `累计节点占比`
ORDER BY b.bin_order;
