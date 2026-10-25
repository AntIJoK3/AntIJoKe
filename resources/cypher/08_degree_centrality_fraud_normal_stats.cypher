// 度中心性 = 同构边 SHARED_IP|CONTENT_SIMILAR 上的邻居数（每站单独计数）
// 欺诈 vs 正常：均值±标准差、中位数、Cohen's d（全 500 标注站，无需 pagerank/betweenness）

MATCH (w:Website)
WHERE w.label IN ['fraud', 'normal']
OPTIONAL MATCH (w)-[:SHARED_IP|CONTENT_SIMILAR]-(n:Website)
WITH w, count(DISTINCT n) AS degree
WITH w.label AS label, degree
WITH label,
     count(*) AS n,
     avg(degree) AS m_deg,
     stDev(degree) AS s_deg,
     percentileCont(degree, 0.5) AS med_deg
WITH collect({
       label: label, n: n,
       m_deg: m_deg, s_deg: s_deg, med_deg: med_deg
     }) AS groups
WITH [g IN groups WHERE g.label = 'fraud'][0] AS f,
     [g IN groups WHERE g.label = 'normal'][0] AS o
WITH f, o,
     ((f.n - 1) * f.s_deg^2 + (o.n - 1) * o.s_deg^2) / (f.n + o.n - 2) AS pooled_var
RETURN '度中心性' AS 特征,
       f.n AS fraud_n,
       o.n AS normal_n,
       round(f.m_deg, 2) AS fraud_mean,
       round(f.s_deg, 2) AS fraud_std,
       round(f.med_deg, 2) AS fraud_median,
       round(o.m_deg, 2) AS normal_mean,
       round(o.s_deg, 2) AS normal_std,
       round(o.med_deg, 2) AS normal_median,
       CASE
         WHEN pooled_var > 0
         THEN round((f.m_deg - o.m_deg) / sqrt(pooled_var), 3)
         ELSE null
       END AS cohens_d;
