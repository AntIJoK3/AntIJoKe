// 欺诈 vs 正常：PageRank — 均值±标准差、中位数、Cohen's d
// 前置：已跑 graph_features（w.pagerank 已写入，建议 GDS + scaler: 'L1Norm'）
// 仅要求 pagerank 非空，不依赖 betweenness / 度中心性

MATCH (w:Website)
WHERE w.label IN ['fraud', 'normal']
  AND w.pagerank IS NOT NULL
WITH w.label AS label, w.pagerank AS pagerank
WITH label,
     count(*) AS n,
     avg(pagerank) AS m_pr,
     stDev(pagerank) AS s_pr,
     percentileCont(pagerank, 0.5) AS med_pr
WITH collect({
       label: label, n: n,
       m_pr: m_pr, s_pr: s_pr, med_pr: med_pr
     }) AS groups
WITH [g IN groups WHERE g.label = 'fraud'][0] AS f,
     [g IN groups WHERE g.label = 'normal'][0] AS o
RETURN 'PageRank' AS 特征,
       f.n AS fraud_n,
       o.n AS normal_n,
       round(f.m_pr, 6) AS fraud_mean,
       round(f.s_pr, 6) AS fraud_std,
       round(f.med_pr, 6) AS fraud_median,
       round(o.m_pr, 6) AS normal_mean,
       round(o.s_pr, 6) AS normal_std,
       round(o.med_pr, 6) AS normal_median,
       round(
         (f.m_pr - o.m_pr) /
         sqrt(((f.n - 1) * f.s_pr^2 + (o.n - 1) * o.s_pr^2) / (f.n + o.n - 2)),
         3
       ) AS cohens_d,
       round(f.m_pr, 6) + ' $\\pm$ ' + round(f.s_pr, 6) AS `fraud_mean_pm_std_latex`,
       toString(round(f.med_pr, 6)) AS `fraud_median_latex`,
       round(o.m_pr, 6) + ' $\\pm$ ' + round(o.s_pr, 6) AS `normal_mean_pm_std_latex`,
       toString(round(o.med_pr, 6)) AS `normal_median_latex`,
       toString(round(
         (f.m_pr - o.m_pr) /
         sqrt(((f.n - 1) * f.s_pr^2 + (o.n - 1) * o.s_pr^2) / (f.n + o.n - 2)),
         3
       )) AS `cohens_d_latex`;
