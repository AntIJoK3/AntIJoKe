// 欺诈 vs 正常：介数中心性（GDS 写回的 w.betweenness）
// 前置：03_gds_project + 04_gds_features（gds.betweenness.write）
// 仅统计 betweenness 非空的 fraud/normal 节点（论文中需注明 n）

MATCH (w:Website)
WHERE w.label IN ['fraud', 'normal']
  AND w.betweenness IS NOT NULL
WITH w.label AS label, toFloat(w.betweenness) AS betweenness
WITH label,
     count(*) AS n,
     avg(betweenness) AS m_btw,
     stDev(betweenness) AS s_btw,
     percentileCont(betweenness, 0.5) AS med_btw
WITH collect({
       label: label, n: n,
       m_btw: m_btw, s_btw: s_btw, med_btw: med_btw
     }) AS groups
WITH [g IN groups WHERE g.label = 'fraud'][0] AS f,
     [g IN groups WHERE g.label = 'normal'][0] AS o
WITH f, o,
     ((f.n - 1) * f.s_btw^2 + (o.n - 1) * o.s_btw^2) / (f.n + o.n - 2) AS pooled_var
RETURN '介数中心性' AS 特征,
       f.n AS fraud_n,
       o.n AS normal_n,
       round(f.m_btw, 2) AS fraud_mean,
       round(f.s_btw, 2) AS fraud_std,
       round(f.med_btw, 4) AS fraud_median,
       round(o.m_btw, 2) AS normal_mean,
       round(o.s_btw, 2) AS normal_std,
       round(o.med_btw, 4) AS normal_median,
       CASE
         WHEN pooled_var > 0
         THEN round((f.m_btw - o.m_btw) / sqrt(pooled_var), 3)
         ELSE null
       END AS cohens_d,
       round(f.m_btw, 2) + ' $\\pm$ ' + round(f.s_btw, 2) AS `fraud_mean_pm_std_latex`,
       toString(round(f.med_btw, 4)) AS `fraud_median_latex`,
       round(o.m_btw, 2) + ' $\\pm$ ' + round(o.s_btw, 2) AS `normal_mean_pm_std_latex`,
       toString(round(o.med_btw, 4)) AS `normal_median_latex`,
       toString(
         CASE
           WHEN pooled_var > 0
           THEN round((f.m_btw - o.m_btw) / sqrt(pooled_var), 3)
           ELSE null
         END
       ) AS `cohens_d_latex`;
