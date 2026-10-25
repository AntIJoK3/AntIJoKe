// 拓扑规模统计：节点、关系、Website 平均度（四种同构 Website—Website 边）
// 在 Neo4j Browser 中执行；结果切换为 Table 视图

MATCH (n)
WITH count(n) AS total_nodes

MATCH (w:Website)
WITH total_nodes, count(w) AS website_cnt

MATCH (ip:IP)
WITH total_nodes, website_cnt, count(ip) AS ip_cnt

MATCH (ns:NameServer)
WITH total_nodes, website_cnt, ip_cnt, count(ns) AS ns_cnt

MATCH (e:RegistrantEmail)
WITH total_nodes, website_cnt, ip_cnt, ns_cnt, count(e) AS email_cnt

MATCH ()-[r]->()
WITH total_nodes, website_cnt, ip_cnt, ns_cnt, email_cnt, count(r) AS total_rels

OPTIONAL MATCH ()-[r:SHARED_IP]->()
WITH total_nodes, website_cnt, ip_cnt, ns_cnt, email_cnt, total_rels, count(r) AS shared_ip

OPTIONAL MATCH ()-[r:CONTENT_SIMILAR]->()
WITH total_nodes, website_cnt, ip_cnt, ns_cnt, email_cnt, total_rels, shared_ip, count(r) AS content_sim

OPTIONAL MATCH ()-[r:SHARED_NS]->()
WITH total_nodes, website_cnt, ip_cnt, ns_cnt, email_cnt, total_rels, shared_ip, content_sim, count(r) AS shared_ns

OPTIONAL MATCH ()-[r:SHARED_EMAIL]->()
WITH total_nodes, website_cnt, ip_cnt, ns_cnt, email_cnt, total_rels,
     shared_ip, content_sim, shared_ns, count(r) AS shared_email

MATCH (w:Website)
OPTIONAL MATCH (w)-[:SHARED_IP|CONTENT_SIMILAR|SHARED_NS|SHARED_EMAIL]-(n:Website)
WITH total_nodes, website_cnt, ip_cnt, ns_cnt, email_cnt, total_rels,
     shared_ip, content_sim, shared_ns, shared_email,
     w, count(DISTINCT n) AS deg
WITH total_nodes, website_cnt, ip_cnt, ns_cnt, email_cnt, total_rels,
     shared_ip, content_sim, shared_ns, shared_email,
     avg(deg) AS avg_website_degree

UNWIND [
  {统计项: '总节点数', 数值: total_nodes},
  {统计项: 'Website 节点数', 数值: website_cnt},
  {统计项: 'IP 节点数', 数值: ip_cnt},
  {统计项: 'NameServer 节点数', 数值: ns_cnt},
  {统计项: 'RegistrantEmail 节点数', 数值: email_cnt},
  {统计项: '总关系数', 数值: total_rels},
  {统计项: 'SHARED_IP 关系数', 数值: shared_ip},
  {统计项: 'CONTENT_SIMILAR 关系数', 数值: content_sim},
  {统计项: 'SHARED_NS 关系数', 数值: shared_ns},
  {统计项: 'SHARED_EMAIL 关系数', 数值: shared_email},
  {统计项: 'Website 节点平均度', 数值: round(avg_website_degree, 4)}
] AS row
RETURN row.统计项 AS 统计项, row.数值 AS 数值;
