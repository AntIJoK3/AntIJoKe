// 唯一约束与向量索引（文本嵌入维度与 all-MiniLM-L6-v2 一致：384）
CREATE CONSTRAINT website_url_unique IF NOT EXISTS
FOR (w:Website) REQUIRE w.url IS UNIQUE;

CREATE CONSTRAINT ip_address_unique IF NOT EXISTS
FOR (ip:IP) REQUIRE ip.address IS UNIQUE;

CREATE CONSTRAINT email_unique IF NOT EXISTS
FOR (e:RegistrantEmail) REQUIRE e.email IS UNIQUE;

CREATE CONSTRAINT ns_unique IF NOT EXISTS
FOR (n:NameServer) REQUIRE n.ns_name IS UNIQUE;

CREATE CONSTRAINT analytics_unique IF NOT EXISTS
FOR (a:AnalyticsID) REQUIRE a.tracker_id IS UNIQUE;

CREATE CONSTRAINT ssl_fp_unique IF NOT EXISTS
FOR (c:SSLCert) REQUIRE c.fingerprint IS UNIQUE;

CREATE VECTOR INDEX content_vec_idx IF NOT EXISTS
FOR (w:Website)
ON (w.text_embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'COSINE'
  }
};
