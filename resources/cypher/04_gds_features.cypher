// 图算法写回 Website（需先执行 03_gds_project）
// PageRank 默认 scaler=None，分数和可远大于 1；训练/统计需 L1Norm（与 NetworkX pagerank 和≈1 一致）
CALL gds.pageRank.write('fraud-projection', {
  writeProperty: 'pagerank',
  dampingFactor: 0.85,
  maxIterations: 20,
  scaler: 'L1Norm'
});
CALL gds.betweenness.write('fraud-projection', {writeProperty: 'betweenness'});

CALL gds.wcc.write('fraud-projection', {writeProperty: 'wcc_id'});

CALL gds.louvain.write('fraud-projection', {writeProperty: 'louvain_community'});

CALL gds.fastRP.write(
  'fraud-projection',
  {
    embeddingDimension: 128,
    writeProperty: 'graph_embedding',
    iterationWeights: [0.0, 1.0, 1.0, 1.0],
    randomSeed: 42
  }
);
