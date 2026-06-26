#!/usr/bin/env node
"use strict";

function main() {
  const inputPath = process.argv[2];
  const outputPath = process.argv[3];
  if (!inputPath || !outputPath) {
    console.error("Usage: node ua-tour-analyze.js <input.json> <output.json>");
    process.exit(1);
  }
  const fs = require("fs");
  const path = require("path");

  const raw = JSON.parse(fs.readFileSync(inputPath, "utf8"));
  const nodes = raw.nodes || [];
  const edges = raw.edges || [];
  const layers = raw.layers || [];

  const nodeById = new Map();
  for (const n of nodes) nodeById.set(n.id, n);
  const nodeIdSet = new Set(nodes.map((n) => n.id));

  // Only consider edges between actual graph nodes (skip function:/class: synthetic nodes)
  const realEdges = edges.filter(
    (e) => nodeIdSet.has(e.source) && nodeIdSet.has(e.target)
  );

  // ---- A & B. Fan-in / Fan-out ----
  const fanIn = new Map();
  const fanOut = new Map();
  for (const id of nodeIdSet) {
    fanIn.set(id, 0);
    fanOut.set(id, 0);
  }
  for (const e of realEdges) {
    fanOut.set(e.source, (fanOut.get(e.source) || 0) + 1);
    fanIn.set(e.target, (fanIn.get(e.target) || 0) + 1);
  }

  const nameOf = (id) => (nodeById.get(id) ? nodeById.get(id).name : id);
  const summaryOf = (id) => (nodeById.get(id) ? nodeById.get(id).summary : "");

  const fanInRanking = [...nodeIdSet]
    .map((id) => ({ id, fanIn: fanIn.get(id), name: nameOf(id) }))
    .sort((a, b) => b.fanIn - a.fanIn)
    .slice(0, 20);

  const fanOutRanking = [...nodeIdSet]
    .map((id) => ({ id, fanOut: fanOut.get(id), name: nameOf(id) }))
    .sort((a, b) => b.fanOut - a.fanOut)
    .slice(0, 20);

  // ---- C. Entry point candidates ----
  const entryNames = new Set([
    "index.ts","index.js","main.ts","main.js","app.ts","app.js","server.ts","server.js",
    "mod.rs","main.go","main.py","main.py","main.rs","manage.py","app.py","wsgi.py","asgi.py",
    "run.py","__main__.py","Application.java","Main.java","Program.cs","config.ru","index.php",
    "App.swift","Application.kt","main.cpp","main.c",
  ]);

  const fanOutVals = [...fanOut.values()].sort((a, b) => b - a);
  const top10pctIdx = Math.max(0, Math.floor(fanOutVals.length * 0.1) - 1);
  const fanOutTop10pctThreshold = fanOutVals.length ? fanOutVals[top10pctIdx] : 0;
  const fanInVals = [...fanIn.values()].sort((a, b) => a - b);
  const bottom25Idx = Math.max(0, Math.floor(fanInVals.length * 0.25) - 1);
  const fanInBottom25Threshold = fanInVals.length ? fanInVals[bottom25Idx] : 0;

  const entryScored = [];
  for (const n of nodes) {
    let score = 0;
    const fp = n.filePath || "";
    const depth = fp.split("/").length;
    if (n.type === "document") {
      if (n.name === "README.md" && depth === 1) score += 5;
      else if (n.name && n.name.endsWith(".md") && depth === 1) score += 2;
    } else if (n.type === "file") {
      if (entryNames.has(n.name)) score += 3;
      if (depth <= 2) score += 1;
      if ((fanOut.get(n.id) || 0) >= fanOutTop10pctThreshold && fanOutTop10pctThreshold > 0) score += 1;
      if ((fanIn.get(n.id) || 0) <= fanInBottom25Threshold) score += 1;
    }
    if (score > 0) entryScored.push({ id: n.id, score, name: n.name, summary: n.summary });
  }
  entryScored.sort((a, b) => b.score - a.score);
  const entryPointCandidates = entryScored.slice(0, 5);

  // ---- D. BFS from top code entry point ----
  // pick top non-document entry candidate
  const codeEntry = entryScored.find((e) => {
    const node = nodeById.get(e.id);
    return node && node.type === "file";
  });
  const startNode = codeEntry ? codeEntry.id : (nodes[0] && nodes[0].id);

  // adjacency for imports/calls forward
  const adj = new Map();
  for (const id of nodeIdSet) adj.set(id, []);
  for (const e of realEdges) {
    if (e.type === "imports" || e.type === "calls") {
      adj.get(e.source).push(e.target);
    }
  }
  const order = [];
  const depthMap = {};
  const visited = new Set();
  if (startNode) {
    const queue = [[startNode, 0]];
    visited.add(startNode);
    while (queue.length) {
      const [cur, d] = queue.shift();
      order.push(cur);
      depthMap[cur] = d;
      for (const nb of adj.get(cur) || []) {
        if (!visited.has(nb)) {
          visited.add(nb);
          queue.push([nb, d + 1]);
        }
      }
    }
  }
  const byDepth = {};
  for (const id of order) {
    const d = String(depthMap[id]);
    if (!byDepth[d]) byDepth[d] = [];
    byDepth[d].push(id);
  }

  // ---- E. Non-code inventory ----
  const nonCodeFiles = { documentation: [], infrastructure: [], data: [], config: [] };
  for (const n of nodes) {
    const entry = { id: n.id, name: n.name, type: n.type, summary: n.summary };
    if (n.type === "document") nonCodeFiles.documentation.push(entry);
    else if (n.type === "service" || n.type === "pipeline" || n.type === "resource")
      nonCodeFiles.infrastructure.push(entry);
    else if (n.type === "table" || n.type === "schema" || n.type === "endpoint")
      nonCodeFiles.data.push(entry);
    else if (n.type === "config") nonCodeFiles.config.push(entry);
  }

  // ---- F. Clusters ----
  // bidirectional pairs over imports/calls
  const edgeSet = new Set();
  for (const e of realEdges) {
    if (e.type === "imports" || e.type === "calls") edgeSet.add(e.source + "||" + e.target);
  }
  const pairKey = (a, b) => (a < b ? a + "||" + b : b + "||" + a);
  const biPairs = new Set();
  for (const e of realEdges) {
    if (e.type === "imports" || e.type === "calls") {
      if (edgeSet.has(e.target + "||" + e.source)) biPairs.add(pairKey(e.source, e.target));
    }
  }
  // Since this codebase is mostly acyclic, also build undirected neighbor map
  // and form clusters of nodes sharing multiple connections within a layer.
  const undAdj = new Map();
  for (const id of nodeIdSet) undAdj.set(id, new Set());
  for (const e of realEdges) {
    if (e.type === "imports" || e.type === "calls") {
      undAdj.get(e.source).add(e.target);
      undAdj.get(e.target).add(e.source);
    }
  }
  // Seed clusters from hub nodes (those with >=2 undirected neighbors), group hub+neighbors
  const clusters = [];
  const seen = new Set();
  const hubs = [...nodeIdSet]
    .map((id) => ({ id, deg: undAdj.get(id).size }))
    .filter((x) => x.deg >= 2)
    .sort((a, b) => b.deg - a.deg);
  for (const { id } of hubs) {
    if (seen.has(id)) continue;
    const members = [id, ...undAdj.get(id)].slice(0, 5);
    // count internal edges
    let edgeCount = 0;
    for (let i = 0; i < members.length; i++)
      for (let j = 0; j < members.length; j++)
        if (i !== j && undAdj.get(members[i]).has(members[j])) edgeCount++;
    edgeCount = Math.floor(edgeCount / 2);
    if (members.length >= 2 && edgeCount >= 1) {
      clusters.push({ nodes: members, edgeCount });
      members.forEach((m) => seen.add(m));
    }
    if (clusters.length >= 10) break;
  }

  // ---- G. Layers ----
  const layerList = layers.map((l) => ({ id: l.id, name: l.name, description: l.description }));

  // ---- H. node summary index ----
  const nodeSummaryIndex = {};
  for (const n of nodes) {
    nodeSummaryIndex[n.id] = { name: n.name, type: n.type, summary: n.summary };
  }

  const out = {
    scriptCompleted: true,
    entryPointCandidates,
    fanInRanking,
    fanOutRanking,
    bfsTraversal: { startNode, order, depthMap, byDepth },
    nonCodeFiles,
    clusters,
    layers: { count: layers.length, list: layerList },
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: realEdges.length,
  };
  fs.writeFileSync(outputPath, JSON.stringify(out, null, 2));
  console.log("done");
}

try {
  main();
} catch (err) {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
}
