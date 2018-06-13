# positive data

MATCH (c1:Concept)-[r1]->(c2:Concept)-[r2]->(c3:Concept) MATCH (c1)-[r3]->(c3) 
WHERE r1.EDGE_YEAR < 2012 AND r2.EDGE_YEAR < 2012 AND r3.EDGE_YEAR >= 2012 AND r1.DOCS_SEEN>=2 AND r2.DOCS_SEEN>=2 AND r3.DOCS_SEEN>=2 AND r1.NEGATED_COUNT = 0 AND r2.NEGATED_COUNT = 0 AND r3.NEGATED_COUNT = 0 AND r1.HEDGED_COUNT = 0 AND r2.HEDGED_COUNT = 0 AND r3.HEDGED_COUNT = 0 AND c1.AVE_IDF >= 1.0 AND c2.AVE_IDF >= 1.0 AND c3.AVE_IDF >= 1.0 
OPTIONAL MATCH (c1),(c3),p = allShortestPaths((c1)-[*..5]->(c3)) 
where NONE (r IN relationships(p) WHERE r.NEGATED_COUNT > 0 or r.HEDGED_COUNT > 0 or r.EDGE_YEAR >= 2012 or r.DOCS_SEEN < 2) AND NONE (c IN NODES(p) WHERE c.NODE_HASH = c2.NODE_HASH or c.AVE_IDF < 1.0) 
WITH collect(p) AS paths,c1,c2,c3,r1,r2,r3
return r3.EDGE_DEDUPLICATION_HASH,c1.AVE_IDF,c2.AVE_IDF,c3.AVE_IDF,r1.DOCS_SEEN,r2.DOCS_SEEN,size((c1)-[]->()) AS source_outdegree,size((c1)<-[]-()) AS source_indegree,size((c3)-[]->()) AS destination_outdegree,size((c3)<-[]-()) AS destination_indegree,apoc.coll.avg(extract(n in reduce(concepts = [], cc in [x IN paths| filter(pc IN nodes(x) WHERE NOT pc IN [c1,c3])] | concepts+cc) | n.AVE_IDF)) AS AVG_IDF,apoc.coll.avg(extract(r in reduce(edges = [], ee in [x IN paths| relationships(x)] | edges+ee) | r.DOCS_SEEN)) AS AVG_SEEN,length(paths) AS path_count,length(paths[0]) AS path_len,c1.NODE_HASH,c2.NODE_HASH,c3.NODE_HASH,c1.NODE_TEXT,c2.NODE_TEXT,c3.NODE_TEXT,r1.EDGE_PMIDS,r2.EDGE_PMIDS,r1.EDGE_LABEL,r2.EDGE_LABEL,r3.EDGE_LABEL;


# negative data

MATCH (c1:Concept)-[r1]->(c2:Concept)-[r2]->(c3:Concept) 
WHERE c1 <> c3 AND NOT (c1)-[]->(c3) AND r1.EDGE_YEAR < 2012 AND r2.EDGE_YEAR < 2012 AND r1.DOCS_SEEN>=2 AND r2.DOCS_SEEN>=2 AND r1.NEGATED_COUNT = 0 AND r2.NEGATED_COUNT = 0 AND r1.HEDGED_COUNT = 0 AND r2.HEDGED_COUNT = 0 AND c1.AVE_IDF >= 1.0 AND c2.AVE_IDF >= 1.0 AND c3.AVE_IDF >= 1.0 
OPTIONAL MATCH (c1),(c3),p = allShortestPaths((c1)-[*..5]->(c3)) 
where NONE (r IN relationships(p) WHERE r.NEGATED_COUNT > 0 or r.HEDGED_COUNT > 0 or r.EDGE_YEAR >= 2012 or r.DOCS_SEEN < 2) AND NONE (c IN NODES(p) WHERE c.NODE_HASH = c2.NODE_HASH or c.AVE_IDF < 1.0) 
WITH collect(p) AS paths,c1,c2,c3,r1,r2
return c1.NODE_HASH,c2.NODE_HASH,c3.NODE_HASH,c1.AVE_IDF,c2.AVE_IDF,c3.AVE_IDF,r1.DOCS_SEEN,r2.DOCS_SEEN,size((c1)-[]->()) AS source_outdegree,size((c1)<-[]-()) AS source_indegree,size((c3)-[]->()) AS destination_outdegree,size((c3)<-[]-()) AS destination_indegree,apoc.coll.avg(extract(n in reduce(concepts = [], cc in [x IN paths| filter(pc IN nodes(x) WHERE NOT pc IN [c1,c3])] | concepts+cc) | n.AVE_IDF)) AS AVG_IDF,apoc.coll.avg(extract(r in reduce(edges = [], ee in [x IN paths| relationships(x)] | edges+ee) | r.DOCS_SEEN)) AS AVG_SEEN,length(paths) AS path_count,length(paths[0]) AS path_len,c1.NODE_TEXT,c2.NODE_TEXT,c3.NODE_TEXT,r1.EDGE_PMIDS,r2.EDGE_PMIDS,r1.EDGE_LABEL,r2.EDGE_LABEL
LIMIT 50160;

