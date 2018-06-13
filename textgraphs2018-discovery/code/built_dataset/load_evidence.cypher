CREATE CONSTRAINT ON (e:evidence) ASSERT e.id IS UNIQUE;
USING PERIODIC COMMIT
LOAD CSV FROM 'file:///EVIDENCE.tsv' AS line FIELDTERMINATOR '\t'             
CREATE (:evidence {id:toInteger(line[0]), edge_dedup:toInteger(line[1]), text:line[7]});