# Scientific Discovery as Link Prediction in Influence and Citation Graphs

Our dataset is avaible to download with this [link](https://arizona.app.box.com/folder/72298595101)

- Neo4j is a good visulization tool to view our influence graph.
- If you have issue to set it up, you can use the raw data files:
  - The influences.csv contains the list of the influence edges such as  A increase B, where A corresponds to column *:START_ID(Concept)* ,B correspondsto *column :END_ID(Concept)*.
  - The concept id can be mapped to text with concepts.csv.
  - Each edge has its id, which can be mapped to the sentences supported this influence relation by looking up evidence.csv .
