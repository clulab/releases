The path extraction phase has several steps that must be run sequentially. The details of each step are as follows:

- Step 0: Some of the sentences in the corpus don't end with any kind of punctuations. In order for Stanford NLP parser to be able to make a distinction between separate sentences, a period character (.) is added to the end of the sentences that are not punctuated.

- Step 1: This step annotates the corpus using FastNLPProcessor and serializes the resulting doc objects to files. Since this is a very time-consuming process, annotating the whole corpus is infeasible. Instead, we divided the corpus into 100 separate files (corpus_part001 ... corpus_part100) and each part is annotated independently. In this way, we are able to annotate all of the 100 files in parallel using separate HPC jobs and speed up the process. When all of the 100 HPC jobs are finished, there should be 100 doc files (doc_part001 ... doc_part100) in "data/docs" directory.

- Step 2: This step generates two tsv files which will be used later. The scala program calls the bash script "call_create_BERTtoken_mappings" internally. Before running this step, execute permission should be given to the bash scrip using chmod command.

- Step 3: For each doc file created in step 1, all of the syntactic paths are extracted and stored in a tsv file. After this step is completed, there should be 100 files (paths_db_part001.tsv ... paths_db_part100.tsv) in "data/paths_dbs" directory.

- Step 4: This step combines "paths_db_partxxx.tsv" files (created in step 3) into a single data structure, filters the paths with frequencies below a threshold (10), and saves a part of the remaining paths in "filtered_paths_db_partxxx.tsv" files. 150 of these files are created (filtered_paths_db_part001.tsv ... filtered_paths_db_part150.tsv). This step has three substeps which have to be run sequentially.

- Steps 5, 6, and 7: Generate the final syntactic paths database which is stored as several tsv files.
