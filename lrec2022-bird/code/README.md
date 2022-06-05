The source code as well as the data for BIRD is placed here. 

BIRD has three main phases:

1. *create corpus*: a corpus of 100,000 randomly selected English Wikipedia articles is created. The code is in Python via Jupyter Notebook.

2. *path extraction*: a database of syntactic paths is created by collecting all of the syntactic paths in the corpus. Some of the code in this phase is in Scala (executed using Maven) and some is in Python.

3. *search*: given a syntactic path, the top *k* most similar paths from the paths database are found. The code in this phase is in Python.

The *data* directory: this directory would contain the corpus as well as the collected path database. See inside of the directory for more details.
