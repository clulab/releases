
# Versions
These are the various versions in the fact verification code development cycle (and what they do) at University of Arizona. Note, there must be only one version of this document and preferably exists in the master branch

| Date of modification |name of the branch |git SHA | change made | New F1 score | New overall accuracy | New average Precision|  Merged with master? |Type of Classifier SVM or Decomp Attn | Notes |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Nov 8th 2018|   person_space_c1 | 9f20b8b8e3e79c6b3410b51c3905f58042d42d28  | Replaced PERSON_C1 with PERSON C1 in the NER replacement code   | 0.46  | 0.5062006200620062  | 0.73| Yes | Decomp Attn | email dated:Fri, Nov 9, 3:26 PM  | 
| Nov 11th 2018|   mrksic | 33667b1e2f68584d1e0fd9611275ca3d5a6aa508  | Does IR retrieval using their FEVERReader instead of our custom function.. We are in the middle of adding mrksic vectors in this branch, so no results. But  This should be added to master main for hand crafted development channel|  NA | NA  | NA | No | SVM | | 
