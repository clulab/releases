# Synthetic Dataset for Evaluating Complex Compositional Knowledge for Natural Language Inference
Code and dataset for Synthetic Dataset for Evaluating Complex Compositional Knowledge for Natural Language Inference. </br>
For any questions related to the dataset and code, please contact Sushma Akoju, Email: sushmaakoju@arizona.edu . </br>
This work is under supervision of Prof. Mihai Surdeanu, Computational Language Understanding Lab at the University of Arizona. </br>
The original repository can be found @ <a href="https://github.com/sushmaakoju/natural-logic/commits/main?after=396e926489ddc9eae51e7cb2cb4ef5270a7f5021+69&branch=main&qualified_name=refs%2Fheads%2Fmain">sushmaakoju/natural-logic Nov 2022 to Jul 2023.</a>.

## Dataset : Sentences Involving Complex Compositional Knowledge (SICCK)
SICCK data: <a href="https://github.com/clulab/releases/tree/master/acl2023-nlrse-sicck/data/SICCK"> Sentences Involving Complex Compositional Knowledge (SICCK) </a>

- Selected 15 examples from SICK dataset and corresponding analysis for compositionality: <a href="https://github.com/clulab/releases/tree/master/acl2023-nlrse-sicck/data/original-sick-examples"> sick-15 </a>

## License 
Derived from <a href="https://marcobaroni.org/composes/sick.html">SICK Dataset</a>

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Note
The results in this version differ from the results in the published paper due to: (a) minor bug fixes, and (b) averaging results across five different random seeds. Please note that the different results reported in this version do not change any of the observations in the published paper.

## Code

Author: sushmaakoju@arizona.edu

The code was run on two different settings: Google Colab using python and Scala respectively. 
- <a href="https://github.com/clulab/releases/tree/sushma/acl2023-nlrse-sicck/code">Colab Notebooks Code for SICCK dataset generation, annotations, zero-shot evaluation and finetuning of NLI models and evaluation</a>. 
- The <a href="https://github.com/clulab/releases/tree/sushma/acl2023-nlrse-sicck/code/generating-modified-sentences/natlog"> scala code for generating modified object parts of sentences using Clulab Processors software </a>.

### Acknowledgements
This work is under supervision of Prof. Mihai Surdeanu, University of Arizona. 

#### Noun Phrase Modification of Subject Part of the Sentences
Author for Leftmost NounPhrase Modification for subject part of sentences and selection of type of Parser i.e. Berkeley Neural Parser: Robert Vacareanu, Email: rvacareanu@arizona.edu.

#### About questions
For any questions related to the dataset and code, please contact: sushmaakoju@arizona.edu. 

#### Annotations:
- Robert Vacareanu, Haris Riaz and Prof. Eduardo Blanco, Prof. Mihai Surdeanu, Sushma Akoju - contributed towards annotations and discussions about Entailment Labels during annotations and which helped to refine the annotations guidelines as well as human annotations for the entire 1304 premise, hypothesis sentence pairs.