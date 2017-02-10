System used in EMNLP 2016 paper:
"Creating Customized Causal Embeddings for Question Answering with Minimal Supervision"

This README explains how to run several components:

  EXTRACTION:
  (a) the tool to extract causal events from freetext
  (b) the tool to extract causal events from anotated gigaword
  
	CREATING EMBEDDINGS:
  (c) the tool to convert the extracted events to the word-pair format needed for the
      Levy&Goldberg vector training
  
  EVALUATING:
  (d) the tool to run the direct evaluation
  (e) the tool to run the causal Yahoo!CQA evaluation
  
  Some details are included here, some are included in the relevant sample properties
	files.
	
	If you have questions not covered by this documentation, please feel free to email
  Becky Sharp (bsharp@email.arizona.edu).
	
	
	**NOTE: please extract the data/ dir before running the example code!

--------------------------------------------------------------------------------------------
  EXTRACTION:
	NOTE: My extracted tuples are included in /data/causalTuples
--------------------------------------------------------------------------------------------
		Both of these modes produces two types of output files.
		(1) The first has a .args extension, and has just the CAUSE --> RESULT extracted events.
		(2) The second has a .detail extension and includes more detail about the source the extracted
		causal even came from, the rule that fired to do the extraction, etc.

  (a) the tool to extract causal events from freetext
		To extract causal events from free text, you need to spcify several parameters using
		a properties file (sample provided in props/extractFreeText.props). These include:
			- input_dir = [path to your free text files] # files should have one document per line
			- input_file_extension = [file extension of the files you want included]
			- output_dir = [path for output file directory]
			- view = the format for the output:
							options:
								-- words
								-- lemmas
								-- lemmasWithTags : corresponds to the lemma_originalPOS, like cat_NNS
								-- wordsWithTags	: corresponds to the originalWord_originalPOS, like runs_VBZ
			- rules_file = the location of your rules file, a sample is provided in src/main/resources
			---------------
			command to run:
					sbt "runMain edu.arizona.sista.extraction.ExtractFromFreeText -props [path to props file]"
			---------------
			
  (b) the tool to extract causal events from anotated gigaword
			This is very similar to (a) except that we assume you have annotated gigaword downloaded.
			A sample properties file is located in props/extractAgiga.props.  Here, your properties file is
			designed to specify:
				# Location of the rules file, when using “/“ at the start it’s relative to 
				# src/main/resources
				rules_file = /CMBasedRules.yml
				
				# Location of your (compressed) annotated gigaword xml files
				data_dir = /data/nlp/corpora/agiga/data/xml/
				
				# The prefix for all outputs generated
				output_prefix = causalOut/causalOut_
				
				# Used to break larger gigaword files into more manageable chunks
				num_docs_per_output_file = 10000
				
				view = lemmasWithTags # same as (a) above
				
				nthreads = 2
				---------------
				command to run:
						sbt "runMain edu.arizona.sista.extraction.ExtractFromAgiga -props [path to props file]"
				---------------
						
--------------------------------------------------------------------------------------------
  CREATING EMBEDDINGS:
	NOTE:  My generated embeddings are included in data/embeddings
--------------------------------------------------------------------------------------------
  (c) the tool to convert the extracted events to the word-pair format needed for the
      Levy&Goldberg vector training
			The properties file here has several arguments, described in detail in the sample
			props file: props/createEmbedInput.props
			---------------
			command to run:
					sbt "runMain preprocessing.CreateGoldbergInput -props [path to props file]"
			---------------
			
			generates 6 files (3 for cause-to-effect and 3 for effect-to-cause alignment) that end with:
				.contexts
				.cv
				.wv
	
			these are to be used as explained in the word2vecf documentation (I am not including instructions
			for that as their tool/API could change...).  If you want more detail, please email us using
			the contact email above.
			
						
--------------------------------------------------------------------------------------------  
  EVALUATING:
--------------------------------------------------------------------------------------------
  (d) the tool to run the direct evaluation
			The direct evaluation provides two metrics: MAP and a precision-recall curve (well, it provides
			the data, you have to plot it...)
			The models used are able to be turned on/off to simplify the eval if desired.  Details are in the
			sample prperties file (props/directEval.props)
			---------------
			command to run:
					sbt "runMain edu.arizona.sista.embeddings.DirectEval -props [path to props file]"
			---------------
	
  (e) the tool to run the causal Yahoo!CQA evaluation
		** NOTE: to run this you need to have svmRank installed!
		** Seems to require ~ 8G RAM to run in some confiugurations
		
		The questions are included in data/yahoo
		A sample properties file for running one of the Cross-validation jobs is included (props/qa.props_CV0_V+cB)
		This properties file can be modified to run any of the experiments, though not all the data is included in
		the release (i.e. some of the cnn files).  If you would like additional detail or data, please don't hesitate
		to contact Becky Sharp (bsharp@email.arizona.edu)
		---------------
			command to run:
					sbt "runMain edu.arizona.sista.qa.ranking.RankerEntryPoint -props [path to props file]"
			---------------
		


  