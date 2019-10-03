# This code base is the first phase/data pre-processing step. Here we use multiple kinds of pre processing step as mentioned in the emnlp 2019 paper

## overlap aware ner convertor

This code takes a claim and evidence pairs, finds where all NER tags exist and replace them smartly. refer examples below. 
This is being done to show the NN model that there is overlap between claim and evidence.

```
conda create --name delex python=3
source activate delex
pip install tqdm
pip install clean-text
pip install git+https://github.com/myedibleenso/py-processors.git
mkdir outputs
mkdir sstagged_files
chmod 700 prereq.sh
./prereq.sh
```

**note**: we are using an in house tool 
called [pyprocessors]((https://py-processors.readthedocs.io/en/latest/)) (which inturn uses taggers like Stanford's CoreNLP)
to do annotation/NER/POS tagging etc. 
We are using the jar option mentioned 
in the link above.
 However if you want 
  to use docker instead for pyprocessors 
  we are mentioning below the commands for that too.


#### commands to run if you are using java server for pyprocessors:

`
python main.py --pyproc_port 8886 --use_docker False --convert_prepositions False --create_smart_NERs True --inputFile data/file_to_delex.jsonl
`
#### commands to run if you are using docker for pyprocessors:

`source activate delex`

`docker pull myedibleenso/processors-server:latest`
` docker run myedibleenso/processors-server`

`docker run -d -e _JAVA_OPTIONS="-Xmx3G" -p 127.0.0.1:8886:8888 --name procserv myedibleenso/processors-server`
or


```docker start procserv``` (if you already created a docker container)

`
python main.py --pyproc_port 8886 --use_docker False --convert_prepositions False --create_smart_NERs True --inputFile data/file_to_delex.jsonl
`

#### Explanation of command line arguments"

`--pyproc_port 8886` By default pyprocessors , the java version runs off port 8888. If you intend to change it/want to run it over another port, you can pass it as
a command line argument like this.

`--use_docker true` if you are using docker for pyprocessors (usually in laptops its easier to use a docker, where as in machines where you don't have
root/sudo access use java processors server)


#### Some sample conversions

```
hypothesis_before_annotation: Isis claims to behead US journalist
hypothesis_ann: ORGANIZATION-c1 claims to behead LOCATION-c1 journalist
premise_before_annotation: BREAKING : Islamic State , in video , beheads American journalist James Wright Foley who was kidnapped in 2012 - @BNONews
premise_ann: BREAKING : ORGANIZATION-e1 , in video , beheads MISC-e1 journalist PERSON-e1 who was kidnapped in DATE-e1 - @BNONews

['The', 'Boston', 'Celtics', 'play', 'their', 'home', 'games', 'at', 'TD', 'Garden', '.']

['The', 'Celtics', 'play', 'their', 'home', 'games', 'at', 'the', 'TD', 'Garden', ',', 'which', 'they', 'share', 'with', 'the', 'National', 'Hockey', 'League', '-LRB-', 'NHL', '-RRB-', "'s", 'Boston', 'Bruins', '.']

****['The', 'ORGANIZATION-c1', 'play', 'their', 'home', 'games', 'at', 'the', 'LOCATION-c1', ',', 'which', 'they', 'share', 'with', 'the', 'ORGANIZATION-e2', '-LRB-', 'ORGANIZATION-e3', '-RRB-', "'s", sed , '.']
```

# Super Sense Tagger

step 2 given [here](#Step_2:_running_sstagger)
Super sense tagging is when you can take a sentence and assign the abstract super sense to it. like NER but more abstract.
Eg:

Before tagging:

`I do n't think he 's afraid to take a strong stand on gun control , what with his upbringing in El Paso .`

After Tagging:
```
I do|`a n't think|cognition he 's|stative afraid to take_a_ strong _stand|cognition on gun_control|ARTIFACT , what_with his upbringing|ATTRIBUTE in El_Paso|LOCATION .
```

For more details on SS taggging refer Noah Schnieder's github [page](https://github.com/nschneid/pysupersensetagger)

I have a folder `amalgram/` in this repo where the code and trained models are replicated

#### Step 1: creating POS tags
 
The SStagger needs as input the POS tag and the tokens of a given sentence, in a particular one line format.

Eg:
```Sounds	VBZ
haunting	VBG
,	,
and	CC
a	DT
```
Refer to Noah's code base above for more details.

This code base of mine, which you are looking at, I am using to generate these tokens/tags in the required format for the claim evidence pairs from [FEVER1.0](http://fever.ai/2018/task.html) data set. To do that run the command below.:

`python superSenseTag.py `
`
--write_pos_tags True --pyproc_port 8887 --use_docker True --inputFile data/fever_train_split_fourlabels.jsonl
`


Notes to self
- on laptop and clara use conda environment `delex` 
steps
    - check list before running pos tagging on server
        - go to a new folder 
         -remove outputs if exists 
         -create outputs folder 
        - git pull 
        - verify the log -1
        - start delex conda env
        - change port
        - change input file
        - change docker false
        - verify atleast one claim file is written
        - verify atleast one evidence file is written
        - sort evidence files by size and pick the biggest one so far+ verify that a new line written after evidences    
            
Notes
- classic fever data has only 3 classes/labels,. viz.,SUPPORTS, REFUTES, NOT ENOUGH INFO. Here we have already converted into 4 classes after that of [fnc](http://www.fakenewschallenge.org/), viz., AGREE, DISAGREE, DISCUSS, UNRELATED 
- this command will create a huge number of files, one per each claim-evidence pair.
- if you can get the docker to run for pyprocessors and then use `--use_docker true` that will be fastest way to run this code.``
    - `turn on docker`
    - `docker start procserv`
    - open `localhost:8886` and confirm that the `pyprocessor` server is running

#Step_2:_running_sstagger
#### Step_2: running_sstagger


- commands for create a conda run environment for running the actual sstagger    
```
 conda create -name py2 python=2.7
 source activate py2
 pip install cython
 pip install nltk
 python
 nltk.download('wordnet')
 exit
```
 
 
- Check list:
    - source activate py2
    - make sure *.tags doesn't exist in input folder or output folder. Might be vestigial/left over from older runs, but yeah, that will be detrimental if a *.tags is provided as input. THe code will crash
    - open up predict.ssh and make sure the values of` --input_folder` points to where you generated the POS tagged
   files from step1. and `--output_folder` exists. 
    - input_folder is where  your pos tagged files are
    - open up src/discriminativeTagger.pyx , and make `run_with_parallelization=False` in the main() function.
    - now run the below command from the place where the file `sst.sh` exists.

`./sst.sh example`
- example is a dummy input file left over for vestigial reasons,which will never be used
- the logs and print statements will be written inside example.log.
- once you start the command and if you see SS tags being printed on screen, you should be fine. nevertheless: got to folder fnc_test_outputs_sstagged and keep an eye on the number of files created. also open up some of the output files created and make sure no surprise sentences are added.
  
```
    ls /work/mithunpaul/neuter_ner_fnc_test/amalgram/pysupersensetagger-2.0/fnc_test_outputs_sstagged/pos_tagged_files_fnc_test
```
###### xargs
also here is a version of the same command that will run the above as multiple processses. 
i suggest avoiding it if you have enough time to run it serially.

 
- CheckList/steps:
    - create conda environment as above
    - create a new folder (pos_tagged_files_fnc_test) at the same location as sst.sh
    - copy all the pos tags generated in step2 to this input folder
    - do an `ls -al` in this input folder and make sure this is 
    double the number you get when you do an `wc -l` on the lexicalized data file
    eg:
    ```
    ls /work/mithunpaul/neuter_ner_fnc_test/amalgram/pysupersensetagger-2.0/pos_tagged_files_fnc_test | wc -l
    50826
    wc -l /work/mithunpaul/neuter_ner_fnc_test/data/fn_test_split_fourlabels.jsonl`
    25413
    ```
    - open up predict.ssh remove the value of` --input_folder`. This is because the find command
    above takes care of it. 
    - create an output folder (eg: `mkdir fnc_test_outputs_sstagged`)
    - inside the output folder create another folder which has exactly the same name as the input folder
    (Eg:`mkdir fnc_test_outputs_sstagged/pos_tagged_files_fnc_test`). This has to be done because xargs passes the entire path of the input file , including the folder name. so then the code tries to create an output file with the same name, which includes the path.
    
    - open up predict.ssh and make sure `--output_folder` exists and points to the path above- only the outer folder. 
    - open predict.ssh and add `--use_xargs` at the end of python command
    - make sure no file with name `*.tags` exist in input folder or output folder. Might be vestigial/left over from older runs, but yeah, that will be detrimental if a *.tags is provided as input. THe code will crash
    - open up src/discriminativeTagger.pyx , and make `run_with_parallelization=True` in the main() function. 
    - source activate py2
    - open up sst.sh and make sure the input file is not written from linux shell scripot also
    i.e it should look like this:
    ```
    ./predict_sst.sh $input 
    ```
 
    and not
    ```
    ./predict_sst.sh $input > $input.pred.tags
    ```
    - before you start the run with xargs, try running bare minimum command tomake sure there are no compile timer errors
    Eg: `./sst.sh dummy_input_file`
    - now run the below command from the folder where the file `sst.sh` exists.    
    
    
`find pos_tagged_files_fnc_test_5k/ -print0 | xargs -0 -n 1 -I{} -t -P 72 ./sst.sh {}`


   
Notes:
- the xargs command will run the sst.sh on each input file from the folder you provide, which in this case is `pos_tagged_files_fnc_test`. Change accordingly
- P is the number of cores you can spare in your machine. Change accordingly. Don't change anything else.
- some files will take really long time to get SSTagged. This is because it most probably contains way too many sentences. Be patient. That is why we are running it over xargs because
even if one process is stuck the rest 71 will keep moving forward.
- the code should stop by itself when the total number of output files are the same as input files (in this case 50826 )- 50k files should take around 500 minutes=8 hours

#### Step 3: merging NER tags and ss tags

In this phase, we need to do the smartner tagging plus ss tagging. Eg:

`input: Daniel Craig was the longest serving James Bond`

`output: PERSONc1 was the longest COGNITIONc1 PERSONc2`


other examples
```
claim: 		'A 	seven 		time 		Formula 	       One 		     World 	Champion 	is 		Michael Shumacher 	.'
combined: '	A 	NUMBERc1 	TIMEc1 		Foodc1 		Numberc2 	MISCc1 			stativec1 	PERSONc1 		O.'



evidence: 'He 	is 		a 	seven-time 	Formula 	One 		World Champion 	and 	is 	widely 	regarded 	as one of the greatest Formula One drivers of all time .'
combined:	'He 	stativee1 	a 	seven-time 	MISCc1 	Numberc2  	MISCc1 			and 	is 	widely      sociale1 	as NUMBERe2 of the greatest Foodc1 Numberc2 PERSONe1 of all time .'
```

This can be run using

`source activate delex`

`python main.py --use_docker true  --inputFile data/fever_train_split_fourlabels.jsonl --convert_prepositions False --create_smart_NERs False --merge_ner_ss True --input_folder_for_smartnersstagging_merging sstagged_files/ --outputFolder sstag_ner_merged_files --remove_punctuations False --log_level ERROR`                
 
 on server:
```
python main.py --use_docker false  --inputFile data/fever_original_dev_our_test_partition.jsonl --convert_prepositions False --create_smart_NERs False --merge_ner_ss True --input_folder_for_smartnersstagging_merging amalgram/pysupersensetagger-2.0/outputs_sstagged/ --outputFolder sstag_ner_merged_files_fever_test --remove_punctuations False --log_level ERROR --pyproc_port 8886 --back_off_to_ner True
```

## checklist to check before starting merging 
        
- git pull
- source activate delex
- make sure you are in the folder that you match the sstagged files with eg: neuter_ner_fever_training
- make sure input folder name is correct (eg:`--input_folder_for_smartnersstagging_merging amalgram/pysupersensetagger-2.0/outputs_sstagged//`)..
- make sure there are no .pred.tags.pred.tags file in the input_folder()
- make sure you are providing the same data/inputfile i.e plain text file (Eg:-inputFile data/fever_dev_split_fourlabels.jsonl)
- create output folder if that doesn't exist : `mkdir sstag_ner_merged_files`
- give a new pyproc port for every run
- you can check the total number of data points written by doing `wc -l sstag_ner_merged_files/smartner_sstags_merged.jsonl `
- you can keep an eye on the number of files converted/skipped by doing `tail -f merging_sstag_smartnertag.log `


## extra info 
- for cleaning/removing punctuations i use [this](https://github.com/jfilter/clean-text)
