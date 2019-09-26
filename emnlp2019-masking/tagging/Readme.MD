# This code base has two pre-processing codes. Mostly to do with POS tagging, NER, Super sense tagging etc

## smart ner convertor

This code takes a claim and evidence pairs, finds where all NER tags exist and replace them smartly. refer examples below. 
This is being done to show the NN model that there is overlap between claim and evidence.

```
conda create --name meanteacher python=3
source activate meanteacher
pip install tqdm
pip install clean-text
pip install git+https://github.com/myedibleenso/py-processors.git
```

**note: we are using an in house tool called [pyprocessors](https://py-processors.readthedocs.io/en/latest/) to do annotation/NER/POS tagging etc. 
We are using the jar option mentioned in the file to run pyprocessors server. However if you are adventurous enough to go the docker route, below are the commands you must use.
**

commands to run:

`source activate meanteacher`

`docker pull myedibleenso/processors-server:latest`
` docker run myedibleenso/processors-server`

`docker run -d -e _JAVA_OPTIONS="-Xmx3G" -p 127.0.0.1:8886:8888 --name procserv myedibleenso/processors-server`
or


```docker start procserv``` (if you are using docker)

If not using docker, run this command below (it should start the pyprocessors as a java server as mentioned [here](https://py-processors.readthedocs.io/en/latest/example.html) under `Option 1`):
`
python main.py --pyproc_port 8887 --use_docker False --convert_prepositions False --convert_NERs True --inputFile data/dev_fourlabels_new.jsonl
`
#### optional command line arguments"

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
- on laptop and clara use conda environment `meanteacher` 
steps
    - check list before running pos tagging on server
        - go to a new folder 
         -remove outputs if exists 
         -create outputs folder 
        - git pull 
        - verify the log -1
        - start meanteacher conda env
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

#### Step 2: running sstagger

the above came these are the commands i used to combine multiple claim files to one

steps 
- move the output of pos tagging to input folder of ss tagging.
    - note: dont do plain `mv * ../amalgram/pysupersensetagger-2.0/input_to_sstagger_output_from_pos_tagger/`. Linux will tell you argument list too long. Instead create a shell script like this:
    ```for each in ./*;
    do
    mv $eachfile ../amalgram/pysupersensetagger-2.0/input_to_sstagger_output_from_pos_tagger/
    done
    ```
- commands for create a conda run environment for running the actual sstagger    
```
 conda create -name py2_decompattn_nonallennlp python=2.7
 source activate py2_decompattn_nonallennlp
 pip install cython
 pip install nltk
 python
 nltk.download('wordnet')
 exit
```
- remember to create a folder inside outputs dir if using xargs, and the output (the sstagged files)
 will be written there.
 
 Example:
 
 ``mkdir outputs_sstagged/input_to_sstagger_output_from_pos_tagger``

- Check two things. *.tags doesn't exist in input folder or output folder. Might be vestigial/left over from older runs, but yeah, that will be detrimental if a *.tags is provided as input. THe code will crash
- open up predict.ssh and make sure the values of` --input_folder` points to where you generated the POS tagged
   files from step1. and `--output_folder` exists. 
- now run the below command from the place where the file `sst.sh` exists.

`./sst.sh example`
- example is a dummy input file left over for vestigial reasons,which will never be used
- the logs and print statements will be written inside example.log.

also here is a version of the same command that will run the above as multiple processses. i suggest avoiding it though.

`find ./input_to_sstagger_output_from_pos_tagger/ -print0 | xargs -0 -n 1 -I{} -t -p ./sst.sh {}`
 
Notes:
- the xargs command will run the sst.sh on each input file from the folder: input_to_sstagger_output_from_pos_tagger
- i have modified the python code to create output file with $inputfilename.pred.tags in the output folder mentioned above
- P is the number of cores you can spare in your machine. 
- this command will create the output (*.pred.tags) at exactly the same location as the input file. Which means, you must do `rm *.tags` before every run else it will
start creating files like `*.pred.tags.pred.tags`

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

`source activate meanteacher`

`python main.py --use_docker true  --inputFile data/fever_train_split_fourlabels.jsonl --convert_prepositions False --create_smart_NERs False --merge_ner_ss True --input_folder_for_smartnersstagging_merging sstagged_files/ --outputFolder sstag_ner_merged_files --remove_punctuations False --log_level ERROR`                
 on server:
`python main.py --use_docker false  --inputFile data/fever_original_dev_our_test_partition.jsonl --convert_prepositions False --create_smart_NERs False --merge_ner_ss True --input_folder_for_smartnersstagging_merging amalgram/pysupersensetagger-2.0/outputs_sstagged/ --outputFolder sstag_ner_merged_files_fever_test --remove_punctuations False --log_level ERROR --pyproc_port 8886`

## extra info 
- for cleaning/removing punctuations i use [this](https://github.com/jfilter/clean-text)


# notes to self/delete later 
  
- when you want to learn something, check if there is a course in stanford or berkeley that covers that topic
    or google: ```xargs site:*.edu```


 
 
checklist for merging runs
        
    - git pull
    - source activate meanteacher
    - make sure you are in the folder that you match the sstagged files with eg: neuter_ner_fever_training
    - make sure input folder name is correct (eg:--input_folder_for_smartnersstagging_merging amalgram/pysupersensetagger-2.0/outputs_sstagged//)..
    - make sure there are no .pred.tags.pred.tags file in the input_folder()
    - make sure you are providing the same data/inputfile i.e plain text file (Eg:-inputFile data/fever_dev_split_fourlabels.jsonl)
    - create output folder if that doesn't exist : `mkdir sstag_ner_merged_files`
    - give a new pyproc port for every run
    

### for emnlp 2019-abstract submission on may 15th 2019 below corners were cut
        - punctuations were left in.
        - there was length mismatch between claim and evidence found while doing merging of super sense and ner tags. I 
        told the code to skip those files and move on. Almost 1% of files were thus skipped in all 4 runs. Should ideally go back and figure out why there is a length mismatch
        
 ### todo for this project (as of today:may 15th 2019- before the actual paper deadline):
 - generate pos tags for fnc-test
 - generate ss tags for fnc-test
 - merge ss and pos tags for fnc-test
 - generate pos tags for fever-test
 - generate ss tags for fever-test
 - merge ss and pos tags for fever-test
 - add tqdm
 - at some point, get back to removing punctuations.
 - comment this line or move it inside if punctuations:`if not (word.lower() in ["lrb","rrb","lcb","rcb","lsb","rsb"]):`
        
                
            
### Commonly encountered errors
    - if you hit compile time error with text as
    `ImportError: Building module discriminativeTagger failed: ["CompileError: command 'gcc' failed with exit status 1\n"]` it means you have a compilation error. Kill the command and scroll up to find the compilation error.
    - if your code is not responding/hangs for more than a minute after you hit `./sst.sh logs` 
    or hit a error which says cannot find compiled *.so files, do this:`rm ~/.pyxbld/lib.linux-x86_64-2.7/*.so`
    - if it says package x not found, that means you haven't turned on conda. do `source activate py2`
    - if nonoe of this works, i.e tmux window is not responding, get out of the tmux window and do `tmux kill-session -t 4`
    - if you just want to check if a kind of file exists and if `ls *.tags` gives you error of `argument too long` you can try :` find amalgram/pysupersensetagger-2.0/outputs_sstagged/ -iname "one*"`
    
    
### daily work log/diary
- may 15th 2019:
    - submitted abstract to emnlp2019
    - still waiting on sandeep to do fnc-train-fever test using features from acl submission, plainlex, smartner etc
    - have generated sstagged files also for all 4 partitions (fnc-fever-train-test). have uploaded them to gdrive [folder](https://drive.google.com/open?id=1QqbMDyGaao1fUIG3rbgS1xqlxwx8d-o4). sandeep still needs to run the same tests on them
    - have started 2 runs for generating pos tags for fnc-test and fever-test. Details can be found [here](https://docs.google.com/spreadsheets/d/1nfewuq33Hxkwp9WaiLldN4LjRJo3vbVH2vxGHRHKKbI/edit?usp=sharing)
 
