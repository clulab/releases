/*
 *  Extracts paths from the input annotated doc
 *  and saves them in a file
 *
 *  Run this file using the following command:
 *  mvn scala:run -DmainClass=Path_Extraction_Step3 "-DaddArgs=arg1"
 *  arg1: the part number for the input annotated doc file
 */

import sys.process._
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Map
import org.clulab.processors.corenlp.CoreNLPProcessor
import org.clulab.processors.shallownlp.ShallowNLPProcessor
import org.clulab.processors.{Document, Processor}
import org.clulab.struct.DirectedGraphEdgeIterator
import org.clulab.processors.fastnlp.FastNLPProcessor
import org.clulab.serialization._
import java.io._

object Path_Extraction_Step3 {
    def main(args:Array[String]) {

        // determines whether path extraction is done for BERT or DIRT
        val BERT_mode = true
        
        // maximum number of input tokens BERT can accept is 512 
        val BERT_MAX_TOKEN_INDEX = 511

        /* list of named entity classes used for filtering paths */
        val namedEntityClasses = List("PERSON", "LOCATION", "ORGANIZATION", "MISC", "MONEY", "NUMBER", "ORDINAL", "PERCENT", "DATE", "TIME",
                                      "DURATION", "SET", "EMAIL", "URL", "CITY", "STATE_OR_PROVINCE", "COUNTRY", "NATIONALITY", "RELIGION",
                                      "TITLE", "IDEOLOGY", "CRIMINAL_CHARGE", "CAUSE_OF_DEATH", "HANDLE")

        /* list of part-of-speech tags for nouns */
        val nounPosTags = List("CD", "NN", "NNS", "NNP", "NNPS", "PRP", "WP", "WDT", "WP$")

        /* paths_db is the container for the extracted paths along with their features and sentences
        *  paths_db = { path , (slotX_words_freq , slotY_words_freq , sentences_info) }
        *  slotX_words_freq = {word , freq}
        *  slotY_words_freq = {word , freq} 
        *  sentences_info = [(slotX_start_char_index, slotX_end_char_index, slotX_word, slotY_start_char_index, slotY_end_char_index, slotY_word, sentence_id)]
        */
        var paths_db: Map[ArrayBuffer[(String, String, String, String)] , 
                          (Map[String,Int] , Map[String,Int] , ArrayBuffer[(Int,Int,String,Int,Int,String,Int)])]  = Map()

        val files_dir = "src/main/scala/code/data/"
        val docs_files_dir = files_dir + "docs/"
        val paths_dbs_files_dir = files_dir + "paths_dbs/"
        
        /* dictionary for storing mappings between sentenceids and sentences.
           sentence_to_sentenceid = {sentence : sentence_id} */
        var sentence_to_sentenceid: Map[String , Int] = Map()

        var sentence_chars_to_BERTtokens_indexes: Map[Int , Map[Int,Int]] = Map()
        var chars_to_BERTtokens_indexes: Map[Int,Int] = Map()
        
        // fill sentence_to_sentenceid dictionary
        // from "sentenceids_to_sentences.tsv"
        val sentence_to_sentenceid_file = files_dir + "sentenceids_to_sentences.tsv"
        val sts_bufferedSource = scala.io.Source.fromFile(sentence_to_sentenceid_file, "UTF-8")
        for (line <- sts_bufferedSource.getLines)
        {
            var fields = line.split("\t")
            var sen_id = fields(0).toInt
            var sent = fields(1)
            // trim double quotation marks at beginning and end
            sent = sent.substring(1 , sent.length-1)
            sent = reverse_process_string(sent)
            sentence_to_sentenceid(sent) = sen_id
        }
        sts_bufferedSource.close
        

        // In BERT mode, fill sentence_chars_to_BERTtokens_indexes dictionary 
        // from "sentence_chars_to_BERTtokens_indexes.tsv"
        if (BERT_mode)
        {
            val sentence_chars_to_BERTtokens_indexes_file = files_dir + "sentence_chars_to_BERTtokens_indexes.tsv"
            val sctbi_bufferedSource = scala.io.Source.fromFile(sentence_chars_to_BERTtokens_indexes_file, "UTF-8")

            for (line <- sctbi_bufferedSource.getLines)
            {
                chars_to_BERTtokens_indexes = Map()
            
                var fields = line.split("\t")
                var sen_id = fields(0).toInt
                
                var col2 = fields(1)
                col2 = col2.substring(1 , col2.length-1)  //trim '{' and '}'
                
                var col2_fields = col2.split(",")
                for (charidx_tokenidx_str <- col2_fields)
                {
                    var charidx_and_tokenidx = charidx_tokenidx_str.split(":")
                    var char_idx = charidx_and_tokenidx(0).toInt
                    var BERT_token_idx = charidx_and_tokenidx(1).toInt
                    chars_to_BERTtokens_indexes(char_idx) = BERT_token_idx
                }
                sentence_chars_to_BERTtokens_indexes(sen_id) = chars_to_BERTtokens_indexes
            }
            sctbi_bufferedSource.close
        }
        
        // load the annotated corpus from the file 
        // created by "path_extraction_step1.scala"
        val part_no = "%03d".format(args(0).toInt)
        val doc_serializer = new DocumentSerializer
        val doc_bufferedReader = new BufferedReader(new FileReader(docs_files_dir + "doc_part" + part_no))
        var doc = doc_serializer.load(doc_bufferedReader)
        doc_bufferedReader.close
        
        // iterate over each sentence in the corpus and extract paths
        var sentenceCount = 0
        var progressMileStone = 0.05
        var sen_id = -1
        val totalSentences: Double = doc.sentences.length
        
        println("\nNumber of total sentences: " + Math.round(totalSentences))
        
        println("\nExtracting paths...")
        
        for (sentence <- doc.sentences) {
            
            // update progress indicator
            if ((sentenceCount/totalSentences) > progressMileStone)
            {
                println(Math.round(progressMileStone * 100) + "% ")
                progressMileStone += 0.05
            }
            
            // get lemmas for the sentence
            val lemmas = sentence.lemmas match {
                            case Some(i) => i
                            case None => Array[String]()}
            
            // get pos tags for the sentence
            val tags = sentence.tags match {
                            case Some(i) => i
                            case None => Array[String]()}
            
            // get named entities for the sentence
            val entities = sentence.entities match {
                            case Some(i) => i
                            case None => Array[String]()}
            
            // get the lemmatized words of the sentence
            val tokens = sentence.words.zipWithIndex.map{ case (element, index) => lemmas(index) }
            
            // get sentence id of the sentence
            val sen = sentence.words.mkString(" ")
            sen_id = sentence_to_sentenceid(sen)
            
            // get the dependency tree for the sentence
            val deps = sentence.dependencies.get
            
            var multi_word_group = create_multi_word_groups(entities, namedEntityClasses)
            
            // store the observed unprocessed paths of the sentence
            var observed_unprocessed_paths : ArrayBuffer[Seq[(Int, Int, String, String)]] = ArrayBuffer()
            
            // if the sentence is incompatible with its corresponding data in 
            // 'sentence_chars_to_BERTtokens_indexes.tsv', discard the sentence. 
            // Otherwise, proceed to extract paths.
            var discard_sentence = false
            if (BERT_mode)
                if ((sen.length-1) != (sentence_chars_to_BERTtokens_indexes(sen_id).keysIterator.max))
                    discard_sentence = true

            // if the sentence is too big, discard it for computational costs reasons.
            // It's probably a junk sentence anyway.
            // 512 is chosen somewhat arbitrarily as a threshold for "too long" sentences.
            if (tokens.length > 512) {
                discard_sentence = true
            }
            
            if (!discard_sentence)
            {
              // extract paths
              for (start <- 0 until tokens.length-1)
                for (end <- 0 until tokens.length-1)
                {
                    if (start != end)
                    {
                        // get the raw path
                        val paths = deps.shortestPathEdges(start, end, ignoreDirection = true)
                        var unprocessed_path = paths.head
                        
                        /* on both endpoints of the path, remove all depencency relations that are 
                           internal parts of multi-words. For example, in the following path:
                           John Smith went to New York
                           John<-compound<-Smith<-nsubj<-go->nmod_to->York->compound->New
                           remove  John<-compound<-Smith  and  York->compound->New
                           so we'll have:
                           Smith<-nsubj<-go->nmod_to->York
                         */
                        var (trimmed_unprocessed_path , slotX_start_idx , slotX_end_idx , slotY_start_idx , slotY_end_idx) 
                            = trim_multiword_slotfillers(unprocessed_path, multi_word_group)
                        unprocessed_path = trimmed_unprocessed_path

                        var slotX_start_char_idx = -1 
                        var slotX_end_char_idx = -1
                        var slotY_start_char_idx = -1 
                        var slotY_end_char_idx = -1
                        
                        // only keep paths that their slot-fillers are named entities or nouns.
                        // also, in BERT mode, only accept paths with 
                        // BERT tokens indexes <= BERT_MAX_TOKEN_INDEX-1.
                        var keep = false
                        if (unprocessed_path.length > 0)
                        {
                            var slotX_char_span = get_char_span_in_sentence(slotX_start_idx , slotX_end_idx , sentence.words)
                            slotX_start_char_idx = slotX_char_span._1
                            slotX_end_char_idx = slotX_char_span._2

                            var slotY_char_span = get_char_span_in_sentence(slotY_start_idx , slotY_end_idx , sentence.words)
                            slotY_start_char_idx = slotY_char_span._1
                            slotY_end_char_idx = slotY_char_span._2
                        
                            if (BERT_mode)
                                keep = is_valid_BERT_token_index(slotX_start_char_idx , slotX_end_char_idx ,
                                                                 slotY_start_char_idx , slotY_end_char_idx ,
                                                                 sentence_chars_to_BERTtokens_indexes , 
                                                                 sen_id , 
                                                                 BERT_MAX_TOKEN_INDEX) &&
                                       keep_path(namedEntityClasses, entities, nounPosTags, tags, slotX_start_idx, slotY_start_idx) &&
                                       path_doesnt_have_invalid_chars(unprocessed_path, tokens, slotX_start_idx , slotX_end_idx , slotY_start_idx , slotY_end_idx)
                            else
                                keep = keep_path(namedEntityClasses, entities, nounPosTags, tags, slotX_start_idx, slotY_start_idx) &&
                                       path_doesnt_have_invalid_chars(unprocessed_path, tokens, slotX_start_idx , slotX_end_idx , slotY_start_idx , slotY_end_idx)
                        }
                        
                        // because of multi-word slot-fillers, there may be redundant
                        // unprocessed paths. make sure to process them only once.
                        // also, only keep the paths we want to keep.
                        if ((!observed_unprocessed_paths.contains(unprocessed_path)) && keep)
                        {
                            // add unprocessed_path to the list of observed paths
                            observed_unprocessed_paths.append(unprocessed_path)
                            
                            var slotX_word = tokens.slice(slotX_start_idx , slotX_end_idx).mkString(" ")
                            var slotY_word = tokens.slice(slotY_start_idx , slotY_end_idx).mkString(" ")
                        
                            // process the path; replace indexes with tokens, strip 
                            // slot-filler words, and store the path in a new data structure 
                            var path = processPath(unprocessed_path, tokens)

                            // if the path is not new, update paths_db accordingly
                            if (paths_db.contains(path))
                            {
                                var (slotX , slotY , sentences_info) = paths_db(path)
                            
                                if (slotX.contains(slotX_word))
                                    slotX(slotX_word) += 1
                                else
                                    slotX(slotX_word) = 1
                                
                                if (slotY.contains(slotY_word))
                                    slotY(slotY_word) += 1
                                else
                                    slotY(slotY_word) = 1
                                
                                sentences_info.append((slotX_start_char_idx, slotX_end_char_idx, slotX_word, 
                                                       slotY_start_char_idx, slotY_end_char_idx, slotY_word,
                                                       sen_id)) 
                            }
                        
                            // if the path is new, add it to paths_db
                            else
                            {
                                paths_db(path) = (Map(slotX_word -> 1) , 
                                                  Map(slotY_word -> 1) , 
                                                  ArrayBuffer((slotX_start_char_idx, slotX_end_char_idx, slotX_word, 
                                                               slotY_start_char_idx, slotY_end_char_idx, slotY_word,
                                                               sen_id)))
                            }
                        }
                    }
                }
            
            }
            sentenceCount += 1
        }
        
        println("100%\n")

        
        // save paths_db to a file
        println("Writing 'paths_db_part" + part_no + ".tsv' to disk...")
        val paths_db_file = new File(paths_dbs_files_dir + "paths_db_part" + part_no + ".tsv")
        val paths_db_writer = new PrintWriter(paths_db_file, "UTF-8")
        
        for((path, (slotX,slotY,sentences_info)) <- paths_db)
        {
            val path_str = get_path_datastructure_string(path)
            val slotX_str = get_pathslot_datastructure_string(slotX)
            val slotY_str = get_pathslot_datastructure_string(slotY)
            val sentences_info_str = get_pathsent_datastructure_string(sentences_info)
            
            paths_db_writer.write("{" + path_str + ":(" + slotX_str + "," + slotY_str + "," + sentences_info_str + ")}" + "\n")
        }
        paths_db_writer.close()
    }



    /* processes a string by replacing a backslash with 
      two backslashes and adds a backslash before a quotation 
      mark. This is necessary for writing strings to a file 
      that will be later read by another program.
    */
    def process_string(s: String) : String =
    {
        var ret_val = s.replace("\\" , "\\\\")
        ret_val = ret_val.replace("\"" , "\\\"")
        return ret_val
    }


    def reverse_process_string(s: String) : String =
    {
        var ret_val = s.replace("\\\\" , "\\")
        ret_val = ret_val.replace("\\\"" , "\"")
        return ret_val
    }
    
    
    def create_multi_word_groups(entities:Array[String] , namedEntityClasses:List[String]) : Map[Int , List[Int]] =
    {
        var multi_word_group : Map[Int , List[Int]] = Map()
        
        var current_list : List[Int] = List(0)
        
        for (i <- 1 until entities.length)
        {
            if ((namedEntityClasses.contains(entities(i))) && (entities(i) == entities(i-1)))
                current_list = i :: current_list
            
            else
            {
                for (j <- current_list)
                    multi_word_group(j) = current_list
                
                current_list = List(i)
            }
        }
        
        for (j <- current_list)
            multi_word_group(j) = current_list
        
        return multi_word_group
    }


    def trim_multiword_slotfillers(unprocessed_path:Seq[(Int, Int, String, String)] , multi_word_group:Map[Int , List[Int]])
        : (Seq[(Int, Int, String, String)] , Int , Int , Int , Int) =
    {    
        var i = 0
        var last_idx = unprocessed_path.length - 1
        var continue = true
        var trimmed_unprocessed_path = unprocessed_path
        var slotX_indexes : List[Int] = List()
        
        // trim from left-hand side
        while (continue)
        {
            if (multi_word_group(unprocessed_path(i)._1) == multi_word_group(unprocessed_path(i)._2))
            {
                // delete left-most element
                trimmed_unprocessed_path = trimmed_unprocessed_path.drop(1)
                
                if (i==0)
                    slotX_indexes = multi_word_group(unprocessed_path(i)._1)
                
                if (i == last_idx)
                    continue = false
                else
                    i += 1
            }
            else
                continue = false
        }
        
        if (trimmed_unprocessed_path.length == 0)
            return (trimmed_unprocessed_path , -1 , -1 , -1 , -1)
        
        if (slotX_indexes.length == 0)
        {
            if (unprocessed_path(0)._4 == "<")
                slotX_indexes = unprocessed_path(0)._2 :: slotX_indexes
            else
                slotX_indexes = unprocessed_path(0)._1 :: slotX_indexes
        }
        
        val slotX_start_idx = slotX_indexes.min
        val slotX_end_idx   = slotX_indexes.max + 1
        
        //trim from right-hand side
        last_idx = trimmed_unprocessed_path.length - 1
        i = last_idx
        continue = true
        var u_p = trimmed_unprocessed_path
        var slotY_indexes : List[Int] = List()
        
        while (continue)
        {
            if (multi_word_group(u_p(i)._1) == multi_word_group(u_p(i)._2))
            {
                //delete right-most element
                trimmed_unprocessed_path = trimmed_unprocessed_path.dropRight(1)
                
                if (i==last_idx)
                    slotY_indexes = multi_word_group(u_p(i)._1)
                    
                if (i == 0)
                    continue = false
                else
                    i -= 1
            }
            else
                continue = false
        }
        
        if (slotY_indexes.length == 0)
        {
            if (u_p(last_idx)._4 == ">")
                slotY_indexes = u_p(last_idx)._2 :: slotY_indexes
            else
                slotY_indexes = u_p(last_idx)._1 :: slotY_indexes
        }
        
        val slotY_start_idx = slotY_indexes.min
        val slotY_end_idx   = slotY_indexes.max + 1
        
        return (trimmed_unprocessed_path , slotX_start_idx , slotX_end_idx , slotY_start_idx , slotY_end_idx)
    }
    
    
    def get_char_span_in_sentence(start_word_index:Int , end_word_index:Int , words:Array[String]) : (Int , Int) =
    {
        var s = words.slice(0 , start_word_index+1).mkString(" ")
        
        var start_char_index = s.length - words(start_word_index).length
        
        s = words.slice(0 , end_word_index).mkString(" ")
        
        var end_char_index = s.length
        
        return (start_char_index , end_char_index)
    }


    def is_valid_BERT_token_index(slotX_start_char_idx:Int , slotX_end_char_idx:Int ,
                                  slotY_start_char_idx:Int , slotY_end_char_idx:Int , 
                                  sentence_chars_to_BERTtokens_indexes:Map[Int , Map[Int,Int]] , 
                                  sen_id:Int , BERT_MAX_TOKEN_INDEX:Int) : Boolean =
    {
        var BERT_token_index = -1
        
        for (char_index <- List(slotX_start_char_idx , slotX_end_char_idx-1 , slotY_start_char_idx , slotY_end_char_idx-1))
        {
            BERT_token_index = sentence_chars_to_BERTtokens_indexes(sen_id)(char_index)
        
            if ((BERT_token_index<0) || (BERT_token_index >= BERT_MAX_TOKEN_INDEX))
                return false
        }
        
        return true
    }


    /* checks whether should keep the given path or filter it.
       slot fillers must be named entities or nouns.
     */
    def keep_path(namedEntityClasses:List[String], entities:Array[String], 
                  nounPosTags:List[String], tags:Array[String],
                  slotX_idx: Int, slotY_idx: Int) : Boolean =
    {
        var ret_val = false
        
        // Slot fillers must be named entities or nouns
        if ((namedEntityClasses.contains(entities(slotX_idx)) || nounPosTags.contains(tags(slotX_idx))) &&
            (namedEntityClasses.contains(entities(slotY_idx)) || nounPosTags.contains(tags(slotY_idx))))

            ret_val = true
    
        return ret_val
    }


    /* process the given raw path:
         - replace token indexes with tokens
         - strip the slot-filler words from the path
         - return the path in a new data structure
     */
    def processPath(unprocessed_path:Seq[(Int, Int, String, String)], tokens:Array[String]) : 
        ArrayBuffer[(String, String, String, String)] = 
    {   
        val slot_filler_string = "_"
        var processed_path = ArrayBuffer[(String, String, String, String)]()
        
        for (t <- unprocessed_path)
        {
            var new_t = (tokens(t._1) , tokens(t._2) , t._3 , t._4)
            processed_path += new_t
        }
        
        if (processed_path(0)._4 == "<")
        {
            processed_path(0) = processed_path(0).copy(_2 = slot_filler_string)
        }
        else
        {
            processed_path(0) = processed_path(0).copy(_1 = slot_filler_string)
        }
        
        val last_idx = processed_path.length - 1
        if (processed_path(last_idx)._4 == ">")
        {
            processed_path(last_idx) = processed_path(last_idx).copy(_2 = slot_filler_string)
        }
        else
        {
            processed_path(last_idx) = processed_path(last_idx).copy(_1 = slot_filler_string)
        }
        
        return processed_path
    }
    
    
    /* creates a text string for a path. This text is meant to be
       written to a file and later be read by the Python program 
       that uses this data. 
     */    
    def get_path_datastructure_string(processed_path:ArrayBuffer[(String, String, String, String)]) : String =
    {
        var counter = 1
        var ret_val = ""
        
        ret_val += "["
        for (t <- processed_path)
        {
            ret_val += "(" +
                       "\"" + process_string(t._1) + "\"," +
                       "\"" + process_string(t._2) + "\"," +
                       "\"" + process_string(t._3) + "\"," +
                       "\"" + process_string(t._4) + "\""  +
                       ")"
            
            if (counter < processed_path.length)
                ret_val += ","
                
            counter +=1
        }
        
        ret_val += "]"
        return ret_val
    }


    /* creates a text string for a slot data (which is a dictionary).
       This text is meant to be written to a file and later be read 
       by the Python program that uses this data. 
     */
    def get_pathslot_datastructure_string(slot:Map[String,Int]) : String =
    {
        var ret_val = ""
        ret_val += "{"
        for((word,freq) <- slot)
        {
            ret_val += "\"" + process_string(word) + "\":" + freq + "," 
        }
        ret_val = ret_val.substring(0 , ret_val.length-1) + "}"
        return ret_val
    }


    /* creates a text string for a path sentence info (third element of 
       values of paths_db dictionary). This text is meant to be written 
       to a file and later be read by the Python program that uses this data. 
     */    
    def get_pathsent_datastructure_string(sentences_info: ArrayBuffer[(Int,Int,String,Int,Int,String,Int)]) : String =
    {
        var counter = 1
        var ret_val = ""
        
        ret_val += "["
        for (t <- sentences_info)
        {
            ret_val += "(" +
                       t._1.toString + "," +
                       t._2.toString + "," +
                       "\"" + process_string(t._3) + "\"," +
                       t._4.toString + "," +
                       t._5.toString + "," +
                       "\"" + process_string(t._6) + "\"," +
                       t._7.toString + ")"
            
            if (counter < sentences_info.length)
                ret_val += ","
                
            counter +=1
        }
        
        ret_val += "]"
        return ret_val    
    }
    
    
    def path_doesnt_have_invalid_chars(unprocessed_path:Seq[(Int, Int, String, String)], tokens:Array[String],
                                       slotX_start_idx:Int , slotX_end_idx:Int , slotY_start_idx:Int , slotY_end_idx:Int) : Boolean =
    {
        for (t <- unprocessed_path)
        {
            if ((tokens(t._1) contains ",") || (tokens(t._1) contains "}") || (tokens(t._1) contains "]") || (tokens(t._1) contains ")") ||
                (tokens(t._2) contains ",") || (tokens(t._2) contains "}") || (tokens(t._2) contains "]") || (tokens(t._2) contains ")") ||
                (t._3 contains ",") || (t._3 contains "}") || (t._3 contains "]") || (t._3 contains ")") ||
                (t._4 contains ",") || (t._4 contains "}") || (t._4 contains "]") || (t._4 contains ")"))
            {
                return false
            }
        }
        
        var s = ""
        for (i <- slotX_start_idx until slotX_end_idx)
        {
            s = tokens(i)
            if ((s contains ",") || (s contains "}") || (s contains "]") || (s contains ")") || (s contains ":"))
            {
                return false
            }
        }
        
        for (i <- slotY_start_idx until slotY_end_idx)
        {
            s = tokens(i)
            if ((s contains ",") || (s contains "}") || (s contains "]") || (s contains ")") || (s contains ":"))
            {
                return false
            }
        }
        
        return true
    }
}
