/*
 *   Creates "sentenceids_to_sentences.tsv"
 *   and "sentence_chars_to_BERTtokens_indexes.tsv"
 * 
 *   Run this file using the following command:
 *   mvn scala:run -DmainClass=Path_Extraction_Step2 "-DaddArgs=arg1|arg2"
 *   In BERT mode, you need to provide two commmand-line arguments:
 *   arg1: virtual environment path for running create_BERTtoken_mappings.py
 *   arg2: BERT model name that will be passed to create_BERTtoken_mappings.py
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

object Path_Extraction_Step2 {
    def main(args:Array[String]) {
    
        // determines whether preprocessing is done for BERT or DIRT
        val BERT_mode = true
    
        val files_dir = "src/main/scala/code/data/"
        val docs_files_dir = files_dir + "docs/"
        val preprocess_dir = "./src/main/scala/code/path_extraction/"
        val call_create_BERTtoken_mappings_filename = "call_create_BERTtoken_mappings"
        
        val num_of_doc_files = 100;
        var new_sen_id = 0
        var total = num_of_doc_files
        var progressMileStone = 0.05
        
        /* dictionary for storing mappings between sentenceids and sentences.
           sentence_to_sentenceid = {sentence : sentence_id} */
        var sentence_to_sentenceid: Map[String , Int] = Map()
        
        println("Extracting sentences...")
        for (i <- 1 until num_of_doc_files+1)
        {
            // update progress indicator
            if ((i/total) > progressMileStone)
            {
                println(Math.round(progressMileStone * 100) + "% ")
                progressMileStone += 0.05
            }
        
            // load the annotated corpus from the file 
            // created by "path_extraction_step1.scala"
            val doc_serializer = new DocumentSerializer
            val doc_bufferedReader = new BufferedReader(new FileReader(docs_files_dir + "doc_part" + "%03d".format(i)))
            var doc = doc_serializer.load(doc_bufferedReader)
            doc_bufferedReader.close

            // fill sentence_to_sentenceid
            for (sentence <- doc.sentences) {
                val sen = sentence.words.mkString(" ")
                if (!sentence_to_sentenceid.contains(sen))
                {
                    sentence_to_sentenceid(sen) = new_sen_id
                    new_sen_id += 1
                }            
            }
        }
        println("100%\n")
        
        // write the mapping of sentenceids to sentences to a file (sentenceids_to_sentences.tsv)
        println("Writing 'sentenceids_to_sentences.tsv' to disk...")
        val sentenceids_to_sentences_file = new File(files_dir + "sentenceids_to_sentences.tsv")
        val sentenceids_to_sentences_writer = new PrintWriter(sentenceids_to_sentences_file, "UTF-8")
        
        var sc = 0
        progressMileStone = 0.05
        val dlm = "\t"
        val totalUniqueSentences: Double = sentence_to_sentenceid.size
        for ((sentence , sentence_id) <- sentence_to_sentenceid)
        {
            if ((sc/totalUniqueSentences) > progressMileStone)
            {
                println(Math.round(progressMileStone * 100) + "% ")
                progressMileStone += 0.05
            }
        
            sentenceids_to_sentences_writer.write(sentence_id.toString +
                                                  dlm +
                                                  "\"" + process_string(sentence) + "\"" +
                                                  "\n")
            sc += 1
        }
        sentenceids_to_sentences_writer.close()
        println("100%\n")

        
        if (BERT_mode)
        {
            // create "sentence_chars_to_BERTtokens_indexes.tsv"
            val virtual_environment_path = args(0)
            val BERT_model_name = args(1)
            preprocess_dir + call_create_BERTtoken_mappings_filename + " " + virtual_environment_path + " " + BERT_model_name !
        }
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
}
