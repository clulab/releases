/*
 *  Reads 'filtered_paths_db_partxxx.tsv' files one by one,
 *  and generates three tsv files based on them:
 *  'pathids_to_paths.tsv', 'paths_features.tsv', and 'paths_sentences.tsv'
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

object Path_Extraction_Step5 {
    def main(args:Array[String]) {

        val files_dir = "src/main/scala/code/data/"
        val filtered_paths_dbs_files_dir = files_dir + "filtered_paths_dbs/"
        
        val num_of_parts = 150
        
        val pathids_to_paths_file = new File(files_dir + "pathids_to_paths.tsv")
        val pathids_to_paths_writer = new PrintWriter(pathids_to_paths_file, "UTF-8")
        
        val paths_features_file = new File(files_dir + "paths_features.tsv")
        val paths_features_writer = new PrintWriter(paths_features_file, "UTF-8")
        
        val paths_sentences_file = new File(files_dir + "paths_sentences.tsv")
        val paths_sentences_writer = new PrintWriter(paths_sentences_file, "UTF-8")

        println()
        println("Writing 'pathids_to_paths.tsv', 'paths_features.tsv', and 'paths_sentences.tsv' to disk...")
        
        val delim = "\t"
        var counter = 0
        
        for (i <- 1 until num_of_parts+1)        
        {
            var part_no_str = "%03d".format(i)
            val paths_db_part_file = filtered_paths_dbs_files_dir + "filtered_paths_db_part" + part_no_str + ".tsv"
            val pdbp_bufferedSource = scala.io.Source.fromFile(paths_db_part_file, "UTF-8")
            
            for (line <- pdbp_bufferedSource.getLines)
            {
                val line_content = line.substring(1 , line.length-1) // remove '{' and '}'
            
                val path_str_last_index = line_content.indexOf(']')
                val path_str = line_content.substring(1 , path_str_last_index)
            
                var dic_val_str = line_content.substring( path_str_last_index+3 , line_content.length-1 )
                val slotX_str_last_index = dic_val_str.indexOf('}')
                val slotX_str = dic_val_str.substring(1 , slotX_str_last_index)
            
                dic_val_str = dic_val_str.substring( slotX_str_last_index+2 , dic_val_str.length )
                val slotY_str_last_index = dic_val_str.indexOf('}')
                val slotY_str = dic_val_str.substring(1 , slotY_str_last_index)
            
                dic_val_str = dic_val_str.substring( slotY_str_last_index+3 , dic_val_str.length-1 )
                val sentences_info_str = dic_val_str
                
                pathids_to_paths_writer.write(counter.toString +
                                            delim +
                                            "[" + path_str + "]" +
                                            "\n")

                paths_features_writer.write(counter.toString +
                                            delim + 
                                            "{" + slotX_str + "}" +
                                            delim +
                                            "{" + slotY_str + "}" +
                                            "\n")
            
                paths_sentences_writer.write(counter.toString +
                                            delim +
                                            "[" + sentences_info_str + "]" +
                                            "\n")
                counter += 1
            }
            
            pdbp_bufferedSource.close
            println("part " + part_no_str + " completed")
        }
        
        pathids_to_paths_writer.close()
        paths_features_writer.close()
        paths_sentences_writer.close()
    }
}
