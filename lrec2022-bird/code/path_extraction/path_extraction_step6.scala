/*
 * Generates 'xfeatures_paths.tsv' and 'yfeatures_paths.tsv'  
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

object Path_Extraction_Step6 {
    def main(args:Array[String]) {

        val files_dir = "src/main/scala/code/data/"
        val filtered_paths_dbs_files_dir = files_dir + "filtered_paths_dbs/"
    
        /* paths_db is the container for the extracted paths along with their features and sentences
        *  paths_db = { path , (slotX_words_freq , slotY_words_freq , sentences_info) }
        *  slotX_words_freq = {word , freq}
        *  slotY_words_freq = {word , freq} 
        *  sentences_info = [(slotX_start_char_index, slotX_end_char_index, slotX_word, slotY_start_char_index, slotY_end_char_index, slotY_word, sentence_id)]
        */
        var paths_db: Map[ArrayBuffer[(String, String, String, String)] , 
                          (Map[String,Int] , Map[String,Int] , ArrayBuffer[(Int,Int,String,Int,Int,String,Int)])]  = Map()
        
        var paths_to_pathids_db: Map[ArrayBuffer[(String, String, String, String)] , Int] = Map()

        /* slotX_features_db and slotY_features_db = {word , List[path_ids]}
           The dictionaries that for each word store the list of path ids 
           of the paths that the word fills a slot of (slotX and slotY).
           These dictionaries are created to speed up the calculation of 
           similary scores later in the pipeline */
        var slotX_features_db: Map[String , List[Int]] = Map()
        var slotY_features_db: Map[String , List[Int]] = Map()

        val num_of_parts = 150
        
        // read 'filtered_paths_db_partxxx.tsv' files from disk to fill paths_db dictionary
        println()
        println("Reading 'filtered_paths_db_partxxx.tsv' files from disk...")

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
            
                var path = create_path_from_string(path_str)
                var slotX = create_slot_from_string(slotX_str)
                var slotY = create_slot_from_string(slotY_str)
                //var sentences_info = create_sentencesinfo_from_string(sentences_info_str)
                var sentences_info : ArrayBuffer[(Int,Int,String,Int,Int,String,Int)] = ArrayBuffer() // it is intractable to fill sentences_info; fortunately we don't need it in this scala file, so we fill it with an empty arraybuffer
                
                paths_db(path) = (slotX, slotY, sentences_info)
            }
            pdbp_bufferedSource.close
            
            println(i + "/150 parts loaded")
        }

        // read 'pathids_to_paths.tsv' from disk 
        // and load it into 'paths_to_pathids_db' dictionary
        println()
        println("Reading 'pathids_to_paths.tsv' from disk...")
        
        val pitp_bufferedSource = scala.io.Source.fromFile(files_dir + "pathids_to_paths.tsv", "UTF-8")
        for (line <- pitp_bufferedSource.getLines)
        {
            val path_data = line.split("\t")
            val path_id = path_data(0).toInt
            val second_col = path_data(1)
            val path_str = second_col.substring(1 , second_col.length - 1)  //remove encompassing '[' and ']'
            val path = create_path_from_string(path_str)
            
            paths_to_pathids_db(path) = path_id
        }
        pitp_bufferedSource.close

        
        println()
        println("Creating slotX and slotY features to paths databases...")
        
        /* create two dictionaries slotX_features_db and slotY_features_db that
         * each contain a mapping from words to the lists of path ids of the paths
         * that the words have been used in as a slot-filler. One for slotX and one 
         * for slotY. */
        var c = 1
        var total_iterations = paths_db.size
        var mileStone: Int = total_iterations / 100
        var progress_status = 0
        
        for((path, (slotX,slotY,_)) <- paths_db)
        {
            var path_id = paths_to_pathids_db(path)
        
            for ((word,_) <- slotX)
            {
                if (slotX_features_db.contains(word))
                    slotX_features_db(word) = path_id :: slotX_features_db(word)
                else
                    slotX_features_db(word) = List(path_id)
            }
            
            for ((word,_) <- slotY)
            {
                if (slotY_features_db.contains(word))
                    slotY_features_db(word) = path_id :: slotY_features_db(word)
                else
                    slotY_features_db(word) = List(path_id)
            }
            
            if (c % mileStone == 0)
            {
                progress_status += 1
                println(progress_status + "%")
            }
            c += 1
        }
        
        
        val totalSlotXFeatures = slotX_features_db.size
        val totalSlotYFeatures = slotY_features_db.size
        println()
        println("Number of total slotX features (words): " + totalSlotXFeatures)
        println("Number of total slotY features (words): " + totalSlotYFeatures)
        println()
        
        println("Writing 'xfeatures_paths.tsv' to disk...")

        // write the previously created dictionaries to files 
        // xfeatures_paths.tsv and yfeatures_paths.tsv
        val Xfeatures_paths_file = new File(files_dir + "xfeatures_paths.tsv")
        val Xfeatures_paths_writer = new PrintWriter(Xfeatures_paths_file, "UTF-8")
        
        c = 1
        total_iterations = totalSlotXFeatures
        mileStone = total_iterations / 100
        progress_status = 0
        
        for ((word,path_ids) <- slotX_features_db)
        {
            var line = "\"" + process_string(word) + "\"\t" + path_ids.mkString("[" , "," , "]") + "\n"
            
            Xfeatures_paths_writer.write(line)

            if (c % mileStone == 0)
            {
                progress_status += 1
                println(progress_status + "%")
            }
            c += 1
        }
        Xfeatures_paths_writer.close()
        
        
        println()
        println("Writing 'yfeatures_paths.tsv' to disk...")
        
        val Yfeatures_paths_file = new File(files_dir + "yfeatures_paths.tsv")
        val Yfeatures_paths_writer = new PrintWriter(Yfeatures_paths_file, "UTF-8")
        
        c = 1
        total_iterations = totalSlotYFeatures
        mileStone = total_iterations / 100
        progress_status = 0
        
        for ((word,path_ids) <- slotY_features_db)
        {
            var line = "\"" + process_string(word) + "\"\t" + path_ids.mkString("[" , "," , "]") + "\n"
            
            Yfeatures_paths_writer.write(line)
            
            if (c % mileStone == 0)
            {
                progress_status += 1
                println(progress_status + "%")
            }
            c += 1
        }
        Yfeatures_paths_writer.close()
    }
    
    
    def create_path_from_string(path_str:String) : ArrayBuffer[(String, String, String, String)] =
    {
        var path : ArrayBuffer[(String, String, String, String)] = ArrayBuffer()
        var str = path_str
    
        var continue = true
        while (continue)
        {
            var first_tuple_last_index = str.indexOf(')')
            if (first_tuple_last_index == -1)
            {
                continue = false
            }
            else
            {
                var tuple_str = str.substring(1 , first_tuple_last_index)
                    
                var str_last_index = str.length - 1
                if (str_last_index == first_tuple_last_index)
                {
                    str = ""
                }
                else
                {
                    str = str.substring(first_tuple_last_index+2 , str_last_index+1)
                }
                    
                var tuple = tuple_str.split(",")
                    
                tuple(0) = tuple(0).substring(1 , tuple(0).length-1)
                tuple(1) = tuple(1).substring(1 , tuple(1).length-1)
                tuple(2) = tuple(2).substring(1 , tuple(2).length-1)
                tuple(3) = tuple(3).substring(1 , tuple(3).length-1)
                    
                tuple(0) = reverse_process_string(tuple(0))
                tuple(1) = reverse_process_string(tuple(1))
                tuple(2) = reverse_process_string(tuple(2))
                tuple(3) = reverse_process_string(tuple(3))
                    
                path.append((tuple(0) , tuple(1) , tuple(2) , tuple(3)))
            }
        }
        
        return path
    }


    def create_slot_from_string(slot_str:String) : Map[String,Int] =
    {
        var slot : Map[String,Int] = Map()
        
        var key_value_pairs = slot_str.split(",")
        
        for (i <- 0 until key_value_pairs.length)
        {
            var key_value_str = key_value_pairs(i)
            
            var key_value = key_value_str.split(":")
            
            var key = key_value(0)
            key = key.substring(1 , key.length-1)
            key = reverse_process_string(key)
            
            var value = key_value(1).toInt
            
            slot(key) = value
        }
        
        return slot
    }


    def create_sentencesinfo_from_string(sentences_info_str:String) : ArrayBuffer[(Int,Int,String,Int,Int,String,Int)] =
    {
        var sentences_info : ArrayBuffer[(Int,Int,String,Int,Int,String,Int)] = ArrayBuffer()
        var str = sentences_info_str
    
        var continue = true
        while (continue)
        {
            var first_tuple_last_index = str.indexOf(')')
            if (first_tuple_last_index == -1)
            {
                continue = false
            }
            else
            {
                var tuple_str = str.substring(1 , first_tuple_last_index)
                    
                var str_last_index = str.length - 1
                if (str_last_index == first_tuple_last_index)
                {
                    str = ""
                }
                else
                {
                    str = str.substring(first_tuple_last_index+2 , str_last_index+1)
                }
                    
                var tuple = tuple_str.split(",")
                    
                tuple(2) = tuple(2).substring(1 , tuple(2).length-1)
                tuple(5) = tuple(5).substring(1 , tuple(5).length-1)
                    
                tuple(2) = reverse_process_string(tuple(2))
                tuple(5) = reverse_process_string(tuple(5))
                    
                sentences_info.append((tuple(0).toInt , tuple(1).toInt , tuple(2) , tuple(3).toInt , tuple(4).toInt , tuple(5) , tuple(6).toInt))
            }
        }
        
        return sentences_info
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
}
