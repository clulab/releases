/*
 *  Combines "paths_db_partxxx.tsv" files into a single data structure, 
 *  filters the paths with frequencies below a threshold, 
 *  and saves a part of the remaining paths in "filtered_paths_db_partxxx.tsv" files.
 *
 *  150 of such tsv files are created as below:
 *    - The filtered paths are divided into 100 equal parts. For each part, 
 *    a tsv file is created:
 *    filtered_paths_db_part001 ... filtered_paths_db_part100
 *
 *    - The 50 most frequent individual paths have huge amount of data. We exclude them
 *    from "filtered_paths_db_part100" and instead we create a tsv file for each of them.
 *    In fact, parts 149 and 150 get special treatment, so we only create up to part 148:
 *    filtered_paths_db_part101 ... filtered_paths_db_part148
 *    
 *    - Parts 149 and 150 are gigantic. They're not intitially created. Instead,
 *    they're divided into 100 subparts in "path_extraction_step4_special_treatment.scala"
 *    and then the subparts are combined in "path_extraction_step4_special_treatment_finalize.scala"
 *    to finally create parts 149 and 150.
 *  
 *
 *  Run this file using the following command:
 *  mvn scala:run -DmainClass=Path_Extraction_Step4 "-DaddArgs=arg1"
 *  arg1: the part number for the output "filtered_paths_db_partxxx.tsv" file
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

object Path_Extraction_Step4 {
    def main(args:Array[String]) {
    
        // the filtered paths will be divided into this many equal parts 
        val total_parts = 100
        
        // the last path entries of 'paths_db' are huge.
        // we create a tsv file for each of them.
        val num_of_last_huge_paths = 50
    
        val files_dir = "src/main/scala/code/data/"
        
        val paths_dbs_files_dir = files_dir + "paths_dbs/"
        
        val filtered_paths_dbs_files_dir = files_dir + "filtered_paths_dbs/"
    
        /* paths_db is the container for the extracted paths along with their features and sentences
        *  paths_db = { path , (slotX_words_freq , slotY_words_freq , sentences_info) }
        *  slotX_words_freq = {word , freq}
        *  slotY_words_freq = {word , freq} 
        *  sentences_info = [(slotX_start_char_index, slotX_end_char_index, slotX_word, slotY_start_char_index, slotY_end_char_index, slotY_word, sentence_id)]
        */
        var paths_db: Map[ArrayBuffer[(String, String, String, String)] , 
                          (Map[String,Int] , Map[String,Int] , ArrayBuffer[(Int,Int,String,Int,Int,String,Int)])]  = Map()

        /* slotX_features_db and slotY_features_db = {word , List[path_ids]}
           The dictionaries that for each word store the list of path ids 
           of the paths that the word fills a slot of (slotX and slotY).
           These dictionaries are created to speed up the calculation of 
           similary scores later in the pipeline */
        var slotX_features_db: Map[String , List[Int]] = Map()
        var slotY_features_db: Map[String , List[Int]] = Map()

        // combining data from paths_db_partxxx.tsv files 
        // into a sinlge dictionary
        println()
        println("Aggregating paths databases...")

        val num_of_doc_files = 100;
        var total = num_of_doc_files
        var progressMileStone = 0.05
        for (i <- 1 until num_of_doc_files+1)        
        {
            var partno = "%03d".format(i)
            val paths_db_part_file = paths_dbs_files_dir + "paths_db_part" + partno + ".tsv"
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
                var sentences_info = create_sentencesinfo_from_string(sentences_info_str)
                
                // add to paths_db
                if (paths_db.contains(path))
                {
                    var (slotX_aggregate, slotY_aggregate, sentences_info_aggregate) = paths_db(path)
                    
                    for ((word , freq) <- slotX)
                    {
                        if (slotX_aggregate.contains(word))
                            slotX_aggregate(word) = slotX_aggregate(word) + freq
                        else
                            slotX_aggregate(word) = freq
                    }

                    for ((word , freq) <- slotY)
                    {
                        if (slotY_aggregate.contains(word))
                            slotY_aggregate(word) = slotY_aggregate(word) + freq
                        else
                            slotY_aggregate(word) = freq
                    }
                    
                    for (t <- sentences_info)
                    {
                        sentences_info_aggregate.append((t._1, t._2, t._3, t._4, t._5, t._6, t._7))
                    }
                    
                    paths_db(path) = (slotX_aggregate, slotY_aggregate, sentences_info_aggregate)
                }
                else
                {
                    paths_db(path) = (slotX, slotY, sentences_info)
                }
            }
            pdbp_bufferedSource.close
            
            println(i + "%")
        }
        println()
        
        
        var totalPaths: Double = paths_db.size
        println("Number of total extracted paths: " + Math.round(totalPaths))
        
        println()
        println("Creating paths frequency database...")
        
        // create a list that stores the frequency of each path
        var paths_freq: List[(ArrayBuffer[(String, String, String, String)] , Int)] = List()
        var c = 0
        progressMileStone = 0.05
        for((path, (slotX,_,_)) <- paths_db)
        {
            if ((c/totalPaths) > progressMileStone)
            {
                println(Math.round(progressMileStone * 100) + "% ")
                progressMileStone += 0.05
            }
        
            var freq = 0
            for( (_,f) <- slotX )
                freq += f
        
            paths_freq = (path,freq) :: paths_freq
            
            c += 1
        }
        
        println("100%\n")
        println("Sorting the paths frequency database...\n")
        
        // sort the list that stores the frequency of each path
        paths_freq = paths_freq.sortBy(_._2)
        
        /*
        // write the list to a file (paths_freq.tsv)
        println("Writing the paths frequency database to disk...")
        c = 0
        progressMileStone = 0.05
        val paths_freq_file = new File(files_dir + "paths_freq.tsv")
        val paths_freq_writer = new PrintWriter(paths_freq_file, "UTF-8")
        for (i <- (paths_freq.length-1) to 0 by -1)
        {
            if ((c/totalPaths) > progressMileStone)
            {
                println(Math.round(progressMileStone * 100) + "% ")
                progressMileStone += 0.05
            }
        
            var path = paths_freq(i)._1
            var frequency = paths_freq(i)._2
            val delim = "\t"
            paths_freq_writer.write(get_path_textual_string(path) +
                                    delim +
                                    frequency +
                                    "\n")
                                    
            c += 1
        }
        paths_freq_writer.close()
        println("100%\n")
        */
        
        // paths that have frequency below this threshold will be filtered
        val freq_threshold = 10
        
        println("Finding out how many paths have frequency equal or above the threshold...")
        
        // find out how many paths have frequency equal or above the threshold
        // by using succesor (next-largest element) finding algorithm which is based on binary search algorithm.
        // For more info see https://en.wikipedia.org/wiki/Binary_search_algorithm#Approximate_matches
        var L_bssuc : Int = 0
        var R_bssuc : Int = paths_freq.length
        var frequency : Int = 0
        
        while (L_bssuc < R_bssuc) {
            var m_bssuc : Int = (L_bssuc + R_bssuc) / 2
            
            frequency = paths_freq(m_bssuc)._2
            if (frequency > freq_threshold - 1)
                R_bssuc = m_bssuc
            else
                L_bssuc = m_bssuc + 1
        }
        val first_above_threshold_index = R_bssuc
        if (first_above_threshold_index == paths_freq.length)
        {
            println()
            println("All paths would be filtered using the current threshold! Cannot continue. Terminating the program.")
            return
        }
        
        val last_above_threshold_index_plus_one = paths_freq.length
        var totalRemainingPaths = last_above_threshold_index_plus_one - first_above_threshold_index
        
        println()
        println("Number of paths with frequency equal or above the threshold: " + totalRemainingPaths)
        println("Applying final paths filtering using frequency threshold of " + freq_threshold + " and writing the remaining paths to disk...")
        
        // filter the paths with frequency below the threshold and write the remaining paths to disk
        val part_no_int_f = args(0).toInt
        val part_no_str_f = "%03d".format(part_no_int_f)
        
        val paths_db_file = new File(filtered_paths_dbs_files_dir + "filtered_paths_db_part" + part_no_str_f + ".tsv")
        val paths_db_writer = new PrintWriter(paths_db_file, "UTF-8")
        
        var path_str_f = ""
        var slotX_str_f = ""
        var slotY_str_f = ""
        var sentences_info_str_f = ""
        
        var path_f : ArrayBuffer[(String, String, String, String)] = ArrayBuffer()
        
        val part_size = totalRemainingPaths / total_parts
        
        var for_loop_start = -1
        if (part_no_int_f <= total_parts)
        {
            for_loop_start = first_above_threshold_index + (part_no_int_f - 1) * part_size
        }
        else
        {
            for_loop_start = last_above_threshold_index_plus_one - num_of_last_huge_paths + (part_no_int_f - total_parts - 1)
        }
        
        var for_loop_end = -1
        if (part_no_int_f == total_parts) 
        {
            for_loop_end = last_above_threshold_index_plus_one - num_of_last_huge_paths
        }
        else if (part_no_int_f < total_parts)
        {
            for_loop_end = first_above_threshold_index + part_no_int_f * part_size
        }
        else
        {
            for_loop_end = for_loop_start + 1
        }
        
        if (part_no_int_f >= total_parts)
        {
            println("for_loop_start: " + for_loop_start)
            println("for_loop_end:   " + for_loop_end)
        }
        
        val part_size_double: Double = part_size
        c = 1
        var mile_stone = part_size / 100
        var progress_percentage = 0.0
        
        for (i <- for_loop_start until for_loop_end)
        {
            path_f = paths_freq(i)._1
            val (slotX_f , slotY_f , sentences_info_f) = paths_db(path_f)
            
            path_str_f = get_path_datastructure_string(path_f)
            slotX_str_f = get_pathslot_datastructure_string(slotX_f)
            slotY_str_f = get_pathslot_datastructure_string(slotY_f)
            sentences_info_str_f = get_pathsent_datastructure_string(sentences_info_f)
            paths_db_writer.write("{" + path_str_f + ":(" + slotX_str_f + "," + slotY_str_f + "," + sentences_info_str_f + ")}" + "\n")

            if (part_no_int_f <= total_parts)
            {
                if (c % mile_stone == 0)
                {
                    progress_percentage = ((c / part_size_double) * 100).ceil
                    println(progress_percentage.toInt + "%")
                }
            }
            
            c += 1
        }
        paths_db_writer.close()
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


    /* creates a text string for a path. This text is meant to be
       written to a file and later be read by the Python program 
       that uses this data. 
     */    
    def get_path_datastructure_string(processed_path:ArrayBuffer[(String, String, String, String)]) : String =
    {
        var counter = 1
        var ret_val = "["
        
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
        var ret_val = "{"
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
        var ret_val = "["
        
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
}
