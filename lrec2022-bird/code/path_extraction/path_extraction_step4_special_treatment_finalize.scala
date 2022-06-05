/*
 *  See "path_extraction_step4.scala" for explanation
 *  
 *  Run this file using the following command:
 *  mvn scala:run -DmainClass=Path_Extraction_Step4_special_treatment_finalize "-DaddArgs=arg1"
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

object Path_Extraction_Step4_special_treatment_finalize {
    def main(args:Array[String]) {
    
        val part_no_int = args(0).toInt
        val part_no_str = "%03d".format(part_no_int)
    
        val subparts_total = 100
        
        val files_dir = "src/main/scala/code/data/"
        val filtered_paths_dbs_files_dir = files_dir + "filtered_paths_dbs/"
        val filtered_paths_dbs_spec_treat_files_dir = filtered_paths_dbs_files_dir + "special_treatment/"
    
        var lineCount = 0
        
        var path_str = ""
        var slotX_str = ""
        var slotY_str = ""
        var sentences_info_str = ""

        for (i <- 1 until subparts_total+1)        
        {
            var subpart_no_str = "%03d".format(i)
            val paths_db_subpart_file = filtered_paths_dbs_spec_treat_files_dir + "filtered_paths_db_part" + part_no_str + "_subpart_" + subpart_no_str + ".tsv"
            val pdbsp_bufferedSource = scala.io.Source.fromFile(paths_db_subpart_file, "UTF-8")
            
            lineCount = 0
            for (line <- pdbsp_bufferedSource.getLines)
            {
                lineCount += 1
                
                if (i == 1)
                {
                    val path_data = line.split("\t")
                    path_str = path_data(0)
                    slotX_str = path_data(1)
                    slotY_str = path_data(2)
                    sentences_info_str = path_data(3)
                }
                else
                {
                    sentences_info_str += "," + line 
                }
            }
            pdbsp_bufferedSource.close
        }
        
        val paths_db_file = new File(filtered_paths_dbs_files_dir + "filtered_paths_db_part" + part_no_str + ".tsv")
        val paths_db_writer = new PrintWriter(paths_db_file, "UTF-8")
        paths_db_writer.write("{" + path_str + ":(" + slotX_str + "," + slotY_str + ",[" + sentences_info_str + "])}" + "\n")
        paths_db_writer.close()
    }
}
