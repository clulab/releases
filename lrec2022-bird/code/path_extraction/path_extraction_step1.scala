/*
 *   Annotates the corpuses using FastNLPProcessor
 *   and serializes the resulting doc objects to files 
 *
 *   Run this file using the following command:
 *   mvn scala:run -DmainClass=Path_Extraction_Step1 "-DaddArgs=arg1"
 *   arg1: the part number for the corpus part to be annotated
 */

import org.clulab.processors.corenlp.CoreNLPProcessor
import org.clulab.processors.shallownlp.ShallowNLPProcessor
import org.clulab.processors.{Document, Processor}
import org.clulab.struct.DirectedGraphEdgeIterator
import org.clulab.processors.fastnlp.FastNLPProcessor
import java.io._
import org.clulab.serialization._

object Path_Extraction_Step1 {
    def main(args:Array[String]) {
        
        val files_dir = "./src/main/scala/code/data/"
        val raws_files_dir = files_dir + "raws/"
        val docs_files_dir = files_dir + "docs/"
        
        
        // create FastNLP Processor
        var proc:Processor = new FastNLPProcessor()
        
        // read the contents of corpus
        val part_no = "%03d".format(args(0).toInt)
        val corpus = scala.io.Source.fromFile(raws_files_dir + "corpus_part" + part_no, "UTF-8").mkString
        
        // run the FastNLP Processer on the corpus
        var doc = proc.annotate(corpus)

        // create PrintWriter for writing the document to a file
        val doc_file = new File(docs_files_dir + "doc_part" + part_no)
        val doc_writer = new PrintWriter(doc_file, "UTF-8")
        
        // serialize the document into a file
        val doc_serializer = new DocumentSerializer
        doc_serializer.save(doc, doc_writer, keepText=false)
        doc_writer.close()
    }
}
