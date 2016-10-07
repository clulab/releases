package edu.arizona.sista.utils

import java.io._

/**
  * Created by dfried on 6/9/14.
  */
object Serialization {

  def serialize[A](obj: A, filename: String): Unit = {
    val fout = new FileOutputStream(filename)
    val oout = new ObjectOutputStream(fout)
    oout.writeObject(obj)
    oout.close
    fout.close
  }


  import java.io.{InputStream, ObjectInputStream, ObjectStreamClass}

  class ClassLoaderObjectInputStream(cl: ClassLoader, is: InputStream) extends ObjectInputStream(is) {
    override def resolveClass(osc: ObjectStreamClass): Class[_] = {
      val c = Class.forName(osc.getName, false, cl)
      if (c != null) c else super.resolveClass(osc)
    }
  }


  def deserialize[A](filename: String): A = {
    val fin = new FileInputStream(filename)
    val oin = new ClassLoaderObjectInputStream(getClass.getClassLoader, fin)
    val obj = oin.readObject.asInstanceOf[A]
    oin.close
    fin.close
    obj
  }
}
