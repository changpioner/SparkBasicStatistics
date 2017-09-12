package test

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * Created by Namhwik on 2017/9/6.
  */
object Test {
  System.setProperty("hadoop.home.dir","C:\\ruanjian\\hadoop")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val os = scala.sys.props.get("os.name").head
    val spark = if(os.startsWith("Windows"))
      SparkSession.builder().appName("test").master("local").getOrCreate()
    else
      SparkSession.builder().appName("test").getOrCreate()
    val vector = Vectors.sparse(4,Seq((0, 1.0), (3, -2.0)))
    println(vector.apply(0)+","+vector.apply(1)+","+vector.apply(2)+","+vector.apply(3))
  }
}
