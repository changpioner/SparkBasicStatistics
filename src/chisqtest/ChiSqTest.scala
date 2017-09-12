package chisqtest

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.mllib
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.mllib.stat.test.ChiSqTestResult
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
  * Created by Namhwik on 2017/9/7.
  */
object ChiSqTest {
  System.setProperty("hadoop.home.dir","C:\\ruanjian\\hadoop")
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val os = scala.sys.props.get("os.name").head
    val spark = if(os.startsWith("Windows"))
      SparkSession.builder().appName("test").master("local").getOrCreate()
    else
      SparkSession.builder().appName("test").getOrCreate()
    import spark.implicits._

    val activities: Array[Double] = Array(1,0,1,0,1,0,1,0,0,1,0,1)
    val matrices = Matrices.dense(2,6,activities)
    val chiSq_Result = stat.Statistics.chiSqTest(matrices)
    println(chiSq_Result)


    val data = Seq(
      (0.0, Vectors.dense(0.5, 10.0)),
      (0.0, Vectors.dense(0.6, 20.0)),
      (1.0, Vectors.dense(1.5, 30.0)),
      (0.0, Vectors.dense(0.4, 30.0)),
      (0.0, Vectors.dense(0.45, 40.0)),
      (1.0, Vectors.dense(1.6, 40.0))
    )

    val df = data.toDF("label", "features")
    df.show()
    val chiSq_Result1 = ChiSquareTest.test(df,"features","label")
    chiSq_Result1.show(false)


    val obs: RDD[LabeledPoint] =
      spark.sparkContext.parallelize(
        Seq(
          LabeledPoint(1.0, mllib.linalg.Vectors.dense(1.0, 0.0, 3.0)),
          LabeledPoint(2.0, mllib.linalg.Vectors.dense(5.0, 2.0, 0.0)),
          LabeledPoint(3.0, mllib.linalg.Vectors.dense(7.0, 0.0, 0.5)
          )
        )
      ) // (feature, label) pairs.

    // The contingency table is constructed from the raw (feature, label) pairs and used to conduct
    // the independence test. Returns an array containing the ChiSquaredTestResult for every feature
    // against the label.
    val featureTestResults: Array[ChiSqTestResult] = Statistics.chiSqTest(obs)
    featureTestResults.zipWithIndex.foreach { case (k, v) =>
      println("Column " + (v + 1).toString + ":")
      println(k)
    }  // summary of the test
    println("**************goodness of fit test***********")
    val vec = mllib.linalg.Vectors.dense(1, 2, 2, 4, 3)

    // compute the goodness of fit. If a second vector to test against is not supplied
    // as a parameter, the test runs against a uniform distribution.
    val goodnessOfFitTestResult = Statistics.chiSqTest(vec)
    // summary of the test including the p-value, degrees of freedom, test statistic, the method
    // used, and the null hypothesis.
    println(s"$goodnessOfFitTestResult\n")
  }
}
