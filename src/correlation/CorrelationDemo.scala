package correlation

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.linalg.{Matrix, Vector, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.sql.{Row, SparkSession}

/**
  * Created by Namhwik on 2017/9/6.
  */
object CorrelationDemo {
  System.setProperty("hadoop.home.dir","C:\\ruanjian\\hadoop")
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val os = scala.sys.props.get("os.name").head
    val spark = if(os.startsWith("Windows"))
      SparkSession.builder().appName("test").master("local").getOrCreate()
    else
      SparkSession.builder().appName("test").getOrCreate()
    import spark.implicits._


    /***
    ***************************************************************************************************************************************
    ML的相关性分析，输入为多个Vector组成的DataFrame,需要指定进行分析的列,输出为由一条Row[Matrix]组成的DF
    Matrix是由Vectors两两进行相关性分析的结果矩阵
    ML使用的是ml.stat.Correlatio的corr方法
    ***************************************************************************************************************************************
    ***/
    val data: Seq[Vector] = Seq(
      Vectors.dense(4.0,2.0,4.0,5.0),
      Vectors.dense(9.0,5.0,2.0,2.0),
      Vectors.dense(3.0,4.0,6.0,4.0),
      Vectors.dense(2.0,7.0,3.0,1.0)
    )

    val df = data.map(Tuple1.apply).toDF("features")
    df.show(false)

    val Row(coeff1: Matrix) = Correlation.corr(df, "features","pearson").head
    println("Pearson correlation matrix:\n" + coeff1.toString)

    val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
    println("Spearman correlation matrix:\n" + coeff2.toString)


    /***
      ***************************************************************************************************************************************
    MLLIB的相关性分析，输入可为
      1.RDD[Vector]
      2.RDD[Double]
    输出为由一条Row[Matrix]组成的DF
    Matrix是由Vectors两两进行相关性分析的结果矩阵
    MLLIB使用的是mllib.stat.Statistics的corr方法
      ***************************************************************************************************************************************
      ***/
    val data1:Array[Double] = Array(106,86,100,101,99,103,97,113,112,110)
    val data2 :Array[Double]= Array(7,0,27,50,28,29,20,12,6,17)
    val data3 = Array(4.0,2.0,6.0,3.0)
    val data4 = Array(5.0,2.0,4.0,1.0)
    val distDataArr = Array(data1,data2,data3,data4).map(x=>spark.sparkContext.parallelize(x))
    val correlation01 = Statistics.corr(distDataArr(0), distDataArr(1), "spearman")
    //val correlation02 = Statistics.corr(distDataArr(0),distDataArr(2), "pearson")
    //val correlation03 = Statistics.corr(distDataArr(0),distDataArr(3), "pearson")
    println(s"correlation01:$correlation01")
   // println(s"correlation02:$correlation02")
   // println(s"correlation03:$correlation03")

  }

  def sum(spark:SparkSession): Unit ={
    val observations = spark.sparkContext.parallelize(
      Seq(
        org.apache.spark.mllib.linalg.Vectors.dense(1.0, 10.0, 100.0),
        org.apache.spark.mllib.linalg.Vectors.dense(2.0, 20.0, 200.0),
        org.apache.spark.mllib.linalg.Vectors.dense(3.0, 30.0, 300.0)
      )
    )

    // Compute column summary statistics.
    val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)
    println(summary.mean)  // a dense vector containing the mean value for each column
    println(summary.variance)  // column-wise variance
    println(summary.numNonzeros)  // number of nonzeros in each column

  }
}
