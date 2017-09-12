package features

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.DoubleParam
import org.apache.spark.sql.SparkSession

/**
  * Created by Namhwik on 2017/9/5.
  *
  *
  ChiSqSelector代表卡方特征选择。它适用于带有类别特征的标签数据。
  ChiSqSelector根据独立卡方检验，然后选取类别标签主要依赖的特征。它类似于选取最有预测能力的特征。
  它支持5种特征选取方法：
    1、numTopFeatures：通过卡方检验选取最具有预测能力的Top(num)个特征；
    2、percentile：类似于上一种方法，但是选取一小部分特征而不是固定(num)个特征；
    3、fpr:选择了所有p值低于阈值的特征，这样就可以控制false positive rate来进行特征选择；
    4、fdr: 使用Benjamini-Hochberg 来选择所有错误发现率低于阈值的特征。
    5、fwe: 选择所有p值低于阈值的特性。阈值按1 / numFeatures缩放，从而控制了选择的家庭错误率。
  默认情况下特征选择方法是numTopFeatures(50)，可以根据setSelectorType()选择特征选取方法。
  *
  */
object ChiSqSelectorDemo {
  System.setProperty("hadoop.home.dir","C:\\ruanjian\\hadoop")

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val os = scala.sys.props.get("os.name").head
    val spark = if(os.startsWith("Windows"))
      SparkSession.builder().appName("test").master("local").getOrCreate()
    else
      SparkSession.builder().appName("test").getOrCreate()
    import spark.implicits._
    val data = Seq(
      (0, Vectors.dense(0.5, 10.0),0.0),
      (1, Vectors.dense(0.6, 20.0),0.0),
      (2, Vectors.dense(1.5, 30.0),1.0),
      (3, Vectors.dense(0.4, 30.0),0.0),
      (4, Vectors.dense(0.45, 40.0),0.0),
      (5, Vectors.dense(1.6, 40.0),1.0)

    )

    val df = spark.createDataset(data).toDF("id", "features", "clicked")

    val selector = new ChiSqSelector()
      .setSelectorType("percentile")
      //.setNumTopFeatures(1)
      .setFeaturesCol("features")
      .setLabelCol("clicked")
      .setOutputCol("selectedFeatures")
    val selectorModel = selector.fit(df)
    //selectorModel.percentile= DoubleParam(2.1)
    val param = new DoubleParam("","","")
    val result = selectorModel.transform(df)

    println(s"ChiSqSelector output with top ${selector.getNumTopFeatures} features selected")
    result.show()
  }
}
