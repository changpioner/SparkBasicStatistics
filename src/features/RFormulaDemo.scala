package features

/**
  * Created by Namhwik on 2017/9/5.
  *
  RFormula通过R模型公式来选择列。支持R操作中的部分操作，包括‘~’, ‘.’, ‘:’, ‘+’以及‘-‘，基本操作如下：
    1、 ~分隔目标和对象
    2、 +合并对象，“+0”意味着删除空格
    3、-删除一个对象，“-1”表示删除空格
    4、 :交互（数值相乘，类别二值化）
    5、 . 除了目标列的全部列
  假设a和b为两列：
    1、 y ~ a + b表示模型y ~ w0 + w1 * a +w2 * b其中w0为截距，w1和w2为相关系数
    2、 y ~a + b + a:b – 1表示模型y ~ w1* a + w2 * b + w3 * a * b，其中w1，w2，w3是相关系数
  RFormula产生一个向量特征列以及一个double或者字符串标签列。
  如果用R进行线性回归，则对String类型的输入列进行one-hot编码、对数值型的输入列进行double类型转化。
  如果类别列是字符串类型，它将通过StringIndexer转换为double类型。
  如果标签列不存在，则输出中将通过规定的响应变量创造一个标签列。
  假设我们有一个DataFrame,它的列名是id, country, hour和clicked。

id | country | hour | clicked
---|---------|------|---------
 7 | "US"    | 18   | 1.0
 8 | "CA"    | 12   | 0.0
 9 | "NZ"    | 15   | 0.0
  如果我们用clicked ~ country + hour(基于country和hour来预测clicked)来作用于RFormula,将会得到下面的结果。

id | country | hour | clicked | features         | label
---|---------|------|---------|------------------|-------
 7 | "US"    | 18   | 1.0     | [0.0, 0.0, 18.0] | 1.0
 8 | "CA"    | 12   | 0.0     | [0.0, 1.0, 12.0] | 0.0
 9 | "NZ"    | 15   | 0.0     | [1.0, 0.0, 15.0] | 0.0
  *
  *
  */
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.sql.{DataFrame, SparkSession}



 object RFormulaDemo {
   System.setProperty("hadoop.home.dir","C:\\ruanjian\\hadoop")
   def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val os = scala.sys.props.get("os.name").head
    val spark = if(os.startsWith("Windows"))
      SparkSession.builder().appName("test").master("local").getOrCreate()
    else
      SparkSession.builder().appName("test").getOrCreate()

     val dataSet = spark.createDataFrame(Seq(
       (1, "US", 18, 1.0),
       (2, "B", 12, 0.0),
       (3, "C", 18, 1.0),
       (4, "US", 12, 0.0),
       (5, "US", 18, 1.0),
       (6, "US", 12, 0.0),
       (7, "US", 18, 1.0),
       (8, "CA", 12, 0.0),
       (9, "US", 18, 1.0),
       (10, "CA", 12, 0.0),
       (11, "NZ", 15, 0.0)
     )).toDF("id", "country", "hour", "clicked")

     dataSet.show()
    val formula = new RFormula()
      .setFormula("clicked ~ country + hour")
      .setFeaturesCol("features")
      .setLabelCol("label")
    val  output: DataFrame = formula.fit(dataSet).transform(dataSet)
     output.printSchema()
    output.select("features", "label").show(false)

  }


}