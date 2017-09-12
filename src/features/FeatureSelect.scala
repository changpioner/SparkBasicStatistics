package features
import java.util.Arrays

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.StructType
/**
  * Created by Namhwik on 2017/9/5.
  */

/**
  *
  * VectorSlicer算法介绍：

         VectorSlicer是一个转换器输入特征向量，输出原始特征向量子集。VectorSlicer接收带有特定索引的向量列，通过对这些索引的值进行筛选得到新的向量集。可接受如下两种索引：

    1、整数索引---代表向量中特征的的索引，setIndices()

    2、字符串索引---代表向量中特征的名字，这要求向量列有AttributeGroup，因为这根据Attribute来匹配名字字段

        指定整数或者字符串类型都是可以的。另外，同时使用整数索引和字符串名字也是可以的。同时注意，至少选择一个特征，不能重复选择同一特征（整数索引和名字索引对应的特征不能叠）。注意如果使用名字特征，当遇到空值的时候将会报错。
        输出向量将会首先按照所选的数字索引排序（按输入顺序），其次按名字排序（按输入顺序）。

        示例：输入一个包含列名为userFeatures的DataFrame：

 userFeatures
------------------
 [0.0, 10.0, 0.5]

        userFeatures是一个向量列包含3个用户特征。假设userFeatures的第一列全为0，我们希望删除它并且只选择后两项。我们可以通过索引setIndices(1,2)来选择后两项并产生一个新的features列：

 userFeatures     | features
------------------|-----------------------------
 [0.0, 10.0, 0.5] | [10.0, 0.5]
  *
  *
  *
  *
  */
object FeatureSelect {
  System.setProperty("hadoop.home.dir","C:\\ruanjian\\hadoop")
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val os = scala.sys.props.get("os.name").head
    val spark = if(os.startsWith("Windows"))
      SparkSession.builder().appName("test").master("local").getOrCreate()
    else
      SparkSession.builder().appName("test").getOrCreate()

    example2(spark)

  }
  def example1(spark: SparkSession): Unit ={

    val data = Arrays.asList(
      Row(Vectors.sparse(3, Seq((0, -2.0), (1, 2.3)))),
      Row(Vectors.dense(-2.0, 2.3, 0.0))
    )

    val defaultAttr = NumericAttribute.defaultAttr
    val attrs = Array("f1", "f2", "f3").map(defaultAttr.withName)
    val attrGroup = new AttributeGroup("userFeatures", attrs.asInstanceOf[Array[Attribute]])

    val dataset = spark.createDataFrame(data, StructType(Array(attrGroup.toStructField())))

    val slicer = new VectorSlicer().setInputCol("userFeatures").setOutputCol("features")

    slicer.setIndices(Array(1)).setNames(Array("f3"))
    // or slicer.setIndices(Array(1, 2)), or slicer.setNames(Array("f2", "f3"))

    val output = slicer.transform(dataset)
    output.show(false)
  }
  def example2(spark:SparkSession): Unit ={
    val data = Array(Row(Vectors.dense(-2.0, 2.3, 0.0)))

    //为特征数组设置属性名（字段名），分别为f1 f2 f3
    val defaultAttr = NumericAttribute.defaultAttr
    val attrs = Array("f1", "f2", "f3").map(defaultAttr.withName)
    val attrGroup = new AttributeGroup("userFeatures", attrs.asInstanceOf[Array[Attribute]])

    //构造DataFrame
    val dataRDD = spark.sparkContext.parallelize(data)
    val dataset = spark.sqlContext.createDataFrame(dataRDD, StructType(Array(attrGroup.toStructField())))

    print("原始特征：")
    dataset.take(1).foreach(println)


    //构造切割器
    var slicer = new VectorSlicer().setInputCol("userFeatures").setOutputCol("features")

    //根据索引号，截取原始特征向量的第1列和第3列
    slicer.setIndices(Array(0,2))
    print("output1: ")
    println(slicer.transform(dataset).select("userFeatures", "features").first())

    //根据字段名，截取原始特征向量的f2和f3
    slicer = new VectorSlicer().setInputCol("userFeatures").setOutputCol("features")
    slicer.setNames(Array("f2","f3"))
    print("output2: ")
    println(slicer.transform(dataset).select("userFeatures", "features").first())

    //索引号和字段名也可以组合使用，截取原始特征向量的第1列和f2
    slicer = new VectorSlicer().setInputCol("userFeatures").setOutputCol("features")
    slicer.setIndices(Array(0)).setNames(Array("f2"))
    print("output3: ")
    println(slicer.transform(dataset).select("userFeatures", "features").first())
  }
}
