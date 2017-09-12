package chisqtest

/**
  * Created by Namhwik on 2017/7/26.
  *
  * getPhiValue计算相关性系数
  * 只适用于四格表
  * 结果介于-1~+1之间，独立性检验是根据两变量独立的假设计算出来的，k^2值大小反应两变量独立距离，k^2；
  */
object Phi {
  def value(array: Array[Long]): Double = {
    getPhiValue(array(0),array(1),array(2),array(3))
  }
  def getPhiValue(n11:Long,n10:Long,n01:Long,n00:Long): Double = {
    (n11*n00 - n10*n01) / math.sqrt((n10+n11)*(n00+n01)*(n10+n00)*(n01+n11))
  }
  def check(binaryArr :Array[Long]): Boolean = {
    val p1 = binaryArr(0).toDouble/(binaryArr(0)+binaryArr(2)).toDouble
    val p2 = binaryArr(1).toDouble/(binaryArr(1)+binaryArr(3)).toDouble
    binaryArr(0)*p1<5 || binaryArr(2)*p1<5 || binaryArr(1)*p2<5 || binaryArr(3)*p2<5
  }
  def checkedChisqr(array: Array[Long]): Double = {
    val binaryArr: Array[Double] = array.map(x=>x.toDouble)
    val a = binaryArr(0)
    val b = binaryArr(1)
    val c = binaryArr(2)
    val d = binaryArr(3)
    val n = a+b+c+d
    Math.pow(Math.abs(a*d-b*c)-n/2.0,2) * n  /((a+b)*(c+d)*(a+c)*(b+d))
  }
  def checkedPValue(array: Array[Long]): Double = {
    val k = checkedChisqr(array)
    println("K^2:=======>"+k)
    val pvalue = Pvalue.chisqr2pValue(1,k)
    println("pvalue:=======>"+pvalue)
    pvalue
  }
}
