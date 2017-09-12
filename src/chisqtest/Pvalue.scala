package chisqtest

/**
  * Created by Namhwik on 2017/7/26.
  */
object Pvalue {
    def chisqr2pValue(dof: Int, chi_squared: Double): Double = {
      if (chi_squared < 0 || dof < 1)
        return 0.0
      val k = dof.toDouble * 0.5
      val v = chi_squared * 0.5
      if (dof == 2)
        return Math.exp(-1.0 * v)
      var incompleteGamma = log_igf(k, v)
      // 如果过小或者非数值或者无穷
      if (Math.exp(incompleteGamma) <= 1e-8 || java.lang.Double.isNaN(Math.exp(incompleteGamma)) || java.lang.Double.isInfinite(Math.exp(incompleteGamma)))
        return 1e-14
      val gamma = Math.log(getApproxGamma(k))
      incompleteGamma -= gamma
      if (Math.exp(incompleteGamma) > 1)
        return 1e-14
      val pValue = 1.0 - Math.exp(incompleteGamma)
      pValue.toDouble
    }

    private def getApproxGamma(n: Double) = {
      // RECIP_E = (E^-1) = (1.0 / E)
      val RECIP_E = 1.0 / math.E //0.36787944117144232159552377016147
      // TWOPI = 2.0 * PI
      val TWOPI = 2.0 * math.Pi //6.283185307179586476925286766559
      var d = 1.0 / (10.0 * n)
      d = 1.0 / ((12 * n) - d)
      d = (d + n) * RECIP_E
      d = Math.pow(d, n)
      d *= Math.sqrt(TWOPI / n)
      d
    }

    private def log_igf(s: Double, z: Double) :Double= {
      if (z < 0.0)
       return 0.0
      val sc = (Math.log(z) * s) - z - Math.log(s)
      val k = KM(s, z)
      Math.log(k) + sc
    }

    private def KM(s: Double, z: Double) = {
      var s1 =s
      var sum = 1.0
      val nom = 1.0
      val denom = 1.0
      var log_nom = Math.log(nom)
      var log_denom = Math.log(denom)
      var log_s = Math.log(s1)
      val log_z = Math.log(z)
      var i = 0
      while (i < 1000) {
        log_nom += log_z
        s1 += 1
        log_s = Math.log(s1)
        log_denom += log_s
        val log_sum = log_nom - log_denom
        sum += Math.exp(log_sum)
        i += 1
      }
      sum
    }


}
