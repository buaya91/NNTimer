import exp._

object NNTimer {

  def test(url: String) = {
    val dataSize = 500
    val networkSize = Seq(784, 60, 10)
    val cpuNo = Seq(4)
    val runSpecs = cpuNo.map { RunSpec(dataSize, networkSize, _, true) }
    runSpecs.map(rs => {
      val r = ExpRunner.run(rs, url)
      IOHelper.writeRowToCSV(r, "run_results.csv")
    })
  }

  def exp1(epochs: Int, url: String) = {
    val dataSize = 60000                  //fixed
    val networkSize = Seq(784, 60, 10)    //fixed
    val cpuNo = Range(4, 85, 4)           //varies
    val runSpecs = cpuNo.map { RunSpec(dataSize, networkSize, _, true) }
    runSpecs.map(rs => {
      val r = ExpRunner.run(rs, url)
      IOHelper.writeRowToCSV(r, "run_results.csv")
    })
  }

  def exp2(epochs: Int, url: String) = {
    val dataSize = Range(10000, 60001, 5000)                  //fixed
    val networkSize = Seq(784, 60, 10)    //fixed
    val cpuNo = 84           //varies
    val runSpecs = dataSize.map { RunSpec(_, networkSize, cpuNo, true)}
    runSpecs.map(rs => {
      val r = ExpRunner.run(rs, url)
      IOHelper.writeRowToCSV(r, "run_results.csv")
    })
  }

  def exp3(epochs: Int, url: String) = {
    val dataSize = 60000                  //fixed
    val networkSize = Range(10, 61, 10).map(Seq(784, _, 10))
    val cpuNo = 84           //varies
    val runSpecs = networkSize.map { RunSpec(dataSize, _, cpuNo, true)}
    runSpecs.map(rs => {
      val r = ExpRunner.run(rs, url)
      IOHelper.writeRowToCSV(r, "run_results.csv")
    })
  }

  def main(args: Array[String]) {
    val (epoch, url) =
      if (args.isEmpty) {
        (200, "local[*]")
      }
      else {
        (args(0).toInt, args(1))
      }

    exp1(epoch, url)
    exp2(epoch, url)
    exp3(epoch, url)
  }
}
