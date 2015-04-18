package exp

import java.io.File
import com.github.tototoshi.csv._



object IOHelper {
  def writeRowsToCSV(results: Seq[RunResult], path: String) = {
    val f = new File(path)
    val writer = CSVWriter.open(f, append = true)
    results.map {r => writer.writeRow(r.paramsList)}
  }

  def writeRowToCSV(result: RunResult, path: String) = {
    val f = new File(path)
    val writer = CSVWriter.open(f, append = true)
    writer.writeRow(result.paramsList)
  }
}
