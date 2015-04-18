

lazy val root = (project in file(".")).
  settings(
    name := "nntimer",
    version := "1.0",
    scalaVersion := "2.11.2",
    organization := "org.usm.cs"
  )

ivyScala := ivyScala.value map { _.copy(overrideScalaVersion = true) }

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "2.2.0" % "test",
  "org.apache.spark" %% "spark-mllib" % "1.2.0",
  "org.apache.spark" %% "spark-core" % "1.2.0",
  "org.scalanlp" %% "breeze" % "0.11.2",
  "com.github.tototoshi" %% "scala-csv" % "1.2.1"
)

assemblyMergeStrategy in assembly := {
  case n if n.startsWith("reference.conf") => MergeStrategy.concat
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.deduplicate
}