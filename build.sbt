name := "deep-nlp-scala"

version := "0.1-SNAPSHOT"

scalaVersion := "2.10.4"

libraryDependencies ++= List(
  "log4j"               % "log4j"               % "1.2.15" exclude("javax.jms", "jms"),
  "org.deeplearning4j"  % "deeplearning4j-core" % "0.0.3.3.2.alpha1",
  "org.deeplearning4j"  % "deeplearning4j-nlp"  % "0.0.3.3.2.alpha1",
  "org.nd4j"            % "canova-parent"       % "0.0.0.1",
  "org.nd4j"            % "nd4j-api"            % "0.0.3.5.5.2",
  "org.nd4j"            % "nd4j-netlib-blas"    % "0.0.3.5.5.2",
//  "org.nd4j"            % "nd4j-jcublas-common" % "0.0.3.5.5.2",
//  "org.nd4j"            % "nd4j-jcublas-6.5"    % "0.0.3.5.5.2",
  "edu.stanford.nlp"    % "stanford-corenlp"    % "3.4.1", // 3.4.1 last version with java 7 support
  "edu.stanford.nlp"    % "stanford-corenlp"    % "3.4.1" classifier "models"
)

resolvers ++= Seq(
  "JBoss repository" at "https://repository.jboss.org/nexus/content/groups/public",
  //"CVUT repository" at "http://repository.fit.cvut.cz/maven/release",
  Resolver.mavenLocal,
  Resolver.file("project-ivy-repo", file("project-ivy-repo"))
)
