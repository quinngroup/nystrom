/** Implementation of Spectral Clustering
  *
  * Have to put something here.  It should
  * provide more details as to how the code works, or what spectral clustering does. 
	*/

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.broadcast.Broadcast
import breeze.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.clustering.{PowerIterationClustering, PowerIterationClusteringModel}


object SpectralClustering{
  
/*
* Calculates affinity matrix of the input file given
*/
  def calculateAffinity(K: RDD[(org.apache.spark.mllib.linalg.Vector, Long)] , g: Double) : RDD[(Long,(Long, Double))] = {
    
    //take cartesian product
    val cProduct = K.cartesian(K)
    
    //rbf kernel used to find affinity rbf(x,y) = exp(-gamma * (squared distance between x,y)
    val affinity =  cProduct.map({case ((vector1,index1),(vector2,index2)) => (index1,(index2,Math.exp(-g * Vectors.sqdist(vector1,vector2))))})
    affinity                
  }  
  
  /*
   * Calculates the normalized affinity as W = D^-1/2 A D^-1/2
   */
  def laplacian(finalAffinity: RDD[IndexedRow] , diagonalMatrix: Broadcast[org.apache.spark.mllib.linalg.Vector]) : IndexedRowMatrix = {
    
    val W = new IndexedRowMatrix (finalAffinity.map(row => {
      
                                        val i=row.index.toInt
                                        val w=row.vector.toArray
                                        val d = diagonalMatrix.value.toArray
                                        for(point <- 0 until w.length)
                                                w(point) *= d(i) * d(point)
                                        new IndexedRow(i,Vectors.dense(w))
                                           }))  
    W    
  }
  
  /*
   * KMeans implementation to caculate the clusters.
   */
  def doKMeans(u:RDD[IndexedRow], numClusters: Integer, numIterations: Integer): RDD[(Long,Int)] = {
    val urdd = u.map(_.vector)
    val clusters = KMeans.train(urdd, numClusters, numIterations)
    val WSSSE = clusters.computeCost(urdd)
    println("Within Set Sum of Squared Errors = " + WSSSE)
    val predictions = u.map( row => (row.index,clusters.predict(row.vector)))
    predictions
    //predictions.foreach(println)
  }
  
  /*
   * PIC implementation to calculate the clusters
   */
  /*def pic(u:RDD[IndexedRow]) ={      
       
    // Cluster the data into two classes using PowerIterationClustering
    val pic = new PowerIterationClustering().setK(3).setMaxIterations(10)
    val model = pic.run(A)    
    model.assignments.foreach { a => println(s"${a.id} -> ${a.cluster}")   
    }  
    }
  */
  /*
   * Visualization of the clusters
   */
  
  /*
   * Rand Index
   */
  def randIndex(indexWithClass: RDD[(Long,Int)], originalData: RDD[(Long,Int)]): Double ={
    
    val points = indexWithClass.join(originalData)
    val allPoints = points.cartesian(points).filter({case((id1,value1),(id2,value2)) => id1<id2})
   
    val aggregate = allPoints.map({ case ((id1,(c1,oc1)),(id2,(c2,oc2))) => 
                                    if(c1==c2 && oc1 == oc2){
                                        (1,0,0,0) // True Positive
                                    }else if(c1!=c2 && oc1!=oc2){
                                        (0,1,0,0)  // True Negative
                                    }else if(c1 ==c2 && oc1!=oc2){
                                        (0,0,1,0)  //False Positive
                                    }else{
                                        (0,0,0,1)  //False Negative
                                    }
                                  }).reduce({case ((tp1,tn1,fp1,fn1),(tp2,tn2,fp2,fn2)) => (tp1+tp2,tn1+tn2,fp1+fp2,fn1+fn2)})
                                  
    val randIndexScore = (aggregate._1 + aggregate._2).toDouble / (aggregate._1+aggregate._2+aggregate._3+aggregate._4).toDouble                             
    println(aggregate)
    randIndexScore
  }
  
  /*
   * Silhoutte validation of clusters
   * References
   * 1. Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
        Interpretation and Validation of Cluster Analysis". Computational
        and Applied Mathematics 20: 53-65. doi:10.1016/0377-0427(87)90125-7.
   * 2. http://en.wikipedia.org/wiki/Silhouette_(clustering)
   */
  def silhoutte(indexPointClass: RDD[(Long,(org.apache.spark.mllib.linalg.Vector,Int))]): Double={         //(A: RDD[(org.apache.spark.mllib.linalg.Vector, Long)],predictions: RDD[(Int, Iterable[Long])]) ={
   
    
    
    //cartesian all the points
    val allPoints = indexPointClass.cartesian(indexPointClass)
    
    //calculate euclidian distance and output: (index1,index2,class1,class2,euclidean distance)
    val distance = allPoints.map({ case((id1,(point1,class1)),(id2,(point2,class2))) => ((id1,class1,class2),(Vectors.sqdist(point1,point2),1))})
    //distance.foreach(println)
    
    //calculate average euclidean for all the points
    val alldistances = distance.reduceByKey({case ((d1,count1), (d2,count2)) =>  ((d1*count1+d2*count2)/(count1+count2), count1+count2)})
    
    //for each point (index,class) we get its distance in the same class and in all other classes. 
    val distances = alldistances.map({case((id1,class1,class2),(distance,count)) => ((id1,class1),(class2,distance))}).groupByKey() 
    
    val silhoutte = distances.map({case((id,c),iter) => { var a =0.0
                                          var b=Double.PositiveInfinity
                                          for( (ci,di) <- iter){
                                            if(c == ci)
                                              a = di
                                            else
                                              if(di < b)
                                                b = di
                                                
                                          }
                                          val s = (b-a)/max(a,b)
                                          ((id,c),s)                            
                             
                                          }
    } )
    //silhoutte.foreach(println)
    
    //overall silhoutte score. Average of all silhoutte scores
    val silhoutteScore = silhoutte.map({case((id,c),score) => (score,1)}).reduce({case ((score1,count1),(score2,count2)) =>((score1*count1+score2*count2)/(count1+count2),(count1+count2))})._1
    
    //println(silhoutteScore)
    silhoutteScore
 }
  
  /*
   * Main control of the program
   */
  
  def main(args:Array[String]){
    val g = args(0).toDouble //gamma
    val numClusters = args(1).toInt //number of clusters to input to kmeans
    val numIterations = args(2).toInt //number of iterations to input to kmeans
    val inputpath = args(3)
    //val outputpath = args(4)
    
    val groundtruthFile = args(4)
    val sampleSize = args(6).toInt
    var sample2Size = 0
    var kpca = 0
    if (args.size == 9){
    sample2Size = args(7).toInt
    kpca = args(8).toInt}

    var conf = new SparkConf().setAppName("Spectral Clustering")
//    if (args.size == 7)
	                            conf = conf.set("spark.executor.memory",args(5))
    val sc = new SparkContext(conf)
      
    //Arguments     
    //Read the data    
    var input = sc.textFile(inputpath)
    
    //each point (x,y) gets assigned an indexval filename = args(1) => ([x,y],index)
    val A = input.map(v =>  Vectors.dense(v.split(",").map(_.toDouble))).zipWithIndex
    
    val beforeSVD = System.currentTimeMillis / 1000.0
    val uu = if (sampleSize > numClusters) {
	    val dataMatrix =new IndexedRowMatrix(A.map(r => new IndexedRow(r._2,r._1)))
    	if (sample2Size == 0) {
	    val (uuu,_) = Nystrom.oneShotNystrom(dataMatrix, sampleSize,numClusters,Nystrom.rbfKernel(g)_, assumeOrdered=true,normalized=true)
		uuu
	}else{
		val (uuu,_) = Nystrom.doubleNystrom(dataMatrix, sampleSize, sample2Size, kpca, numClusters, Nystrom.rbfKernel(g), assumeOrdered=true)
		uuu
	}
    }else{
    
	    //calculate affinity of all points. For points ([1,(x1,y1)],[2,(x2,y2)]) => (1,(2,affinityValue))
	    val affinity = SpectralClustering.calculateAffinity(A,g)
	    
	    //Affinity of one point with all other points. (x, ((1,affinityValue)..(x,affinityValue)..(n,affinityValue)))
	    val allAffinity = affinity.groupByKey()
	    
	    //remove the second index, and make an IndexedRow => [index1 ,(affinity1 , ...., affinityn)]
	    val finalAffinity =allAffinity.map({case (index,affinities) => new IndexedRow(index,Vectors.dense(affinities.toSeq.map(_._2).toArray))})
	    
	    //build a diagonal matrix
	    val diagonalMatrix = sc.broadcast(Vectors.dense(finalAffinity.map( v => 1/Math.sqrt(v.vector.toArray.sum)).collect()))
	    
	    //Calculate the normalized Affinity.    
	    val W = SpectralClustering.laplacian(finalAffinity,diagonalMatrix)
	    
	    //compute the SVD
	    val svd = W.computeSVD(numClusters,computeU = true)
	    svd.U
    }
    val afterSVD = System.currentTimeMillis/1000.0 - beforeSVD
    val u = uu.rows
    
    
    //KMeans step
    val predictions = doKMeans(u, numClusters,numIterations)
    
    //plot final clusters
    
    //silhoutte validation or randIndex evaluation of the clusters by
    //(point,index) -> (index,point)
    val indexWithPoint = A.map(_.swap)   
    //(class , all indexes) to (index,class)    
    //val indexWithClass = predictions.flatMap({case (cluster,indexes) => indexes.map( index => (index,cluster))})
    
    //join above two on index to get: (index,point,cluster)
    val indexPointClass = indexWithPoint.join(predictions)
    
    val silhouetteScore = silhoutte(indexPointClass)
    
    val groundtruth = sc.textFile(groundtruthFile).map(_.toInt).zipWithIndex.map(_.swap)
    val randIndexScore = randIndex(predictions,groundtruth)
    println(randIndexScore)
    println(silhouetteScore)
    println(afterSVD)
	}
}
