//import org.apache.mahout.math._

//import org.apache.mahout.sparkbindings._
//import scalabindings._
//import RLikeOps._

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._

object Nystrom { 
	type KernelFunc = (Vector,Vector) => Double
	def sampleIndexedRows(m:IndexedRowMatrix, count:Int, returnComplement:Boolean = false, seed:Long = scala.util.Random.nextLong):(IndexedRowMatrix,IndexedRowMatrix) = {
		var indicies = m.rows.map(_.index.toInt).collect.toArray
		assert(indicies.size>= count, "Sample size is bigger than available rows")
		indicies = indicies.sorted
		val r = new scala.util.Random(seed)
		val ss = if (count > (indicies.size*0.75).toInt){
				val s =scala.collection.mutable.Set[Int](indicies:_*)
				while(s.size > count)
					s.remove(indicies(r.nextInt(indicies.size)))
				s
			} else {
				val s = scala.collection.mutable.Set[Int]()
				while(s.size < count)
					s.add(indicies(r.nextInt(indicies.size)))
				s
			}
		
		val sampleSet = scala.collection.immutable.Set[Int](ss.toSeq:_*)
		val sampleMatrix = new IndexedRowMatrix(m.rows.filter(sampleSet contains _.index.toInt).cache())
		if(!returnComplement){
			(sampleMatrix, null)
		}else{
			val complementMatrix = new IndexedRowMatrix(m.rows.filter(row => !(sampleSet contains row.index.toInt)))
			(sampleMatrix, complementMatrix)
		}

	}
	/*
	def getSampleSet(min:Int, max:Int, count:Int, seed:Long =scala.util.Random.nextLong ):Set[Int] = { 
		val s = scala.collection.mutable.Set[Int]()
		val r = new scala.util.Random(seed)
		assert(max-min >= count, "Sample count is bigger than the range")
		if (max - min == count) 
			scala.collection.immutable.Set[Int](min to max:_*)
		else {
			while(s.size < count)
				s.add(r.nextInt(max-min) + min)
			scala.collection.immutable.Set[Int](s.toSeq:_*)
		}
	}*/

	def computeKernelMatrix(A: IndexedRowMatrix, B: IndexedRowMatrix, kernelFunc: KernelFunc): IndexedRowMatrix = 

		new IndexedRowMatrix(  A.rows.cartesian(B.rows).map({
				case (rA, rB) => (rA.index, (rB.index, kernelFunc(rA.vector,rB.vector) ) )
				})
			.groupByKey()
			.map({
				case (i,iter) => new IndexedRow(i, Vectors.dense(iter.toSeq.sortWith(_._1<_._1).map(_._2).toArray))
			})
		)
	/*def dotProductKernel(a :Vector, b :Vector) :Double = {
		a.toArray.zip(b.toArray).map({case (v1,v2)=>v1*v2}).sum
	}*/
       	def dotProductKernel( a :Vector, b :Vector) :Double = BDV(a.toArray) dot BDV(b.toArray)
	def rbfKernel(g :Double) (a :Vector, b :Vector) :Double = math.exp(-Vectors.sqdist(a,b) * g)
	def matrixProduct(A: IndexedRowMatrix, B: IndexedRowMatrix) : IndexedRowMatrix = computeKernelMatrix(A,B, dotProductKernel)
	def vector_pinv( s: Vector,  tol: Double= -1.0) :Vector = {
			//Moore-penrose pseudo-inverse of the singular values vector
			val t = if (tol == -1.0) tol else java.lang.Double.MIN_VALUE*s.size*s.toArray.max
			Vectors.dense(s.toArray.map( x => if (x > t) 1/x else x))
		}

	def matrixDotDiag(m :IndexedRowMatrix, v :Vector, op: (Double) => Double = identity) : IndexedRowMatrix = 
		new IndexedRowMatrix(
			m.rows.map( row => new IndexedRow(row.index, Vectors.dense(row.vector.toArray.zip(v.toArray).map(x=> x._1*op(x._2)) )  ))
		)
	/*def sampleIndexedRows(m:IndexedRowMatrix, s:Int):IndexedRowMatrix ={
		val samples =	getSampleSet(0,m.rows.count.toInt,s)
		new IndexedRowMatrix(m.rows.filter(samples contains _.index.toInt).map(_.vector).zipWithIndex.map({case (v,i) => new IndexedRow(i,v)}))
	}*/
	def transpose(m :IndexedRowMatrix) :IndexedRowMatrix = new IndexedRowMatrix(
		m.rows.flatMap({
			row => row.vector.toArray.zipWithIndex.map({case (x,i) => (i,(row.index, x))})
			}).groupByKey.map({ case (i, iter) => new IndexedRow(i, Vectors.dense(iter.toSeq.sortWith(_._1<_._1).map(_._2).toArray))}).cache())
		
	
	
	def oneShotNystrom(originalMatrix :IndexedRowMatrix, s: Int, k:Int, kernelFunc: KernelFunc = dotProductKernel): (IndexedRowMatrix, Vector) = {
		//val chosen = Set(1,10,50,321,135,512,124,562,845,73,245,327,194,865,14,642,724,634,758)
		//val W =new IndexedRowMatrix( originalMatrix.rows.filter(chosen contains _.index.toInt ).cache())
		//val W =new IndexedRowMatrix( originalMatrix.rows.sample(false,s.toDouble/originalMatrix.numRows().toDouble).sortBy(r=>r.index))
		val (w , cp) = sampleIndexedRows(originalMatrix, s,true)
		val Kw = computeKernelMatrix(w,w,kernelFunc)
		val C = new IndexedRowMatrix( computeKernelMatrix(cp, w, kernelFunc).rows.union(Kw.rows))
		val Kw_svd= Kw.computeSVD(Kw.rows.count.toInt) //Do not use numRows() as it counts the rows by looking up the max index + 1 	
		val S_kw_pinv = vector_pinv(Kw_svd.s)
		val G = matrixDotDiag( C.multiply(Kw_svd.V) , S_kw_pinv, math.sqrt)
		val G_svd = G.computeSVD(k)
		val Vosnys = matrixDotDiag( G.multiply(G_svd.V) , vector_pinv(G_svd.s))
		
				
	
		(Vosnys,Vectors.dense(G_svd.s.toArray.map(x=>x*x)))
		
		
	}
	def doubleNystrom(originalMatrix :IndexedRowMatrix, s :Int, m :Int, l :Int, k :Int, kernelFunc: KernelFunc = dotProductKernel) : (IndexedRowMatrix, Vector) = {


		//val S =new IndexedRowMatrix( originalMatrix.rows.sample(false,s.toDouble/originalMatrix.numRows().toDouble).sortBy(r=>r.index))
		val (sM,cM) = sampleIndexedRows(originalMatrix,s,true)
		val V_s_l_T = transpose(oneShotNystrom(sM,m,l, kernelFunc)._1)
		//val V_s_l = VV._1
		//val V_s_l_T = transpose(V_s_l)
		val Ks = computeKernelMatrix(sM,sM,kernelFunc)
		val Kw = matrixProduct( matrixProduct(V_s_l_T,Ks), V_s_l_T) ///  V_s_l.T*Ks*V_s_l
		val Kw_svd= Kw.computeSVD(Kw.rows.count.toInt) //Do not use numRows() as it counts the rows by looking up the max index + 1 	
		val S_kw_pinv = vector_pinv(Kw_svd.s)
		val C0 = new IndexedRowMatrix(computeKernelMatrix(cM, sM, kernelFunc).rows.union(Ks.rows))
		val C = matrixProduct(C0,V_s_l_T)    ///  C0*V_s_l

		//println((Kw.rows.count(),Kw.numCols()))
		//System.exit(1)
		val G = matrixDotDiag( C.multiply(Kw_svd.V) , S_kw_pinv, math.sqrt)
		val G_svd = G.computeSVD(k)
		val Vdnys = matrixDotDiag( G.multiply(G_svd.V) , vector_pinv(G_svd.s))
		
				
	
		(Vdnys,Vectors.dense(G_svd.s.toArray.map(x=>x*x)))
	}

	def main(args: Array[String]){
		//println(getSampleSet(0, 10, 13))
		//System.exit(1)	
		
		val datapath = args(0)
		val delimiter = if (args(1) == "SPACE") " " else args(1)
		val kernel:KernelFunc = if (args(2).startsWith("rbf")) rbfKernel(args(2).split("_")(1).toDouble)_ else dotProductKernel
		val k = args(3).toInt
		val m = args(4).toInt
		println("Data in: " + datapath)
		println("Delimiter: "+ delimiter)
		println("Kernel:"+ kernel.toString)
		println("k: "+k.toString)
		println("m: "+m.toString)
		val sc = new SparkContext(new SparkConf().setAppName("DoubleApp"))
		val data = new IndexedRowMatrix(sc.textFile(datapath).map(line => Vectors.dense(line.split(delimiter).map(_.toDouble))).zipWithIndex.map({case (v,i) => new IndexedRow(i,v)}) )
		//println(dotProductKernel(Vectors.dense(Array(1.0,2.0,3.0)),Vectors.dense(Array(5.0,6.0,7.0))))
		//println(matrixProduct(data,data).numRows())
		//val kernel = rbfKernel(g)_
		//val k = 10
		val beforeSVD = System.currentTimeMillis / 1000.0
		val a = oneShotNystrom(data,m,k,kernel)._2
		//val b = doubleNystrom(data,300,100,60, k,kernel)._2
		//val c = computeKernelMatrix(data,data,kernel).computeSVD(k).s
		val afterSVD = System.currentTimeMillis/1000.0 - beforeSVD
		println(afterSVD)
		//println(a)
		//println(b)
		//println(c)
		
	}
}
