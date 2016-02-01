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

	def computeKernelMatrix(A: IndexedRowMatrix, B: IndexedRowMatrix, kernelFunc: (Vector,Vector) => Double): IndexedRowMatrix = 

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
	
	def oneShotNystrom(originalMatrix :IndexedRowMatrix, s: Int, k:Int, kernelFunc: (Vector,Vector) => Double = dotProductKernel): (IndexedRowMatrix, Vector) = {
		//val chosen = Set(1,10,50,321,135,512,124,562,845,73,245,327,194,865,14,642,724,634,758)
		//val W =new IndexedRowMatrix( originalMatrix.rows.filter(chosen contains _.index.toInt ).cache())
		val W =new IndexedRowMatrix( originalMatrix.rows.sample(false,s.toDouble/originalMatrix.numRows().toDouble).sortBy(r=>r.index))
		
		val Kw = computeKernelMatrix(W,W,kernelFunc)
		val C = computeKernelMatrix(originalMatrix, W, kernelFunc)
		val Kw_svd= Kw.computeSVD(Kw.rows.count.toInt) //Do not use numRows() as it counts the rows by looking up the max index + 1 	
		val S_kw_pinv = vector_pinv(Kw_svd.s)
		val G = matrixDotDiag( C.multiply(Kw_svd.V) , S_kw_pinv, math.sqrt)
		val G_svd = G.computeSVD(k)
		val Vosnys = matrixDotDiag( G.multiply(G_svd.V) , vector_pinv(G_svd.s))
		
				
	
		(Vosnys,Vectors.dense(G_svd.s.toArray.map(x=>x*x)))
		
		
	}
	def transpose(m :IndexedRowMatrix) :IndexedRowMatrix = new IndexedRowMatrix(
		m.rows.flatMap({
			row => row.vector.toArray.zipWithIndex.map({case (x,i) => (i,(row.index, x))})
			}).groupByKey.map({ case (i, iter) => new IndexedRow(i, Vectors.dense(iter.toSeq.sortWith(_._1<_._1).map(_._2).toArray))}).cache())
		
	
	def doubleNystrom(originalMatrix :IndexedRowMatrix, s :Int, m :Int, l :Int, k :Int, kernelFunc: (Vector, Vector) => Double = dotProductKernel) : (IndexedRowMatrix, Vector) = {

		//TODO

		val S =new IndexedRowMatrix( originalMatrix.rows.sample(false,s.toDouble/originalMatrix.numRows().toDouble).sortBy(r=>r.index))
		val C0 = computeKernelMatrix(originalMatrix, S, kernelFunc)
		val VV = oneShotNystrom(S,m,l, kernelFunc)
		val V_s_l = VV._1
		val V_s_l_T = transpose(V_s_l)
		val C = matrixProduct(C0,V_s_l_T)   ///  C0*V_s_l
		val Ks = computeKernelMatrix(S,S,kernelFunc)
		val Kw = matrixProduct( matrixProduct(V_s_l,Ks), V_s_l_T) ///  V_s_l.T*Ks*V_s_l

		val Kw_svd= Kw.computeSVD(Kw.rows.count.toInt) //Do not use numRows() as it counts the rows by looking up the max index + 1 	
		val S_kw_pinv = vector_pinv(Kw_svd.s)
		val G = matrixDotDiag( C.multiply(Kw_svd.V) , S_kw_pinv, math.sqrt)
		val G_svd = G.computeSVD(k)
		val Vdnys = matrixDotDiag( G.multiply(G_svd.V) , vector_pinv(G_svd.s))
		
				
	
		(Vdnys,Vectors.dense(G_svd.s.toArray.map(x=>x*x)))
	}

	def main(args: Array[String]){
		val sc = new SparkContext(new SparkConf().setAppName("DoubleApp"))
		val data = new IndexedRowMatrix(sc.textFile("data/inputData.csv").map(line => Vectors.dense(line.split(",").map(_.toDouble))).zipWithIndex.map({case (v,i) => new IndexedRow(i,v)}) )
		//println(dotProductKernel(Vectors.dense(Array(1.0,2.0,3.0)),Vectors.dense(Array(5.0,6.0,7.0))))
		//println(matrixProduct(data,data).numRows())
		//val a = oneShotNystrom(data,300, 10,rbfKernel(0.01)_)._2
		val b = doubleNystrom(data,300,50,10, 10,rbfKernel(0.01)_)._2
		//println(a)
		println(b)
	}
}
