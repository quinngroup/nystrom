//import org.apache.mahout.math._

//import org.apache.mahout.sparkbindings._
//import scalabindings._
//import RLikeOps._

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.storage.StorageLevel
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._

object Nystrom { 
	type KernelFunc = (Vector,Vector) => Double
	def sampleIndexedRows(m:IndexedRowMatrix, count:Int, returnComplement:Boolean = false, seed:Long = scala.util.Random.nextLong, assumeOrdered:Boolean = false):(IndexedRowMatrix,IndexedRowMatrix) = {
		val r = new scala.util.Random(seed)
		val sampleSet = if (assumeOrdered) {
				getSampleSet(0,m.rows.count.toInt,count,seed)

			}else{
				var indicies = m.rows.map(_.index.toInt).collect.toArray

				assert(indicies.size>= count, "Sample size is bigger than available rows")
				//indicies = indicies.sorted
				val ss= if (count > (indicies.size*0.75).toInt){
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
				scala.collection.immutable.Set[Int](ss.toSeq:_*)

			}
		val sampleMatrix = new IndexedRowMatrix(m.rows.filter(sampleSet contains _.index.toInt))
		if(!returnComplement){
			(sampleMatrix, null)
		}else{
			val complementMatrix = new IndexedRowMatrix(m.rows.filter(row => !(sampleSet contains row.index.toInt)))
			(sampleMatrix, complementMatrix)
		}

	}
	
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
	}

	def computeKernelMatrix(A: IndexedRowMatrix, B: IndexedRowMatrix, kernelFunc: KernelFunc): IndexedRowMatrix = 

		new IndexedRowMatrix(  A.rows.cartesian(B.rows).map({
				case (rA, rB) => (rA.index, (rB.index, kernelFunc(rA.vector,rB.vector) ) )
				})
			.groupByKey()
			.map({
				case (i,iter) => new IndexedRow(i, Vectors.dense(iter.toArray.sorted.map(_._2).toArray)) //iter.toSeq.sortWith(_._1<_._1).map(_._2).toArray))
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
			}).groupByKey.map({ case (i, iter) => new IndexedRow(i, Vectors.dense(iter.toArray.sorted.map(_._2).toArray))}))//iter.toSeq.sortWith(_._1<_._1).map(_._2).toArray))}).cache())
		
	
	
	def columnSum(originalMatrix :IndexedRowMatrix) :Vector = 
		Vectors.dense(originalMatrix.rows.flatMap(_.vector.toArray.zipWithIndex.map(_.swap)).reduceByKey(_+_).collect.sorted.map(_._2).toArray)
	def rowSum(originalMatrix :IndexedRowMatrix) :Vector =
		Vectors.dense(originalMatrix.rows.map(row => (row.index, row.vector.toArray.sum)).collect.sorted.map(_._2).toArray)
	def matrixDotColumn(originalMatrix :IndexedRowMatrix, colVec :Vector, op: Double =>Double = identity) :Vector = {
		//val col = 
		Vectors.dense(originalMatrix.rows.map(row => (row.index, row.vector.toArray.zip(colVec.toArray).map(m=> m._1*m._2).sum)).collect.sorted.map(r=>op(r._2)).toArray)
	}
	def diagDotMatrix(diag :Vector, originalMatrix :IndexedRowMatrix) :IndexedRowMatrix = {
		val d = originalMatrix.rows.context.broadcast(diag.toArray)
		new IndexedRowMatrix(originalMatrix.rows.map( row => new IndexedRow(row.index, Vectors.dense(row.vector.toArray.map(r => r * d.value(row.index.toInt)).toArray))))
	}
	def oneShotNystrom(originalMatrix :IndexedRowMatrix, s: Int, k:Int, kernelFunc: KernelFunc = dotProductKernel, assumeOrdered:Boolean=false, normalized:Boolean = false): (IndexedRowMatrix, Vector) = {
		//val chosen = Set(1,10,50,321,135,512,124,562,845,73,245,327,194,865,14,642,724,634,758)
		//val W =new IndexedRowMatrix( originalMatrix.rows.filter(chosen contains _.index.toInt ).cache())
		//val W =new IndexedRowMatrix( originalMatrix.rows.sample(false,s.toDouble/originalMatrix.numRows().toDouble).sortBy(r=>r.index))
		val (w , cp) = sampleIndexedRows(originalMatrix, s,true,assumeOrdered=assumeOrdered)
//		cp.rows.persist(StorageLevel.MEMORY_AND_DISK)
		val Kw = computeKernelMatrix(w,w,kernelFunc)
		Kw.rows.persist(StorageLevel.MEMORY_AND_DISK)
//		w.rows.unpersist()
		val Kw_svd= Kw.computeSVD(Kw.rows.count.toInt) //Do not use numRows() as it counts the rows by looking up the max index + 1 	
		val S_kw_pinv = vector_pinv(Kw_svd.s)
		val C = new IndexedRowMatrix( computeKernelMatrix(cp, w, kernelFunc).rows.union(Kw.rows))
//		C.rows.persist(StorageLevel.MEMORY_AND_DISK)
//		cp.rows.unpersist()
		var G = matrixDotDiag( C.multiply(Kw_svd.V) , S_kw_pinv, math.sqrt)
		if(normalized){
			//calculating the row sum for the approximate kernel matrix
			//without constructing K:
			//since a row sum of K, vector d,  can be represented as K dot ones(n,1)
			//and since K = GG.T, it follows that d = GG.T dot ones(n,1)
			//ie, d = G dot dg,  where dg is the row sum of G.T (column sum of G)
			val d = matrixDotColumn(G,columnSum(G), op= (x) => (1.0/math.sqrt(x))) // applying 1/sqrt(x) once and for all
			G = diagDotMatrix(d,G)
		}
		G.rows.persist(StorageLevel.MEMORY_AND_DISK)
		Kw.rows.unpersist()

//		C.rows.unpersist()
		val G_svd = G.computeSVD(k)
		val Vosnys = matrixDotDiag( G.multiply(G_svd.V) , vector_pinv(G_svd.s))
		Vosnys.rows.persist(StorageLevel.MEMORY_AND_DISK)
		G.rows.unpersist()
		(Vosnys,Vectors.dense(G_svd.s.toArray.map(x=>x*x)))
		
		
	}
	def doubleNystrom(originalMatrix :IndexedRowMatrix, s :Int, m :Int, l :Int, k :Int, kernelFunc: KernelFunc = dotProductKernel, assumeOrdered:Boolean=false, normalized:Boolean = false) : (IndexedRowMatrix, Vector) = {


		//val S =new IndexedRowMatrix( originalMatrix.rows.sample(false,s.toDouble/originalMatrix.numRows().toDouble).sortBy(r=>r.index))
		val (sM,cM) = sampleIndexedRows(originalMatrix,s,true,assumeOrdered=assumeOrdered)
//		cM.rows.persist(StorageLevel.MEMORY_AND_DISK)
		val V_s_l_T = transpose(oneShotNystrom(sM,m,l, kernelFunc)._1)
		V_s_l_T.rows.persist(StorageLevel.MEMORY_AND_DISK)
		//val V_s_l = VV._1
		//val V_s_l_T = transpose(V_s_l)
		val Ks = computeKernelMatrix(sM,sM,kernelFunc)
		
//		Ks.rows.persist(StorageLevel.MEMORY_AND_DISK)
		val Kw = matrixProduct( matrixProduct(V_s_l_T,Ks), V_s_l_T) ///  V_s_l.T*Ks*V_s_l
		Kw.rows.persist(StorageLevel.MEMORY_AND_DISK)
		val Kw_svd= Kw.computeSVD(Kw.rows.count.toInt) //Do not use numRows() as it counts the rows by looking up the max index + 1 	
		Kw.rows.unpersist()
		val S_kw_pinv = vector_pinv(Kw_svd.s)
		val C0 = new IndexedRowMatrix(computeKernelMatrix(cM, sM, kernelFunc).rows.union(Ks.rows))
//		C0.rows.persist(StorageLevel.MEMORY_AND_DISK)
		val C = matrixProduct(C0,V_s_l_T)    ///  C0*V_s_l
//		C.rows.persist(StorageLevel.MEMORY_AND_DISK)
//		C0.rows.unpersist()
//		Kw.rows.unpersist()

		
		//println((Kw.rows.count(),Kw.numCols()))
		//System.exit(1)
		var G = matrixDotDiag( C.multiply(Kw_svd.V) , S_kw_pinv, math.sqrt)
		if(normalized){
			//TODO: shows problems since G is not positive-definite after kPCA on Kw
/*			//wille have to think of a way to calculate the row sum, or normalize before kPCA
			G.rows.cache()
			val d = matrixDotColumn(G,columnSum(G), op= (x) => (1.0/math.sqrt(x)))
			val KG = matrixProduct(G,G)
			//KG.rows.cache()
			val d2 = rowSum(KG)
			val diff = d.toArray.zip(d2.toArray).map(r=>math.abs(r._1-r._2)).sum
			println(d2)
			println(d)
			println("Differences in sum")
			println(diff)
			for(i <- 0 until d.toArray.size)
				if(math.abs(d(i))<1) println(i,d(i))
			System.exit(0)
			G = diagDotMatrix(d,G)*/
		}
		G.rows.persist(StorageLevel.MEMORY_AND_DISK)
		V_s_l_T.rows.unpersist()

		val G_svd = G.computeSVD(k)
		val Vdnys = matrixDotDiag( G.multiply(G_svd.V) , vector_pinv(G_svd.s))
		Vdnys.rows.persist(StorageLevel.MEMORY_AND_DISK)
		G.rows.unpersist()
	
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
		val s = args(5).toInt
		val l = args(6).toInt
		println("Data in: " + datapath)
		println("Delimiter: "+ delimiter)
		println("Kernel:"+ kernel.toString)
		println("k: "+k.toString)
		println("m: "+m.toString)
		println("s: "+s.toString)
		println("l: "+l.toString)
		var sconf = new SparkConf()
						.setAppName("DoubleApp")
						//.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
						//.set("spark.shuffle.compress","true")
						//.set("spark.rdd.compress","true")
						//.set("spark.shuffle.spill.compress","true")
						
		if (args.size == 8)
			sconf = sconf.set("spark.executor.memory",args(7))
		val sc = new SparkContext(sconf)
		val data = new IndexedRowMatrix(sc.textFile(datapath).map(line => Vectors.dense(line.split(delimiter).map(_.toDouble))).zipWithIndex.map({case (v,i) => new IndexedRow(i,v)}) )
		//println(dotProductKernel(Vectors.dense(Array(1.0,2.0,3.0)),Vectors.dense(Array(5.0,6.0,7.0))))
		//println(matrixProduct(data,data).numRows())
		//val kernel = rbfKernel(g)_
		//val k = 10
		val beforeSVD = System.currentTimeMillis / 1000.0
		val a = oneShotNystrom(data,m,k,kernel, assumeOrdered=true, normalized=true)._2
		//val b = doubleNystrom(data,m,s,l, k,kernel,assumeOrdered=true, normalized=true)._2
		//val c = computeKernelMatrix(data,data,kernel).computeSVD(k).s
		val afterSVD = System.currentTimeMillis/1000.0 - beforeSVD
		println(afterSVD)
		//println(a)
		//println(b)
		//println(c)
		
	}
}
