using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Jobs;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SimdMatrixProduct
{
    //[Config(typeof(MyConfig))]
    [SimpleJob]
    public class Program
    {
        float[] globalA = null!;
        float[] globalB = null!;
        int N = 2049;
        [GlobalSetup]
        public void Setup()
        {
            Random rand = new Random();
            globalA = new float[N*N];
            globalB = new float[N*N];
            for(int i = 0;i < N*N; i++)
            {
                globalA[i] = rand.Next();
                globalB[i] = rand.Next();
            }
        }
        [Benchmark]
        public float[] MatrixProductBench() => MatrixProduct(globalA, globalB, N);
        [Benchmark]
        public float[] MatrixProductRecerseBench() => MatrixProductReverse(globalA, globalB, N);
        [Benchmark]
        public float[] MatrixProductSIMDBench() => MatrixProductSIMD(globalA, globalB, N);
        [Benchmark]
        public float[] MatrixProductVectorAPIBench() => MatrixProductVectorAPI(globalA, globalB, N);
        [Benchmark]
        public float[] MatrixProductSIMDGatherScatterBench() => MatrixProductSIMDGatherScatter(globalA, globalB, N);
        [Benchmark]
        public float[] MatrixProductSIMDGatherScatterUnrollBench() => MatrixProductSIMDGatherScatterUnroll(globalA, globalB, N);
        [Benchmark]
        public float[] MatrixProductSIMDGatherScatterUnrollFMABench() => MatrixProductSIMDGatherScatterUnrollFMA(globalA, globalB, N);

        [SkipLocalsInit]
        static unsafe float[] MatrixProduct(float[] A, float[] B, int N)
        {
            float[] ret = new float[A.Length];
            fixed (float* a = A) fixed (float* b = B) fixed (float* c = ret)
            {
                for (int i = 0; i < N; ++i)
                    for(int j = 0; j < N; ++j)
                    {
                        int ii = i * N;
                        float sum = 0;
                        for (int k = 0, kk = 0; k < N; ++k, kk += N)
                            sum += a[ii + k] * b[j + kk];
                        c[ii + j] = sum;
                    }
            }
            return ret;
        }
        [SkipLocalsInit]
        static unsafe float[] MatrixProductReverse(float[] A, float[] B, int N)
        {
            float[] ret = new float[A.Length];
            float[] B2 = new float[B.Length];
            fixed (float* b = B) fixed(float *b2 = B2)
            {
                for (int i = 0, ii = 0; i < N; ++i, ii += N)
                    for (int j = 0, jj = 0; j < N; ++j, jj += N)
                        b2[ii + j] = b[jj + i];
            }
            fixed (float* a = A) fixed (float* b = B2) fixed (float* c = ret)
            {
                for (int i = 0, ii = 0; i < N; ++i, ii += N)
                    for (int j = 0, jj = 0; j < N; ++j, jj += N)
                    {
                        float sum = 0;
                        for (int k = 0; k < N; ++k)
                            sum += a[ii + k] * b[jj + k];
                        c[ii + j] = sum;
                    }
            }
            return ret;
        }

        [SkipLocalsInit]
        static unsafe float[] MatrixProductVectorAPI(float[] A, float[] B, int N)
        {
            int n = (N / Vector<float>.Count) * Vector<float>.Count;
            float[] ret = new float[A.Length];
            float[] B2 = new float[B.Length];
            fixed (float* b = B) fixed (float* b2 = B2)
            {
                for (int i = 0, ii = 0; i < N; ++i, ii += N)
                    for (int j = 0, jj = 0; j < N; ++j, jj += N)
                        b2[ii + j] = b[jj + i];
            }
            fixed (float* a = A) fixed (float* b = B2) fixed (float* c = ret)
            {
                for (int i = 0, ii = 0; i < N; ++i, ii += N)
                    for (int j = 0, jj = 0; j < N; ++j, jj += N)
                    {
                        float sum = 0;
                        int k = 0;
                        {
                            Vector<float> vsum = new Vector<float>(0);
                            for (; k < n; k += Vector<float>.Count)
                            {

                                Vector<float> va = new Vector<float>(new ReadOnlySpan<float>(&a[ii + k], Vector<float>.Count));
                                Vector<float> vb = new Vector<float>(new ReadOnlySpan<float>(&b[jj + k], Vector<float>.Count));
                                vsum = vsum + (va * vb);
                            }
                            for (int l = 0; l < Vector<float>.Count; ++l) sum += vsum[l];
                        }
                        for (; k < N; ++k)
                            sum += a[ii + k] * b[jj + k];
                        c[ii + j] = sum;
                    }
            }
            return ret;
        }

        [SkipLocalsInit]
        static unsafe float[] MatrixProductSIMD(float[] A, float[] B, int N)
        {
            if (!Avx2.IsSupported) Console.WriteLine("AVX2 is not supported.");
            float * sumspace = stackalloc float[Vector256<float>.Count]; // 0-ed

            int n = (N / Vector256<float>.Count) * Vector256<float>.Count;
            float[] ret = new float[A.Length];
            float[] B2 = new float[B.Length];
            fixed (float* b = B) fixed (float* b2 = B2)
            {
                for (int i = 0, ii = 0; i < N; ++i, ii += N)
                    for (int j = 0, jj = 0; j < N; ++j, jj += N)
                        b2[ii + j] = b[jj + i];
            }
            fixed (float* a = A) fixed (float* b = B2) fixed (float* c = ret)
            {
                for (int i = 0, ii = 0; i < N; ++i, ii += N)
                    for (int j = 0, jj = 0; j < N; ++j, jj += N)
                    {
                        float sum = 0;
                        int k = 0;
                        {
                            Vector256<float> vsum = Vector256<float>.Zero;
                            for (; k < n; k += Vector256<float>.Count)
                            {
                                Vector256<float> va = Avx.LoadVector256(&a[ii + k]);
                                Vector256<float> vb = Avx.LoadVector256(&b[jj + k]);
                                vsum = Avx.Add(vsum, Avx.Multiply(va, vb));
                            }
                            Avx.Store(sumspace, vsum);
                        }
                        for (int l = 0; l < Vector256<float>.Count; ++l) sum += sumspace[l];


                        for (; k < N; ++k)
                            sum += a[ii + k] * b[jj + k];
                        c[ii + j] = sum;
                    }
            }
            return ret;
        }

        [SkipLocalsInit]
        static unsafe float[] MatrixProductSIMDGatherScatter(float[] A, float[] B, int N)
        {
            if (!Avx2.IsSupported) Console.WriteLine("AVX2 is not supported.");
            float* sumspace = stackalloc float[Vector256<float>.Count];
            int* ptrDelta = stackalloc int[Vector256<int>.Count];
            for (int i = 0; i < Vector256<float>.Count; ++i) ptrDelta[i] = (i * N * Vector256<float>.Count) >> 3;
            int n = (N / Vector256<float>.Count) * Vector256<float>.Count;
            float[] ret = new float[A.Length];
            float[] B2 = new float[B.Length];
            fixed (float* b = B) fixed (float* b2 = B2)
            {
                int nn = N * Vector256<float>.Count;
                for (int i = 0, ii = 0; i < N; ++i, ii += N)
                {
                    IntPtr p = (IntPtr)(&b[i]);
                    int j = 0, jj = 0;
                    for (; j < n; j += Vector256<float>.Count, jj += nn)
                    {
                        //p = (IntPtr)(&b[jj + i]);
                        float* ptr = (float*)(p & unchecked((IntPtr)0xfffffffe00000000));
                        int lowptr = (int)(((long)p >> 2) & 0x7fffffff);
                        Vector256<int> vb_ptr_lower = Avx2.Add(Avx2.LoadVector256(ptrDelta), Avx2.BroadcastScalarToVector256(&lowptr));
                        Vector256<float> vb = Avx2.GatherVector256(ptr, vb_ptr_lower, 4);

                        Avx.Store(&b2[ii + j], vb);
                        p = p + ((IntPtr)(sizeof(float)) * nn);
                    }
                    for (; j < N; ++j, jj += N)
                        b2[ii + j] = b[jj + i];
                }
            }
            //for (int i = 0; i < N; ++i)
            //{
            //    for (int j = 0; j < N; ++j)
            //        Console.Write("{0} ", B[i * N + j]);
            //    Console.WriteLine();
            //}
            //Console.WriteLine();
            //for (int i = 0; i < N; ++i)
            //{
            //    for (int j = 0; j < N; ++j)
            //        Console.Write("{0} ", B2[i * N + j]);
            //    Console.WriteLine();
            //}
            fixed (float* a = A) fixed (float* b = B2) fixed (float* c = ret)
            {
                for (int i = 0, ii = 0; i < N; ++i, ii += N)
                    for (int j = 0, jj = 0; j < N; ++j, jj += N)
                    {
                        float sum = 0;
                        int k = 0;
                        {
                            Vector256<float> vsum = Vector256<float>.Zero;
                            for (; k < n; k += Vector256<float>.Count)
                            {
                                Vector256<float> va = Avx.LoadVector256(&a[ii + k]);
                                Vector256<float> vb = Avx.LoadVector256(&b[jj + k]);
                                vsum = Avx.Add(vsum, Avx.Multiply(va, vb));
                            }
                            Avx.Store(sumspace, vsum);
                        }
                        for (int l = 0; l < Vector256<float>.Count; ++l) sum += sumspace[l];
                        for (; k < N; ++k)
                            sum += a[ii + k] * b[jj + k];
                        c[ii + j] = sum;
                    }
            }
            return ret;
        }

        [SkipLocalsInit]
        static unsafe float[] MatrixProductSIMDGatherScatterUnroll(float[] A, float[] B, int N)
        {
            if (!Avx2.IsSupported) Console.WriteLine("AVX2 is not supported.");
            float* sumspace = stackalloc float[Vector256<float>.Count];
            int* ptrDelta = stackalloc int[Vector256<int>.Count];
            for (int i = 0; i < Vector256<float>.Count; ++i) ptrDelta[i] = (i * N * Vector256<float>.Count) >> 3;
            int n = (N / Vector256<float>.Count) * Vector256<float>.Count;
            float[] ret = new float[A.Length];
            float[] B2 = new float[B.Length];
            fixed (float* b = B) fixed (float* b2 = B2)
            {
                int nn = N * Vector256<float>.Count;
                for (int i = 0, ii = 0; i < N; ++i, ii += N)
                {
                    IntPtr p = (IntPtr)(&b[i]);
                    int j = 0, jj = 0;
                    for (; j < n; j += Vector256<float>.Count, jj += nn)
                    {
                        //p = (IntPtr)(&b[jj + i]);
                        float* ptr = (float*)(p & unchecked((IntPtr)0xfffffffe00000000));
                        int lowptr = (int)(((long)p >> 2) & 0x7fffffff);
                        Vector256<int> vb_ptr_lower = Avx2.Add(Avx2.LoadVector256(ptrDelta), Avx2.BroadcastScalarToVector256(&lowptr));
                        Vector256<float> vb = Avx2.GatherVector256(ptr, vb_ptr_lower, 4);

                        Avx.Store(&b2[ii + j], vb);
                        p = p + ((IntPtr)(sizeof(float)) * nn);
                    }
                    for (; j < N; ++j, jj += N)
                        b2[ii + j] = b[jj + i];
                }
            }
            fixed (float* a = A) fixed (float* b = B2) fixed (float* c = ret)
            {
                n = (N / (Vector256<float>.Count * 8)) * (Vector256<float>.Count * 8);
                int deltaK = Vector256<float>.Count * 8;
                for (int i = 0, ii = 0; i < N; ++i, ii += N)
                    for (int j = 0, jj = 0; j < N; ++j, jj += N)
                    {
                        float sum = 0;
                        int k = 0;
                        {
                            Vector256<float> vsum1 = Vector256<float>.Zero;
                            Vector256<float> vsum2 = Vector256<float>.Zero;
                            Vector256<float> vsum3 = Vector256<float>.Zero;
                            Vector256<float> vsum4 = Vector256<float>.Zero;
                            Vector256<float> vsum5 = Vector256<float>.Zero;
                            Vector256<float> vsum6 = Vector256<float>.Zero;
                            Vector256<float> vsum7 = Vector256<float>.Zero;
                            Vector256<float> vsum8 = Vector256<float>.Zero;
                            for (; k < n; k += deltaK)
                            {
                                vsum1 = Avx.Add(vsum1, Avx.Multiply(Avx.LoadVector256(&a[ii + k]), Avx.LoadVector256(&b[jj + k])));
                                vsum2 = Avx.Add(vsum2, Avx.Multiply(Avx.LoadVector256(&a[ii + k + Vector256<float>.Count]), Avx.LoadVector256(&b[jj + k + Vector256<float>.Count])));
                                vsum3 = Avx.Add(vsum3, Avx.Multiply(Avx.LoadVector256(&a[ii + k + (Vector256<float>.Count * 2)]), Avx.LoadVector256(&b[jj + k + (Vector256<float>.Count * 2)])));
                                vsum4 = Avx.Add(vsum4, Avx.Multiply(Avx.LoadVector256(&a[ii + k + (Vector256<float>.Count * 3)]), Avx.LoadVector256(&b[jj + k + (Vector256<float>.Count * 3)])));
                                vsum5 = Avx.Add(vsum1, Avx.Multiply(Avx.LoadVector256(&a[ii + k + (Vector256<float>.Count * 4)]), Avx.LoadVector256(&b[jj + k + (Vector256<float>.Count * 4)])));
                                vsum6 = Avx.Add(vsum2, Avx.Multiply(Avx.LoadVector256(&a[ii + k + (Vector256<float>.Count * 5)]), Avx.LoadVector256(&b[jj + k + (Vector256<float>.Count * 5)])));
                                vsum7 = Avx.Add(vsum3, Avx.Multiply(Avx.LoadVector256(&a[ii + k + (Vector256<float>.Count * 6)]), Avx.LoadVector256(&b[jj + k + (Vector256<float>.Count * 6)])));
                                vsum8 = Avx.Add(vsum4, Avx.Multiply(Avx.LoadVector256(&a[ii + k + (Vector256<float>.Count * 7)]), Avx.LoadVector256(&b[jj + k + (Vector256<float>.Count * 7)])));
                            }
                            Avx.Store(sumspace, Avx.Add(Avx.Add(Avx.Add(vsum1, vsum2), Avx.Add(vsum3, vsum4)), Avx.Add(Avx.Add(vsum5, vsum6), Avx.Add(vsum7, vsum8))));
                        }
                        for (int l = 0; l < Vector256<float>.Count; ++l) sum += sumspace[l];
                        for (; k < N; ++k)
                            sum += a[ii + k] * b[jj + k];
                        c[ii + j] = sum;
                    }
            }
            return ret;
        }


        [SkipLocalsInit]
        static unsafe float[] MatrixProductSIMDGatherScatterUnrollFMA(float[] A, float[] B, int N)
        {
            if (!Avx2.IsSupported) Console.WriteLine("AVX2 is not supported.");
            float* sumspace = stackalloc float[Vector256<float>.Count];
            int* ptrDelta = stackalloc int[Vector256<int>.Count];
            for (int i = 0; i < Vector256<float>.Count; ++i) ptrDelta[i] = (i * N * Vector256<float>.Count) >> 3;
            int n = (N / Vector256<float>.Count) * Vector256<float>.Count;
            float[] ret = new float[A.Length];
            float[] B2 = new float[B.Length];
            fixed (float* b = B) fixed (float* b2 = B2)
            {
                int nn = N * Vector256<float>.Count;
                for (int i = 0, ii = 0; i < N; ++i, ii += N)
                {
                    IntPtr p = (IntPtr)(&b[i]);
                    int j = 0, jj = 0;
                    for (; j < n; j += Vector256<float>.Count, jj += nn)
                    {
                        //p = (IntPtr)(&b[jj + i]);
                        float* ptr = (float*)(p & unchecked((IntPtr)0xfffffffe00000000));
                        int lowptr = (int)(((long)p >> 2) & 0x7fffffff);
                        Vector256<int> vb_ptr_lower = Avx2.Add(Avx2.LoadVector256(ptrDelta), Avx2.BroadcastScalarToVector256(&lowptr));
                        Vector256<float> vb = Avx2.GatherVector256(ptr, vb_ptr_lower, 4);

                        Avx.Store(&b2[ii + j], vb);
                        p = p + ((IntPtr)(sizeof(float)) * nn);
                    }
                    for (; j < N; ++j, jj += N)
                        b2[ii + j] = b[jj + i];
                }
            }
            fixed (float* a = A) fixed (float* b = B2) fixed (float* c = ret)
            {
                n = (N / (Vector256<float>.Count * 8)) * (Vector256<float>.Count * 8);
                int deltaK = Vector256<float>.Count * 8;
                for (int i = 0, ii = 0; i < N; ++i, ii += N)
                    for (int j = 0, jj = 0; j < N; ++j, jj += N)
                    {
                        float sum = 0;
                        int k = 0;
                        {
                            Vector256<float> vsum1 = Vector256<float>.Zero;
                            Vector256<float> vsum2 = Vector256<float>.Zero;
                            Vector256<float> vsum3 = Vector256<float>.Zero;
                            Vector256<float> vsum4 = Vector256<float>.Zero;
                            Vector256<float> vsum5 = Vector256<float>.Zero;
                            Vector256<float> vsum6 = Vector256<float>.Zero;
                            Vector256<float> vsum7 = Vector256<float>.Zero;
                            Vector256<float> vsum8 = Vector256<float>.Zero;
                            for (; k < n; k += deltaK)
                            {
                                vsum1 = Fma.MultiplyAdd(Avx.LoadVector256(&a[ii + k]), Avx.LoadVector256(&b[jj + k]), vsum1);
                                vsum2 = Fma.MultiplyAdd(Avx.LoadVector256(&a[ii + k + Vector256<float>.Count]), Avx.LoadVector256(&b[jj + k + Vector256<float>.Count]), vsum2);
                                vsum3 = Fma.MultiplyAdd(Avx.LoadVector256(&a[ii + k + (Vector256<float>.Count * 2)]), Avx.LoadVector256(&b[jj + k + (Vector256<float>.Count * 2)]), vsum3);
                                vsum4 = Fma.MultiplyAdd(Avx.LoadVector256(&a[ii + k + (Vector256<float>.Count * 3)]), Avx.LoadVector256(&b[jj + k + (Vector256<float>.Count * 3)]), vsum4);
                                vsum5 = Fma.MultiplyAdd(Avx.LoadVector256(&a[ii + k + (Vector256<float>.Count * 4)]), Avx.LoadVector256(&b[jj + k + (Vector256<float>.Count * 4)]), vsum5);
                                vsum6 = Fma.MultiplyAdd(Avx.LoadVector256(&a[ii + k + (Vector256<float>.Count * 5)]), Avx.LoadVector256(&b[jj + k + (Vector256<float>.Count * 5)]), vsum6);
                                vsum7 = Fma.MultiplyAdd(Avx.LoadVector256(&a[ii + k + (Vector256<float>.Count * 6)]), Avx.LoadVector256(&b[jj + k + (Vector256<float>.Count * 6)]), vsum7);
                                vsum8 = Fma.MultiplyAdd(Avx.LoadVector256(&a[ii + k + (Vector256<float>.Count * 7)]), Avx.LoadVector256(&b[jj + k + (Vector256<float>.Count * 7)]), vsum8);
                            }
                            Avx.Store(sumspace, Avx.Add(Avx.Add(Avx.Add(vsum1, vsum2), Avx.Add(vsum3, vsum4)), Avx.Add(Avx.Add(vsum5, vsum6), Avx.Add(vsum7, vsum8))));
                        }
                        for (int l = 0; l < Vector256<float>.Count; ++l) sum += sumspace[l];
                        for (; k < N; ++k)
                            sum += a[ii + k] * b[jj + k];
                        c[ii + j] = sum;
                    }
            }
            return ret;
        }

        static void Check()
        {
            Random rand = new Random();
            (float[], Matrix4x4) gen()
            {
                float[] ret = new float[3 * 3];
                for (int k = 0; k < ret.Length; ++k) ret[k] = rand.Next(1024);
                return (ret, new Matrix4x4(ret[0], ret[1], ret[2], 0, ret[3], ret[4], ret[5], 0, ret[6], ret[7], ret[8], 0,0,0,0,1));
            }
            for(float i = 0;i < 100; ++i)
            {
                var (f1, m1) = gen();
                var (f2, m2) = gen();
                var f3 = MatrixProduct(f1, f2, 3);
                var m3 = m1 * m2;
                Debug.Assert(m3.M11 == f3[0] && m3.M12 == f3[1] && m3.M13 == f3[2] && m3.M21 == f3[3] && m3.M22 == f3[4] && m3.M23 == f3[5] && m3.M31 == f3[6] && m3.M32 == f3[7] && m3.M33 == f3[8]);
            }
        }
        static void Check2()
        {
            Random rand = new Random();
            int N = 33;
            float[] gen()
            {
                float[] ret = new float[N*N];
                for (int k = 0; k < ret.Length; ++k) ret[k] = rand.Next(1024);
                return ret;
            }
            for (float i = 0; i < 10; ++i)
            {
                var f1 = gen(); var f2 = gen();
                var f3 = MatrixProduct(f1, f2, N);
                var f4 = MatrixProductSIMDGatherScatter(f1, f2, N);
                Debug.Assert(f3.Zip(f4).All((v) => MathF.Abs(v.First - v.Second) <= 0.001f));
                // Nが大きいと誤差が大きくなってダメ
            }
        }
        private class MyConfig : ManualConfig
        {
            public MyConfig()
            {
                AddJob(Job.ShortRun.WithPowerPlan(PowerPlan.Balanced));
            }
        }
        static void Main(string[] args)
        {
            //Check();
            //Check2();
            BenchmarkDotNet.Running.BenchmarkRunner.Run<Program>();
        }
    }
}