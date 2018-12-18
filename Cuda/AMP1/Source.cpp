#include "stdafx.h"
#include <amp.h>  
#include <omp.h>
#include <ppl.h>
#include <iostream>  
#include <array> 
using namespace concurrency;
using namespace std;

//print info for default accelerator, to see all accelerators, please, disable 'if'
inline void acceleratorInfo()
{
	Concurrency::accelerator::accelerator();
	vector<accelerator> desc = accelerator::get_all();
	for (int i = 0; i < desc.size(); i++)
	{
		if (desc[i].is_emulated != true)
		{
			wcout << desc[i].description << " info:\n";
			cout << "\tversion: " << desc[i].version << "\n";
			cout << "\tmemory: " << desc[i].dedicated_memory << "\n";
			wcout << "\tpath: " << desc[i].device_path << "\n";
			cout << "\tdisplay: " << std::boolalpha << desc[i].has_display << "\n";
			cout << "\tis debug: " << std::boolalpha << desc[i].is_debug << "\n";
			cout << "\tsupports cpu shared memory: " << std::boolalpha << desc[i].supports_cpu_shared_memory << "\n";
			cout << "\tsupports double precision: " << std::boolalpha << desc[i].supports_double_precision << "\n";
			cout << "\tsupports limited double precision: " << std::boolalpha << desc[i].supports_limited_double_precision << "\n";
		}
	}
}

//sum of two vectors without AMP
inline void vectorSum(int a[], int b[], const int N)
{
	int* sum = new int[N];

	for (int idx = 0; idx < N; idx++)
	{
		sum[idx] = a[idx] + b[idx];
		//cout << "a " << a[idx] << "b " << b[idx] << "sum" << sum[idx] << endl;
	}
	delete[] sum;
}

//sum of two vectors with AMP
inline void AMP_vectorSum(int a[], int b[], const int N)
{
	int* sum = new int[N];
	array_view<const int, 1> a_AMP(N, a);
	array_view<const int, 1> b_AMP(N, b);
	array_view<int, 1> sum_AMP(N, sum);
	sum_AMP.discard_data();

	parallel_for_each(
		// Define the compute domain, which is the set of threads that are created.  
		sum_AMP.extent,
		// Define the code to run on each thread on the accelerator.  
		[=](index<1> idx) restrict(amp)
	{
		sum_AMP[idx] = a_AMP[idx] + b_AMP[idx];
	}
	);
	sum_AMP.synchronize();

	/*for (int idx = 0; idx < N; idx++)
	{
		cout << "a " << a[idx] << "b " << b[idx] << "sum" << sum_AMP[idx] << endl;
	}*/

}

const int num = 511;
//Multiply matrix by number
inline void multiplyMatrix(int matrix[], const int m)
{
	int* out = new int[m*m];
	for (int i = 0; i < m*m; i++)
		out[i] = matrix[i];
	for (int i = 0; i < m; i++)
		for (int j = 0; j < m; j++)
			out[j + (m*i)] *= num;
	delete[] out;
}

//Multiply matrix by number with AMP
inline void AMP_multiplyMatrix(int matrix[], const int m)
{
	array_view<int, 2> a(m, m, matrix);
	parallel_for_each(
		a.extent,
		[=](index<2> idx) restrict(amp) {
		a[idx] *= num;
	});
	a.synchronize();
}

//Matrix transponse
inline void transponseMatrix(int matrix[], const int m)
{
	int* out = new int[m*m];
	for (int i = 0; i < m*m; i++)
		out[i] = matrix[i];
	for (int rows = 0; rows < m; rows++)
		for (int cols = rows + 1; cols < m; cols++)
			swap(out[cols*m + rows], out[rows*m + cols]);
	delete[] out;
}

//Matrix transponse AMP
inline void AMP_transponseMatrix(int matrix[], const int m)
{
	array_view<int, 2> a(m, m, matrix);
	parallel_for_each(
		a.extent,
		[=](index<2> idx) restrict(amp) {
		int row = idx[0];
		int col = idx[1];
		int temp = 0;
		temp = a(row, col);
		a(row, col) = a(col, row);
		a(col, row) = temp;
	});
	a.synchronize();
}

//Matrix multiply
inline void matrixMmatrix(int matrix[], const int m)
{
	int* out = new int[m*m];
	for (int i = 0; i < m*m; i++)
		out[i] = matrix[i];

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < m; j++)
		{
			out[m * i + j] = 0;
			for (int k = 0; k < m; k++)
			{
				out[m * i + j] += matrix[m * i + k] * out[m * k + j];
			}
		}
	}
	delete[] out;
}

//Matrix multiply AMP
inline void AMP_matrixMmatrix(int matrix[], const int m)
{
	//Edit 1
	//use cont on read only matrix
	array_view<const int, 2> a(m, m, matrix);
	array_view<int, 2> b(m, m, matrix);
	//Edit 2
	b.discard_data();
	parallel_for_each(
		a.extent,
		[=](index<2> idx) restrict(amp) {
		int row = idx[0];
		int col = idx[1];
	//Edit 3
		// add local param to comulate sum in each thread
		int temp = 0;
		for (int i = 0; i < m; i++)
			temp += a(row, i)*a(i, row);
		b[idx] += temp;
	});
	a.synchronize();
}

//Matrix multiply OMP
inline void OMP_matrixMmatrix(int matrix[], const int m)
{
	int* out = new int[m*m];
	for (int i = 0; i < m*m; i++)
		out[i] = matrix[i];

#pragma omp parallel for
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < m; j++)
		{
			out[m * i + j] = 0;
			int temp = 0;
			for (int k = 0; k < m; k++)
			{
				temp += matrix[m * i + k] * out[m * k + j];
			}
			out[m * i + j] += temp;
		}
	}
	delete[] out;
}

//Matrix multiply PPL
inline void PPL_matrixMmatrix(int matrix[], const int m)
{
	int* out = new int[m*m];
	for (int i = 0; i < m*m; i++)
		out[i] = matrix[i];

	parallel_for (0,m, [=] (int i)
	{
		for (int j = 0; j < m; j++)
		{
			out[m * i + j] = 0;
			int temp = 0;
			for (int k = 0; k < m; k++)
			{
				temp += matrix[m * i + k] * out[m * k + j];
			}
			out[m * i + j] += temp;
		}
	});
	delete[] out;
}