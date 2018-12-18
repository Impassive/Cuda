#include "stdafx.h"
#include <amp.h>  
#include <iostream> 
#include <array>
#include <omp.h>
#include <ppl.h>

using namespace std;
using namespace Concurrency;


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

inline void copyMatrix(int matrix[], const int m)
{
	int* out = new int[m*m];
	for (int i = 0; i < m*m; i++)
		out[i] = matrix[i];
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
	parallel_for_each(a.extent, [=](index<2> idx)
		restrict(amp)
	{
		int row = idx[0];
		int col = idx[1];
		int temp = 0;
		temp = a(row, col);
		a(row, col) = a(col, row);
		a(col, row) = temp;
	});
	a.synchronize();
}

//Matrix transponse AMP tiled
template <int tileSize>
inline void AMP__block_transposeMatrix_shared(int matrix[], const int m)
{
	array_view<const int, 2> a(m, m, matrix);
	array_view<int, 2> b(m, m, matrix);
	b.discard_data();
	parallel_for_each(a.extent.tile<tileSize, tileSize>(), [=](tiled_index<tileSize, tileSize> tidx)
		restrict(amp)
	{
		tile_static float local_buffer[tileSize][tileSize];
		local_buffer[tidx.local[1]][tidx.local[0]] = a[tidx.global];
		tidx.barrier.wait();
		index<2> outIdx(index<2>(tidx.tile_origin[1], tidx.tile_origin[0]) + tidx.local);
		b[outIdx] = local_buffer[tidx.local[0]][tidx.local[1]];
	});
}


//MULTIPLY


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

	parallel_for(0, m, [=](int i)
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



//Matrix multiply AMP tiled non shared
inline void AMP__multiply_nonshared(int matrix[], const int m)
{
	array_view<const int, 2> input(m, m, matrix);
	array_view<int, 2> output(m, m, matrix);
	output.discard_data();
	const int tileSize = 16;
	parallel_for_each(output.extent.tile <tileSize, tileSize>(),
		[=](tiled_index<tileSize, tileSize> tidx) restrict(amp)
	{
		float temp = 0;
		for (int k = 0; k < m; k++) {
			temp += input[tidx.global[0]][k] * input[k][tidx.global[1]];
		}
		output[tidx.global] = temp;
	});
}

//Matrix multiply AMP tiled shared
template <int tileSize>
inline void MultiplyWithTiling(int matrix[], const int m)
{
	array_view<const int, 2> input(m, m, matrix);
	array_view<int, 2> output(m, m, matrix);
	output.discard_data();
	parallel_for_each(output.extent.tile<tileSize, tileSize>(),
		[=](tiled_index<tileSize, tileSize> t_idx) restrict(amp)
	{
		int row = t_idx.local[0];
		int col = t_idx.local[1];
		int rowGlobal = t_idx.global[0];
		int colGlobal = t_idx.global[1];
		int sum = 0;
		for (int i = 0; i < m; i += tileSize) {
			tile_static int a[tileSize][tileSize];
			tile_static int b[tileSize][tileSize];
			a[row][col] = input(rowGlobal, col + i);
			b[row][col] = input(row + i, colGlobal);
			// The threads in the tile all wait here until locA and locB are filled.  
			t_idx.barrier.wait();

			for (int k = 0; k < tileSize; k++) {
				sum += a[row][k] * b[k][col];
			}
			t_idx.barrier.wait();
		}
		output[t_idx.global] = sum;
	});
	output.synchronize();
}

inline void MultiplyLarged(int matrix[], const int m)
{
	array_view<const int, 2> a(m, m, matrix);
	array_view<int, 2> b(m, m, matrix);
	concurrency::extent<1> ext(m);
	b.discard_data();
	parallel_for_each(
		ext,
		[=](index<1> idx) restrict(amp) {
		int row = idx[0];
		for (int i = 0; i < m; i++)
		{
			int temp = 0;
			for (int j = 0; j < m; j++)
				temp += a(row, j)*a(j, i);
			b[idx[0]][i] = temp;
		}
	});
	a.synchronize();
}

