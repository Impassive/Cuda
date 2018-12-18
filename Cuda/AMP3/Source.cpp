#include "stdafx.h"
#include "stdafx.h"
#include <stdio.h>
#include <amp.h>
#include <amp_math.h>
#include <iostream>
#include "time.h"
#include <omp.h>
#include <ppl.h>

using namespace concurrency;
using namespace std;

inline void acceleratorInfo(vector<accelerator> desc)
{
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

inline void Reduction1(int array[], const int m, accelerator_view acc)
{
	array_view<int, 1> a(m, array);

	for (unsigned stride = m / 2; stride > 0; stride /= 2)
	{
		parallel_for_each(acc, concurrency::extent<1>(stride),
			[=](index<1> idx) restrict(amp)
		{
			a[idx] = a[idx] + a[idx + stride];

			if ((idx[0] == stride - 1) && (stride & 0x1) && (stride != 1))
			{
				a[0] += a[stride - 1];
			}
		});
	}
	a.synchronize();
	cout << "HERE IS RESULT:  " << a[0] << endl;
}

inline void Reduction2(int array[], const int m, const int width, accelerator_view acc)
{
	array_view<int, 1> a(m, array);

	for (int stride = m / width; stride > 0; stride /= width)
	{
		parallel_for_each(acc, concurrency::extent<1>(stride),
			[=](index<1> idx) restrict(amp)
		{
			int sum = 0;
			for (int i = 0; i < width; i++)
			{
				sum += a[idx + i * stride];
			}
			a[idx] = sum;

		});
	}
	a.synchronize();
	cout << "HERE IS RESULT:  " << a[0] << endl;
}


template <unsigned tileSize>
inline void Reduction3(int source[], const int m, accelerator_view acc)
{
	unsigned elementCount = m;
	unsigned tile = tileSize;
	concurrency::array<double, 1> arr_1(elementCount, source);
	concurrency::array<double, 1> arr_2((elementCount / tileSize) != 0 ?
		(elementCount / tileSize) :	1);
	concurrency::array_view<double, 1> av_src(arr_1);
	concurrency::array_view<double, 1> av_dst(arr_2);
	av_dst.discard_data();
	// Reduce using parallel_for_each as long as the sequence length
	// is evenly divisable to the number of threads in the tile.
	while ((elementCount % tileSize) == 0)
	{
		concurrency::parallel_for_each(
			concurrency::extent<1>(m).tile<tileSize>(),
			[=](tiled_index<tileSize> tidx) restrict(amp)
		{
			tile_static double tile_data[tileSize];

			unsigned local_idx = tidx.local[0];
			tile_data[local_idx] = av_src[tidx.global];
			tidx.barrier.wait();
			for (unsigned s = 1; s < tileSize; s *= 2)
			{
				if (local_idx % (2 * s) == 0)
				{
					tile_data[local_idx] += tile_data[local_idx + s];
				}
				tidx.barrier.wait();
			}
			// Store the tile result in the global memory.
			if (local_idx == 0)
			{
				av_dst[tidx.tile] = tile_data[0];
			}
		});
		elementCount = elementCount / tileSize;
		std::swap(av_src, av_dst);
		av_dst.discard_data();

	}
}

template <int tileSize, int tileCount, int batchSize>
inline void Reduction4(int array[], const int m, accelerator_view acc)
{
	int elements = m;
	unsigned stride = tileSize * tileCount * batchSize;

	double tailSum = 0.0;


	unsigned tailLength = elements % stride;
	if (tailLength != 0) {
		tailSum = vectorSumWindow(array, elements - tailLength, elements);
		elements -= tailLength;
	}

	concurrency::array<double, 1> arr(elements, array);
	concurrency::array<double, 1> partial_result(tileCount);
	parallel_for_each(concurrency::extent<1>(tileCount * tileSize).tile<tileSize>(),
		[=, &arr, &partial_result](tiled_index<tileSize> tidx) restrict(amp)
	{
		tile_static double tile_data[tileSize];

		unsigned local_idx = tidx.local[0];

		unsigned targetStartIndex = (tidx.tile[0] * batchSize * tileSize) + local_idx;
		tile_data[local_idx] = 0.0;

		double temp = 0.0;
		do
		{
			for (unsigned i = 0; i < batchSize; i++)
			{
				temp += arr[targetStartIndex + tileSize * i];
			}

			targetStartIndex += stride;
		} while (targetStartIndex < elements);
		tile_data[local_idx] = temp;
		tidx.barrier.wait();

		// Reduce local result in tileData to tileData[0]
		for (int localStride = tileSize / 2; localStride > 0; localStride /= 2)
		{
			if (local_idx < localStride) {
				tile_data[local_idx] += tile_data[local_idx + localStride];
			}

			tidx.barrier.wait();
		}

		// Store result to partial result in global memory 
		if (local_idx == 0) {
			partial_result[tidx.tile[0]] = tile_data[0];
		}
	});
}
template<typename T>
T vectorSumWindow(T* source, int ind, int end) {
	T result = 0;
	for (int i = ind; i < end; i++) {
		result += source[i];
	}
	return result;
}
