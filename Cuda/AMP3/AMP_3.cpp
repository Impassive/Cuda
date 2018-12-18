// AMP_3.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "stdafx.h"
#include "stdafx.h"
#include <algorithm> // for copy() and copy_n()
#include <vector>
#include <stdio.h>
#include <amp.h>
#include <amp_math.h>
#include <iostream>
#include "time.h"
#include <omp.h>
#include <ppl.h>
#include "Source.cpp"

using namespace std;

void populateArray(int arr[], int m)
{
	for (int i = 0; i < m; i++)
		//arr[i] = (int)rand() % 1000;
		arr[i] = 1;
}

int main()
{
	timer t;
	//Data
	int window = 8;
	int m = 100000 * window;
	int* vect = new int[m];
	populateArray(vect, m);

	//prepare similar arrays:
	int* data1 = new int[m];
	populateArray(data1, m);
	int* data2 = new int[m];
	populateArray(data2, m);
	int* data3 = new int[m];
	populateArray(data3, m);
	int* data4 = new int[m];
	populateArray(data4, m);
	int* data5 = new int[m];
	populateArray(data5, m);
	int* data6 = new int[m];
	populateArray(data6, m);
	int* data7 = new int[m];
	populateArray(data5, m);
	int* data8 = new int[m];
	populateArray(data6, m);


	vector<accelerator> accs = accelerator::get_all();
	accelerator_view intel = accs[1].default_view;
	accelerator_view nvidia = accs[0].default_view;
	acceleratorInfo(accs);

	//Non-tiled reduction
	cout << "\n\nNon-tiled reduction\n";
	t.start();
	Reduction1(data1, m, nvidia);
	t.stop();
	double elapsed = t.elapsed_seconds();
	cout << "Nvidia: " << elapsed << "\n";
	t.start();
	Reduction1(data2, m, intel);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "Intel: " << elapsed << "\n";


	//Non-tiled windowed reduction
	cout << "\nNon-tiled windowed reduction\n";
	t.start();
	Reduction2(data3, m, window, nvidia);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "Nvidia: " << elapsed << "\n";
	t.start();
	Reduction2(data4, m, window, intel);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "Intel: " << elapsed << "\n";

	//Tiled windowed reduction
	cout << "\nTiled reduction\n";
	t.start();

	Reduction3<8>(data5, m, nvidia);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "Nvidia: " << elapsed << "\n";
	t.start();
	//Reduction3<16>(data6, m, intel);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "Intel: " << elapsed << "\n";

	//Cascade reduction
	cout << "\nCascade reduction\n";
	t.start();

	//Reduction4<32, 4, 8>(data5, m, nvidia);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "Nvidia: " << elapsed << "\n";
	t.start();
	//Reduction4<32, 4, 8>(data6, m, intel);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "Intel: " << elapsed << "\n";


	system("pause");
	return 0;
}

