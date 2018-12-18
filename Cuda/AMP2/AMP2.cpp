// AMP2.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Source.cpp"
#include <amp.h>  
#include <iostream> 
#include <array>
#include "time.h"

using namespace std;
using namespace Concurrency;

int main()
{
	timer t;

	//Print Info
	acceleratorInfo();

	//Data
	int m = 100 * 16;
	int* matrix = new int[m*m];
	for (int i = 0; i < m*m; i++)
	{
		matrix[i] = rand();
	}

	cout << "\n\nTransponse\n";
	//matrixTransponse
	cout << "standard";
	t.start();
	transponseMatrix(matrix, m);
	t.stop();
	double elapsed = t.elapsed_seconds();
	cout << "\telapsed - " << elapsed << "s\n";

	//matrixTransponseAMP 
	cout << "AMP";
	t.start();
	AMP_transponseMatrix(matrix, m);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "\telapsed - " << elapsed << "s\n";

	//matrixTransponseBlockAMP tiled
	cout << "AMP tiled";
	t.start();
	AMP__block_transposeMatrix_shared<16>(matrix, m);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "\telapsed - " << elapsed << "s\n";

	//MULTIPLY
	//Standard
	cout << "\nmatrix multiply\n";
	cout << "matrix multiply standard";
	t.start();
	matrixMmatrix(matrix, m);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "\telapsed - " << elapsed << "s\n";


	//OMP
	cout << "OMP";
	t.start();
	OMP_matrixMmatrix(matrix, m);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "\telapsed - " << elapsed << "s\n";

	//PPL
	cout << "PPL";
	t.start();
	PPL_matrixMmatrix(matrix, m);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "\telapsed - " << elapsed << "s\n";

	//Simple AMP
	cout << "simple AMP";
	t.start();
	AMP_matrixMmatrix(matrix, m);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "\telapsed - " << elapsed << "s\n";

	//matrix multiply AMP tiled nonshared
	cout << "AMP tiled non shared";
	t.start();
	AMP__multiply_nonshared(matrix, m);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "\telapsed - " << elapsed << "s\n";

	//Matrix multiply AMP tiled shared
	cout << "AMP tiled shared";
	t.start();
	MultiplyWithTiling<16>(matrix, m);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "\telapsed - " << elapsed << "s\n";

	//Matrix large decompose
	cout << "AMP large decompose";
	t.start();
	MultiplyLarged(matrix, m);
	t.stop();
	elapsed = t.elapsed_seconds();
	cout << "\telapsed - " << elapsed << "s\n";

	system("pause");
	return 0;
}

