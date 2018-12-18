// AMP1.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <amp.h>  
#include <iostream> 
#include <array>
#include "Source.cpp"

using namespace concurrency;
using namespace std;


int main()
{
	//Print accelerator device info
	acceleratorInfo();
	const int N = 20000000;
	int m = 3000;
	cout << "\nCalculations (N=" << N / 1000000 << "kk) (matrix=" << m / 1000 << "k x " << m / 1000 << "k)\n";

	//Sum two vectors
	int* a = new int[N];
	int* b = new int[N];
	for (int i = 0; i < N; i++)
	{
		a[i] = rand();
		b[i] = rand();
	}

	cout << "    Vector sum:\n";
	//Standard
	clock_t begin = clock();
	vectorSum(a, b, N);
	clock_t end = clock();
	double elapsed = double(end - begin);
	cout << "\tStandard - " << 1.0*elapsed / CLOCKS_PER_SEC << "s\n";
	//AMP
	begin = clock();
	AMP_vectorSum(a, b, N);
	end = clock();
	elapsed = double(end - begin);
	cout << "\tAMP - " << 1.0*elapsed / CLOCKS_PER_SEC << "s\n";

	//Matrix multiplication by number

	int* matrix = new int[m*m];
	for (int i = 0; i < m*m; i++)
	{
		matrix[i] = rand();
	}

	cout << "    Matrix multiplication by number:\n";
	//Standard
	begin = clock();
	multiplyMatrix(matrix, m);
	end = clock();
	elapsed = double(end - begin);
	cout << "\tStandard - " << 1.0*elapsed / CLOCKS_PER_SEC << "s\n";
	//AMP
	begin = clock();
	AMP_multiplyMatrix(matrix, m);
	end = clock();
	elapsed = double(end - begin);
	cout << "\tAMP - " << 1.0*elapsed / CLOCKS_PER_SEC << "s\n";


	//Matrix transponse
	cout << "    Matrix transponse:\n";
	//Standard
	begin = clock();
	transponseMatrix(matrix, m);
	end = clock();
	elapsed = double(end - begin);
	cout << "\tStandard - " << 1.0*elapsed / CLOCKS_PER_SEC << "s\n";
	//AMP
	begin = clock();
	AMP_transponseMatrix(matrix, m);
	end = clock();
	elapsed = double(end - begin);
	cout << "\tAMP - " << 1.0*elapsed / CLOCKS_PER_SEC << "s\n";

	//Matrix multiply
	cout << "    Matrix * Matrix:\n";
	//Standard
	begin = clock();
	matrixMmatrix(matrix, m);
	end = clock();
	elapsed = double(end - begin);
	cout << "\tStandard - " << 1.0*elapsed / CLOCKS_PER_SEC << "s\n";
	//AMP
	begin = clock();
	AMP_matrixMmatrix(matrix, m);
	end = clock();
	elapsed = double(end - begin);
	cout << "\tAMP - " << 1.0*elapsed / CLOCKS_PER_SEC << "s\n";
	//OMP
	begin = clock();
	OMP_matrixMmatrix(matrix, m);
	end = clock();
	elapsed = double(end - begin);
	cout << "\tOMP - " << 1.0*elapsed / CLOCKS_PER_SEC << "s\n";
	//PPL
	begin = clock();
	PPL_matrixMmatrix(matrix, m);
	end = clock();
	elapsed = double(end - begin);
	cout << "\tPPL - " << 1.0*elapsed / CLOCKS_PER_SEC << "s\n";
	system("pause");
}




