#pragma once
#include <winnt.h>
#include <winbase.h>

class timer {
private:
	LARGE_INTEGER StartingTime, EndingTime, Elapsed, Frequency;

public:
	timer::timer() {
		QueryPerformanceFrequency(&Frequency);
	}

	void timer::start() {
		QueryPerformanceCounter(&StartingTime);
	}

	void timer::stop() {
		QueryPerformanceCounter(&EndingTime);
	}

	LONGLONG timer::elapsed_microseconds() {
		Elapsed.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
		Elapsed.QuadPart *= 1000000;
		Elapsed.QuadPart /= Frequency.QuadPart;
		return (long long)Elapsed.QuadPart;
	}

	double timer::elapsed_seconds() {
		return (this->elapsed_microseconds() / 1000000.0);
	}

	timer::~timer() {}
};

