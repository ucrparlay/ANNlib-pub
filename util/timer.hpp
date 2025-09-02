#ifndef _ANN_UTIL_TIMER_HPP_
#define _ANN_UTIL_TIMER_HPP_

#include <chrono> // TODO: remove
#include <atomic> // TODO: remove

struct timer{
	timer(const char *name) :
		name(name), time_begin(high_resolution_clock::now()){
	}
	auto click(){
		const auto time_end = high_resolution_clock::now();
		const auto duration = time_end - time_begin;
		time_begin = time_end;
		return duration_cast<milliseconds>(duration).count();
	}
	/*
	~timer(){
		const auto msec = click();
		printf("msec_%s: %lu\n", name, msec);
	}
	*/
	const char *name;
	time_point<high_resolution_clock> time_begin;
};

// _ANN_UTIL_TIMER_HPP_
