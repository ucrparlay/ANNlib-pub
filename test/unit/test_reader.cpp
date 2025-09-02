#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include "custom/undef.hpp"
#include "io/reader.hpp"
using std::cout;
using std::endl;

int main()
{
	ANN::io::mmap_reader r("/tmp/t.out");
	// auto x = r(); // FAIL
	char a = r();
	cout << a << endl;
	cout << (char)r() << endl;

	std::vector<std::vector<int>> v = r({4,5,2});
	// std::vector<std::vector<int>> v = r(std::vector{4,5,2});
	for(const auto &t : v)
	{
		for(auto e : t)
			cout << e << ' ';
		cout << endl;
	}

	r.set_pos(2);

	std::vector<int> v2 = r(11);
	for(auto e : v2)
		cout << e << ' ';
	cout << endl;

	return 0;
}
