#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include "custom/undef.hpp"
#include "io/writer.hpp"

int main()
{
	ANN::io::buffered_writer w("/tmp/t.out");
	w('A');
	w('0');
	std::vector v{
		std::vector{48,49,50,51},
		std::vector{97,98,99,100,101},
		std::vector{65,66}
	};
	// std::string s = "hello";
	w(v);
	w(std::vector<uint32_t>(2'000'000'000));
	return 0;
}
