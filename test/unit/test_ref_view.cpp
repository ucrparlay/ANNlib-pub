#include <cstdio>
#include <vector>
#include <utility>
#include "util/intrin.hpp"
#include "util/ref.hpp"

using vec_t = std::vector<int>;

int main()
{
	vec_t a{1,2,3};
	/*-------------- switch here -------------*/
	// ANN::util::ref_view r(a);
	ANN::util::ref_view r(std::move(a));
	/*----------------------------------------*/
	for(auto e : r)
		printf("%d ", e);
	putchar('\n');
	auto b = ANN::util::to<vec_t>(r);
	printf("a: %lu\n", a.size());
	printf("b: %lu\n", b.size());
	return 0;
}
