#include <vector>
#include "../utils.hpp"

template<class U, class S1, class S2>
auto get_gt(const S1& base, const S2& query, uint32_t dim, uint32_t k) {
  return ConstructKnng<U>(base, query, dim, k);
}

template<class U, class S1, class S2, typename L>
auto get_gt(const S1& base, const S2& query, uint32_t dim, uint32_t k,
            const parlay::sequence<parlay::sequence<L>>& base_filters,
            const parlay::sequence<parlay::sequence<L>>& query_filters) {
  return ConstructKnng<U>(base, query, dim, k, base_filters, query_filters);
}
