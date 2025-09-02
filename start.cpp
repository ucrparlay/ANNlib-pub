#include <cstdio>
#include <cassert>
#include <vector>
#include "custom/undef.hpp"
#include "graph/adj.hpp"
#include "algo/vamana.hpp"
using std::vector;

struct Point{
    const char *name;
    vector<float> coord;

    /* define types and funcs for ANNlib to use */
    using id_t = const char*;
    using elem_t = float;
    using coord_t = vector<float>;

    id_t get_id() const {return name;}
    const coord_t& get_coord() const {return coord;}
};

struct Desc{
    using point_t = Point;

    using dist_t = float;
    static dist_t distance(const auto &cu, const auto &cv, uint32_t dim){
        // user-defined distance function: 2d L2 norm
        assert(dim==2);
        auto d0=cu[0]-cv[0], d1=cu[1]-cv[1];
        return d0*d0 + d1*d1;
    }

    template<typename Nid, class Ext, class Edge>
    using graph_t = typename ANN::graph::adj_map<Nid,Ext,Edge>;
};

int main(){
    vector<Point> ps{
        {"left", {0,0}}, //3,3 = 18
        {"right", {4,0}}, // 1,3 = 10
        {"top", {2,1}} // 1,2 = 5
    };
    ANN::vamana<Desc> index(2); // dimension = 2
    index.insert(ps.begin(), ps.end());
    auto res = index.search(vector<float>{3,3}, 2, 10);
    for(auto [d,u] : res)
        printf("[%s] in distance of %.1f\n", u, d);
    return 0;
}
