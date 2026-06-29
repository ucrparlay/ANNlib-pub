#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "algo/algo.hpp"
#include "algo/vamana.hpp"
#include "algo/vamana_filtered.hpp"
#include "algo/vamana_stitched.hpp"
// #include "algo/HNSW.hpp"
// #include "algo/hnsw_filtered.hpp"
#include "benchUtils.h"
#include "parlay.hpp"
// #include "cpam.hpp"
#include "dist.hpp"
#include "graph/adj.hpp"
#include "modules/ground_truth.hpp"
#include "parse_points.hpp"
#include "util/intrin.hpp"
#include "utils.hpp"

// Vamana
#include "modules/filtered_vamana.hpp"
#include "modules/post_vamana.hpp"
#include "modules/stitched_vamana.hpp"
#include "modules/vamana.hpp"
// HNSW
// #include "modules/filtered_hnsw.hpp"
// #include "modules/hnsw.hpp"
// #include "modules/post_hnsw.hpp"

// using ANN::filtered_vamana;
// using ANN::stitched_vamana;
// using ANN::vamana;


// parlay::sequence<size_t> per_visited;
// parlay::sequence<size_t> per_eval;
// parlay::sequence<size_t> per_size_C;

namespace ANN::external {

  auto def_custom_tag() {
    return custom_tag_parlay{};
  }

}  // namespace ANN::external

template<typename T>
point_converter_default<T> to_point;

template<typename T>
class gt_converter {
 public:
  using type = parlay::sequence<T>;
  template<typename Iter>
  type operator()([[maybe_unused]] uint32_t id, Iter begin, Iter end) {
    using type_src = typename std::iterator_traits<Iter>::value_type;
    static_assert(std::is_convertible_v<type_src, T>, "Cannot convert to the target type");

    const uint32_t n = std::distance(begin, end);

    // T *gt = new T[n];
    auto gt = parlay::sequence<T>(n);
    for (uint32_t i = 0; i < n; ++i) gt[i] = *(begin + i);
    return gt;
  }
};

template<class DescLegacy>
struct desc {
  using point_t = point<typename DescLegacy::type_elem>;
  using coord_t = typename point_t::coord_t;
  using dist_t = float;
  static dist_t distance(const coord_t &cu, const coord_t &cv, uint32_t dim) {
    return DescLegacy::distance(cu, cv, dim);
  }

  template<typename Nid, class Ext, class Edge>
  using graph_t = ANN::graph::adj_map<Nid, Ext, Edge>;

  template<typename Nid, class Ext, class Edge>
  using graph_aux = ANN::graph::adj_map<Nid, Ext, Edge>;
};

// template<class DescLegacy>
// struct desc_cpam : desc<DescLegacy> {
//   template<typename Nid, class Ext, class Edge = Nid>
//   using graph_t = graph_cpam<Nid, Ext, Edge>;

//   template<typename Nid, class Ext, class Edge = Nid>
//   using graph_aux = graph_cpam<Nid, Ext, Edge>;
// };

// Visit all the vectors in the given 2D array of points
// This triggers the page fetching if the vectors are mmap-ed
// template<class T>
// void visit_point(const T &array, size_t dim0, size_t dim1) {
//   parlay::parallel_for(0, dim0, [&](size_t i) {
//     const auto &a = array[i];
//     [[maybe_unused]] volatile auto elem = a.get_coord()[0];
//     for (size_t j = 1; j < dim1; ++j) elem = a.get_coord()[j];
//   });
// }

// template<typename F>
// auto parse_array(const std::string &s, F f){
// 	std::stringstream ss;
// 	ss << s;
// 	std::string current;
// 	std::vector<decltype(f((char*)NULL))> res;
// 	while(std::getline(ss, current, ',')) {
// 		res.push_back(f(current.c_str()));
//   }
// 	// std::sort(res.begin(), res.end());
// 	return res;
// };

template<typename U>
void run_test(commandLine parameter)  // intend to be pass-by-value manner
{
  const char *file_in = parameter.getOptionValue("-in");
  [[maybe_unused]] const size_t size_init = parameter.getOptionLongValue("-init", 0);
  [[maybe_unused]] const size_t size_step = parameter.getOptionLongValue("-step", 0);
  size_t size_max = parameter.getOptionLongValue("-max", 0);
  const auto &R = parse_array(parameter.getOptionValue("-R"), atoi);                            // filtered vamana
  const auto &r = parse_array(parameter.getOptionValue("-r"), atoi);                            // stitched vamana small
  [[maybe_unused]] const uint32_t stitched_R = parameter.getOptionIntValue("-stitched_R", 64);  // stitched vamana large
  const auto &beam = parse_array(parameter.getOptionValue("-beam"), atoi);
  [[maybe_unused]] const uint32_t m = parameter.getOptionIntValue("-m", 40);
  [[maybe_unused]] const float ml = parameter.getOptionDoubleValue("-ml", 0.4);
  [[maybe_unused]] const uint32_t efc = parameter.getOptionIntValue("-efc", 60);
  [[maybe_unused]] const float alpha = parameter.getOptionDoubleValue("-alpha", 1);
  [[maybe_unused]] const float batch_base = parameter.getOptionDoubleValue("-b", 2);
  [[maybe_unused]] const uint32_t rounds = parameter.getOptionIntValue("-rounds", 5);
  const char *file_query = parameter.getOptionValue("-q");
  const uint32_t k = parameter.getOptionIntValue("-k", 10);
  // const uint32_t ef = parameter.getOptionIntValue("-ef", m * 20);
  // const char *vamana_type = parameter.getOptionValue("-vt");
  // const char *ground_truth = parameter.getOptionValue("-gt");
  const auto &gt_labels = parse_array(parameter.getOptionValue("-gts"), atoi);
  const char *file_label_in = parameter.getOptionValue("-lb");
  [[maybe_unused]] const char *file_label_query = parameter.getOptionValue("-lq");

  auto gen_dir = [&](const char *dir) -> std::string {
    std::vector<std::string> parts;
    const std::string s(dir);
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, '/')) {
      parts.push_back(part);
    }
    std::string res;
    for (size_t i = 0; i < parts.size() - 1; ++i) {
      res += parts[i];
      res += "/";
    }
    return res;
  };

  parlay::internal::timer t("run_test:prepare", true);

  using L = typename U::point_t::label_t;
  using T = typename U::point_t::elem_t;

  auto [ps, dim] = load_point(file_in, to_point<T>, size_max);
  t.next("Load the base set");
  printf("%s: [%lu,%u]\n", file_in, ps.size(), dim);

  if (ps.size() < size_max) {
    size_max = ps.size();
    printf("size_max is corrected to %lu\n", size_max);
  }

  auto [q, dd] = load_point(file_query, to_point<T>);
  t.next("Load queries");
  printf("%s: [%lu,%u]\n", file_query, q.size(), dd);
  assert(dd == dim);

  // parallel pre-fetch
  visit_point(ps, size_max, dim);
  visit_point(q, q.size(), dim);
  t.next("Prefetch vectors");

  // auto [F_b, P_b, F_q, _] =
  //     load_label_helper<L>(uint32_t{}, file_label_in, file_label_query, size_max);

  auto [F_b, P_b] = load_label_helper<L>(uint32_t{}, file_label_in, size_max);

  auto gt_exists = [](const char *gt_path) -> bool {
    struct stat buffer;
    return (stat(gt_path, &buffer) == 0);
  };

  std::vector<std::string> gt_paths;
  const std::string base_dir = gen_dir(file_label_in);
  std::string normal_gt_path = base_dir + "gt.bin";
  gt_paths.push_back(normal_gt_path);
  for (const auto &num : gt_labels) {
    std::string filter_gt_path = std::format("{}N{}L{}.gt.bin", base_dir, ps.size(), num);
    gt_paths.push_back(filter_gt_path);
  }

/*
  std::cout << "Generating Ground Truth..." << std::endl;
  // auto gt = (gt_paths.size() >= 1 && gt_exists(gt_paths[0]) ? load_point(gt_paths[0], gt_converter<uint32_t>{}).first : get_gt<U>(ps, q, dim, k));
  parlay::sequence<parlay::sequence<typename U::point_t::id_t>> gt;
  if (gt_paths.size() >= 1 && gt_exists(gt_paths[0].c_str())) {
    std::cout << "Load ground truth from " << std::string(gt_paths[0]) << std::endl;
    try {
      gt = load_point((gt_paths[0] + ":ubin").c_str(), gt_converter<uint32_t>{}).first;
    } catch (const std::invalid_argument &e) {
      std::cerr << e.what() << '\n';
    } 
  } else {
    std::cout << "Calculating ground truth..." << std::endl;
    gt = get_gt<U>(ps, q, dim, k);
    if (gt_paths.size() >= 1) {
      std::cout << "Writing ground truth to " << std::string(gt_paths[0]) << "..." << std::endl;
      write_to_bin(gt_paths[0].c_str(), gt, dim);
    }
  }

  std::cout << ">>> Vamana >>>" << std::endl;
  for (const uint32_t r : R) {
    const auto &vamana_index = run_vamana_insert<U>(dim, r, 100, alpha, batch_base, size_init, size_step, size_max, ps);
    for (const uint32_t l : beam) {
      std::cout << std::endl << ">>> Vamana R = " << r << ", " << "L = " << l << std::endl;
      // run_vamana<U>(dim, r, 100, alpha, batch_base, k, l, size_init, size_step, size_max, ps, q, gt);
      run_vamana_search(vamana_index, k, l, q, gt);
    }
  }

  std::cout << ">>> HNSW >>>" << std::endl;
  const auto &hnsw_index = run_hnsw_insert<U>(dim, ml, m, efc, alpha, batch_base, size_init, size_step, size_max, ps);
  for (const uint32_t ef : beam) {
    std::cout << std::endl << ">>> HNSW M = " << m << ", " << "Ef = " << ef << std::endl;
    // run_hnsw<U>(dim, ml, m, efc, alpha, batch_base, k, ef, size_init, size_step, size_max, ps, q, gt);
    run_hnsw_search(hnsw_index, k, ef, q, gt);
  }
*/

  // test specificity
  const size_t spec_labels_size = gt_labels.size();
  parlay::sequence<decltype(F_b)> Fqs(spec_labels_size, decltype(F_b)(q.size()));
  // std::vector<decltype(gt)> filtered_gts(spec_labels_size);
  parlay::sequence<parlay::sequence<parlay::sequence<typename U::point_t::id_t>>> filtered_gts(spec_labels_size);
  
  std::cout << "Generating Filtered Ground Truth..." << std::endl;
  for (size_t i = 0; i < spec_labels_size; ++i) {
    for (size_t j = 0; j < q.size(); ++j) {
  // parlay::parallel_for(0, spec_labels_size, [&](size_t i) {
  //   parlay::parallel_for(0, q.size(), [&](size_t j) {
      Fqs[i][j].push_back(gt_labels[i]);
    // });
    }
    if (gt_paths.size() - 1 >= spec_labels_size && gt_exists(gt_paths[i + 1].c_str())) {
      std::cout << "Load ground truth from " << std::string(gt_paths[i + 1]) << std::endl;
      filtered_gts[i] = load_point((gt_paths[i + 1] + ":ubin").c_str(), gt_converter<uint32_t>{}).first;
    } else {
      std::cout << "Calculating ground truth..." << std::endl;
      filtered_gts[i] = get_gt<U>(ps, q, dim, k, F_b, Fqs[i]);
      if (gt_paths.size() - 1 >= spec_labels_size) {
        std::cout << "Writing ground truth to " << std::string(gt_paths[i + 1]) << "..." << std::endl;
        write_to_bin(gt_paths[i + 1].c_str(), filtered_gts[i], k);
      }
    }
  // });
  }
  std::cout << "Generated " << filtered_gts.size() << " Filtered Ground Truth File(s)." << std::endl << std::endl;

/****************************************************************
 * Filtered Vamana
*****************************************************************/

  std::cout << ">>> Filtered Vamana >>>" << std::endl;
  for (const uint32_t fr : R) {
    // const auto &entrance = find_medoid(P_b, F_b.size(), 0.1);
    // std::cout << "Entrance point size: " << entrance.size() << std::endl;
    const auto &filtered_vamana_index = run_filtered_vamana_insert<U>(dim, fr, efc, alpha, batch_base, size_init, size_step, size_max, ps, F_b);
    for (size_t i = 0; i < spec_labels_size; ++i) {
      std::cout << ">> Round: " << i + 1 << ", Label Value: " << gt_labels[i] << ", Label Num: " << P_b[gt_labels[i]].size() << std::endl;
      for (const uint32_t l : beam) {
        std::cout << std::endl << ">>> Filtered Vamana R = " << fr << ", " << "L = " << l << std::endl;
        // run_filtered_vamana<U>(dim, r, 100, alpha, batch_base, k, l, size_init, size_step, size_max, ps, q, F_b, P_b, F_q, filtered_gt, false);
        run_filtered_vamana_search(filtered_vamana_index, k, l, q, Fqs[i], filtered_gts[i], rounds);
      }
    }
  }

  // auto pps = decltype(ps)(ps.size());
  // auto Fbb = decltype(F_b)(F_b.size());
  // std::vector<size_t> idx(ps.size());
  // std::iota(idx.begin(), idx.end(), 0);
  // std::random_device rd;
  // std::mt19937 generator(rd());
  // std::shuffle(idx.begin(), idx.end(), generator);
  // parlay::parallel_for(0, ps.size(), [&](size_t i) {
  //   pps[i] = ps[idx[i]];
  //   Fbb[i] = F_b[idx[i]];
  // });
  // std::cout << ">>> Filtered Vamana >>>" << std::endl;
  // for (const uint32_t fr : r) {
  //   // const auto &entrance = find_medoid(P_b, F_b.size(), 0.1);
  //   // std::cout << "Entrance point size: " << entrance.size() << std::endl;
  //   const auto &filtered_vamana_index = run_filtered_vamana_insert<U>(dim, fr, efc, alpha, batch_base, size_init, size_step, size_max, pps, Fbb);
  //   for (size_t i = 0; i < spec_labels_size; ++i) {
  //     std::cout << ">> Round: " << i + 1 << ", Label Value: " << gt_labels[i] << ", Label Num: " << P_b[gt_labels[i]].size() << std::endl;
  //     for (const uint32_t l : beam) {
  //       std::cout << std::endl << ">>> Filtered Vamana R = " << fr << ", " << "L = " << l << std::endl;
  //       // run_filtered_vamana<U>(dim, r, 100, alpha, batch_base, k, l, size_init, size_step, size_max, ps, q, F_b, P_b, F_q, filtered_gt, false);
  //       run_filtered_vamana_search(filtered_vamana_index, k, l, q, Fqs[i], filtered_gts[i]);
  //     }
  //   }
  // }

/****************************************************************
 * Stitched Vamana
*****************************************************************/

  std::cout << ">>> Stitched Vamana >>>" << std::endl;
  for (const uint32_t sr : r) {
    // const auto &entrance = find_medoid(P_b, F_b.size(), 0.2);
    // std::cout << "Entrance point size: " << entrance.size() << std::endl;
    const auto &stitched_vamana_index = run_stitched_vamana_insert<U>(dim, sr, stitched_R, efc, alpha, batch_base, size_max, ps, F_b, P_b);
    for (size_t i = 0; i < spec_labels_size; ++i) {
      std::cout << ">> Round: " << i + 1 << ", Label Value: " << gt_labels[i] << ", Label Num: " << P_b[gt_labels[i]].size() << std::endl;
      for (const uint32_t l : beam) {
        std::cout << std::endl << ">>> Stitched Vamana R = " << sr << ", Stitched_R = " << stitched_R << ", L = " << l << std::endl;
        // run_stitched_vamana<U>(dim, r, 100, alpha, batch_base, k, l, size_max, ps, q, F_b, P_b, F_q, filtered_gt, false);
        run_stitched_vamana_search(stitched_vamana_index, k, l, q, Fqs[i], filtered_gts[i], rounds);
      }
    }
  }

/****************************************************************
 * ANN Constraint-Optimized Retrieval Network (ACORN)
*****************************************************************/
  // TODO: ACORN

  // std::cout << ">>> Filtered HNSW >>>" << std::endl;
  // const auto &entrance = find_medoid(P_b, F_b.size(), 0.5);
  // std::cout << "Entrance point size: " << entrance.size() << std::endl;
  // const auto &filtered_hnsw_index = run_filtered_hnsw_insert<U>(dim, ml, m, efc, alpha, batch_base, size_init, size_step, size_max, ps, F_b, entrance);
  // for (size_t i = 0; i < spec_labels_size; ++i) {
  //   std::cout << ">> Round: " << i + 1 << ", Label Value: " << gt_labels[i] << ", Label Num: " << P_b[gt_labels[i]].size() << std::endl;
  //   for (const uint32_t ef : beam) {
  //     std::cout << std::endl << ">>> Filtered HNSW M = " << m << ", " << "Ef = " << ef << std::endl;
  //     // run_filtered_hnsw<U>(dim, ml, m, efc, alpha, batch_base, k, ef, size_init, size_step, size_max, ps, q, F_b, P_b, F_q, filtered_gt);
  //     run_filtered_hnsw_search(filtered_hnsw_index, entrance, k, ef, q, Fqs[i], P_b, filtered_gts[i]);
  //   }
  // }

  // std::cout << ">>> HNSW Post Processing >>>" << std::endl;
  // // const auto &hnsw_index = run_hnsw_insert<U>(dim, ml, m, efc, alpha, batch_base, size_init, size_step, size_max, ps);
  // for (size_t i = 0; i < spec_labels_size; ++i) {
  //   std::cout << ">> Round: " << i + 1 << ", Label Value: " << gt_labels[i] << ", Label Num: " << P_b[gt_labels[i]].size() << std::endl;
  //   for (const uint32_t ef : beam) {
  //     std::cout << std::endl << ">>> HNSW Post M = " << m << ", " << "Ef = " << ef << std::endl;
  //     // run_post_hnsw<U>(dim, ml, m, efc, alpha, batch_base, k, ef, size_init, size_step, size_max, ps, q, F_b, F_q, filtered_gt);
  //     run_post_hnsw(hnsw_index, k, ef, q, F_b, Fqs[i], filtered_gts[i]);
  //   }
  // }
}

int main(int argc, char **argv) {
  for (int i = 0; i < argc; ++i) printf("%s ", argv[i]);
  putchar('\n');

  commandLine parameter(argc, argv,
                        "-type <elemType> -dist <distance>"
                        "-ml <ml> -m <m> -efc <ef_construction> -alpha <alpha> "
                        "-r <filtered_R> -R <stitched_large_R> -stitched_R <stitched_small_R> -beam <beam_size>"
                        "-in <baseset> -q <queries> "
                        "-init <init_size> -step <step_size> -max <max_size> -b <batch>"
                        "-k <recall@k> -rounds <query_rounds> -gts <specificity_label_set>"
                        "-lb <base_labelset> -lq <query_labelset>");

  const char *dist_func = parameter.getOptionValue("-dist");
  // const char *vamana_type = parameter.getOptionValue("-vt");
  // const char *file_label_in = parameter.getOptionValue("-in");
  // const char *file_label_query = parameter.getOptionValue("-q");
  // if (strcmp(vamana_type, "vamana")) {
  //   assert(file_label_in && file_label_query);
  // }

  auto run_test_helper = [&](auto type) {  // emulate a generic lambda in C++20
    using T = decltype(type);
    if (!strcmp(dist_func, "L2")) {
      run_test<desc<descr_l2<T>>>(parameter);
    } else if (!strcmp(dist_func, "angular")) {
      run_test<desc<descr_ang<T>>>(parameter);
    }
    /*
    else if(!strcmp(dist_func,"ndot"))
            run_test<desc<descr_ndot<T>>>(parameter);
    */
    // else throw std::invalid_argument("Unsupported distance type");
  };

  const char *type = parameter.getOptionValue("-type");

  if (!strcmp(type, "float")) {
    run_test_helper(float{});
  } else if (!strcmp(type, "uint8")) {
    run_test_helper(uint8_t{});
  }
  /*
  else if(!strcmp(type,"int8"))
          run_test_helper(int8_t{});
  else if(!strcmp(type,"float"))
          run_test_helper(float{});
  */
  else {
    throw std::invalid_argument("Unsupported element type");
  }
  return 0;
}
