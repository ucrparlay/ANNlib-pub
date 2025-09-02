#include <math.h>
#include <parlay/io.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <chrono>

struct commandLine {
  int argc;
  char** argv;
  std::string comLine;
  commandLine(int _c, char** _v, std::string _cl) : argc(_c), argv(_v), comLine(_cl) {
    if (getOption("-h") || getOption("-help")) badArgument();
  }

  commandLine(int _c, char** _v) : argc(_c), argv(_v), comLine("bad arguments") {}

  void badArgument() {
    std::cout << "usage: " << argv[0] << " " << comLine << std::endl;
    exit(0);
  }

  // get an argument
  // i is indexed from the last argument = 0, second to last indexed 1, ..
  char* getArgument(int i) {
    if (argc < 2 + i) badArgument();
    return argv[argc - 1 - i];
  }

  // looks for two filenames
  std::pair<char*, char*> IOFileNames() {
    if (argc < 3) badArgument();
    return std::pair<char*, char*>(argv[argc - 2], argv[argc - 1]);
  }

  std::pair<size_t, char*> sizeAndFileName() {
    if (argc < 3) badArgument();
    return std::pair<size_t, char*>(std::atoi(argv[argc - 2]), (char*)argv[argc - 1]);
  }

  bool getOption(std::string option) {
    for (int i = 1; i < argc; i++)
      if ((std::string)argv[i] == option) return true;
    return false;
  }

  char* getOptionValue(std::string option) {
    for (int i = 1; i < argc - 1; i++)
      if ((std::string)argv[i] == option) return argv[i + 1];
    return NULL;
  }

  std::string getOptionValue(std::string option, std::string defaultValue) {
    for (int i = 1; i < argc - 1; i++)
      if ((std::string)argv[i] == option) return (std::string)argv[i + 1];
    return defaultValue;
  }

  long getOptionLongValue(std::string option, long defaultValue) {
    for (int i = 1; i < argc - 1; i++)
      if ((std::string)argv[i] == option) {
        long r = atol(argv[i + 1]);
        if (r < 0) badArgument();
        return r;
      }
    return defaultValue;
  }

  int getOptionIntValue(std::string option, int defaultValue) {
    for (int i = 1; i < argc - 1; i++)
      if ((std::string)argv[i] == option) {
        int r = atoi(argv[i + 1]);
        if (r < 0) badArgument();
        return r;
      }
    return defaultValue;
  }

  double getOptionDoubleValue(std::string option, double defaultValue) {
    for (int i = 1; i < argc - 1; i++)
      if ((std::string)argv[i] == option) {
        double val;
        if (sscanf(argv[i + 1], "%lf", &val) == EOF) {
          badArgument();
        }
        return val;
      }
    return defaultValue;
  }
};

class ZipfDistribution {
 public:
  ZipfDistribution(uint64_t num_points, uint32_t num_labels)
      : num_labels(num_labels),
        num_points(num_points),
        uniform_zero_to_one(std::uniform_real_distribution<>(0.0, 1.0)) {}

  std::unordered_map<uint32_t, uint32_t> createDistributionMap() {
    std::unordered_map<uint32_t, uint32_t> map;
    uint32_t primary_label_freq = (uint32_t)ceil(num_points * distribution_factor);
    for (uint32_t i{1}; i < num_labels + 1; i++) {
      map[i] = (uint32_t)ceil(primary_label_freq / i);
    }
    return map;
  }

  int writeDistribution(std::ofstream& outfile) {
    auto distribution_map = createDistributionMap();
    for (uint32_t i{0}; i < num_points; i++) {
      bool label_written = false;
      for (auto it = distribution_map.cbegin(); it != distribution_map.cend(); it++) {
        auto label_selection_probability =
            std::bernoulli_distribution(distribution_factor / (double)it->first);
        if (label_selection_probability(rand_engine) && distribution_map[it->first] > 0) {
          if (label_written) {
            outfile << ',';
          }
          outfile << it->first;
          label_written = true;
          // remove label from map if we have used all labels
          distribution_map[it->first] -= 1;
        }
      }
      if (!label_written) {
        outfile << 0;
      }
      if (i < num_points - 1) {
        outfile << std::endl;
      }
    }
    return 0;
  }

  int writeDistribution(std::string filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
      std::cerr << "Error: could not open output file " << filename << std::endl;
      return -1;
    }
    writeDistribution(outfile);
    outfile.close();
    return 0;
  }

 private:
  const uint32_t num_labels;
  const uint64_t num_points;
  const double distribution_factor = 0.7;
  std::knuth_b rand_engine;
  const std::uniform_real_distribution<double> uniform_zero_to_one;
};

int main(int argc, char** argv) {
  commandLine P(argc, argv,
                "[-output_file <of>] [-num_points <np>]"
                "[-num_labels <nl> ] [-distribution_type <dt>]");

  char* ofc = P.getOptionValue("-output_file");
  if (ofc == NULL) P.badArgument();
  std::string output_file = std::string(ofc);
  uint64_t num_points = P.getOptionIntValue("-num_points", -1);
  if (num_points <= 0) P.badArgument();
  uint32_t num_labels = P.getOptionIntValue("-num_labels", -1);
  if (num_labels <= 0) P.badArgument();
  char* dist_type = P.getOptionValue("-distribution_type");
  if (dist_type == NULL) P.badArgument();
  std::string distribution_type = std::string(dist_type);

  std::cout << "Generating synthetic labels for " << num_points << " points with " << num_labels
            << " unique labels" << std::endl;

  try {
    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
      std::cerr << "Error: could not open output file " << output_file << std::endl;
      return -1;
    }

    // add total points to the first line
    outfile << num_points << std::endl;

    if (distribution_type == "zipf") {
      ZipfDistribution zipf(num_points, num_labels);
      zipf.writeDistribution(outfile);
    } else if (distribution_type == "random") {
      // std::random_device rd;
      // std::mt19937 gen(rd());
      // std::uniform_int_distribution<> large(10, 20);
      // std::uniform_int_distribution<> small(1, 2);
      // std::mt19937 rng((unsigned int) std::chrono::steady_clock::now().time_since_epoch().count());
      for (size_t i = 0; i < num_points; i++) {
        bool label_written = false;
      //   // auto rand_frac = dis(gen);
      //   if (rand() < (RAND_MAX / 8)) {
      //     size_t len = large(gen);
      //     size_t begin = std::max<size_t>(1, rng() % (num_labels - len + 1));
      //     for (size_t j = 0; j < len; j++) {
      //       if (label_written) {
      //         outfile << ',';
      //       }
      //       outfile << begin + j;
      //       label_written = true;
      //     }
      //   } else {
      //     size_t len = small(gen);
      //     size_t begin = std::max<size_t>(1, rng() % (num_labels - len));
      //     for (size_t j = 0; j < len; j++) {
      //       if (label_written) {
      //         outfile << ',';
      //       }
      //       outfile << begin + j;
      //       label_written = true;
      //     }
      //   }
        for (size_t j = 1; j <= num_labels; j++) {
          if (rand() < (RAND_MAX / 2)) {
            if (label_written) {
              outfile << ',';
            }
            outfile << j;
            label_written = true;
          }
        }
        if (!label_written) {
          outfile << 0;
        }
        if (i < num_points - 1) {
          outfile << std::endl;
        }
      }
    } else if (distribution_type == "one_per_point") {
      std::random_device rd;                                 // obtain a random number from hardware
      std::mt19937 gen(rd());                                // seed the generator
      std::uniform_int_distribution<> distr(1, num_labels);  // define the range

      for (size_t i = 0; i < num_points; i++) {
        outfile << distr(gen);
        if (i != num_points - 1) outfile << std::endl;
      }
    }
    if (outfile.is_open()) {
      outfile.close();
    }

    std::cout << "Labels written to " << output_file << std::endl;
  } catch (const std::exception& ex) {
    std::cerr << "Label generation failed: " << ex.what() << std::endl;
    return -1;
  }
  return 0;
}