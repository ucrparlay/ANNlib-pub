import argparse
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class ZipfDistribution:
    def __init__(self, num_points: int, cardinality: int, distribution_factor: float = 0.7):
        self.cardinality = cardinality
        self.num_points = num_points
        self.distribution_factor = distribution_factor
        self.rand_engine = random.Random()
        self.counter = [0] * cardinality
        self.total = 0.

    """
    def create_distribution_map(self):
        distribution_map = {}
        primary_label_freq = math.ceil(
            self.num_points * self.distribution_factor)
        for i in range(1, self.cardinality + 1):
            distribution_map[i] = math.ceil(primary_label_freq / i)
        return distribution_map
    """

    def write_distribution(self, outfile):
        # distribution_map = self.create_distribution_map()
        for i in tqdm(range(self.num_points), desc="Generating"):
            label_written = False
            # for label, count in distribution_map.items():
            for label in range(1, self.cardinality + 1):
                label_selection_probability = self.distribution_factor / label
                if self.rand_engine.random() < label_selection_probability: # and distribution_map[label] > 0:
                    if label_written:
                        outfile.write(',')
                    outfile.write(str(label))
                    label_written = True
                    self.counter[label - 1] += 1
                    self.total += 1.
                    # distribution_map[label] -= 1
            if not label_written:
                outfile.write('0')
            if i < self.num_points - 1:
                outfile.write('\n')

        print(f"average label num: {self.total / self.num_points}")
        print("percentile")
        for i, c in enumerate(self.counter):
            print(f"label {i + 1} with {c / self.num_points} pc.")


def gen_random_labels(chunk_size: int, cardinality: int, ):
    """
    chunk_size: int: generating chunk size
    cardinality: int: num. of unique labels
    """
    result = []
    label_probs = {i: random.random() ** 3 for i in range(1, cardinality + 1)}
    labels = list(label_probs.keys())

    for _ in range(chunk_size):
        sample = []
        for label in labels:
            if random.random() < label_probs[label]:
                sample.append(label)
        
        if not sample:
            sample.append(random.choice(labels))
        result.append(','.join(map(str, sample)))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic label distribution for dataset points")
    parser.add_argument("-o", "--output_file",
                        required=True, help="Filename for saving the label file")
    parser.add_argument("-n", "--num_points", required=True,
                        type=int, help="Number of points in dataset")
    parser.add_argument("-card", "--cardinality",
                        required=True, type=int, help="Number of unique labels")
    parser.add_argument("-dist", "--distribution_type",
                        default="random", help="Distribution type (random/zipf/one_per_point)")
    parser.add_argument("-zipf_factor", "--distribution_factor",
                        default=0.7, type=float, help="Distribution factor for zipf distribution")

    args = parser.parse_args()

    if args.num_points <= 0:
        print("Error: num_points must be greater than 0")
        return -1

    print(
        f"Generating synthetic labels for {args.num_points} points with {args.cardinality} unique labels")

    with open(args.output_file, 'w') as outfile:
        if args.distribution_type == "zipf":
            zipf = ZipfDistribution(
                args.num_points, args.cardinality, args.distribution_factor)
            zipf.write_distribution(outfile)
        elif args.distribution_type == "random":
            threshold = 1_000_000
            if args.num_points // threshold <= 1:
                labels = gen_random_labels(
                    args.num_points, args.cardinality, )
            else:
                num_proc = cpu_count()
                chunk_size = args.num_points // num_proc
                chunks = [(chunk_size, args.cardinality, )
                          for _ in range(num_proc)]
                # adjust remainings
                if args.num_points % chunk_size != 0:
                    chunks[-1] = (chunks[-1][0] + (args.num_points % chunk_size), args.cardinality, )

                with Pool(processes=num_proc) as pool:
                    results = list(pool.starmap(gen_random_labels, chunks))

                # Flatten
                print(f"Sync flattening ...")
                labels = [label for chunk in results for label in chunk]

            print("Writing ...")
            outfile.write('\n'.join(labels))

        elif args.distribution_type == "one_per_point":
            for i in tqdm(range(args.num_points), desc="Generating"):
                outfile.write(str(random.randint(1, args.cardinality)))
                if i < args.num_points - 1:
                    outfile.write('\n')

    print(f"Labels written to {args.output_file}")


if __name__ == "__main__":
    main()
