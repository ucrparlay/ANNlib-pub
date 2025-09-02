import struct
import numpy as np
from pathlib import Path
from collections import defaultdict
from utils.datasets import YFCC100MDataset


class MarcoProcessor:
    def __init__(self, base_dir: str):
        """
        Initialize file paths and load header information from the binary files.
        """
        self.marco_dir = Path(base_dir)
        self.marco_embed_dir = self.marco_dir / "embedding"
        self.marco_query_dir = self.marco_dir / "query"

        self.embedding_file = self.marco_embed_dir / "marco.768D.10M.euclidean.fbin"
        self.meta_file = self.marco_embed_dir / "meta.bin"
        self.metaidx_file = self.marco_embed_dir / "metaidx.bin"
        self.filter_file = self.marco_embed_dir / "marco.filter.base.10M.txt"

        self.marco_embed = defaultdict(int)
        self.labels_all = []

        self.N = None               # Number of points (from embedding data)
        self.meta = None            # memmap of meta.bin
        self.metaidx = None         # memmap of metaidx.bin

        # Read header information and create memory maps.
        self._read_headers()

    def _read_headers(self):
        """
        Read header information from the embedding, meta, and metaidx files.
        """
        with open(self.embedding_file, "rb") as f:
            self.N = struct.unpack('I', f.read(4))[0]
            dim = struct.unpack('I', f.read(4))[0]
            print(
                f"Dataset MS-MARCO in dimension {dim}, with distance euclidean, size: {self.N}")

        self.metaidx = np.memmap(
            self.metaidx_file,
            dtype=np.uint64,
            mode='r',
            offset=4,
        )

        self.meta = np.memmap(self.meta_file, dtype=np.uint8, mode='r')

    def process(self):
        """
        Process the metaidx data and build a list of labels for each point while updating
        the marco_embed dictionary with counts.
        """
        counter = defaultdict(int)
        avg_label_size = 0.
        max_label_size = -1
        min_label_size = 10000000

        for i in range(self.N):
            start = self.metaidx[i]
            end = self.metaidx[i + 1]
            labels = self.meta[start:end]
            self.labels_all.append(list(set(labels)))
            label_size = len(labels)
            max_label_size = max(max_label_size, label_size)
            min_label_size = min(min_label_size, label_size)
            avg_label_size += label_size

            for label in list(set(labels)):
                counter[label] += 1

        print(f"avg label size: {avg_label_size / self.N}\nmax_label_size: {max_label_size}\nmin_label_size: {min_label_size}\ncounter dict: {counter}")

    def write_output(self):
        """
        Write the processed labels to a text file.
        Each point's labels are written on a new line, separated by commas.
        """
        total_labels = 0
        cnt = 0
        with open(self.filter_file, "w") as f:
            for labels in self.labels_all:
                cnt += 1
                # Write the labels separated by commas.
                line = ",".join(str(label) for label in labels)
                total_labels += len(labels)
                # Write a newline after each point except the last.
                if cnt < len(self.labels_all):
                    f.write(line + "\n")
                else:
                    f.write(line)
        print(f"{len(self.labels_all)} points with {total_labels} labels are written to {self.filter_file}")

    def run(self):
        """
        Run the full processing: process the data and write the output.
        """
        self.process()
        self.write_output()


class YFCCProcessor:
    def __init__(self, base_dir: str):
        """
        Initialize file paths, dataset instance, and other containers.

        Args:
            base_dir (str): The base directory for the dataset.
        """
        self.base_dir = Path(base_dir)
        self.ds = YFCC100MDataset(base_dir=base_dir)
        print(self.ds)  # ds.__str__()

        self.dataset_metadata = self.ds.get_dataset_metadata()
        # self.query_metadata = self.ds.get_queries_metadata()

        self.labels_dir = self.base_dir
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.labels_dir / "yfcc.filter.base.txt"

        self.filter_dict = defaultdict(list)
        self.data = []  # list of labels per point.

    def process_metadata(self):
        """
        Process the dataset metadata to build a dictionary mapping from point index
        to its list of labels, and then create a complete list (data) of labels per point.
        Points that are missing in the dictionary are represented by a default label [0].
        """
        rows, cols = self.dataset_metadata.nonzero()

        for idx, v in zip(rows, cols):
            self.filter_dict[idx].append(v)

        total_points = self.dataset_metadata.shape[0]
        not_in_list = [i for i in range(
            total_points) if i not in self.filter_dict]

        # For indices that are missing in filter_dict, assign [0].
        it = 0
        for idx, labels in sorted(self.filter_dict.items()):
            while it < len(not_in_list) and idx > not_in_list[it]:
                self.data.append([0])
                it += 1
            self.data.append(labels)

        while len(self.data) < total_points:
            self.data.append([0])

        assert len(
            self.data) == total_points, f"Data length is {len(self.data)}, expected {total_points}."

    def write_output(self):
        """
        Write the processed labels to a text file.
        Each point's labels are written on a new line, with multiple labels separated by commas.
        """
        total_labels = 0
        cnt = 0

        with open(self.file_path, "w") as f:
            for labels in self.data:
                cnt += 1
                label_written = False
                # Write labels separated by commas.
                for label in labels:
                    if label_written:
                        f.write(',')
                    f.write(str(label))
                    label_written = True
                    total_labels += 1
                # Add newline for every point except the last.
                if cnt < len(self.data):
                    f.write('\n')

        print(
            f"{len(self.data)} points with {total_labels} labels are written to {self.file_path}")

    def run(self):
        """
        Run the full pipeline: process the metadata and write the output file.
        """
        self.process_metadata()
        self.write_output()


if __name__ == '__main__':
    data_dir = Path("/data/jsu068")

    # MS-MARCO
    processor = MarcoProcessor(base_dir=data_dir / "marco")
    processor.run()

    # YFCC
    processor = YFCCProcessor(base_dir=data_dir / "yfcc")
    processor.run()
