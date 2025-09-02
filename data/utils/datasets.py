import os
import gzip
import shutil

import numpy as np

from utils.dataset_io import (
    xbin_mmap, download_accelerated, download, sanitize,
    knn_result_read, read_sparse_matrix,
)


BASEDIR = "/data/jsu068/"


class Dataset():
    def prepare(self):
        """
        Download and prepare dataset, queries, groundtruth.
        """
        pass

    def get_dataset_fn(self):
        """
        Return filename of dataset file.
        """
        pass

    def get_dataset(self):
        """
        Return memmapped version of the dataset.
        """
        pass

    def get_dataset_iterator(self, bs=512, split=(1, 0)):
        """
        Return iterator over blocks of dataset of size at most 512.
        The split argument takes a pair of integers (n, p) where p = 0..n-1
        The dataset is split in n shards, and the iterator returns only shard #p
        This makes it possible to process the dataset independently from several
        processes / threads.
        """
        pass

    def get_queries(self):
        """
        Return (nq, d) array containing the nq queries.
        """
        pass

    def get_private_queries(self):
        """
        Return (private_nq, d) array containing the private_nq private queries.
        """
        pass

    def get_groundtruth(self, k=None):
        """
        Return (nq, k) array containing groundtruth indices
        for each query."""
        pass

    def search_type(self):
        """
        "knn" or "range" or "knn_filtered"
        """
        pass

    def distance(self):
        """
        "euclidean" or "ip" or "angular"
        """
        pass

    def data_type(self):
        """
        "dense" or "sparse"
        """
        pass

    def default_count(self):
        """ number of neighbors to return """
        return 10

    def short_name(self):
        return f"{self.__class__.__name__}-{self.nb}"

    def __str__(self):
        return (
            f"Dataset {self.__class__.__name__} in dimension {self.d}, with distance {self.distance()}, "
            f"search_type {self.search_type()}, size: Q {self.nq} B {self.nb}")


#############################################################################
# Datasets for the competition
##############################################################################


class DatasetCompetitionFormat(Dataset):
    """
    Dataset in the native competition format, that is able to read the
    files in the https://big-ann-benchmarks.com/ page.
    The constructor should set all fields. The functions below are generic.

    For the 10M versions of the dataset, the database files are downloaded in
    part and stored with a specific suffix. This is to avoid having to maintain
    two versions of the file.
    """

    def prepare(self, skip_data=False, original_size=10**9):
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        # start with the small ones...
        for fn in [self.qs_fn, self.gt_fn]:
            if fn is None:
                continue
            if fn.startswith("https://"):
                sourceurl = fn
                outfile = os.path.join(self.basedir, fn.split("/")[-1])
            else:
                sourceurl = os.path.join(self.base_url, fn)
                outfile = os.path.join(self.basedir, fn)
            if os.path.exists(outfile):
                print("file %s already exists" % outfile)
                continue
            download(sourceurl, outfile)

        # private qs url
        if self.private_qs_url:
            outfile = os.path.join(
                self.basedir, self.private_qs_url.split("/")[-1])
            if os.path.exists(outfile):
                print("file %s already exists" % outfile)
            else:
                download(self.private_qs_url, outfile)

        # private gt url
        if self.private_gt_url:
            outfile = os.path.join(
                self.basedir, self.private_gt_url.split("/")[-1])
            if os.path.exists(outfile):
                print("file %s already exists" % outfile)
            else:
                download(self.private_gt_url, outfile)

        if skip_data:
            return

        fn = self.ds_fn
        sourceurl = os.path.join(self.base_url, fn)
        outfile = os.path.join(self.basedir, fn)
        if os.path.exists(outfile):
            print("file %s already exists" % outfile)
            return
        if self.nb == 10**9:
            download_accelerated(sourceurl, outfile)
        else:
            # download cropped version of file
            file_size = 8 + self.d * self.nb * np.dtype(self.dtype).itemsize
            outfile = outfile + '.crop_nb_%d' % self.nb
            if os.path.exists(outfile):
                print("file %s already exists" % outfile)
                return
            download(sourceurl, outfile, max_size=file_size)
            # then overwrite the header...
            header = np.memmap(outfile, shape=2, dtype='uint32', mode="r+")

            assert header[0] == original_size
            assert header[1] == self.d
            header[0] = self.nb

    def get_dataset_fn(self):
        fn = os.path.join(self.basedir, self.ds_fn)
        if self.nb != 10**9:
            fn += '.crop_nb_%d' % self.nb
        if os.path.exists(fn):
            return fn
        else:
            raise RuntimeError("file not found")

    def get_dataset_iterator(self, bs=512, split=(1, 0)):
        nsplit, rank = split
        i0, i1 = self.nb * rank // nsplit, self.nb * (rank + 1) // nsplit
        filename = self.get_dataset_fn()
        x = xbin_mmap(filename, dtype=self.dtype, maxn=self.nb)
        assert x.shape == (self.nb, self.d)
        for j0 in range(i0, i1, bs):
            j1 = min(j0 + bs, i1)
            yield sanitize(x[j0:j1])

    def get_data_in_range(self, start, end):
        assert start >= 0
        assert end <= self.nb
        filename = self.get_dataset_fn()
        x = xbin_mmap(filename, dtype=self.dtype, maxn=self.nb)
        return x[start:end]

    def search_type(self):
        return "knn"

    def data_type(self):
        return "dense"

    def get_groundtruth(self, k=None):
        assert self.gt_fn is not None
        fn = self.gt_fn.split("/")[-1]   # in case it's a URL
        assert self.search_type() in ("knn", "knn_filtered")

        I, D = knn_result_read(os.path.join(self.basedir, fn))
        assert I.shape[0] == self.nq
        if k is not None:
            assert k <= 100
            I = I[:, :k]
            D = D[:, :k]
        return I, D

    def get_dataset(self):
        assert self.nb <= 10**7, "dataset too large, use iterator"
        slice = next(self.get_dataset_iterator(bs=self.nb))
        return sanitize(slice)

    def get_queries(self):
        filename = os.path.join(self.basedir, self.qs_fn)
        x = xbin_mmap(filename, dtype=self.dtype)
        assert x.shape == (self.nq, self.d)
        return sanitize(x)

    def get_private_queries(self):
        filename = os.path.join(self.basedir, self.qs_private_fn)
        x = xbin_mmap(filename, dtype=self.dtype)
        assert x.shape == (self.private_nq, self.d)
        return sanitize(x)

    def get_private_groundtruth(self, k=None):
        assert self.private_gt_fn is not None
        assert self.search_type() in ("knn", "knn_filtered")

        I, D = knn_result_read(os.path.join(self.basedir, self.private_gt_fn))
        assert I.shape[0] == self.private_nq
        if k is not None:
            assert k <= 100
            I = I[:, :k]
            D = D[:, :k]
        return I, D


class BillionScaleDatasetCompetitionFormat(DatasetCompetitionFormat):

    def get_dataset_fn(self):
        fn = os.path.join(self.basedir, self.ds_fn)
        if self.nb != 10**9:
            fn += '.crop_nb_%d' % self.nb
        if os.path.exists(fn):
            return fn
        else:
            raise RuntimeError("file %s not found" % fn)


subset_url = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/"


class BigANNDataset(DatasetCompetitionFormat):
    def __init__(self, nb_M=1000):
        self.nb_M = nb_M
        self.nb = 10**6 * nb_M
        self.d = 128
        self.nq = 10000
        self.dtype = "uint8"
        self.ds_fn = "base.1B.u8bin"
        self.qs_fn = "query.public.10K.u8bin"
        self.gt_fn = (
            "GT.public.1B.ibin" if self.nb_M == 1000 else
            subset_url + "GT_100M/bigann-100M" if self.nb_M == 100 else
            subset_url + "GT_10M/bigann-10M" if self.nb_M == 10 else
            None
        )
        # self.gt_fn = "https://comp21storage.z5.web.core.windows.net/comp21/bigann/public_query_gt100.bin" if self.nb == 10**9 else None
        self.base_url = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/"
        self.basedir = os.path.join(BASEDIR, "bigann")

        self.private_nq = 10000
        # https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.private.799253207.10K.u8bin"
        self.private_qs_url = ""
        # https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_1B_final_2bf4748c7817/bigann-1B.bin"
        self.private_gt_url = ""

    def distance(self):
        return "euclidean"


class YFCC100MDataset(DatasetCompetitionFormat):
    """ the 2023 competition """

    def __init__(self, base_dir: str = BASEDIR, filtered=True, dummy=False):
        self.filtered = filtered
        nb_M = 10
        self.nb_M = nb_M
        self.nb = 10**6 * nb_M
        self.d = 192
        self.nq = 100000
        self.dtype = "uint8"
        private_key = 2727415019
        self.gt_private_fn = ""

        if dummy:
            # for now it's dummy because we don't have the descriptors yet
            self.ds_fn = "dummy2.base.10M.u8bin"
            self.qs_fn = "dummy2.query.public.100K.u8bin"
            self.qs_private_fn = "dummy2.query.private.%d.100K.u8bin" % private_key
            self.ds_metadata_fn = "dummy2.base.metadata.10M.spmat"
            self.qs_metadata_fn = "dummy2.query.metadata.public.100K.spmat"
            self.qs_private_metadata_fn = "dummy2.query.metadata.private.%d.100K.spmat" % private_key
            if filtered:
                # no subset as the database is pretty small.
                self.gt_fn = "dummy2.GT.public.ibin"
            else:
                self.gt_fn = "dummy2.unfiltered.GT.public.ibin"

        else:
            # with Zilliz' CLIP descriptors
            self.ds_fn = "base.10M.u8bin"
            self.qs_fn = "query.public.100K.u8bin"
            self.qs_private_fn = "query.private.%d.100K.u8bin" % private_key
            self.ds_metadata_fn = "base.metadata.10M.spmat"
            self.qs_metadata_fn = "query.metadata.public.100K.spmat"
            self.qs_private_metadata_fn = "query.metadata.private.%d.100K.spmat" % private_key
            if filtered:
                # no subset as the database is pretty small.
                self.gt_fn = "GT.public.ibin"
                self.gt_private_fn = "GT.private.%d.ibin" % private_key
            else:
                self.gt_fn = "unfiltered.GT.public.ibin"

            self.private_gt_fn = "GT.private.%d.ibin" % private_key

            # data is uploaded but download script not ready.
        self.base_url = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/"
        self.basedir = base_dir

        self.private_nq = 100000
        self.private_qs_url = self.base_url + self.qs_private_fn
        self.private_gt_url = self.base_url + self.gt_private_fn

        self.metadata_base_url = self.base_url + self.ds_metadata_fn
        self.metadata_queries_url = self.base_url + self.qs_metadata_fn
        self.metadata_private_queries_url = self.base_url + self.qs_private_metadata_fn

    def prepare(self, skip_data=False):
        super().prepare(skip_data, 10**7)
        for fn in (self.metadata_base_url, self.metadata_queries_url,
                   self.metadata_private_queries_url):
            if fn:
                outfile = os.path.join(self.basedir, fn.split("/")[-1])
                if os.path.exists(outfile):
                    print("file %s already exists" % outfile)
                else:
                    download(fn, outfile)

    def get_dataset_metadata(self):
        return read_sparse_matrix(os.path.join(self.basedir, self.ds_metadata_fn))

    def get_queries_metadata(self):
        return read_sparse_matrix(os.path.join(self.basedir, self.qs_metadata_fn))

    def get_private_queries_metadata(self):
        return read_sparse_matrix(os.path.join(self.basedir, self.qs_private_metadata_fn))

    def distance(self):
        return "euclidean"

    def search_type(self):
        if self.filtered:
            return "knn_filtered"
        else:
            return "knn"


def _strip_gz(filename):
    if not filename.endswith('.gz'):
        raise RuntimeError(
            f"expected a filename ending with '.gz'. Received: {filename}")
    return filename[:-3]


def _gunzip_if_needed(filename):
    if filename.endswith('.gz'):
        print('unzipping', filename, '...', end=" ", flush=True)

        with gzip.open(filename, 'rb') as f_in, open(_strip_gz(filename), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        os.remove(filename)
        print('done.')


class OpenAIEmbedding1M(DatasetCompetitionFormat):
    def __init__(self, query_selection_random_seed):
        self.seed = query_selection_random_seed
        self.basedir = os.path.join(BASEDIR, "openai-embedding-1M")
        self.d = 1536
        self.nb = 1000000
        self.nq = 100000
        self.ds_fn = "base1m.fbin"
        self.qs_fn = "queries_100k.fbin"
        self.gt_fn = "gt_1m_100k.bin"

    def prepare(self, skip_data=False):
        from data.utils.datasets import load_dataset
        import tqdm
        import numpy as np
        import faiss

        os.makedirs(self.basedir, exist_ok=True)

        print("Downloading dataset...")
        dataset = load_dataset(
            "KShivendu/dbpedia-entities-openai-1M", split="train")

        print("Converting to competition format...")
        data = []
        for row in tqdm.tqdm(dataset, desc="Collecting vectors", unit="vector"):
            data.append(np.array(row["openai"], dtype=np.float32))
        data = np.array(data)
        with open(os.path.join(self.basedir, self.ds_fn), "wb") as f:
            np.array(data.shape, dtype=np.uint32).tofile(f)
            data.tofile(f)

        print(f"Selecting queries using random seed: {self.seed}...")
        rand = np.random.RandomState(self.seed)
        rand.shuffle(data)
        queries = data[:100000]
        with open(os.path.join(self.basedir, self.qs_fn), "wb") as f:
            np.array(queries.shape, dtype=np.uint32).tofile(f)
            queries.tofile(f)

        k = 100
        batch_size = 1000
        print(
            f"Computing groundtruth for k = {k} using batch_size = {batch_size}...")
        index = faiss.IndexFlatIP(data.shape[1])
        index.add(data)
        ids = np.zeros((len(queries), k), dtype=np.uint32)
        distances = np.zeros((len(queries), k), dtype=np.float32)
        for i in tqdm.tqdm(
            range(0, len(queries), batch_size),
            total=len(queries) // batch_size,
            desc="Brute force scan using FAISS",
            unit=" batch",
        ):
            D, I = index.search(queries[i: i + batch_size], k)
            ids[i: i + batch_size] = I
            distances[i: i + batch_size] = D
        with open(os.path.join(self.basedir, self.gt_fn), "wb") as f:
            np.array(ids.shape, dtype=np.uint32).tofile(f)
            ids.tofile(f)
            distances.tofile(f)

        print(f"Done. Dataset files are in {self.basedir}.")

    def search_type(self):
        """
        "knn" or "range" or "knn_filtered"
        """
        return "knn"

    def distance(self):
        """
        "euclidean" or "ip" or "angular"
        """
        return "eculidean"

    def data_type(self):
        """
        "dense" or "sparse"
        """
        return "dense"

    def default_count(self):
        """number of neighbors to return"""
        return 10

    def short_name(self):
        return f"{self.__class__.__name__}-{self.nb}"


class MSTuringANNS(BillionScaleDatasetCompetitionFormat):
    def __init__(self, nb_M=1000):
        self.nb_M = nb_M
        self.nb = 2599968  # 10**6 * nb_M
        self.d = 100
        self.nq = 100000
        self.dtype = "float32"
        self.ds_fn = "base1b.fbin"
        self.qs_fn = "query100K.fbin"
        self.gt_fn = (
            "query_gt100.bin" if self.nb_M == 1000 else
            # back up subset_url + "GT_100M/msturing-100M"
            "msturing-gt-100M" if self.nb_M == 100 else
            # back up subset_url + "GT_100M/msturing-10M"
            "msturing-gt-10M" if self.nb_M == 10 else
            "msturing-gt-1M" if self.nb_M == 1 else
            None
        )
        self.base_url = "https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/"
        self.basedir = os.path.join(BASEDIR, "MSTuringANNS")

        self.private_nq = 10000
        self.private_qs_url = "https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/testQuery10K.fbin"
        self.private_gt_url = "https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/gt100-private10K-queries.bin"

        self.private_nq_large = 99605
        self.private_qs_large_url = "https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/testQuery99605.fbin"
        self.private_gt_large_url = "https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/gt100-private99605-queries.bin"

    def distance(self):
        return "euclidean"
