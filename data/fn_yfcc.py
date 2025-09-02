def process_yfcc(base_dir: str):
    ds = YFCC100MDataset(base_dir=base_dir)
    print(ds.__str__())
    dataset_metadata = ds.get_dataset_metadata()
    # query_metadata = ds.get_queries_metadata()
    rows, cols = dataset_metadata.nonzero()
    filter_dict = defaultdict(list)
    for idx, v in zip(rows, cols):
        filter_dict[idx].append(v)
    not_in_list = [i for i in range(
        dataset_metadata.shape[0]) if i not in filter_dict]

    labels_dir = data_dir / "yfcc"
    file_path = os.path.join(labels_dir, "yfcc.filter.base.txt")
    # label_type = np.int32

    data = []
    it = 0

    for idx, labels in filter_dict.items():
        while it < len(not_in_list) and idx > not_in_list[it]:
            data.append([0])
            it += 1
        data.append(labels)

    assert (len(data) == 10000000)
    total_labels = 0

    cnt = 0
    with open(file_path, "w") as f:
        for labels in data:
            cnt += 1
            label_written = False
            for label in labels:
                if label_written:
                    f.write(',')
                f.write(str(label))
                label_written = True
                total_labels += 1
            if cnt < len(data):
                f.write('\n')

    print(f"{len(data)} points with {total_labels} labels are written to {file_path}")