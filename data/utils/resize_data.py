import struct
import argparse

def resize_binary(input_file: str, output_file: str, dtype: str, new_num_points: int):
    # Read the original binary file
    if dtype == "float":
        esize = 4
        sformat = 'f'
    elif dtype == "uint":
        esize = 4
        sformat = 'I'
    
    with open(input_file, 'rb') as f:
        # Read the first two uint32 values (total number and dimension)
        total_num = struct.unpack(sformat, f.read(4))[0]
        dim = struct.unpack(sformat, f.read(4))[0]

        print(f"#Total number: {total_num}, #Dim: {dim}, #Type: {dtype}")

        # Make sure the new number of points is less than or equal to the original total number
        if new_num_points > total_num:
            raise ValueError("New number of points is larger than the original dataset size.")

        # Read the points data for the new dataset size (only first new_num_points points)
        data = f.read(new_num_points * dim * esize)  # Each element is a 4-byte float

    # Write to the output file
    with open(output_file, 'wb') as f:
        # Write new number of points and original dimension
        f.write(struct.pack(sformat, new_num_points))
        f.write(struct.pack(sformat, dim))
        
        print(f"#Total number: {new_num_points}, #Dim: {dim}")
        
        # Write only the first new_num_points * dim points
        f.write(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize a binary dataset.")
    parser.add_argument("input_file", type=str, help="Path to the original binary file.")
    parser.add_argument("output_file", type=str, help="Path to save the resized binary file.")
    parser.add_argument("dtype", choices=["float", "uint"], help="Point data type.")
    parser.add_argument(
        "new_num_points",
        type=int,
        default=10_000_000,
        help="New number of points to keep in the dataset (default: 10,000,000)."
    )

    args = parser.parse_args()

    resize_binary(args.input_file, args.output_file, args.dtype, args.new_num_points)
