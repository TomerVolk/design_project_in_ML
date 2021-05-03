

def read_sample(path):
    with open(path, "r") as f:
        count = 0
        out_str = ""
        for row in f:
            out_str += row + "\n"
            count += 1
            if count == 5:
                break
    pass


if __name__ == '__main__':
    read_sample("C:\\Users\\tomer\\Downloads\\track2\\training.txt")
