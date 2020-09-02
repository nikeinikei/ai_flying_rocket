import numpy as np


def main():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    data2 = np.pad(data, [(0, 0), (0, 0)])
    print(data2)

    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[7, 8, 9]])
    z = np.append(x, y, axis=0)
    print(z)


if __name__ == "__main__":
    main()