import numpy as np

arr = np.random.randint(1, 51, size=(5, 4))
print("Array:\n", arr)

anti_diag = [arr[i, -1 - i] for i in range(min(arr.shape))]
print("Anti-diagonal elements:", anti_diag)

row_max = np.max(arr, axis=1)
print("Max in each row:", row_max)

mean_val = np.mean(arr)
less_equal_mean = arr[arr <= mean_val]
print("Elements <= mean (%.2f):" % mean_val, less_equal_mean)

def numpy_boundary_traversal(matrix):
    result = []
    if matrix.size == 0:
        return result
    rows, cols = matrix.shape
    result.extend(matrix[0, :])                   # top row
    result.extend(matrix[1:rows-1, -1])           # right col (excluding corners)
    if rows > 1:
        result.extend(matrix[-1, ::-1])           # bottom row reversed
    if cols > 1:
        result.extend(matrix[rows-2:0:-1, 0])     # left col reversed
    return list(result)

print("Boundary traversal:", numpy_boundary_traversal(arr))
