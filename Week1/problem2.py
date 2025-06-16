import numpy as np

arr = np.random.uniform(0, 10, 20)
arr = np.round(arr, 2)
print("Array:", arr)

print("Min:", np.min(arr))
print("Max:", np.max(arr))
print("Median:", np.median(arr))

arr_transformed = np.where(arr < 5, arr ** 2, arr)
print("Transformed Array:", arr_transformed)

def numpy_alternate_sort(array):
    sorted_arr = np.sort(array)
    result = []
    left, right = 0, len(sorted_arr) - 1    #Two pointer
    while left <= right:
        result.append(sorted_arr[left])
        if left != right:
            result.append(sorted_arr[right])
        left += 1
        right -= 1
    return np.array(result)

print("Alternate sorted array:", numpy_alternate_sort(arr))
