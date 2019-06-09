print(78)

# Complete the kangaroo function below.
def kangaroo(x1, v1, x2, v2):
    while x1 < x2:
        x1 += v1
        x2 += v2
        flag = 'NO'
        if x1 == x2:
            flag == 'YES'
            break
    return flag


f = kangaroo(0, 3, 4, 2)
print(f)