def solution(A, B):
    # The fution returns the number of bits
    binary = bin(A * B)
    setBits = [ones for ones in binary[2:] if ones == '1']
    return len(setBits)

f = solution (3, 7)
print(f)

