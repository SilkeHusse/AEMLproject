cols = 10
rows = 20

board = [
        [ 0 for x in range(cols) ]
        for y in range(rows)
    ]
board += [[ 1 for x in range(cols)]]

def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy+off_y-1 ][cx+off_x] += val
    return mat1

tetrimonio = [[1,1,1],[0,1,0]]

def check_collision(board, shape, offset):
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                if cell and board[ cy + off_y ][ cx + off_x ]:
                    return True
            except IndexError:
                return True
    return False

print(join_matrixes(board,tetrimonio,(9,3)))
print(check_collision(board, tetrimonio,(9,3)))