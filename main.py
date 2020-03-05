import sys
import numpy as np


def read(file):
    m = []
    rows = file.readlines()
    N = len(rows)
    for row in rows:
        print(row.rstrip().split())
        m.append([float(i) for i in row.rstrip().split()])
        assert len(m[len(m)-1]) == N
    return m, N


def eliminate_n_max(arr, n):
    m = dict()
    for i in range(len(arr)):
        ind = arr[i]
        l = m.get(arr[i], [])
        l.append(i)
        m[ind] = l
    for key in sorted(m.keys(), key=lambda x: abs(x)):
        indexes = m[key]
        for i in indexes:
            if n == 0:
                break
            arr[i] = 0
            n -= 1
    return arr


def tismenetsky_incomplete(A, alpha):
    N = len(A)
    Ai_s = A.copy()

    L = np.zeros((N, N))
    U = np.zeros((N, N))

    for i in range(N):
        print("i:", i)

        I = np.eye(N)
        ei = I[:, i]
        ei.shape = (N, 1)
        ei_T = ei.copy()
        ei_T.shape = (1, N)

        di = ei_T @ Ai_s @ ei

        # nrow ncol
        r = np.count_nonzero(Ai_s[i, :])
        nrow = alpha * r
        s = np.count_nonzero(Ai_s[:, i])
        ncol = alpha * s

        fi = eliminate_n_max(Ai_s[i].copy(), nrow)
        fi.shape = (N, 1)
        gi_T = eliminate_n_max(Ai_s[:, i].copy(), ncol)
        gi_T.shape = (1, N)

        ui = Ai_s @ ei - fi
        vi_T = ei_T @ Ai_s - gi_T

        # L
        L[:, i] = ((Ai_s @ ei - fi) / di).flatten()

        # U
        U[i, :] = (ei_T @ Ai_s - gi_T) / di

        # A (An = D)
        Ai_w = Ai_s - (ui @ vi_T + ui @ gi_T + fi @ vi_T + fi @ gi_T) / di
        Ai_w[i][i] = di
        Ai_s = Ai_w.copy()

    return L, Ai_s, U


def main():
    np.set_printoptions(precision=2, suppress=True)
    if len(sys.argv) != 3:
        print("usage python main.py <in.txt> <alpha>")
        sys.exit(-1)
    _, filename, alpha = sys.argv
    alpha = int(alpha)
    with open(filename) as f:
        m, N = read(f)
        A = np.array(m)
        assert np.linalg.matrix_rank(A) == N

        L_hat, D_hat, U_hat = tismenetsky_incomplete(A, alpha)
        print("L\n", L_hat)
        print("D\n", D_hat)
        print("U\n", U_hat)

        ldu = L_hat @ D_hat @ U_hat
        print("LDU\n", ldu)
        print("R = A - LDU\n", A - ldu)


if __name__ == "__main__":
    main()

