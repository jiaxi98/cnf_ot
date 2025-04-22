from sympy import diff, simplify, symbols

x, xk, xk1, yk, yk1, deltak, deltak1 = symbols(
  'x xk xk1 yk yk1, deltak, deltak1'
)
sk = (yk1 - yk) / (xk1 - xk)
xi = (x - xk) / (xk1 - xk)
alpha = (yk1 - yk) * (sk * xi**2 + deltak * xi * (1 - xi))
beta = sk + (deltak1 + deltak - 2 * sk) * xi * (1 - xi)
f = yk + alpha / beta

dfddeltak = simplify(diff(f, deltak))
print(dfddeltak)
