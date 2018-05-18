
def gradient_descent(f, x0, args, alpha=0.01, maxiter=1500):
  x = x0
  for _ in range(maxiter):
    _, grad = f(x, *args)
    x = x - alpha * grad
  return x