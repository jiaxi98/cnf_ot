# This script extract the part for computing reference solution for 2d WPO doublewell problem
import os
import pickle

import numpy as np
import torch

dtype = torch.float
device = torch.device('cpu')

T = 1.0
d = 2
beta = 0.1
# a = 0.5
a = 1.0
c1 = torch.tensor([[a, a]], dtype=dtype, device=device)  # center 1
c2 = torch.tensor([[-a, -a]], dtype=dtype, device=device)  # center 2
c1_np = c1.cpu().numpy()
c2_np = c2.cpu().numpy()
coef = 4
dxyz = 0.01  # step size

if not os.path.isfile('data/fcn4a1_interp.pkl'):

  def rho0(x, d):  # initial density N(0,1)
    return torch.exp(-torch.sum(x**2, dim=-1, keepdim=True) / 2
                     ) / ((2 * np.pi)**(d / 2))

  def rho0_np(x):  # initial density N(0,1) in numpy
    return np.exp(-np.sum(x**2, axis=-1, keepdims=True) / 2
                  ) / ((2 * np.pi)**(d / 2))

  def score0(x):  # initial score
    return -x

  def g(x):  # terminal cost
    # input x is N x 2, output is N x 1
    part1 = torch.sum((x - c1)**2, dim=-1, keepdim=True)
    part2 = torch.sum((x - c2)**2, dim=-1, keepdim=True)
    return part1 * part2 / coef

  def g_np(x):  # terminal cost in numpy
    # input x is N x 2, output is N x 1
    part1 = np.sum((x - c1_np)**2, axis=-1, keepdims=True)
    part2 = np.sum((x - c2_np)**2, axis=-1, keepdims=True)
    return part1 * part2 / coef

  def V_x(x):  # gradient of terminal cost
    # input x is N x 2, output is N x 2
    part1 = torch.sum((x - c1)**2, dim=-1, keepdim=True)
    part2 = torch.sum((x - c2)**2, dim=-1, keepdim=True)
    return 2 * (x - c1) * part2 / coef + 2 * (x - c2) * part1 / coef

  def V_x_np(x):  # gradient of terminal cost in numpy
    # input x is N x 2, output is N x 2
    part1 = np.sum((x - c1_np)**2, axis=-1, keepdims=True)
    part2 = np.sum((x - c2_np)**2, axis=-1, keepdims=True)
    return 2 * (x - c1_np) * part2 / coef + 2 * (x - c2_np) * part1 / coef

  int_range = 2.0
  int_num = 100
  dx = int_range / int_num  # 0.01
  dx2 = dx**2
  int_num2 = int_num * 2 + 1

  # first test integration
  y = torch.tensor(
    [[0.0, 0.0], [-0.5, -0.5], [0.5, 0.5]], dtype=dtype, device=device
  )  # Ny x 2
  yt_y1 = torch.linspace(
    -int_range, int_range, int_num2, dtype=dtype, device=device
  )  # int_num2 x 1
  yt_y2 = torch.linspace(
    -int_range, int_range, int_num2, dtype=dtype, device=device
  )  # int_num2 x 1
  yt_y1, yt_y2 = torch.meshgrid(yt_y1, yt_y2)  # int_num2 x int_num2
  yt_y = torch.stack(
    (yt_y1.flatten(), yt_y2.flatten()), dim=1
  )  # int_num2^2 x 2
  # yt_y2 = yt_y**2

  yt_y_unsq = yt_y.unsqueeze(0)  # 1 x int_num2^2 x 2
  y_unsq = y.unsqueeze(1)  # Ny x 1 x 2
  yt = yt_y_unsq + y_unsq  # Ny x int_num2^2 x 2
  temp = g(yt) + torch.sum(yt_y_unsq**2, dim=-1,
                           keepdim=True) / (2 * T)  # Ny x int_num2^2 x 1
  result = torch.sum(torch.exp(-temp / (2 * beta)), dim=1) * dx2  # Ny x 1
  print('range:', int_range, 'num:', int_num, 'result:', result)

  # below we store the result for
  # h(y) = int exp(-(g(z) + |y-z|^2 / (2T)) / (2*beta)) dz
  # where y is in the range [-4,4] x [-4,4], dy = dz = 0.01
  # the result is stored in hy2d.npz file
  # need to save: a, dy, range for y, beta, g, result h(y)
  # the integration is truncated to y-2 <= z <= y+2 for z
  y_range = 4.0
  int_range = 6.0  # range for z
  dy = dxyz
  dz = dxyz
  dz2 = dz**2
  int_numy = int(y_range / dy)  # 400
  int_numy2 = int_numy * 2 + 1  # 801
  int_numz = int(int_range / dz)
  int_numz2 = int_numz * 2 + 1
  y1 = np.linspace(-y_range, y_range, int_numy2)  # int_numy2
  y2 = np.linspace(-y_range, y_range, int_numy2)  # int_numy2
  y1, y2 = np.meshgrid(y1, y2)  # int_numy2 x int_numy2
  y = np.stack((y1.flatten(), y2.flatten()), axis=1)  # int_numy2^2 x 2

  # compute h(y)
  hy = np.zeros((int_numy2**2, 1))
  y_z1 = np.linspace(-int_range, int_range, int_numz2)  # int_numz2
  y_z2 = np.linspace(-int_range, int_range, int_numz2)  # int_numz2
  y_z1, y_z2 = np.meshgrid(y_z1, y_z2)  # int_numz2 x int_numz2
  y_z = np.stack((y_z1.flatten(), y_z2.flatten()), axis=1)  # int_numz2^2 x 2
  y_z2 = np.sum(y_z**2, axis=1, keepdims=True)  # int_numz2^2 x 1
  for i in range(int_numy2**2):
    if i % 1000 == 0:  # print every 1000 steps
      print('i:', i)
    z = y_z + y[i:(i + 1), :]  # int_numz2^2 x 2
    gz = g_np(z)  # int_numz2^2 x 1
    temp = gz + y_z2 / (2 * T)  # int_num2^2 x 1
    hy[i, 0] = np.sum(np.exp(-temp / (2 * beta))) * dz2
  # save the result
  np.savez(
    './results/WPONG/hy2d.npz',
    a=a,
    dy=dy,
    y_range=y_range,
    beta=beta,
    coef=coef,
    hy=hy
  )
  print('h(y) saved')

  # load the result
  npzfile = np.load('./results/WPONG/hy2d.npz')
  hy = npzfile['hy']  # int_num2^2 x 1

  # # 2d plot of h(y)
  # hy = hy.reshape((int_numy2,int_numy2))
  # plt.figure()
  # # the ::-1 is to flip the y2 axis because the y2 axis is from -4 to 4
  # # for example, for hy = [[1,2],[3,4]], we hope to see image of h(y) as [[3,4],[1,2]]
  # plt.imshow(hy[::-1,:], extent=[-y_range,y_range,-y_range,y_range])
  # plt.colorbar()
  # plt.title('h(y)')
  # plt.show()

  x_range = 2.0
  dx = dxyz
  dy = dxyz
  dy2 = dy**2
  int_numx = int(x_range / dx)  # 200
  int_numx2 = int_numx * 2 + 1  # 401
  x1 = np.linspace(-x_range, x_range, int_numx2)  # int_numx2
  x2 = np.linspace(-x_range, x_range, int_numx2)  # int_numx2
  X1, X2 = np.meshgrid(x1, x2)  # int_numx2 x int_numx2
  # X11, X22 = np.meshgrid(x1, x2, indexing='ij') # int_numx2 x int_numx2
  x = np.stack((X1.flatten(), X2.flatten()), axis=1)  # int_numx2^2 x 2
  x2x2 = x

  # compute rho(T,x), scoreT(x), w(0,x), w(T,x)
  rhoT = np.zeros((int_numx2**2, 1))  # to store the density at T
  scoreT = np.zeros((int_numx2**2, 2))  # to store the score at T
  w0 = np.zeros((int_numx2**2, 2))  # to store the w at 0
  gx = g_np(x)  # int_numx2^2 x 1
  gy = g_np(y)  # int_numy2^2 x 1
  rho0y = rho0_np(y)  # int_numy2^2 x 1
  g_primex = V_x_np(x)  # int_numy2^2 x 2
  for i in range(int_numx2**2):
    if i % 1000 == 0:  # print every 1000 steps
      print('i:', i)
    x_i = x[i:(i + 1), :]  # 1 x 2
    x_i_y = x_i - y  # int_numy2^2 x 2
    x_i_y2 = np.sum(x_i_y**2, axis=1, keepdims=True)  # int_numy2^2 x 1
    temp = np.exp(
      -(gx[i:(i + 1), :] + x_i_y2 / (2 * T)) / (2 * beta)
    ) * rho0y / hy  # int_numy2^2 x 1
    rhoT[i, 0] = np.sum(temp) * dy2
    temp2 = -temp * (g_primex[i:(i + 1), :] +
                     x_i_y / T) / (2 * beta)  # int_numy2^2 x 2
    scoreT[i, :] = np.sum(temp2, axis=0) * dy2 / rhoT[i, 0]  # shape: 2
    temp = np.exp(-(gy + x_i_y2 / (2 * T)) / (2 * beta))  # int_numy2^2 x 1
    temp2 = -(x_i_y / T) * temp  # int_numy2^2 x 2
    w0[i, :] = np.sum(temp2, axis=0) / np.sum(temp) + beta * x_i  # shape: 2
  wT = -g_primex - beta * scoreT

  # # Below is the code for save and load the current results
  # # save rho(T,x), scoreT(x), w(0,x), w(T,x)
  # np.savez('./results/WPONG/fcn4.npz', a=a, dx=dx, x_range=x_range, beta=beta, coef=coef, rhoT=rhoT, scoreT=scoreT, w0=w0, wT=wT)
  # # Note: here I rename the file to fcn4a5.npz or fcn4a1.npz based on the value of a

  # # load rho(T,x), scoreT(x), w(0,x), w(T,x)
  # if a == 0.5:
  #     npzfile = np.load('./results/WPONG/fcn4a5.npz')
  # elif a == 1.0:
  #     npzfile = np.load('./results/WPONG/fcn4a1.npz')
  # else:
  #     raise ValueError('a must be 0.5 or 1.0')
  # rhoT = npzfile['rhoT'] # int_numx2^2 x 1
  # scoreT = npzfile['scoreT'] # int_numx2^2 x 2
  # w0 = npzfile['w0'] # int_numx2^2 x 2
  # wT = npzfile['wT'] # int_numx2^2 x 2

  # The following 7 matrices are the reference solution at mesh points
  rhoT = rhoT.reshape((int_numx2, int_numx2))  # int_numx2 x int_numx2
  scoreT_1 = scoreT[:,
                    0].reshape((int_numx2, int_numx2))  # int_numx2 x int_numx2
  scoreT_2 = scoreT[:,
                    1].reshape((int_numx2, int_numx2))  # int_numx2 x int_numx2
  w0_1 = w0[:, 0].reshape((int_numx2, int_numx2))  # int_numx2 x int_numx2
  w0_2 = w0[:, 1].reshape((int_numx2, int_numx2))  # int_numx2 x int_numx2
  wT_1 = wT[:, 0].reshape((int_numx2, int_numx2))  # int_numx2 x int_numx2
  wT_2 = wT[:, 1].reshape((int_numx2, int_numx2))  # int_numx2 x int_numx2

  # # plot rho(T,x)
  # plt.figure()
  # plt.imshow(rhoT[::-1,:], extent=[-x_range,x_range,-x_range,x_range])
  # plt.colorbar()
  # # add two centers
  # plt.scatter(c1_np[0,0], c1_np[0,1], c='r', marker='x', s=100)
  # plt.scatter(c2_np[0,0], c2_np[0,1], c='r', marker='x', s=100)
  # plt.title('rho(T,x)')
  # plt.savefig('./results/WPONG/rhoT2d.png')
  # # plt.show()

  # compute the interpolation of rho(T,x), scoreT(x), w(0,x), w(T,x)
  rhoT_interp = RegularGridInterpolator((x1, x2), rhoT.transpose())
  scoreT_1_interp = RegularGridInterpolator((x1, x2), scoreT_1.transpose())
  scoreT_2_interp = RegularGridInterpolator((x1, x2), scoreT_2.transpose())
  w0_1_interp = RegularGridInterpolator((x1, x2), w0_1.transpose())
  w0_2_interp = RegularGridInterpolator((x1, x2), w0_2.transpose())
  wT_1_interp = RegularGridInterpolator((x1, x2), wT_1.transpose())
  wT_2_interp = RegularGridInterpolator((x1, x2), wT_2.transpose())
  # the reason to transpose is that our meshgrid has indexing xy,
  # but RegularGridInterpolator needs meshgrid with default indexing ij

  # save the interpolators
  with open('data/fcn4_interp.pkl', 'wb') as f:
    pickle.dump(
      {
        'rhoT_interp': rhoT_interp,
        'scoreT_1_interp': scoreT_1_interp,
        'scoreT_2_interp': scoreT_2_interp,
        'w0_1_interp': w0_1_interp,
        'w0_2_interp': w0_2_interp,
        'wT_1_interp': wT_1_interp,
        'wT_2_interp': wT_2_interp
      }, f
    )
  # Note that here I rename the file to fcn4a5_interp.pkl or fcn4a1_interp.pkl based on the value of a

else:
  # load the interpolator
  if a == 0.5:
    with open('data/fcn4a5_interp.pkl', 'rb') as f:
      interpolators = pickle.load(f)
  elif a == 1.0:
    with open('data/fcn4a1_interp.pkl', 'rb') as f:
      interpolators = pickle.load(f)
      breakpoint()
  else:
    raise ValueError('a must be 0.5 or 1.0')

rhoT_interp = interpolators['rhoT_interp']
scoreT_1_interp = interpolators['scoreT_1_interp']
scoreT_2_interp = interpolators['scoreT_2_interp']
w0_1_interp = interpolators['w0_1_interp']
w0_2_interp = interpolators['w0_2_interp']
wT_1_interp = interpolators['wT_1_interp']
wT_2_interp = interpolators['wT_2_interp']


def scoreT(x):  # terminal score
  # input x is N x 2, output is N x 2
  scoreT_1 = scoreT_1_interp(x)
  scoreT_2 = scoreT_2_interp(x)
  return np.stack([scoreT_1, scoreT_2], axis=-1)


def w0(x):  # initial drift
  # input x is N x 2, output is N x 2
  w0_1 = w0_1_interp(x)
  w0_2 = w0_2_interp(x)
  return np.stack([w0_1, w0_2], axis=-1)


def wT(x):  # terminal drift
  # input x is N x 2, output is N x 2
  wT_1 = wT_1_interp(x)
  wT_2 = wT_2_interp(x)
  return np.stack([wT_1, wT_2], axis=-1)
