import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


x = loadmat("x_pred.mat")
x_pred = x["pred"]
x_gt = x["gt"]

x2 = loadmat("x_pred2.mat")
x_pred2 = x2["pred"]
x_gt2 = x2["gt"]

print(x_gt[:5], x_gt2[:5])
print(x_pred - x_pred2)
print(x_gt - x_gt2)

# length = x_pred.shape[1]
# obj = x_gt.T
# pred = x_pred.T
# print(x_pred, x_gt, length)
#
#
# # obj = np.array(list(range(length)))
# # error = (np.random.noncentral_chisquare(3, 0.1, length) *
# #          (np.array(list(range(length))) / 100 + 1)) * 20
# # error -= np.min(error)
# # pred = obj + error
# # pred = pred.reshape((-1, 1))
# # obj = obj.reshape((-1, 1))
#
#
#
#
# plt.subplot(221)
# plt.scatter(obj, pred, s=5)
# line = list(range(np.max(obj)))
# plt.plot(line, line, color='red')
#
# A = np.hstack((pred, np.ones((length, 1))))
# P = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.dot(np.transpose(A), obj))
# projection_pred = np.dot(A, P)
# plt.subplot(222)
# plt.scatter(obj, projection_pred, s=5, color='green')
# plt.plot(line, line, color='red')
#
# A = np.hstack((pred**2, pred, np.ones((length, 1))))
# P = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.dot(np.transpose(A), obj))
# projection_pred = np.dot(A, P)
# plt.subplot(223)
# plt.scatter(obj, projection_pred, s=5, color='green')
# plt.plot(line, line, color='red')
#
# A = np.hstack((pred**3, pred**2, pred, np.ones((length, 1))))
# P = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.dot(np.transpose(A), obj))
# projection_pred = np.dot(A, P)
# plt.subplot(224)
# plt.scatter(obj, projection_pred, s=5, color='green')
# plt.plot(line, line, color='red')
#
# plt.show()
