import matplotlib.pyplot as plt
import torch
import numpy as np
import sys

def test():
    # print("=== test test test ===")
    x = torch.randn(3)
    y = torch.randn(3)
    z = torch.randn(3,1)
    u = torch.randn(3,1)
    # x.append(2)
    # plt.plot(x)
    # plt.show()
    print(np.dot(x, y))
    print(np.dot(x, z))
    # 会报错!
    # print(np.dot(z, x))
    print(np.dot(u, z.T))
test()

# if __name__ == "__main__":
#     if len(sys.argv) > 1:
#         function_name = sys.argv[1]
#         if function_name == "test":
#             test()
#         else:
#             print(f"Function '{function_name}' not recognized.")
#     else:
#         print("No function name provided.")