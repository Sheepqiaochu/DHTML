# # import torch.nn as nn
# # import numpy as np
# # import torch
# # import torch
# # from torch.nn import functional as F
# #
# # loss_func = nn.KLDivLoss(reduction='mean')
# # loss_func2 = nn.L1Loss(reduction='mean')
# # a = torch.tensor([[0, 0, 0, 0],
# #                   [0, 0, 0, 0]], dtype=float)
# #
# # b = torch.tensor([[1, 0, 0, 0],
# #                   [0, 0, 0, 0]], dtype=float)
# # # a = a.view(4)
# # # b = b.view(4)
# # # a = F.log_softmax(a, dim=0)
# # # b = F.softmax(b, dim=0)
# # c = torch.tensor([0.5, 0.5], dtype=float)
# # d = torch.tensor([0.5, 0.5], dtype=float)
# #
# # print(a.size(), b.size())
# # print(loss_func2(a, b))
# # # print(loss_func(a, b))
# # # print(loss_func(a.view(8), b.view(8)))
# # # print(loss_func(c.view(-1), d.view(-1)))
# # # print(0.0139 * 4)
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def loss_figure(loss=[]):
#     epoch = []
#     i = 100
#
#     for k in range(2000):
#         loss.append(i)
#         i -= 1
#         k += 1
#
#     for i in range(len(loss)):
#         epoch.append(i)
#         i += 1
#
#     plt.plot(epoch, loss)
#     plt.ylabel('Loss')
#     plt.xlabel('epoch')
#     plt.show()
#
#
# loss_figure()

import matplotlib.pyplot as plt
import torch
import numpy as np

