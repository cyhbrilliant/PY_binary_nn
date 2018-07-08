import torch
import MMXNOR
from time import clock

x=3000
n=3000
y=3000

a = torch.randn(x, n)
b = torch.randn(n, y)
c = torch.FloatTensor(a.size()[0], b.size()[1])

# The following line will give an error
# a += b
# a[a>0.5]=1
# a[a<=0.5]=-1
# b[b>0]=1
# b[b<=0]=-1

a_cuda=a.cuda()
b_cuda=b.cuda()
c_cuda=c.cuda()

a_cuda=torch.sign(a_cuda)
b_cuda=torch.sign(b_cuda)

#
# print(a_cuda.size())
# print(a_cuda.cpu())
# print(b_cuda.cpu())
start = clock()
mmR=torch.mm(a_cuda, b_cuda)
finish1 = clock()
# xnorR=MMXNOR.matmulXnor.apply(a_cuda, b_cuda, c_cuda)
finish2 = clock()

# print(mmR)
# print(xnorR)
print('mm time=',(finish1-start)*1000,'xnor time=',(finish2-finish1)*1000)