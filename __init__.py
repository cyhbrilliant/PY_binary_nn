import numpy as np
import torch
from torch.autograd import Function, Variable
import MMXNOR.mmxnor as Mm


class matmulXnor(Function):
  """Accumulate x += y using broadcasting sum.
  """
  def forward(self, a, b, c):
    Mm.mmxnor(a, b, c, a.size()[0], a.size()[1], b.size()[1])
    return c