
from enum import Enum


class Test(str, Enum):
  A = "A"
  B = "B"
  C = "C"

print(Test.C in { Test.A, Test.B })