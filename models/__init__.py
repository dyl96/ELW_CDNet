from models import *
from models.model import BaseNet

BaseNet = BaseNet
if __name__ == '__main__':
    x = BaseNet(3, 1)
    print(x)

