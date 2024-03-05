import torch
import time
from thop import profile
from models.lglcdnet_1_sn2_3stage import BaseNet


def main():
    # models.lglcdnet_1_sn2_3stage训练配置
    model = BaseNet(model_size='1.0x', fe_nc=48).cuda(0)
    model.eval()
    x1 = torch.zeros(16, 3, 256, 256).cuda(0)
    x2 = torch.zeros(16, 3, 256, 256).cuda(0)

    # thop
    print('=============================thop===============================')
    flops, params = profile(model, (x1, x2))
    print('FLOPs:', flops / 1000 ** 3 / 16, 'G')
    print('Params:', params / 1000 ** 2, 'M')

    t1 = time.time()
    for i in range(100):
        y = model(x1, x2)

    print('FPS:', 16/((time.time()-t1)/100))


if __name__ == '__main__':
    main()
