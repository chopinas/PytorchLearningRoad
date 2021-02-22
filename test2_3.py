#控制流，权重共享
import random
import torch
class DynamicNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self,x):
        """
        对于模型的前向传播，我们随机选择0，1，2，3
        并重用了多次计算隐藏层的middle_linear模块
        由于每个前向传播构建一个动态计算图
        我们在定义模型的前向传播时使用常规的python控制流运算符，如循环或条件语句
        在这里，我们还看到，在定义计算图形时多次重用同一个模块是安全的，
        :param x:
        :return:
        """
        h_relu=self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0,3)):
            h_relu=self.middle_linear(h_relu).clamp(min=0)
        y_pred=self.output_linear(h_relu)
        return y_pred

N,D_in,D_out,H=64,1000,10,100

x=torch.randn(N,D_in)
y=torch.randn(N,D_out)

model=DynamicNet(D_in,H,D_out)

loss_fn=torch.nn.MSELoss(reduction="sum")
optimizer=torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)
for t in range(500):
    y_pred=model(x)

    loss=loss_fn(y_pred,y)
    print(t,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()