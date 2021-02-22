import torch
#定义一个自动求导函数
class MyReLu(torch.autograd.Function):
    """
    通过建立autograd的子类来实现我们自定义的autograd函数，并且完成张量的正向和反向传播
    """
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx,grad_output):
        """
         在反向传播中，我们接收到上下文对象和一个张量，
         其包含了相对于正向传播过程中产生的输出的损失的梯度。
         我们可以从上下文对象中检索缓存的数据，
         并且必须计算并返回与正向传播的输入相关的损失的梯度.
        :param ctx:
        :param grad_output:
        :return:
        """
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x < 0] = 0
        return grad_x

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

x=torch.randn(N,D_in,device=device)
y=torch.randn(N,D_out,device=device)

w1=torch.randn(D_in,H,device=device,requires_grad=True)
w2=torch.randn(H,D_out,device=device,requires_grad=True)

learning_rate=1e-6

for t in range(600):
    y_pred = MyReLu.apply(x.mm(w1)).mm(w2)

    loss=(y_pred-y).pow(2).sum()
    print(t,loss.item())
    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()