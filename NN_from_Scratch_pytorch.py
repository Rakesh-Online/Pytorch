import torch

X = torch.tensor([[1.,0.,1.,0.], [1.,0.,1.,1.],[0.,1.,0.,1.]])
y = torch.tensor([[1.],[1.],[0.]])
#sigmoid Function
def sigmoid(x):
    return 1/(1+torch.exp(-x))

#derivative of sigmiod Function
def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

#variable initialization
epoch = 7000
lr = 0.1
inputlayer = X.shape[1]
hiddenlayer = 3
output = 1

#Weights and bias initialization

weights = torch.randn(inputlayer,hiddenlayer).type(torch.FloatTensor)
bias= torch.randn(1,hiddenlayer).type(torch.FloatTensor)
weights_out = torch.randn(hiddenlayer, output)
bias_out = torch.randn(1,output)

#model
for i in range(epoch):
    #forward pass
    layer1 = torch.mm(X,weights)
    layer1_1 = layer1+bias
    layer_AF = sigmoid(layer1_1)

    output_layer = torch.mm(layer_AF,weights_out)
    output_layer1_1 = output_layer+bias_out
    output_layer_AF = sigmoid(output_layer1_1)

    #backpropagation
    error = y-output_layer_AF
    derv = derivative_sigmoid(output_layer_AF)
    derv_hidden = derivative_sigmoid(layer_AF)
    der_output =  error*derv
    error_at_hidden = torch.mm(der_output,weights_out.t())
    der_hidden = error_at_hidden*derv_hidden

    weights_out += torch.mm(layer_AF,der_output)*lr
    bias_out += der_output.sum()*lr
    weights += torch.mm(X.t(),der_hidden)*lr
    bias += der_hidden.sum()*lr

print('actual :', y)
print("predicted", output_layer_AF)