#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Pytorch requirements
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import dgl
import dgl.function as fn

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.cuda.LongTensor

#构建G的作用是利用图结构进行传递更新，更新之后的值是要pop出来的
def build_fewshot_graph(num_nodes):
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            g.add_edge(i,j)
    return g

def build_batch_graph(bs,num_nodes):
    G = []
    for _ in range(bs):
        g = build_fewshot_graph(num_nodes)
        G.append(g)
    G = dgl.batch(G)
    return G


# def gmul(input):
#     W, x = input
#     # x is a tensor of size (bs, N, num_features)
#     # W is a tensor of size (bs, N, N, J)
#     x_size = x.size()
#     W_size = W.size()
#     N = W_size[-2]
#     W = W.split(1, 3)
#     W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
#     output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
#     output = output.split(N, 1)
#     output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
#     return output


class Gconv(nn.Module):
    def __init__(self, G, nf_input, nf_output, J, bn_bool=True):
        super(Gconv, self).__init__()
        self.G = G
        self.J = J
        self.nf_input = nf_input
        self.num_inputs = J*nf_input
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)
    
    def reduce_fc(self, edge):
        m0 = edge.data['w0'] * edge.src['h']
        m1 = edge.data['w1'] * edge.src['h']
        return {'m':torch.cat((m0,m1), dim = 1)} #nn*2

    def node_fc(self,nodes):
        x = nodes.data['h'] #=================x.size??=========================
        #print("x_size001:",x.size())
        x = x.view(-1,self.num_inputs)
        #print("x.size002:",x.size())
        x = self.fc(x)
        if self.bn_bool:
            x = self.bn(x)
        return {'h':x}

    def forward(self, input):
        #set the node features
        W, x = input
        #print(x.size())
        self.G.ndata['h'] = x.view(-1,self.nf_input) # (bs*n) * num_feature_inputs
        self.G.edata['w0'], self.G.edata['w1'] = [w.view(-1,1) for w in W.split(1,3)]
        self.G.update_all(self.reduce_fc, fn.sum(msg='m', out='h'))
        self.G.apply_nodes(self.node_fc)
        x = self.G.ndata.pop('h')
        x = x.view(bs,n,-1) 
        print("x.size003:",x.size())
        print("w.size004:",w.size())
        return W,x #x:bs,n,out_feature

        # # fix W, update x
        # # W = input[0]
        # # x = gmul(input) # out has size (bs, N, num_inputs)
        # #if self.J == 1:
        # #    x = torch.abs(x)
        # x_size = x.size()
        # x = x.contiguous()
        # x = x.view(-1, self.num_inputs)
        # x = self.fc(x) # has size (bs*N, num_outputs)

        # if self.bn_bool:
        #     x = self.bn(x)

        # x = x.view(*x_size[:-1], self.num_outputs)
        # return W, x

#A matrix
class Wcompute(nn.Module):
    def __init__(self, G, input_features, nf, operator='J2', activation='softmax', ratio=[2,2,1,1], num_operators=1, drop=False):
        super(Wcompute, self).__init__()
        self.G = G 
        self.input_features = input_features
        self.num_features = nf
        self.operator = operator                                                                                         
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1) #input_channel,output_channel,kernel_size
        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf*ratio[2], 1, stride=1)
        self.bn_3 = nn.BatchNorm2d(nf*ratio[2])
        self.conv2d_4 = nn.Conv2d(nf*ratio[2], nf*ratio[3], 1, stride=1)
        self.bn_4 = nn.BatchNorm2d(nf*ratio[3])
        self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1)
        self.activation = activation

    def msg_fc(self, edges):
        W_new = torch.abs(edges.src['h'] - edges.dst['h']) #bs*n*n(num_edges),nf 
        print(W_new.size()) #27040(40*26*26),133
        W_new = W_new.view(-1,26,26,133) #先这样手动设置一下哈！
        W_new = torch.transpose(W_new,1,3)
        
        W_new = self.conv2d_1(W_new)
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_3(W_new)
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)#n,n,1
        W_new = torch.transpose(W_new, 1, 3) #size: bs x N x N x 1
        print("w_new_size:",W_new.size())
        W_new = W_new.contiguous()
        W_new = W_new.view(-1,1) #(bs*N*N),1
        return {'w0':edges.data['w0'], 'w1':W_new - edges.data['w0'], 'H':edges.src['h']} #=========================
       
    
    def edge_fc(self,nodes):
        w = nodes.mailbox['w1'].squeeze() #一个点的边数
        w1 = F.softmax(w)
        print("w1_size:",w1.size())
        w1 = w1.unsqueeze(2)
        m0 = nodes.mailbox['w0'] * nodes.mailbox['H']
        m1 = w1 * nodes.mailbox['H']
        h = torch.cat((m0,m1),dim=2).sum(1) ###???
        return {'h':h}



    def forward(self, x, W_id):
        bs,n,_=x.size()
        self.G.ndata['h'] = x.view(-1,x.size()[-1])
        self.G.edata['w0'] = W_id.view(-1,1)

        self.G.update_all(self.msg_fc,self.edge_fc)
        x = self.G.ndata['h']
        w_new = torch.cat((self.G.edata['w0'].view(bs,n,n).unsqueeze(3),self.G.edata['w1'].view(bs,n,n).unsqueeze(3)),dim=3)
        return w_new, x.view(bs,n,-1)


        # W1 = x.unsqueeze(2)
        # W2 = torch.transpose(W1, 1, 2) #size: bs x 1 x N x num_features
        # W_new = torch.abs(W1 - W2) #size: bs x N x N x num_features
        # W_new = torch.transpose(W_new, 1, 3) #size: bs x num_features x N x N
        

        # W_new = self.conv2d_1(W_new)
        # W_new = self.bn_1(W_new)
        # W_new = F.leaky_relu(W_new)
        # if self.drop:
        #     W_new = self.dropout(W_new)

        # W_new = self.conv2d_2(W_new)
        # W_new = self.bn_2(W_new)
        # W_new = F.leaky_relu(W_new)

        # W_new = self.conv2d_3(W_new)
        # W_new = self.bn_3(W_new)
        # W_new = F.leaky_relu(W_new)

        # W_new = self.conv2d_4(W_new)
        # W_new = self.bn_4(W_new)
        # W_new = F.leaky_relu(W_new)

        # W_new = self.conv2d_last(W_new)
        # W_new = torch.transpose(W_new, 1, 3) #size: bs x N x N x 1

        # if self.activation == 'softmax':
        #     W_new = W_new - W_id.expand_as(W_new) * 1e8
        #     W_new = torch.transpose(W_new, 2, 3) #bs,N,1,N
        #     # Applying Softmax
        #     W_new = W_new.contiguous()
        #     W_new_size = W_new.size()#[bs,N,1,N]
        #     W_new = W_new.view(-1, W_new.size(3))#(bs*N),N
        #     W_new = F.softmax(W_new)
        #     W_new = W_new.view(W_new_size)#bs,N,1,N
        #     # Softmax applied
        #     W_new = torch.transpose(W_new, 2, 3) #bs,N,N,1

        # elif self.activation == 'sigmoid':
        #     W_new = F.sigmoid(W_new)
        #     W_new *= (1 - W_id)
        # elif self.activation == 'none':
        #     W_new *= (1 - W_id)
        # else:
        #     raise (NotImplementedError)

        # if self.operator == 'laplace':
        #     W_new = W_id - W_new
        # elif self.operator == 'J2':
        #     W_new = torch.cat([W_id, W_new], 3)
        # else:
        #     raise(NotImplementedError)

        # return W_new


class AGC(nn.Module):
    def __init__(self, input_features, nf_output, nf, J, operator='J2', activation='softmax', ratio=[2,2,1,1], num_operators=1, drop=False, bn_bool=True ):
        super(AGC,self).__init__()
        self.J = J


        self.num_inputs = J*input_features
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)
        

        self.operator = operator                                                                                         
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1) #input_channel,output_channel,kernel_size
        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf*ratio[2], 1, stride=1)
        self.bn_3 = nn.BatchNorm2d(nf*ratio[2])
        self.conv2d_4 = nn.Conv2d(nf*ratio[2], nf*ratio[3], 1, stride=1)
        self.bn_4 = nn.BatchNorm2d(nf*ratio[3])
        self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1)
        self.activation = activation

    def msg_fc(self, edges):
        W_new = torch.abs(edges.src['h'] - edges.dst['h']) #bs*n*n(num_edges),nf 
        #print("W_new_size:",W_new.size()) #27040(40*26*26),133 bingo
        W_new = W_new.view(-1,26,26,W_new.size()[-1]) #先这样手动设置一下哈！
        W_new = torch.transpose(W_new,1,3)
        
        W_new = self.conv2d_1(W_new)
        W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)
        W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_3(W_new)
        W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)
        W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)#n,n,1
        W_new = torch.transpose(W_new, 1, 3) #size: bs x N x N x 1
        #print("w_new_size1:",W_new.size())#40,26,26,1 bingo
        W_new = W_new.contiguous()
        W_new = W_new.view(-1,1) #(bs*N*N),1
        return {'w0':edges.data['w0'], 'w1':W_new - edges.data['w0'], 'H':edges.src['h']} #=========================
       
    
    def edge_fc(self,nodes):
        w = nodes.mailbox['w1'].squeeze() #一个点的边数
        w1 = F.softmax(w)
        #print("w1_size:",w1.size()) #1040*26
        w1 = w1.unsqueeze(2)
        m0 = nodes.mailbox['w0'] * nodes.mailbox['H']
        m1 = w1 * nodes.mailbox['H']
        h = torch.cat((m0,m1),dim=2).sum(1) ###???
        return {'h':h}

    def node_fc(self,nodes):
        x = nodes.data['h'] #=================x.size??=========================
        #print("x_size001:",x.size()) #1040(40*26)*266 bingo
        # x = x.view(-1,self.num_inputs)
        # print("x.size002:",x.size())
        x = self.fc(x)
        if self.bn_bool:
            x = self.bn(x)
        return {'h':x}

    def forward(self, input):
        #set the node features
        w_init, x = input#############顺序
        #print("xxx_size:",x.size()) #40*26*133 bingo
        bs, n, xnf = x.size()
        G = []
        for _ in range(bs):
            g = dgl.DGLGraph()
            g.add_nodes(n)
            # for i in range(n):
            #     for j in range(n):
            #         g.add_edge(i,j)
            for i in range(n):
                g.add_edge(0,i)
            G.append(g)
        G = dgl.batch(G)
            
        self.G = G
        self.G.ndata['h'] = x.view(-1,xnf)
        self.G.edata['w0'] = w_init.view(-1,1)
        self.G.update_all(self.msg_fc,self.edge_fc)
        self.G.apply_nodes(self.node_fc)
        x = self.G.ndata.pop('h')

        return 0, x.view(bs,n,-1)



class GNN_nl(nn.Module):
    def __init__(self, args, input_features, nf, J):
        super(GNN_nl, self).__init__()
        self.args = args
        self.input_features = input_features
        self.nf = nf
        self.J = J

        if args.dataset == 'mini_imagenet':
            self.num_layers = 2
        else:
            self.num_layers = 2

        for i in range(self.num_layers):
            if i == 0:
                # module_w = Wcompute(self.G, self.input_features, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                # module_l = Gconv(self.G, self.input_features, int(nf / 2), 2) #out_feature:int(nf/2)自定义
                module_a = AGC(self.input_features, int(nf / 2), nf, 2, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
            else:
                # module_w = Wcompute(self.G, self.input_features + int(nf / 2) * i, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                # module_l = Gconv(self.G, self.input_features + int(nf / 2) * i, int(nf / 2), 2)
                module_a = AGC(self.input_features + int(nf / 2) * i, int(nf / 2), nf, 2, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
                #self.input_features+int(nf/2): concatenate
            # self.add_module('layer_w{}'.format(i), module_w)
            # self.add_module('layer_l{}'.format(i), module_l)
            self.add_module('layer_a{}'.format(i),module_a)

        # self.w_comp_last = Wcompute(self.G, self.input_features + int(self.nf / 2) * self.num_layers, nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        # self.layer_last = Gconv(self.G, self.input_features + int(self.nf / 2) * self.num_layers, args.train_N_way, 2, bn_bool=False)
        
        self.a_last = AGC(self.input_features + int(self.nf / 2) * self.num_layers, args.train_N_way, nf, 2, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
    
    def forward(self, x):
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3)) #bs,n,n,1
        if self.args.cuda:
            W_init = W_init.cuda()

        for i in range(self.num_layers):
            x_new = F.leaky_relu(self._modules['layer_a{}'.format(i)]([W_init, x])[1])
            x = torch.cat([x, x_new], 2)
            # Wi = self._modules['layer_w{}'.format(i)](x, W_init)

            # x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1])
            # x = torch.cat([x, x_new], 2)

        # Wl=self.w_comp_last(x, W_init)
        # out = self.layer_last([Wl, x])[1] #x
        out = self.a_last([W_init, x])[1]

        return out[:, 0, :] #第0个是待测样本


if __name__ == '__main__':
    # test modules
    bs =  4
    nf = 10 #number of features for each conv layer
    num_layers = 5
    N = 8
    x = torch.ones((bs, N, nf))
    W1 = torch.eye(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    W2 = torch.ones(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    J = 2 #number of 2 operators: identity matrix/ adjacency matrix
    W = torch.cat((W1, W2), 3)
    input = [Variable(W), Variable(x)]
    ######################### test gmul ##############################
    # feature_maps = [num_features, num_features, num_features]
    # out = gmul(input)
    # print(out[0, :, num_features:])
    ######################### test gconv ##############################
    # feature_maps = [num_features, num_features, num_features]
    # gconv = Gconv(feature_maps, J)
    # _, out = gconv(input)
    # print(out.size())
    ######################### test gnn ##############################
    # x = torch.ones((bs, N, 1))
    # input = [Variable(W), Variable(x)]
    # gnn = GNN(num_features, num_layers, J)
    # out = gnn(input)
    # print(out.size())


