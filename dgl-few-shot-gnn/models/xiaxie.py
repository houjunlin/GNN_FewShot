import dgl


def build_fewshot_graph(num_nodes):
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            g.add_edge(i,j)
    return g

G = build_fewshot_graph(6)
print(type(G))
print('we have %d nodes.' % G.number_of_nodes())
print("we have %d edges." % G.number_of_edges())

import torch
#x.size(1)=N
#x.size(0)=bs
w = torch.eye(6).unsqueeze(0)
print(w)
print("w_size",w.size())
w1 = w.repeat(12, 1, 1)
#print(w1)
print("w1_size",w1.size())
w2 = w1.unsqueeze(3)
#print(w2)
print("w2_size",w2.size())

