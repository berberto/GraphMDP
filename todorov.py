#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# author: A. Pezzotta -- pezzota [AT] crick.ac.uk

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from fpt import meanFPT
from solvers import directSolve

# cost parameters
q = 1.		# cost per jump
eps = 1.	# weight of KL
tradeoff = q/eps	# only thing that matters

basename = "%.2f_whaleshark"%(tradeoff)

#
#	define from NX graph
#
print("Loading NetworkX graph")
G = nx.read_graphml("v_graph_9Oct2020.graphml")
isolates = list(nx.isolates(G))
print("Removing isolated nodes: ", isolates)
G.remove_nodes_from(isolates)
print("Extracting adjacency matrix")
adj = np.squeeze(np.asarray(nx.adjacency_matrix(G).todense()))
N = G.number_of_nodes()
print("%d nodes"%N)
end = np.sort(np.unique(
		[  434,  3229,  3895,  4424,  4512,  5139,  5551,  5657,  5696,
        5995,  6818,  7439,  7852,  8389,  9100,  9407,  9749,  9809,
        9825, 10028, 10722, 10969, 10979, 11087, 11421, 11455, 11569,
       11584, 11599, 11754, 11955, 12124, 12388, 12663, 12670, 12920,
       12995, 13172, 13410, 13824, 13869, 13963, 14594, 15101, 15196,
       15605, 15623, 15711, 16377, 16490, 16580, 16914, 16968, 16982,
       17130, 17141, 17197, 17294, 17296, 17499, 18183, 18603, 18787,
       19230, 19596, 19937, 19963, 20038, 20085, 20273, 20310, 20433,
       20515, 20527, 20644, 20935, 21013]))


# uncontrolled transition probability
# transitions from a given node have equal probabilities
print("Define random walk on graph")
p = adj.copy().astype(float)
for k in range(N):
	p[:,k] /= np.sum(p[:,k])	# normalize each column


# "end" nodes are absorbing
for term in end:
	p[:,term] = 0.
	p[term,term] = 1.

# objective function parameters
print("Setting up tilted generator")
print("\tcost per jump: ", q)
print("\tweight for KL: ", eps)
pt = p.copy()
for k in range(N):
	if not k in end:
		pt[:,k] *= np.exp(-tradeoff)

# solve for desirability (at non-absorbing states)
print("Solving for the desirability")
Z = directSolve(pt, end, method='lsqr') # solve linear problem

# controlled transition probability
print("Define controlled transition probabilities")
u = pt.copy()
for kp in range(N):
	u[kp] *= Z[kp]
for k in range(N):
	u[:,k] /= Z[k]


# test solution:
print("\nRun checks...")
correct = np.allclose(np.dot(pt.T, Z), Z, rtol=.0001)
normlzd = np.allclose(np.sum(u,axis=0), 1, rtol=.001)
print("correct solution: ", correct)
print("u correctly normalized: ", normlzd)
if not correct:
	print("\tmax error: ", np.max(Z - np.dot(pt.T, Z)))
	exit()
if not normlzd:
	print("\tmax error: ", np.max(np.sum(u,axis=0) - np.ones(N)))
	exit()
print("")

#	create graph from (weighted) edges -- tr. pr. matrix
print("Create graph")
G=nx.Graph()
G = nx.from_numpy_matrix(np.asmatrix(u))
Zdict = dict([x for x in zip(range(len(Z)), Z)])
nx.set_node_attributes(G, Zdict, 'desirability')
filename = basename+".graphml"
print("\tSaving graph in markdown: ", filename)
nx.write_graphml(G, filename)

exit()

#
#	PLOTS
#
print("Plot results")
fig, ax = plt.subplots(2,2, figsize=(8,8))

plt.sca(ax[0,0])
ax[0,0].set_title("Adjacency matrix")
f = ax[0,0].imshow(adj)
# fig.colorbar(f, ax=ax[0,0])

plt.sca(ax[0,1])
ax[0,1].set_title("Graph")
pos = nx.spring_layout(G, iterations=50)
nx.draw(G, pos, node_color=np.log(Z), node_size=80, cmap=plt.cm.Spectral, with_labels=True, font_size=6)

plt.sca(ax[1,0])
ax[1,0].set_title("Reference tr. prob.")
f = ax[1,0].imshow(p)
# fig.colorbar(f, ax=ax[1,0])

plt.sca(ax[1,1])
ax[1,1].set_title("Optimal tr. prob.")
f = ax[1,1].imshow(u)
# fig.colorbar(f, ax=ax[1,1])

filename = basename+".png"
print("\tSaving figures: ", filename)
plt.savefig(filename)
