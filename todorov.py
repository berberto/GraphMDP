import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from fpt import meanFPT


def solveSVD (A, b):
	u, s, vh = np.linalg.svd(A, full_matrices=False)
	c = np.dot(u.T, b)
	w = np.linalg.solve(np.diag(s), c)
	z = np.dot(vh.T, w)
	return z


# #
# #	define from NX graph
# #
# print("Loading NetworkX graph")
# G = nx.read_graphml("v_graph_9Oct2020.graphml")
# isolates = list(nx.isolates(G))
# print("Removing isolated nodes: ", isolates)
# G.remove_nodes_from(isolates)
# adj = nx.adjacency_matrix(G).todense()
# N = G.number_of_nodes()
# print("%d nodes"%N)
# end = [  434,  3229,  3895,  4424,  4512,  5139,  5551,  5657,  5696,
#         5995,  6818,  7439,  7852,  8389,  9100,  9407,  9749,  9809,
#         9825, 10028, 10722, 10969, 10979, 11087, 11421, 11455, 11569,
#        11584, 11599, 11754, 11955, 12124, 12388, 12663, 12670, 12920,
#        12995, 13172, 13410, 13824, 13869, 13963, 14594, 15101, 15196,
#        15605, 15623, 15711, 16377, 16490, 16580, 16914, 16968, 16982,
#        17130, 17141, 17197, 17294, 17296, 17499, 18183, 18603, 18787,
#        19230, 19596, 19937, 19963, 20038, 20085, 20273, 20310, 20433,
#        20515, 20527, 20644, 20935, 21013]


#
#	define random (Erdos-Renyi) graph
#
print("Generating ER graph")
np.random.seed(1990)
N = 20			# number of nodes
deg = 4			# average degree
adj = np.zeros((N,N), dtype=int)
for i in range(N):
	for j in range(i+1,N):
		if np.random.rand() < deg/(N-1):	
			adj[i,j] += 1
			adj[j,i] += 1
# define target points
end = np.sort(np.unique([N-2, N-1]))


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
print("Setting up costs and tilted generator")
q = 1.		# cost per jump
eps = 1.	# weight of KL
tradeoff = q/eps	# only thing that matters

# tilted generator
pt = p.copy()
for k in range(N):
	if not k in end:
		pt[:,k] *= np.exp(-tradeoff)

# solve for desirability (at non-absorbing states)
print("Solving for the desirability")
M = pt.T - np.eye(N)			  # tilted generator - identity
mat = np.delete(M, end, axis=1)   # delete columns corresponding to target nodes
vec = - np.sum(M[:,end], axis=1)  # vector implementing "boundary" conditions
Z = np.ones(N)
Z[np.delete(np.arange(N), end)] = np.squeeze(solveSVD(mat, vec))


# controlled transition probability
print("Define controlled transition probabilities")
u = pt.copy()
for kp in range(N):
	u[kp] *= Z[kp]
for k in range(N):
	u[:,k] /= Z[k]


# test solution:
print("\nRun checks...")
print("correct solution: ", np.allclose( Z - np.dot(pt.T, Z), 0))
print("u correctly normalized: ", np.allclose(np.sum(u,axis=0), 1, atol=1e-15))
print("")

#	create graph from (weighted) edges -- tr. pr. matrix
print("Create graph")
G=nx.Graph()
G = nx.from_numpy_matrix(u)
Zdict = dict([x for x in zip(range(len(Z)), Z)])
nx.set_node_attributes(G, Zdict, 'desirability')


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

# plt.savefig("%.1f_realstuff.png"%tradeoff)
# nx.write_graphml_lxml(G,"%.1f_realstuff.graphml"%tradeoff)
plt.savefig("N%d_deg%d_%.1f_random.png"%(N,deg,tradeoff))
nx.write_graphml(G,"N%d_deg%d_%.1f_random.graphml"%(N,deg,tradeoff))
