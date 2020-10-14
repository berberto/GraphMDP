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
# adj = nx.adjacency_matrix(G).todense()

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

# uncontrolled transition probability
# transitions from a given node have equal probabilities
print("Define random walk on graph")
p = adj.copy().astype(float)
for k in range(N):
	p[:,k] /= np.sum(p[:,k])	# normalize each column

# define target points
end = np.sort(np.unique([N-2, N-1]))

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
Z[np.delete(np.arange(N), end)] = solveSVD(mat, vec)


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
pos = nx.spring_layout(G, iterations=50)

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
nx.draw(G, pos, node_color=np.log(Z), node_size=80, cmap=plt.cm.Spectral, with_labels=True, font_size=6)

plt.sca(ax[1,0])
ax[1,0].set_title("Reference tr. prob.")
f = ax[1,0].imshow(p)
# fig.colorbar(f, ax=ax[1,0])

plt.sca(ax[1,1])
ax[1,1].set_title("Optimal tr. prob.")
f = ax[1,1].imshow(u)
# fig.colorbar(f, ax=ax[1,1])

plt.savefig("matrices_%.1f.png"%tradeoff)
