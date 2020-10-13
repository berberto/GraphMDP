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
# graph = nx.read_graphml("v_graph_9Oct2020.graphml")
# adj = np.adjacency_matrix(graph).todense()

#
#	define random (Erdos-Renyi) graph
#
np.random.seed(1990)
N = 20			# number of nodes
deg = (N-1)/5.	# average degree (a fifth of the total nodes)
adj = np.zeros((N,N), dtype=int)
for i in range(N):
	for j in range(i+1,N):
		if np.random.rand() < deg/(N-1):
			adj[i,j] += 1
			adj[j,i] += 1

# uncontrolled transition probability
# transitions from a given node have equal probabilities
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
q = 1.		# cost per jump
eps = .6	# weight of KL
tradeoff = q/eps	# only thing that matters

# tilted generator
pt = p.copy()
for k in range(N):
	if not k in end:
		pt[:,k] *= np.exp(-tradeoff)

# solve for desirability (at non-absorbing states)
M = pt.T - np.eye(N)			  # tilted generator - identity
mat = np.delete(M, end, axis=1)   # delete columns corresponding to target nodes
vec = - np.sum(M[:,end], axis=1)  # vector implementing "boundary" conditions
Z = np.ones(N)
Z[np.delete(np.arange(N), end)] = solveSVD(mat, vec)

# test solution:
print("correct solution: ", np.allclose( Z - np.dot(pt.T, Z), 0), "\n")
print("desirability vector:    ", Z, "\n")


# controlled transition probability
u = pt.copy()
for kp in range(N):
	u[kp] *= Z[kp]
for k in range(N):
	u[:,k] /= Z[k]
print("u correctly normalized: ", np.allclose(np.sum(u,axis=0), 1, atol=1e-15), "\n")

#
#	PLOTS
#
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)

ax1.set_title("Adjacency matrix")
f = ax1.imshow(adj)
fig.colorbar(f, ax=ax1)

ax2.set_title("Reference tr. prob.")
f = ax2.imshow(p)
fig.colorbar(f, ax=ax2)

ax3.set_title("Optimal tr. prob.")
f = ax3.imshow(u)
fig.colorbar(f, ax=ax3)

plt.savefig("matrices_%.1f.png"%tradeoff)
plt.show()
