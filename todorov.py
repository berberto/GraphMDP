import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



# #
# #	define from NX graph
# #
# graph = nx.read_graphml("v_graph_9Oct2020.graphml")
# adj = np.adjacency_matrix(graph).todense()


#
#	define random (Erdos-Renyi) graph
#
np.random.seed(1990)
N = 20		# number of nodes
deg = N/5.		# average degree
adj = np.zeros((N,N), dtype=int)
for i in range(N):
	for j in range(i+1,N):
		if np.random.rand() < deg/(N-1):
			adj[i,j] += 1
			adj[j,i] += 1

# vector of non vanishing edges (entries of adjacency matrix)
edges = np.stack(np.where(adj == 1)).T

# uncontrolled transition probability
# transitions from a given node have equal probabilities
p = adj.copy().astype(float)
for k in range(N):
	p[:,k] /= np.sum(p[:,k])	# normalize each column

# define target points
end = [N-1,N-2]

# "end" nodes are absorbing
for term in end:
	p[:,term] = 0.
	p[term,term] = 1.

# objective function parameters
q = 1.		# cost per jump
eps = 1.	# weight of KL

# tilted generator
pt = p.copy()
for k in range(N):
	if not k in end:
		pt[:,k] *= np.exp(-q/eps)

# solve for desirability
Nter = len(end)
Nrec = N - Nter
M = pt.T
Zrec = np.linalg.solve(M[:Nrec,:Nrec] - np.eye(Nrec), -np.sum(M[:Nrec,Nrec:], axis=1))
Zter = np.ones(Nter)
print(Zrec)
print(Zter)
Z = np.concatenate((Zrec,Zter))
print("\ndesirability vector:    ", Z)

# controlled transition probability
u = pt.copy()
for kp in range(N):
	u[kp] *= Z[kp]
for k in range(N):
	u[:,k] /= Z[k]
print("u correctly normalized: ", np.allclose(np.sum(u,axis=0), 1, atol=1e-15))

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

plt.savefig("matrices.png")
plt.show()