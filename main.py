import numpy as np
import networkx as nx
import random

def calculate_w(i,j,z):
    return theta[j,z] * alpha[i,j]

def calculate_g(i, y, z):
    """eq. 1"""

    if i != y:
        ret = w[i,y,z] / sum( w[i,j,z] + w[j,i,z] for j in NB(i) )
        return ret
    else:
        ret = sum( w[j,i,z] for j in NB(i) ) / sum( w[i,j,z] + w[j,i,z] for j in NB(i) )
        return ret

def calculate_b(i,j,z):
    """eq. 8"""
    
    _a = g[i,j,z]
    _b = sum( [ g[i,k,z] for k in NB(i) + [i] ] )
    return np.log( _a / _b  )

def update_r(i,j,z):
    """eq. 5"""

    ret = b[i,j,z] - np.max( [ (b[i,k,z] + a[i,k,z]) for k in NB(j) ])
    return ret

def update_a6(j, z):
    """eq. 6 & 7"""
    
    ret = max(  min(r[k,j,z], 0.) for k in NB(j) )
#    ret = sum([max(r[k,j,z], 0) for k in NB(j)] )
    return ret

def update_a7(i,j,z):

    if i in NB(j):
        _a = max( r[j,j,z], 0. )
        _b1 = min( r[j,j,z], 0. )
        _b2 = max( [ min( r[k,j,z], 0. ) for k in NB(j) if k != i ] )
        _b = 0. - _b1 - _b2

#        np.min( 0, [ r[j,j,z]] )

        return min(_a, _b)

def compute_mu(s,t,z):
    """eq. 9"""

    return 1. / ( 1. + math.exp( -1.* (r[t,s,z] + a[t,s,z]) ) )

def NB(i):
    """neighbors of i"""

    return G[i].keys()

N = 100
Z = 5
degree = 3

print "Generate random graph."
G = nx.random_regular_graph(degree, N)
for i,j in G.edges():
    G[i][j]['weight'] = random.random()

print "Generate alpha & theta"
alpha = nx.to_numpy_matrix(G)
theta_ = np.random.rand(N,Z)
theta = np.zeros((N,Z))
for i in xrange(N):
    theta[i,:] = np.array([theta_[i,j]/np.sum(theta_[i,:]) for j in xrange(Z)])

print "Generate w"
w = np.zeros((N, N, Z))
for i,j in G.edges():
    for z in xrange(Z):
        w[i,j,z] = calculate_w(i,j,z)

print "1.1 Calculate the node feature function g(v_i, y_i, z);"
g = np.zeros((N, N, Z))
for i in xrange(N):
    for j in xrange(N):
        for z in xrange(Z):
            g[i,j,z] = calculate_g(i,j,z)

print "1.2 Calculate b_ij^z according to Eq. 8;"
b = np.zeros((N, N, Z))
for i,j in G.edges():
    for z in xrange(Z):
        b[i,j,z] = calculate_b(i, j, z)

print "1.3 Initalize all {r_ij^z} <- 0;"
r = np.zeros((N, N, Z)) # 1.3
#a = np.random.rand(N, N, Z)
a = np.zeros((N, N, Z))

print np.min(b)
print np.mean(b)
print np.max(b)

for i in xrange(3):

    print "="*40
    print "iteration {0}".format(i)
    
    print "-"*40
    print "1.4 - 1.7"
    for i,j in G.edges():
        for z in xrange(Z):
            new_r = update_r(i,j,z)
            r[i,j,z] = new_r

    print np.min(r)
    print np.mean(r)
    print np.max(r)

    print "1.8 - 1.10"
    for j in xrange(N):
        for z in xrange(Z):
            a[j,j,z] = update_a6(j,z)

    print "1.11 - 1.13"
    for i,j in G.edges():
        for z in xrange(Z):
            if i != j:
                a[i,j,z] = update_a7(i,j,z)

    print np.min(a)
    print np.mean(a)
    print np.max(a)

#    nx.write_edgelist(G, "./edges")
