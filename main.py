"""
Toy implementation of the Topical Affinity Propagation algorithm for estimating
topic-based social influence graphs, from Tang et al. 2009. "Social influence
analysis in large-scale networks." Conference on Knowledge Discovery & Data 
Mining 2009, June 28 - July 1, 2009, Paris, France. <http://bit.ly/1nOUh9e>.
"""

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
        ret = sum( w[j,i,z] for j in NB(i) ) / sum( w[i,j,z] + w[j,i,z] 
                                                                for j in NB(i) )
        return ret

def calculate_b(i,j,z):
    """eq. 8"""
    
    _a = g[i,j,z]
    _b = sum( [ g[i,k,z] for k in NB(i) + [i] ] )
    return np.log(_a / _b)*-1.

def update_r(i,j,z):
    """eq. 5"""

    a_ = b[i,j,z]
    b_ = np.max( [ b[i,k,z] + a[i,k,z] for k in NB(j) ])
    
    ret = a_ - b_
    return ret

def update_a6(j, z):
    """eq. 6"""

    ret = np.max( [ np.max([r[k,j,z], 0.]) for k in NB(j) ])
    
    return ret

def update_a7(i,j,z):
    """eq. 7"""
    if i in NB(j):
        _a = min( r[j,i,z], 0. )
        _b1 = min( r[j,i,z], 0. )
        try:
            _b2 = max( [ min( r[k,j,z], 0. ) for k in NB(j) if k != i ] )
        except:
            _b2 = 0.
        
        _b = 0. - _b1 - _b2
        return min(_a, _b)

def compute_mu(s,t,z):
    """eq. 9"""
    
    return 1. / ( 1. + math.exp( -1.* (r[t,s,z] + a[t,s,z]) ) )

def NB(i):
    """neighbors of i"""

    return G[i].keys()
    
def a_step():
    """1.4 - 1.7"""
    for i,j in G.edges():
        for z in xrange(Z):
            r[i,j,z] = update_r(i,j,z)
            r[j,i,z] = update_r(j,i,z)
    for z in xrange(Z):
        shift = r[:,:,z] + np.min(r[:,:,z])
        r[:,:,z] = shift/np.sum(shift)
        
def b_step():
    """1.8 - 1.10"""
    for j in xrange(N):
        for z in xrange(Z):
            a[j,j,z] = update_a6(j,z)

def c_step():
    """1.11 - 1.13"""
    for i,j in G.edges():
        for z in xrange(Z):
            if i != j:
                a[i,j,z] = update_a7(i,j,z)
                a[j,i,z] = update_a7(j,i,z)
    for z in xrange(Z):
        shift = a[:,:,z] + np.min(a[:,:,z])
        a[:,:,z] = shift/np.sum(shift)  

def iterate():
    a_step()
    b_step()
    c_step()

if __name__ == '__main__':
    ############################################################################
    # Topic modeling: LDA Gibbs algorithm in the Indiana Philosophy Ontology   #
    #  Project's Vector Space Model package (https://github.com/inpho/vsm).    #
    #                                                                          #
    # Coauthorship network: built with Tethne http://diging.github.io/tethne/  #
    #  using data from the ISI Web of Science.                                 #
    ############################################################################

    from vsm.model import ldacgsmulti
    import tethne.networks as nt
    import pickle
    import math

    # Load topic modeling results.
    modelpath = "/Users/erickpeirson/Dropbox/DigitalHPS/ED Journals/models/20131208_LSA_Gibbs_EvoDevo.npz"
    l = ldacgsmulti.LdaCgsMulti.load(modelpath)
    Z = 40
    
    # Build a coauthorship network.
    print "Build a coauthorship network."
    papers = pickle.load(open("/Users/erickpeirson/Dropbox/DigitalHPS/ED Journals/models/20131208_papers.pickle", "r"))
    G = nt.authors.coauthors(papers, edge_attribs=['wosid'])
    
    #   Build weights for coauthorship network; this should be unnecessary soon.
    for i,j,d in G.edges(data=True):
        if type(d['wosid']) is list:
            G[i][j]['weight'] = float(len(d['wosid']))
        else:
            G.remove_edge(i,j)
    
    #   Looking only at the largest connected component, for now.
    G = nx.connected_component_subgraphs(G)[0]
    N = len(G.nodes())
    
    # Build theta = { P(topic|author) } matrix.
    print "Build theta = { P(topic|author) } matrix."
    theta = np.zeros((N, Z))
    at = {}
    
    #   Get representation of each topic in each author's papers.
    for q in xrange(len(papers)):
        p = papers[q]
        pa = [ ' '.join([p['aulast'][i], p['auinit'][i]]) 
                for i in xrange(len(p['aulast'])) ]
        for a in pa:
            if a in G.nodes():
                try:
                    at[a].append(l.doc_top[q,:])
                except KeyError:
                    at[a] = [ l.doc_top[q,:] ]

    #   Now generate the P(t|a) matrix.
    q = 0
    for a,values in at.iteritems():
        if len(values) > 1:
            varray = np.array(values)
            V = np.zeros((Z))
            for i in xrange(Z):
                V[i] = np.sum(varray[:,i])
        else:
            V = values[0]

        val = np.array([ v/np.sum(V) for v in V])            
        theta[q,:] = val
        q+=1
        
    # Re-index coauthorship graph.
    print "Re-index coauthorship graph."
    G_ = nx.Graph()
    adict = {}
    adict_ = {}
    q = 0
    for a in at.keys():
        adict[a] = q
        adict_[q] = a
        q += 1
        
    for i,j,d in G.edges(data=True):
        G_.add_edge(adict[i],adict[j],weight=d['weight'])
        
    G = G_
    
    # Clear up some memory.
    del l, G_, papers
    
    #####################
    #   the good stuff  #
    #####################
    
    alpha = nx.to_numpy_matrix(G)
    
    # Calculate w_ij
    print "Calculate w_ij"
    w = np.zeros((N, N, Z))
    for i,j in G.edges():
        for z in xrange(Z):
            w[i,j,z] = calculate_w(i,j,z)
            w[j,i,z] = calculate_w(j,i,z)
            
    # 1.1 Calculate the node feature function g(v_i, y_i, z).
    print "1.1 Calculate the node feature function g(v_i, y_i, z)."
    g = np.zeros((N, N, Z))
    for i,j in G.edges():
        for z in xrange(Z):
                g[i,j,z] = calculate_g(i,j,z)
                g[j,i,z] = calculate_g(j,i,z)
    del w
    
    # 1.2 Calculate b_ij^z according to Eq. 8.
    print "1.2 Calculate b_ij^z according to Eq. 8."
    b = np.zeros((N, N, Z))
    for i,j in G.edges():
        for z in xrange(Z):
            b[i,j,z] = calculate_b(i, j, z)
            b[j,i,z] = calculate_b(j, i, z)
    del g
    
    # 1.3 Initalize all {r_ij^z} <- 0.
    print "1.3 Initalize all {r_ij^z} <- 0."
    r = np.zeros((N, N, Z))
    a = np.zeros((N, N, Z))
    
    # 1.4. Start iterations.
    print "1.4. Start iterations."
    r_m = []    # Crude way to watch for coalescence.
    
    iterations = 50
    for i in xrange(iterations):
        if i % 10 == 0:
            print "iteration " + str(i) + ": " + str(np.mean(r))
        iterate()
        r_m.append(np.mean(r))
    
    # 1.15 - 1.20
    #   Compute mu_st for each pair of neighboring nodes.
    #   Generate G_z = (V_z, E_z) for every topic z according to {mu_st}.
    
    print "1.15 - 1.20"
    print "Compute mu_st for each pair of neighboring nodes."
    print "Generate G_z = (V_z, E_z) for every topic z according to {mu_st}."
    with open("./evo_devo.csv", "w") as f:
        for i,j in G.edges():
            f.write("\t".join([ str(i),str(j)] + 
                              [ str(compute_mu(i, j, z)) 
                                     for z in xrange(Z) ]) + "\n")
            f.write("\t".join([ str(j),str(i)] + 
                              [ str(compute_mu(j, i, z)) 
                                     for z in xrange(Z) ]) + "\n")

    # P(t|a) as node attributes.
    with open("./evo_devo_topics.csv", "w") as f:
        for i in G.nodes():
            f.write( "\t".join([ str(i)] + 
                               [ str(theta[i,z]) 
                                  for z in xrange(Z) ]) + "\n")
    print "Done."
    # TODO: pruning networks