
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 14:49:08 2023

@author: evi
"""

import networkx as nx
from plotting import plot_inside_response_heatmaps
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import config_file as c
import matplotlib.patches as patches
from utility import get_result_paths,get_timing_responses
import torch

def plot_network(timing_res, timing_labels,x0="duration",y0="period",split = 0,normalize = True):
    """
    Shows activation and weights of the network for all timings
    :param timing_res: timing response of shape (num_gates, num_timings, num_layers, num_hidden)
    :param timing_labels: fit_labels for timing response of shape (num_timings, 2)
    :return: None
    """
    
    num_splits, num_timings, num_layers, num_nodes = timing_res.shape
    timing_res = timing_res[split, ...]
    
    
    
    G = nx.DiGraph()
    node_counter = 0
    img=mpimg.imread(c.RESULTS_DIR + '/black.png')
    G.add_node(node_counter,image= img, pos=(-1, num_nodes//2))
    
    
    
    for node_id in np.arange(num_nodes):
        for layer_id in np.arange(num_layers):
            node_counter = node_counter + 1
            bool_exist = plot_inside_response_heatmaps(timing_res, timing_labels, node_id, layer_id, x0, y0,normalize = normalize)
            if bool_exist == False:
                continue
            img=mpimg.imread(c.RESULTS_DIR + '/node_activation.png')
            G.add_node(node_counter,image= img, pos=(layer_id, num_nodes - 1 - node_id))            

            file = 'node_activation.png'  
            location = c.RESULTS_DIR
            path = os.path.join(location, file)  
            os.remove(path)
          
    node_counter = node_counter + 1
    img=mpimg.imread(c.RESULTS_DIR + '/black.png')
    G.add_node(node_counter,image= img, pos=(num_layers, num_nodes//2))
    
    fig=plt.figure(figsize=(12,3))
    ax=plt.subplot(111)
    #ax.set_aspect('equal') # <-- not set the aspect ratio to equal
    
    #network = torch.load(net_path)
    
    '''
    # draw vertical edges using a larger node size
    G.add_edge(4,7)
    G.add_edge(5,7)
    G.add_edge(6,7)
    nx.draw_networkx_edges(G,pos,ax=ax, node_size=8000, arrowsize=30)
    
    # add the horizontal edges with a smaller node size
    G.add_edge(0,1)
    G.add_edge(1,2)
    G.add_edge(2,3)
    G.add_edge(3,4)
    G.add_edge(4,5)
    G.add_edge(6,5)
    nx.draw_networkx_edges(G,pos,ax=ax, node_size=3000, arrowsize=30)
    
    '''
    # trans=ax.transData.transform
    # trans2=fig.transFigure.inverted().transform
   
    piesize=0.4 # this is the image size
    p2=piesize/2.0
    pos=nx.get_node_attributes(G,'pos')

    for n in G:
        # xx,yy=trans(pos[n]) # figure coordinates
        # xa,ya=trans2((xx,yy)) # axes coordinates
        # a = plt.axes([xa-p2,ya-p2, piesize, piesize])
        a = plt.axes([pos[n][0],pos[n][1], piesize, piesize])
        a.set_aspect('equal')
        a.imshow(G.nodes[n]['image'])
        
            
        a.axis('off')
    ax.axis('off')
    plt.show()
    
    
    '''
    G=nx.Graph()
 
    print(G.nodes())
    G.add_edge(0,1)
    G.add_edge(0,2)
    G.add_edge(0,3)
    G.add_edge(0,4)
    G.add_edge(0,5)
    print(G.edges())
    pos=nx.circular_layout(G)
    
    fig=plt.figure(figsize=(5,5))
    ax=plt.subplot(111)
    ax.set_aspect('equal')
    nx.draw_networkx_edges(G,pos,ax=ax)
    
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    
    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform
    
    piesize=0.2 # this is the image size
    p2=piesize/2.0
    for n in G:
        xx,yy=trans(pos[n]) # figure coordinates
        xa,ya=trans2((xx,yy)) # axes coordinates
        a = plt.axes([xa-p2,ya-p2, piesize, piesize])
        a.set_aspect('equal')
        a.imshow(G.node[n]['image'])
        a.axis('off')
    ax.axis('off')
    plt.show()
    
plot_network()'''



def plot_network_including_edges(results_dir,kwarg,x0="duration",y0="period",split = 0,normalize = True):
    """
    Shows activation and weights of the network for all timings
    :param timing_res: timing response of shape (num_gates, num_timings, num_layers, num_hidden)
    :param timing_labels: fit_labels for timing response of shape (num_timings, 2)
    :return: None
    """
    
    

    net_path, _, _, _, _, _, _, _, _, _ = get_result_paths(results_dir, kwarg,None)
    timing_res, timing_labels = get_timing_responses(results_dir, kwarg)
    
    checkpoint = torch.load(net_path, map_location=c.DEVICE)
    network_weights = checkpoint["model_state_dict"]

    # predictions network
    response_labels = timing_labels/1000
    response_labels = np.asarray(response_labels.T)    
    
    
    num_splits, num_timings, num_layers, num_nodes = timing_res.shape
    timing_res = timing_res[split, ...]
    
    
    
    G = nx.DiGraph()
    node_counter = 0
    img=mpimg.imread(c.RESULTS_DIR + '/black.png')
    G.add_node(node_counter,image= img, pos=(-1, num_nodes//2))
    
    edge_color = []
    previous_nodes = [0]
    for layer_id in np.arange(num_layers):
        store_previous_nodes = []
        for node_id in np.arange(num_nodes):
        
            node_counter = node_counter + 1
            bool_exist = plot_inside_response_heatmaps(timing_res, timing_labels, node_id, layer_id, x0, y0,normalize = normalize)
            if bool_exist == False:
                continue
            img=mpimg.imread(c.RESULTS_DIR + '/node_activation.png')
            G.add_node(node_counter,image= img, pos=(layer_id, num_nodes-1-node_id))            

            file = 'node_activation.png'  
            location = c.RESULTS_DIR
            path = os.path.join(location, file)  
            os.remove(path)
            
            for previous_node_id, previous_node in enumerate(previous_nodes):
                u = list(G.nodes)[-1]
                v = previous_node
                string_get_weights = 'rnn.layers.' + str(layer_id) + '.rnn_cell.weight_ih'
                weight_between_nodes = network_weights[string_get_weights]
                d = abs(weight_between_nodes[previous_node_id][node_id])
                if float(weight_between_nodes[previous_node_id][node_id]) < 0:
                    edge_color.append('red')
                else:
                    edge_color.append('blue')
                G.add_weighted_edges_from([(u,v,d)])
                
                string_get_recurrent_weights = 'rnn.layers.' + str(layer_id) + '.rnn_cell.weight_hh'
                recurrent_weights = network_weights[string_get_recurrent_weights]
                u = list(G.nodes)[-1]
                v = list(G.nodes)[-1]
                d = abs(recurrent_weights[node_id])
                if float(recurrent_weights[node_id]) < 0:
                    edge_color.append('red')
                else:
                    edge_color.append('blue')
                G.add_weighted_edges_from([(u,v,d)])
            
            store_previous_nodes.append(node_counter)
        previous_nodes = store_previous_nodes
          
    node_counter = node_counter + 1
    img=mpimg.imread(c.RESULTS_DIR + '/black.png')
    G.add_node(node_counter,image= img, pos=(num_layers, num_nodes//2))
    
    # out node has differently organized weights because nnLinear transposes weights before
    for previous_node_id, previous_node in enumerate(previous_nodes):
        u = list(G.nodes)[-1]
        v = previous_node
        string_get_weights = 'out.0.weight'
        weight_between_nodes = network_weights[string_get_weights]
        d = abs(weight_between_nodes[0][previous_node_id])
        if float(weight_between_nodes[0][previous_node_id]) < 0:
            edge_color.append('red')
        else:
            edge_color.append('blue')
        G.add_weighted_edges_from([(u,v,d)])
    
    fig=plt.figure(figsize=(12,3))
    ax=plt.subplot(111)
    #ax.set_aspect('equal') # <-- not set the aspect ratio to equal
    
    #network = torch.load(net_path)    
    # G.add_edge(0,4)
    # G.add_edge(0,41)
    # G.add_edge(4,10)
###    G.add_weighted_edges_from([(0,4,1),(0,23,10),(4,10,0.1)])
    edgewidth = [d['weight'] for (u,v,d) in G.edges(data=True)]
   
    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform
   
    piesize=0.1 # this is the image size
    p2=piesize/2.0
    pos=nx.get_node_attributes(G,'pos')
    nx.draw_networkx_edges(G, pos,width=edgewidth,edge_color=edge_color)

    for n in G:
        xx,yy=trans(pos[n]) # figure coordinates
        xa,ya=trans2((xx,yy)) # axes coordinates
        a = plt.axes([xa-p2,ya-p2, piesize, piesize])
        # a = plt.axes([pos[n][0],pos[n][1], piesize, piesize])
        a.set_aspect('equal')
        a.imshow(G.nodes[n]['image'])
        
            
        a.axis('off')
    ax.axis('off')
    

    
    plt.show()

