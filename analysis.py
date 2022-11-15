import pandas as pd
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

options = {
    # 'node_color': 'blue',
    # 'node_size': 100,
    'width': 3,
    'arrowstyle': '-|>',
    'arrowsize': 12,
    'font_size' :22,
    'font_color':'whitesmoke'
}

roompos = {
    'I': np.array((0, 1.5)),
    1  : np.array((-2.5, 1.25)),
    6  : np.array((2.5, 1.25)),
    3  : np.array((-1, 0)),
    4  : np.array((1, 0)),
    2  : np.array((-3.5, 0)),
    5  : np.array((3.5, 0)),
    'A': np.array((4, .7)),
    'B': np.array((-4,.7))
}



def format_raw(filename='Observation-dataTable-2022-Raffy-Traas.xlsx'):
    
    # column renaming
    raw = pd.read_excel('Observation-dataTable-2022-Raffy-Traas.xlsx', header=10)
    oldcolnames = list(raw.columns)
    newcolnames = ['t', 'activity', 'table', 'si', 'ti', 'ii', 'ui', 'tid', 'to', 'tspan', 'comments']
    raw.rename(columns = {old : new for old, new in zip(*(list(raw.columns), newcolnames))}, inplace=True)
    raw = raw.convert_dtypes()


    # create interaction dataframe
    idf = raw.iloc[np.where(~pd.isna(raw['si']))]

        # column reformatting
    initiator = []
    initiator_ids = []
    for i, row in idf.iterrows():
        if pd.isna(row['si']) or pd.isna(row['ti']) or pd.isna(row['ii']) or pd.isna(row['ui']):
            initiator.append(None)
            initiator_ids.append(None)
        if row['si']:
            initiator.append('si')
            if pd.isna(row['tid']):
                initiator_ids.append('U')
            else:
                initiator_ids.append(row['tid'])
            # initiator_ids.append('S')
        elif row['ti']:
            initiator.append('ti')
            initiator_ids.append(row['tid'])
        elif row['ii']:
            initiator.append('ii')
            initiator_ids.append('I')
        elif row['ui']:
            initiator.append('ui')
            if pd.isna(row['tid']):
                initiator_ids.append('U')
            else:
                initiator_ids.append(row['tid'])
    new = idf[['t', 'activity', 'table']]
    new = new.assign(i = initiator)
    new = new.assign(iid = initiator_ids)
    new = new.assign(to = idf['to'])
    new = new.assign(tspan = idf['tspan'])
    new = new.assign(comments = idf['comments'])
    idf = new
    idf = idf.convert_dtypes()
    idf['t'] = idf['t'].apply(pd.Timestamp)
    idf['to'] = idf['to'].apply(str).apply(pd.Timestamp)
    idf['dt'] = pd.to_timedelta(idf['to'] - idf['t'])
    idf['dt'] = idf['dt'].apply(lambda x: x.seconds / 60)
    idf = idf.assign(tfromprevious = pd.to_timedelta(idf['to'].diff()))
    idf['tfromprevious'] = idf['tfromprevious'].apply(lambda x: x.seconds / 60)

    # Resolve missing data
        # Take care of row 64, where the instructor takes over the interaction at table 6
    idf.loc[64, 'table'] = 6

        # row 102: the `iid` should be the TA "A", because they answered this question 
        # (about center of gravity among others) when they were in the middle of the 
        # interaction on row 99.
    idf.loc[idf['comments'].str.contains('center of gravity'), 'iid'] = 'A'

        # row 121: a similar case to row 102.  `iid` should be TA "A"
    idf.loc[idf['to'] == pd.Timestamp('2022-11-01 14:27:48.809'), 'iid'] = 'A'

        # row 114: the `iid` should be TA "B"
    idf.loc[idf['to'] == pd.Timestamp('2022-11-01 14:24:56.832'), 'iid'] = 'B'

        # row 103: the `iid` should be the instructor "I"
    idf.loc[idf['comments'].str.contains('qeustion about class schedule'), 'iid'] = 'I'

        # row 110: the `iid` should be the instructor "I"
    idf.loc[idf['comments'].str.contains('~3min'), 'iid'] = 'I'

        # row 124: the `iid` should be the instructor "I"
    idf.loc[idf['comments'].str.contains('student handed in assignment'), 'iid'] = 'I'

        # row 92: the `iid` should be the instructor "I"
    idf.loc[idf['comments'].str.contains('Informing student of their office hours'), 'iid'] = 'I'

        # row 127 and row 128: the `iid` should be the instructor "I"
    idf.loc[idf['comments'].str.contains("Asked when instructor's office hours"), 'iid'] = 'I'
    idf.loc[idf['comments'].str.contains("asked question after class"), 'iid'] = 'I'

    # create dataframe for directed graph
    didf = idf.copy()
    didf = didf.assign(todrop = [np.nan if i == 'ui' else i for i in didf['i']])
    didf = didf.dropna(subset=['todrop'])
    source = []
    target = []
    for i, row in didf.iterrows():
        if row['i'] == 'ii':
            source.append('I')
            target.append(row['table'])
        elif row['i'] == 'ti':
            source.append(row['iid'])
            target.append(row['table'])
        elif row['i'] == 'si':
            source.append(row['table'])
            target.append(row['iid'])
    didf = didf.assign(source = source)
    didf = didf.assign(target = target)
    
    idf = idf.convert_dtypes()
    idf['table'] = idf['table'].astype('category')
    idf['activity'] = idf['activity'].astype('category')
    idf['iid'] = idf['iid'].astype('category')
    idf['tspan'] = idf['tspan'].astype('category')
    
    

    return raw, idf, didf


def unweighted_graph(df, source='source', target='target'):

    # Create an unweighted undirected graph using the NetworkX's from_pandas_edgelist method.
    # The column participantID.A is used as the source and participantID.B as the target.
    G = nx.from_pandas_edgelist(df, 
                                 source=source, 
                                 target=target,
                                 
                                 create_using=nx.Graph())
    return G

def weighted_graph(df, source='source', target='target'):
    # Get the count of interactions between participants and display the top 5 rows.
    grp_interactions = pd.DataFrame(df.groupby([source, target]).size(), columns=['counts']).reset_index()
    # Create a directed graph with an edge_attribute labeled counts.
    g = nx.from_pandas_edgelist(grp_interactions, 
                                 source=source, 
                                 target=target, 
                                 edge_attr='counts', 
                                 create_using=nx.DiGraph())
    
    # Set all the weights to 0 at this stage. We will add the correct weight information in the next step.
    G = nx.Graph()
    G.add_edges_from(g.edges(), counts=0)
    
    for u, v, d in g.edges(data=True):
        G[u][v]['counts'] += d['counts']
        
    return grp_interactions, G, g
    



def degree_distribution(G):
    
    ## DEGREE DISTRIBUTION
    
    # Extract the degree values for all the nodes of G
    degrees = []
    for (nd,val) in G.degree():
        degrees.append(val)
    
    # Plot the degree distribution histogram.
    out = plt.hist(degrees, bins=50)
    plt.title("Degree Histogram")
    plt.ylabel("Frequency Count")
    plt.xlabel("Degree")
    
    # Logarithmic plot of the degree distribution.
    values = sorted(set(degrees))
    hist = [list(degrees).count(x) for x in values]
    out = plt.loglog(values, hist, marker='o')
    plt.title("Degree Histogram")
    plt.ylabel("Log(Frequency Count)")
    plt.xlabel("Log(Degree)")

def degree_centrality(df, G, source='source', target='target', ax=None):
    
    ## DEGREE CENTRALITY
    
    # Plot degree centrality.
    if ax is None:
        fig, ax = plt.subplots()
    call_degree_centrality = nx.degree_centrality(G)
    colors =[call_degree_centrality[node] for node in G.nodes()]
    # pos = graphviz_layout(G, prog='dot')
    # nx.draw_networkx(G, pos, node_color=colors, node_size=300, with_labels=False, arrows=True, **options)
    nx.draw_networkx(G, node_color=colors, node_size=300, with_labels=True, arrows=True, **options)
    ax.axis('off')
    
    # Arrange in descending order of centrality and return the result as a tuple, i.e. (participant_id, deg_centrality).
    t_call_deg_centrality_sorted = sorted(call_degree_centrality.items(), key=lambda kv: kv[1], reverse=True)

    # Convert tuple to pandas dataframe.
    df_call_deg_centrality_sorted = pd.DataFrame([[x,y] for (x,y) in t_call_deg_centrality_sorted], 
                                                 columns=['participantID', 'deg.centrality'])
    
    # Top 5 participants with the highest degree centrality measure.
    print(df_call_deg_centrality_sorted.head())
    
    # Number of unique actors associated with each of the five participants with highest degree centrality measure.
    for node in df_call_deg_centrality_sorted.head().participantID:
        print('Node: {0}, \t num_neighbors: {1}'.format(node, len(list(G.neighbors(node)))))
    
    # Total call interactions are associated with each of these five participants with highest degree centrality measure.
    for node in df_call_deg_centrality_sorted.head().participantID:
        outgoing_call_interactions = df['source']==node
        incoming_call_interactions = df['target']==node
        all_call_int = df[outgoing_call_interactions | incoming_call_interactions]
        print('Node: {0}, \t total number of calls: {1}'.format(node, all_call_int.shape[0]))
        
    return df_call_deg_centrality_sorted, call_degree_centrality
        
def closeness_centrality(df, G, source='source', target='target', ax=None):
        
    ## CLOSENESS CENTRALITY
    
    # Plot closeness centrality.
    if ax is None:
        fig, ax = plt.subplots()
    call_degree_centrality = nx.degree_centrality(G)
    call_closeness_centrality = nx.closeness_centrality(G)
    colors = [call_closeness_centrality[node] for node in G.nodes()]
    # pos = graphviz_layout(G, prog='dot')
    # nx.draw_networkx(G, pos=pos,node_color=colors, with_labels=False)
    nx.draw_networkx(G, node_color=colors, node_size=300, with_labels=True, arrows=True, **options)
    _ = plt.axis('off')
    
    # Arrange participants according to closeness centrality measure, in descending order. 
    # Return the result as a tuple, i.e. (participant_id, cl_centrality).
    t_call_clo_centrality_sorted = sorted(call_closeness_centrality.items(), key=lambda kv: kv[1], reverse=True)

    # Convert tuple to pandas dataframe.
    df_call_clo_centrality_sorted = pd.DataFrame([[x,y] for (x,y) in t_call_clo_centrality_sorted], columns=['participantID', 'clo.centrality'])
    
    return df_call_clo_centrality_sorted, call_closeness_centrality

def betweenness_centrality(df, G, source='source', target='target', ax=None):
    
    # BETWEENNESS CENTRALITY
    # Plot betweenness centrality.
    if ax is None:
        fig, ax = plt.subplots()
    call_degree_centrality = nx.degree_centrality(G)
    call_betweenness_centrality = nx.betweenness_centrality(G)
    colors =[call_betweenness_centrality[node] for node in G.nodes()]
    # pos = graphviz_layout(G, prog='dot')
    # nx.draw_networkx(G, pos=pos, node_color=colors, with_labels=False)
    nx.draw_networkx(G, node_color=colors, node_size=300, with_labels=True, arrows=True, **options)
    _ = plt.axis('off')
    
    # Arrange participants according to betweenness centrality measure, in descending order. 
    # Return the result as a tuple, i.e. (participant_id, btn_centrality). 
    t_call_btn_centrality_sorted = sorted(call_betweenness_centrality.items(), key=lambda kv: kv[1], reverse=True)

    # Convert tuple to a Pandas DataFrame.
    df_call_btn_centrality_sorted = pd.DataFrame([[x,y] for (x,y) in t_call_btn_centrality_sorted], 
                                                 columns=['participantID', 'btn.centrality'])
    return df_call_btn_centrality_sorted, call_betweenness_centrality
    
def eigenvector_centrality(df, G, source='source', target='target', ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    call_degree_centrality = nx.degree_centrality(G)
    # Plot eigenvector centrality.
    call_eigenvector_centrality = nx.eigenvector_centrality(G)
    colors = [call_eigenvector_centrality[node] for node in G.nodes()]
    # pos = graphviz_layout(G, prog='dot')
    # nx.draw_networkx(G, pos=pos, node_color=colors,with_labels=False)
    nx.draw_networkx(G, node_color=colors, node_size=300, with_labels=True, arrows=True, **options)
    _ = plt.axis('off')
    
    # Arrange participants according to eigenvector centrality measure, in descending order. 
    # Return the result as a tuple, i.e. (participant_id, eig_centrality).
    t_call_eig_centrality_sorted = sorted(call_eigenvector_centrality.items(), key=lambda kv: kv[1], reverse=True)

    # Convert tuple to pandas dataframe.
    df_call_eig_centrality_sorted = pd.DataFrame([[x,y] for (x,y) in t_call_eig_centrality_sorted], 
                                                 columns=['participantID', 'eig.centrality'])
    
    return df_call_eig_centrality_sorted, call_eigenvector_centrality

# Execute this cell to define a function that produces a scatter plot.
def centrality_scatter(dict1,dict2,path="",ylab="",xlab="",title="",line=False):
    '''
    The function accepts two dicts containing centrality measures and outputs a scatter plot
    showing the relationship between the two centrality measures
    '''
    # Create figure and drawing axis.
    fig = plt.figure(figsize=(7,7))
    
    # Set up figure and axis.
    fig, ax1 = plt.subplots(figsize=(8,8))
    # Create items and extract centralities.
    
    items1 = sorted(list(dict1.items()), key=lambda kv: kv[1], reverse=True)
    
    items2 = sorted(list(dict2.items()), key=lambda kv: kv[1], reverse=True)
    xdata=[b for a,b in items1]
    ydata=[b for a,b in items2]
    ax1.scatter(xdata, ydata)

    if line:
        # Use NumPy to calculate the best fit.
        slope, yint = np.polyfit(xdata,ydata,1)
        xline = plt.xticks()[0]
        
        yline = [slope*x+yint for x in xline]
        ax1.plot(xline,yline,ls='--',color='b')
        # Set new x- and y-axis limits.
        plt.xlim((0.0,max(xdata)+(.15*max(xdata))))
        plt.ylim((0.0,max(ydata)+(.15*max(ydata))))
        # Add labels.
        ax1.set_title(title)
        ax1.set_xlabel(xlab)
        ax1.set_ylabel(ylab)
        
def centrality_measures(df, G, source='source', target='target'):
    df_call_btn_centrality_sorted, _ = betweenness_centrality(df, G, source, target)
    df_call_clo_centrality_sorted, _ = closeness_centrality(df, G, source, target)
    df_call_deg_centrality_sorted, _ = degree_centrality(df, G, source, target)
    df_call_eig_centrality_sorted, _ = eigenvector_centrality(df, G, source, target)
    
    m1 = pd.merge(df_call_btn_centrality_sorted, df_call_clo_centrality_sorted)
    m2 = pd.merge(m1, df_call_deg_centrality_sorted)
    df_merged  = pd.merge(m2, df_call_eig_centrality_sorted)
    
    return df_merged

def avgpath_diam(G):
    print('Diameter {}'.format(nx.diameter(G)))
    print('Average path length {:0.2f}'.format(nx.average_shortest_path_length(G)))
    return nx.average_shortest_path_length(G), nx.diameter(G)


def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


    
# def plot_graphs_v2(df):
#     ug, G = graphs(df)
#     room_positions = {
#         'I': np.array((0, 1.5)),
#         1  : np.array((-2.5, 1.25)),
#         6  : np.array((2.5, 1.25)),
#         3  : np.array((-1, 0)),
#         4  : np.array((1, 0)),
#         2  : np.array((-3.5, 0)),
#         5  : np.array((3.5, 0)),
#         'R': np.array((4, .7)),
#         'L': np.array((-4,.7))
#     }
#     pos = room_positions
#     nx.draw_networkx_nodes(G, pos, node_color = 'r', node_size = 100, alpha = 1)
#     nx.draw_networkx_labels(G, pos)
#     ax = plt.gca()
#     for e in G.edges:
#         ax.annotate(
#             "",
#             xy=pos[e[0]], xycoords='data',     
#             xytext=pos[e[1]], textcoords='data',
#             arrowprops=dict(
#                 arrowstyle="->", color="0.5",
#                 shrinkA=5, shrinkB=5,
#                 patchA=None, patchB=None,
#                 connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*1)),#2])),
#             ),
#         )
#     plt.axis('off')
#     plt.show()

# def testplot():
#     G=nx.MultiGraph ([(1,2),(1,2),(1,2),(3,1),(3,2),(2,3)])
#     pos = nx.random_layout(G)
#     nx.draw_networkx_nodes(G, pos, node_color = 'r', node_size = 100, alpha = 1)
#     nx.draw_networkx_labels(G, pos)#, labels={n:lab for n,lab in labels.items() if n in pos})
#     ax = plt.gca()
#     for e in G.edges:
#         ax.annotate("",
#                     xy=pos[e[0]], xycoords='data',
#                     xytext=pos[e[1]], textcoords='data',
#                     arrowprops=dict(arrowstyle="->", color="0.5",
#                                     shrinkA=5, shrinkB=5,
#                                     patchA=None, patchB=None,
#                                     connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])
#                                     ),
#                                     ),
#                     )
#     plt.axis('off')
#     plt.show()