
# coding: utf-8

# In[21]:

try:
    import graphlab as gl
    import graphlab.aggregate as agg
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import datetime as dt
    get_ipython().magic(u'matplotlib inline')
    
except:
    raise ImportError("Key libraries cannot be loaded.")


# In[22]:

rcParams['figure.figsize'] = (10,10)
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 22
rcParams['xtick.labelsize'] = 16
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'

def clean_top(ax):
    for x in ax.spines.values():
        x.set_visible(false)
    ax.grid(True, 'major',color ='w', linestyle = '-',linewidth = 1.4)
    ax.path.set_facecolor(0.92)
    ax.set_axisbelow(True)
    ax.xaxis.set_tricks_position('bottom')
    ax.yaxis.set_tricks_position('left')
    


# In[23]:

import os

f_userdata = 'user_edges_2011-07-13'

if os.path.exists(f_userdata):
    sf = gl.SFrame(f_userdata)
else:
    url_userdata = 'https://static.turi.com/datasets/bitcoin/{}.txt'.format(f_userdata)
    sf = gl.SFrame.read_csv(url_userdata, delimiter='\t', header=False,
                              column_type_hints={'X1': int, 'X2': int, 'X3': float})
    sf.rename({'X1': 'src', 'X2': 'dst', 'X3': 'btc', 'X4': 'timestamp'})
    sf.save(f_userdata)


# In[24]:

# Show graphs and sframes inside ipython notebook
gl.canvas.set_target('ipynb')

sf.show()


# In[25]:

sf['timestamp'] = sf['timestamp'].str_to_datetime('%Y-%m-%d-%H-%M-%S')
sf.add_columns(sf['timestamp'].split_datetime(column_name_prefix=None, limit=['year', 'month', 'day']))


# In[26]:

f_price  = 'https://static.turi.com/datasets/bitcoin/market-price.csv'
sf_price  = gl.SFrame.read_csv(f_price, delimiter= ',', header=False, column_type_hints=[str, float])
sf_price.rename({'X1': 'timestamp' , 'X2': 'close-price'})
sf_price['timestamp'] = sf_price['timestamp'].str_to_datetime('%d/%m/%Y')
sf_price.add_columns(sf_price['timestamp'].split_datetime(column_name_prefix = None, limit = ['year', 'month', 'day']))
sf_price.head(5)


# In[27]:

sf = sf.join(sf_price, on=['year', 'month', 'day'], how='left')
sf.remove_column('timestamp.1')

sf['dollar'] = sf.apply(lambda x: x['btc'] * x['close-price'])
sf['dollar_label'] = sf['dollar'].apply(lambda x: '$' + str(round(x, 2)))


# In[28]:

g = gl.SGraph().add_edges(sf, src_field='src', dst_field='dst')


# In[29]:

g.summary()


# In[30]:

transaction_count = sf.groupby(['year' , 'month'], agg.COUNT).sort(['year', 'month'], ascending=True)
# sort the results by year and month
n_count = transaction_count.num_rows()

# add a column with x-axis plot labels
transaction_count['label'] = transaction_count['month'].astype(str) + "/" + transaction_count['year'].astype(str)

#plotfig, 

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(transaction_count['Count'], lw =3,color = '#0a8cc4')
ax.set_title("Bitcoin transactions per month")
ax.set_ylabel("Transactions")
ax.set_xlabel("Month")

xticks = ax.get_xticks().astype(int)
transaction_count_label = list(transaction_count['label'])
ticklabels = list([transaction_count_label[i] for i in xticks])
ax.set_xticklabels(ticklabels)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

clean_plot(ax)
fig.show()



# In[31]:

# find the number of unique source users by month
month_source_count = sf.groupby(['year', 'month', 'src'], agg.COUNT).sort(['year', 'month'], ascending=True)
source_count = month_source_count.groupby(['year', 'month'], agg.COUNT).sort(['year', 'month'], ascending=True)

# find the number of unique destination users by month
month_dest_count = sf.groupby(['year', 'month', 'dst'], agg.COUNT).sort(['year', 'month'], ascending=True)
dest_count = month_dest_count.groupby(['year', 'month'], agg.COUNT).sort(['year', 'month'], ascending=True)

# add columns with x-axis plot labels
source_count['label'] = source_count['month'].astype(str) + "/" + source_count['year'].astype(str)

# plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(source_count['Count'], lw=3, label='source users')
ax.plot(dest_count['Count'], lw=3, label='destination users')
ax.set_title("Bitcoin users per month")
ax.set_ylabel("Unique users")
ax.set_xlabel("Month")
ax.legend(loc='upper left')

xticks = ax.get_xticks().astype(int)
source_count_label = list(source_count['label'])
ticklabels = list([source_count_label[i] for i in xticks])
ax.set_xticklabels(ticklabels)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

clean_plot(ax)
fig.show()


# In[36]:

def count_degree(src, edge, dst):
    dst['in_degree'] += 1
    src['out_degree'] += 1
    return (src, edge, dst)

def get_degree(g):
    new_g = gl.SGraph(g.vertices, g.edges)
    new_g.vertices['in_degree'] = 0
    new_g.vertices['out_degree'] = 0
    return new_g.triple_apply(count_degree, ['in_degree', 'out_degree']).get_vertices()

sf_degree = get_degree(g)
sf_degree['total_degree'] = sf_degree['in_degree'] + sf_degree['out_degree']


# In[45]:

import numpy as np
import  matplotlib
fig, ax = plt.subplots()
for deg_type in ['total_degree', 'in_degree', 'out_degree']:
    counts = np.bincount(list(sf_degree[deg_type]))
    ecdf = np.cumsum(counts) / float(sf_degree.num_rows())
    ax.step(range(len(ecdf)), 1 - ecdf, lw = 3, where= 'post' ,label = deg_type)
    
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel("Degree")
ax.set_ylabel("Fraction of users")
ax.set_title("Degree Distributions")
clean_plot(ax)
fig.show()


# In[46]:

print "In-degree outliers"
print sf_degree.topk('in_degree', k=5)

print "Out-degree outliers"
print sf_degree.topk('out_degree', k=5)


# In[47]:

btc_small = sf[sf['btc']<=5]['btc']
print "Number of small transactions:", len(btc_small)
print "Proportion of all transactions:", float(len(btc_small)) / sf.num_rows()
btc_small.show()


# In[48]:

sf[['src', 'dst', 'btc', 'timestamp', 'dollar']].topk('btc', k=5)


# In[49]:

u307659_edges = g.get_edges(src_ids=[307659, None], dst_ids=[None, 307659])
u307659_edges.head()


# In[51]:

pr = gl.pagerank.create(g,verbose = False)
pr_out = pr.get('pagerank')
pr_out.topk('pagerank', k=5)


# In[56]:

idx_wleaks = 9264
wleaks_net = g.get_neighborhood(ids=idx_wleaks, radius=1, full_subgraph=False)
wleaks_net.show(highlight=[idx_wleaks])


# In[57]:

sf_wleaks = wleaks_net.get_edges().sort('timestamp', ascending=True)
sf_wleaks = sf_wleaks.add_row_number()


# In[59]:

balance = [0.] * sf_wleaks.num_rows()
balance[0] = sf_wleaks[0]['dollar']

for i in range(1, sf_wleaks.num_rows()):
    if sf_wleaks[i]['__src_id'] == idx_wleaks and sf_wleaks[i]['__dst_id'] != idx_wleaks:
        balance[i] = balance[i - 1] - sf_wleaks[i]['dollar']
        
    elif sf_wleaks[i]['__src_id'] != idx_wleaks and sf_wleaks[i]['__dst_id'] == idx_wleaks:
        balance[i] = balance[i - 1] + sf_wleaks[i]['dollar']

    else:
        balance[i] = balance[i - 1]
        
sf_wleaks['balance'] = balance


# In[60]:

fig, ax = plt.subplots()
ax.plot(sf_wleaks['timestamp'].astype(int), sf_wleaks['balance'], lw=3, color='#0a8cc4')
ax.set_ylabel("Balance (USD)")
ax.set_xlabel("Date")

xlim_dates = gl.SArray([sf_wleaks[0]['timestamp'], sf_wleaks[-1]['timestamp']])
ax.set_xlim(xlim_dates.astype(int))
ax.set_xticks(xlim_dates.astype(int))
ax.set_xticklabels(xlim_dates.datetime_to_str('%b %d %Y'))

clean_plot(ax)
fig.show()


# In[61]:

net1 = gl.SGraph().add_edges(wleaks_net.get_edges(src_ids=[idx_wleaks]),
                             src_field='__src_id', dst_field='__dst_id')
net1.show(vlabel='id', highlight=[idx_wleaks], elabel='dollar_label',
          arrows=True, vlabel_hover=False, elabel_hover=False)


# In[62]:

targets = net1.get_edges()['__dst_id']
net2 = g.get_neighborhood(ids=targets, radius=0, full_subgraph=True)
net2.show(vlabel='id', highlight=targets)


# In[63]:

net2_edges = net2.get_edges()
targets = net2_edges[net2_edges['btc'] > 100]['__dst_id']


# In[64]:

net3 = g.get_edges(src_ids=targets)
net3


# In[65]:

net3 = g.get_edges(dst_ids=targets)
net3


# In[66]:

idx_thief = 16657
idx_victim = 27783

thief_net = g.get_neighborhood(ids=idx_thief, radius=1, full_subgraph=False)
thief_net.show(vlabel='id', elabel='dollar_label', arrows=True, highlight=[idx_thief, idx_victim])


# In[68]:

net = g.get_neighborhood(ids = idx_thief,radius =2, full_subgraph=False)
edges = net.get_edges()
mask = (edges['__src_id'] != 23) * (edges['__dst_id'] != 23)  # get only edges that don't involve 23

thief_net2 = gl.SGraph().add_edges(edges[mask], src_field='__src_id', dst_field='__dst_id')
thief_net2.show(vlabel='id', highlight=[idx_thief, idx_victim], vlabel_hover=True)


# In[71]:

net = g.get_neighborhood(ids=idx_victim, radius=2, full_subgraph=False)
vic_edges = net.get_edges()
vic_edges = vic_edges[vic_edges['timestamp'] >= dt.datetime.strptime('2011-06-01', '%Y-%m-%d')]
vic_net2 = gl.SGraph().add_edges(vic_edges, src_field='__src_id', dst_field='__dst_id')


# In[72]:

thief_verts = set(thief_net2.get_vertices()['__id'])
vic_verts = set(vic_net2.get_vertices()['__id'])
common_verts = set.intersection(thief_verts, vic_verts)


# In[73]:

','.join([str(i) for i in common_verts])


# In[76]:

theft_edge = g.get_edges(src_ids=idx_victim, dst_ids=idx_thief)
theft_time = theft_edge.topk('timestamp', k=1)['timestamp'][0]
print "Theft occurred at: ", theft_time, '\n'
theft_edge.head()


# In[79]:

def follow_the_money(g, vertex, attribute, threshold, radius=1):
    if radius == 0:
        return

    else:
        # find all outgoing edges from vertex
        edges = g.get_edges(src_ids=vertex)

        if len(edges) == 0:
            return

        else:
            edges = edges[edges[attribute] > threshold]  #only keep 'later' edges
            out_edges = edges[edges.column_names()]

            # recurse
            for row in edges:
                new_edges = follow_the_money(g, vertex=row['__dst_id'],
                                            attribute=attribute,
                                            threshold=row[attribute],
                                            radius=radius - 1)
                if new_edges is not None:
                    out_edges = out_edges.append(new_edges)

            out_edges = out_edges.groupby(out_edges.column_names(), {})
            return out_edges


# In[80]:

tainted_edges = follow_the_money(g, idx_thief, attribute='timestamp',
                                 threshold=theft_time, radius=1)
net = gl.SGraph().add_edges(tainted_edges, src_field='__src_id', dst_field='__dst_id')
net.show(vlabel='id', elabel='dollar_label', arrows=True, highlight=[idx_thief])


# In[81]:

launder_edge = g.get_edges(src_ids=112654, dst_ids=185593)
launder_time = launder_edge.topk('timestamp', k=1)['timestamp'][0]

edges = follow_the_money(g, vertex=185593, attribute='timestamp',
                         threshold=launder_time, radius=2)
edges['label'] = edges['timestamp'].datetime_to_str('%Y-%m-%d-%H-%M-%S')


# In[82]:

net = gl.SGraph().add_edges(edges, src_field='__src_id', dst_field='__dst_id')
net.show(vlabel='id', elabel='label', arrows=True, highlight=[idx_thief, 185593])


# In[ ]:



