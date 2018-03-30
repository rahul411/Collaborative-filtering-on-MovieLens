import pandas as pd
import numpy as np
import math
import collections
max_rating = 5

user_item_matrix = pd.read_pickle('user_item_matrix.pickle')
s_uv = pd.read_pickle('s_uv.pickle')
s_ij = pd.read_pickle('s_ij.pickle')


class itemNode:
	def __init__(self,itemId, y,neighbors):
		self.itemId = itemId
		self.y = y
		self.neighbors = list(neighbors)
		# print(itemId)
		# print(self.neighbors)
		if itemId in self.neighbors:
			self.neighbors.remove(itemId)
		self.phi = []
		self.msgs = {}
		self.psy = collections.defaultdict(list) 
		self.set_phi()
		self.set_psy()
		self.set_msgs()

	def set_msgs(self):
		for n in self.neighbors:
			self.msgs[n] = np.array([0.2,0.2,0.2,0.2,0.2])

	def set_phi(self):
		if self.y < 1:
			self.phi = [1,0,0,0,0]
		elif self.y > 5:
			self.phi = [0,0,0,0,1]
		else:
			for i in range(1,max_rating+1):
				flag = 0
				if i==math.ceil(self.y):
					self.phi.append(math.ceil(self.y)-self.y)
					flag = 1
				if i==math.floor(self.y):
					self.phi.append(1 - (math.ceil(self.y)-self.y))
					flag = 1
				if flag==0:
					self.phi.append(0)

	def set_psy(self):
		global s_ij
		global max_rating
		for n in self.neighbors:
			p = []
			sigma = (((1-s_ij.iloc[self.itemId,n])/(1-0.35)) + 1)/math.sqrt(2)
			for i in range(1,max_rating+1):
				for j in range(1,max_rating+1):
					p.append(-((i-j)**2)/sigma)
			self.psy[n] = np.exp(p)
			self.psy[n] = self.psy[n]/sum(self.psy[n])
			self.psy[n] = self.psy[n].reshape([5,5])

def getIndexOfNan(df):
	index = df.index[df.apply(np.isnan)]




print(s_ij.iloc[4,4])
print(s_ij.iloc[5,5])
print(user_item_matrix.iloc[1,0])

print(user_item_matrix.shape)
#################Remember to drop this when the PCC is corrected##################
# user_item_matrix = user_item_matrix.drop(user_item_matrix.index[0],axis=0)
# user_item_matrix = user_item_matrix.drop(user_item_matrix.index[0],axis=1)
##################################################################################

user_item_matrix = user_item_matrix.transpose()
# print(user_item_matrix.columns)
r_cap = user_item_matrix.mean(axis=0)
mean_normalized_user_item_matrix = user_item_matrix - r_cap

item_rated = user_item_matrix.notnull()

user = 1000
deno = pd.DataFrame(item_rated.values * s_uv.iloc[user].values)
deno = (deno.abs()).sum(axis=1)
# deno = deno.as_matrix()
# indices = np.where(deno==0)
# print(indices)
# user_item_matrix_numpy = user_item_matrix.as_matrix()
# # user_item_matrix_numpy = user_item_matrix_numpy.transpose()
# print(np.count_nonzero(np.isnan(user_item_matrix_numpy[389])))
# print(deno.isnull().values.any())
numerator = pd.DataFrame(mean_normalized_user_item_matrix.values*s_uv.iloc[user].values).sum(axis=1)
# print(numerator.isnull().values.any())
y_i = r_cap.iloc[user] + numerator.values/deno.values
# print(np.count_nonzero(np.isnan(y_i)))
y_i = pd.DataFrame(y_i)
y_i = y_i.fillna(r_cap.iloc[user])
y_i = y_i.as_matrix()
# print(y_i.isnull().values.any())
top_K = 10

kmax_value = s_ij.fillna(-2).as_matrix()
print(kmax_value.shape)
kmax_value = np.sort(kmax_value)[:,::-1]
kmax_value = kmax_value[:,top_K]
print('kmaz',kmax_value.shape)

rows,cols = s_ij.shape
neighbors = []
#########################################################
# for i in range(cols):
# 	neighbors.append(s_ij.iloc[:,i] >= kmax_value.transpose())

# neighbors = pd.concat(neighbors,axis=0)
# neighbors = neighbors.as_matrix()
######################################################
s_ij_np_matrix = s_ij.as_matrix()
final_neighbors = np.zeros([cols,cols])
for i in range(cols):
	neighbors.append(np.where(s_ij_np_matrix[:,i]>=kmax_value[i])[0][:10])
	final_neighbors[i,neighbors[i]] = 1
print(len(neighbors))
print(neighbors[0].shape)
print(final_neighbors[0, neighbors[0]])

final_neighbors = final_neighbors + final_neighbors.transpose()

neighbors = []
for i in range(cols):
	neighbors.append(np.where(final_neighbors[i,:]>0)[0])

print(len(neighbors))
print(neighbors[0].shape)
print(neighbors[55])
# ########################Creating item Nodes#######################################
items = len(neighbors)
itemNodes = []
for i in range(items):
	itemNodes.append(itemNode(i, y_i[i], neighbors[i]))

print(itemNodes[0].phi)
maxIter = 2
c = 0
for i in range(maxIter):
	print('--------------------' + str(i) + '---------------------------')
	visited = []
	stack = [itemNodes[0]]
	while stack:
		node = stack.pop()
		visited.append(node.itemId)
		for n in node.neighbors:
			local_msgs = np.array([1,1,1,1,1])
			for k in node.neighbors:
				if k != n and node.itemId in itemNodes[k].neighbors:
					# if (local_msgs*itemNodes[k].msgs[node.itemId]).sum()==0:
					# 	print('incoming K',k)
					# 	print(node.itemId)
					# 	print(itemNodes[k].msgs[node.itemId])
					# 	print('local',local_msgs)
					local_msgs =  local_msgs*itemNodes[k].msgs[node.itemId]
					c+=1
			factor = node.psy[n]*node.phi
			node.msgs[n] = local_msgs*(factor.sum(axis=0))
			# if node.msgs[n].sum()==0:
			# 	print('itemNeigbor',n)
			# 	print('itemId', node.itemId)
			# 	print('factor', factor)
			# 	print(local_msgs)
			node.msgs[n] = node.msgs[n]/node.msgs[n].sum()
			if n not in visited:
				stack.append(itemNodes[n])
		# if c==0:
		# 	print(node.itemId)
		# 	print(node.neighbors)
		# print('c',c)
		c=0
for i in range(len(itemNodes)):
	m = np.array([1,1,1,1,1])
	for n in itemNodes[0].neighbors:
		m = m*itemNodes[0].msgs[n]
	Pz = itemNodes[0].phi*m
	Pz = Pz/Pz.sum()
	r = [1,2,3,4,5]
	final_rating = sum(r*Pz)
	print("item:",i)
	print(final_rating)


