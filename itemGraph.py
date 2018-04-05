import pandas as pd
import numpy as np
import math
import collections

#Read the data: user_item_matrix, sij, suv
user_item_matrix = pd.read_pickle('user_item_matrix.pickle')
# print(user_item_matrix.shape)
# user_item_matrix = np.array([[2,2,4],[np.nan,5,2],[2,2,5],[4,4,1]])
# user_item_matrix = pd.DataFrame(user_item_matrix)
items, users = user_item_matrix.shape
s_uv = pd.read_pickle('s_uv.pickle')
s_ij = pd.read_pickle('s_ij.pickle')

max_rating = 5
maxIter = 10


class itemNode:
	def __init__(self,itemId, y,neighbors):
		self.itemId = itemId
		self.y = y
		self.neighbors = list(neighbors)
		#The neighbors list contains self, which should not be considered.
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
			self.phi = [0.95,0,0,0,0]
		elif self.y > 5:
			self.phi = [0,0,0,0,0.95]
		else:
			for i in range(1,max_rating+1):
				flag = 0
				if i==math.ceil(self.y):
					self.phi.append(0.95 - (math.ceil(self.y)-self.y))
					flag = 1
				if i==math.floor(self.y) and flag==0:
					self.phi.append(math.ceil(self.y)-self.y)
					flag = 1
				if flag==0:
					self.phi.append(0)
		zero_indices = np.where(np.array(self.phi)==0)
		self.phi = np.array(self.phi)
		self.phi.put(zero_indices[0],(1-sum(self.phi))/len(zero_indices[0]))	

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

def compute_yi(mask, user, mean_normalized_user_item_matrix, r_cap):
	deno = pd.DataFrame(mask.values * s_uv.iloc[user].values)
	deno = (deno.abs()).sum(axis=1)

	numerator = pd.DataFrame(mean_normalized_user_item_matrix.values*s_uv.iloc[user].values).sum(axis=1)

	y_i = r_cap.iloc[user] + numerator/deno
	y_i = y_i.fillna(r_cap.iloc[user])
	y_i = y_i.as_matrix()
	return y_i

def graph_traversal(maxIter, itemNodes):
	for i in range(maxIter):
		visited = []
		stack = [itemNodes[0]]
		#DFS Graph traversal
		while stack:
			node = stack.pop()
			visited.append(node.itemId)
			for n in node.neighbors:
				local_msgs = np.array([1,1,1,1,1])
				for k in node.neighbors:
					if k != n:
						local_msgs =  local_msgs*itemNodes[k].msgs[node.itemId]
				
				factor = node.psy[n]*node.phi
				for i in range(max_rating):
					factor[i] = factor[i]*local_msgs[i]

				node.msgs[n] = factor.sum(axis=0)
				node.msgs[n] = node.msgs[n]/float(node.msgs[n].sum())
				if n not in visited:
					stack.append(itemNodes[n])

def inference(itemNodes):
	final_ratings = []
	for i in range(len(itemNodes)):
		m = np.array([1,1,1,1,1])
		for n in itemNodes[i].neighbors:
			m = m*itemNodes[i].msgs[n]
		Pz = itemNodes[i].phi*m
		Pz = Pz/Pz.sum()
		r = [1,2,3,4,5]
		final_ratings.append(sum(r*Pz))
		
	return final_ratings


#Calculate the mean rating for all user
r_cap = user_item_matrix.mean(axis=0)

#Mean normalize all ratings, (rui - ru)
mean_normalized_user_item_matrix = user_item_matrix - r_cap

#This is a mask, since we only have to consider those items which are being rated
item_rated = user_item_matrix.notnull()

user_ratings = []
for user in range(users):
	print('Computing for User:',user)
	#Compute y_i's
	y_i = compute_yi(item_rated, user, mean_normalized_user_item_matrix, r_cap)

	top_K = 2
	kmax_value = s_ij.fillna(-2).as_matrix()
	kmax_value = np.sort(kmax_value)[:,::-1]
	kmax_value = kmax_value[:,top_K]

	rows,cols = s_ij.shape
	neighbors = []

	s_ij_np_matrix = s_ij.as_matrix()
	final_neighbors = np.zeros([cols,cols])
	for i in range(cols):
		neighbors.append(np.where(s_ij_np_matrix[i,:]>=kmax_value[i])[0][:top_K+1])
		final_neighbors[i,neighbors[i]] = 1

	final_neighbors = final_neighbors + final_neighbors.transpose()

	neighbors = []
	for i in range(cols):
		neighbors.append(np.where(final_neighbors[i,:]>0)[0])

	#Creating item Nodes of the graph
	items = len(neighbors)
	itemNodes = []
	for i in range(items):
		itemNodes.append(itemNode(i, y_i[i], neighbors[i]))

	graph_traversal(maxIter, itemNodes)
	final_ratings = inference(itemNodes)
	user_ratings.append(final_ratings)

print(user_ratings)
user_ratings = np.array(user_ratings)
np.save('user_ratings.npy',user_ratings)

