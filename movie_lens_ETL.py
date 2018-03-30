#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:59:20 2018

@author: arindam93
"""
import os
import numpy as np
import pandas as pd
import networkx as nx

def calculate_similarity(mean_normalized, user):
    root_mean_square_normalized = (mean_normalized**2).sum(axis=0).pow(1/2)

    cols = len(mean_normalized.columns)
    iteraction_frame = mean_normalized.copy()
    for i in range(cols):
        # if(i%100 == 0):
        #     print(i)
        iteraction_frame.iloc[:,i] = mean_normalized.iloc[:,user]*mean_normalized.iloc[:,i]
    # print('ek zhala')
    similarity = []
    for i in range(cols):
        # if(i%100 == 0):
        #     print(i)
        similarity.append(iteraction_frame.iloc[:,i].sum(axis=0)/(root_mean_square_normalized.iloc[user]*root_mean_square_normalized.iloc[i]))
    # similarity.insert(user,1)
    return similarity



#ratings = open('ml-1m/ratings.dat')
ages = { 1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+" }
occupations = { 0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
                4: "college/grad student", 5: "customer service", 6: "doctor/health care",
                7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
                12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
                17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer" }




ratings = pd.read_csv(os.path.join('ml-1m/', 'ratings.dat'), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['userid', 'movieid', 'rating', 'timestamp'])

users = pd.read_csv(os.path.join('ml-1m/', 'users.dat'), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['userid', 'gender', 'age', 'occupation', 'zipcode'])


#users['age_desc'] = users['age'].apply(lambda x: ages[x])
#users['occ_desc'] = users['occupation'].apply(lambda x: occupations[x])

movies = pd.read_csv(os.path.join('ml-1m/', 'movies.dat'), 
                    sep='::', 
#                    engine='python', 
                    encoding='latin-1',
                    engine = 'python',
                    names=['movieid', 'title', 'genre'])

user=1
user_item_matrix = ratings.pivot(index = 'userid', columns ='movieid', values = 'rating')
# user_item_matrix.to_pickle('user_item_matrix.pickle')
# print(user_item_matrix.shape)
# print(user_item_matrix.iloc[0])
##########################################Using in built function#########################################################
# user_item_matrix = user_item_matrix.transpose() # converting to item user matrix
# rows, cols = user_item_matrix.shape
# s_uv = []
# for user in  range(cols):
#     similarity = []
#     for i in range(cols):
#         similarity.append(user_item_matrix.iloc[:,user].corr(user_item_matrix.iloc[:,i]))
#     s_uv.append(similarity)
# s_uv = np.array(s_uv)
# s_uv = pd.DataFrame(s_uv)
# print(s_uv.shape)
# s_uv.to_pickle('s_uv_using_inbuilt_function.pickle')

# user_item_matrix = user_item_matrix.transpose() # converting to item user matrix
# rows, cols = user_item_matrix.shape
# s_ij = []
# for user in  range(cols):
#     similarity = []
#     for i in range(cols):
#         similarity.append(user_item_matrix.iloc[:,user].corr(user_item_matrix.iloc[:,i]))
#     s_ij.append(similarity)
# s_ij = np.array(s_ij)
# s_ij = pd.DataFrame(s_ij)
# print(s_ij.shape)
# s_ij.to_pickle('s_ij_using_inbuilt_function.pickle')
#########################################################################################################################
user_item_matrix = user_item_matrix.transpose() # converting to item user matrix
r_cap = user_item_matrix.mean(axis=0)
# print(r_cap)
mean_normalized_user_item_matrix = user_item_matrix - r_cap

items, userNodes = user_item_matrix.shape
s_uv = []
for user in range(userNodes):
    if(user%10 == 0):
        print(user)
    s_uv.append(pd.DataFrame(calculate_similarity(mean_normalized_user_item_matrix, user)))

s_uv = pd.concat(s_uv,axis=1)
print(s_uv.iloc[:2,5])
print(s_uv.iloc[5,:2])
s_uv.to_pickle('s_uv.pickle')

print(s_uv.shape)

user_item_matrix = user_item_matrix.transpose() # converting to item user matrix
# print(user_item_matrix.iloc[:,51])
r_cap = user_item_matrix.mean(axis=0)
# print(r_cap)
mean_normalized_user_item_matrix = user_item_matrix - r_cap

userNodes, items = user_item_matrix.shape
s_ij = []
for item in range(items):
    if(item%10 == 0):
        print(item)
    s_ij.append(pd.DataFrame(calculate_similarity(mean_normalized_user_item_matrix, item)))

s_ij = pd.concat(s_ij,axis=1)
print(s_ij.iloc[:2,5])
print(s_ij.iloc[5,:2])
s_ij.to_pickle('s_ij.pickle')
print(s_ij.shape)


# print(s_uv.head(5))

# user_item_matrix = user_item_matrix.transpose() # converting to item user matrix
# r_cap = user_item_matrix.mean(axis=0)
# # print(r_cap)
# mean_normalized_user_item_matrix = user_item_matrix - r_cap
# s_uv = pd.DataFrame(calculate_suv(mean_normalized_user_item_matrix, user))

# user_item_matrix = user_item_matrix.fillna(value=0)



# print(user_item_matrix.head(5))
# print(user_item_matrix.columns)
# cols = user_item_matrix.columns

# idx = user_item_matrix.index


# G = nx.from_pandas_dataframe(user_item_matrix, 1, 2)



# user_item_matrix = user_item_matrix.groupby['1']

# #user_item_matrix = user_item_matrix.as_matrix(columns=None)
# #user_item_matrix = user_item_matrix.values

# factor_graph_matrix = []
# factor_graph_matrix.append(user_item_matrix.iloc[0,:])
# #

# cols = user_item_matrix.columns
# for i in range(len(user_item_matrix)-1):
#     flag = True
#     for j in range(user_item_matrix.columns.size)-1:
        
#         if flag == False:
#             continue
    
#         if (user_item_matrix.iloc[0,cols[j]] != 0 and user_item_matrix.iloc[i+1,cols[j]] != 0):
            
#             factor_graph_matrix.append(user_item_matrix.iloc[i+1,:])
#             flag = False

# factor_graph_matrix = np.asarray(factor_graph_matrix)


#
#for i in range(len(user_item_matrix)-1):
#    flag = True
#    for j in range(np.size(user_item_matrix,1)):
#        
#        if flag == False:
#            continue
#    
#        if (user_item_matrix[0,j] != 0 and user_item_matrix[i+1,j] != 0):
#            
#            factor_graph_matrix.append(user_item_matrix[i+1,:])
#            flag = False
#            
#factor_graph_matrix = np.asarray(factor_graph_matrix)


#            factor_graph_matrix = np.append(factor_graph_matrix, user_item_matrix[i+1,:])

            

#ratings = ratings.values      # Dataframe to Numpy array
#max_userid = ratings['userid'].drop_duplicates().max()
#max_movieid = ratings['movieid'].drop_duplicates().max()
#
#ratings['user_emb_id'] = ratings['userid'] - 1
#ratings['movie_emb_id'] = ratings['movieid'] - 1
#
#ratings_csv_file = 'm1-1m_ratings.csv'
#ratings.to_csv(ratings_csv_file, 
#               sep='\t', 
#               header=True, 
#               encoding='latin-1', 
#               columns=['userid', 'movieid', 'rating', 'timestamp', 'user_emb_id', 'movie_emb_id'])
#print 'Saved to', ratings_csv_file
