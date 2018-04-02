import os
import numpy as np
import pandas as pd

def calculate_user_similarity(mean_normalized, user):
    root_mean_square_normalized = (mean_normalized**2).sum(axis=0).pow(1.0/2)
    cols = len(mean_normalized.columns)
    iteraction_frame = mean_normalized.copy()
    for i in range(cols):
        iteraction_frame.iloc[:,i] = mean_normalized.iloc[:,user]*mean_normalized.iloc[:,i]
    similarity = []
    for i in range(cols):
        similarity.append(iteraction_frame.iloc[:,i].sum(axis=0)/(root_mean_square_normalized.iloc[user]*root_mean_square_normalized.iloc[i]))
    return similarity

def calculate_item_similarity(mean_normalized, item):
    root_mean_square_normalized = (mean_normalized**2).sum(axis=1).pow(1.0/2)
    rows , cols = mean_normalized.shape
    iteraction_frame = mean_normalized.copy()
    for i in range(rows):
        iteraction_frame.iloc[i,:] = mean_normalized.iloc[item,:]*mean_normalized.iloc[i,:]
    similarity = []
    for i in range(rows):
        similarity.append(iteraction_frame.iloc[i,:].sum(axis=0)/(root_mean_square_normalized.iloc[item]*root_mean_square_normalized.iloc[i]))
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

movies = pd.read_csv(os.path.join('ml-1m/', 'movies.dat'), 
                    sep='::', 
                    encoding='latin-1',
                    engine = 'python',
                    names=['movieid', 'title', 'genre'])

user_item_matrix = ratings.pivot(index = 'movieid', columns ='userid', values = 'rating')

user_item_matrix.to_pickle('user_item_matrix.pickle')
# user_item_matrix = np.array([[5,2],[np.nan,4],[3,3]])
# user_item_matrix = pd.DataFrame(user_item_matrix)
print(user_item_matrix.shape)

r_cap = user_item_matrix.mean(axis=0)
mean_normalized_user_item_matrix = user_item_matrix - r_cap

items, userNodes = user_item_matrix.shape
s_uv = []
for user in range(userNodes):
    if(user%10 == 0):
        print(user)
    s_uv.append(pd.DataFrame(calculate_user_similarity(mean_normalized_user_item_matrix, user)))

s_uv = pd.concat(s_uv,axis=1)
s_uv.to_pickle('s_uv.pickle')

print('---------------------------sij--------------------')

r_cap = user_item_matrix.mean(axis=0)

mean_normalized_user_item_matrix = user_item_matrix - r_cap

items, userNodes = user_item_matrix.shape
s_ij = []
for item in range(items):
    if(item%10 == 0):
        print(item)
    s_ij.append(pd.DataFrame(calculate_item_similarity(mean_normalized_user_item_matrix, item)))

s_ij = pd.concat(s_ij,axis=1)
s_ij.to_pickle('s_ij.pickle')


