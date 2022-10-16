# Decision Making With Matrices

# This is a pretty simple assignment.  You will do something you do everyday, but today it will be with matrix manipulations. 

# The problem is: you and your work friends are trying to decide where to go for lunch. You have to pick a restaurant that’s best for everyone.  Then you should decide if you should split into two groups so everyone is happier.  

# Despite the simplicity of the process you will need to make decisions regarding how to process the data.
  
# This process was thoroughly investigated in the operation research community.  This approach can prove helpful on any number of decision making problems that are currently not leveraging machine learning.  

# Transform the restaurant data into a matrix (M_resturants) using the same column index.

# The most important idea in this project is the idea of a linear combination.  
# Informally describe what a linear combination is  and how it will relate to our restaurant matrix.

# Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent? 

# Next, compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent? 

# Sum all columns in M_usr_x_rest to get the optimal restaurant for all users.  What do the entries represent?

# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal restaurant choice.  

# Why is there a difference between the two?  What problem arrives?  What does it represent in the real world?

# How should you preprocess your data to remove this problem? 

# Find  user profiles that are problematic, explain why?

# Think of two metrics to compute the disatistifaction with the group.  

# Should you split into two groups today? 

# Ok. Now you just found out the boss is paying for the meal. How should you adjust? Now what is the best restaurant?

# Tomorrow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants. Can you find their weight matrix? 


# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.  
people = {'Manjot': {'willingness to travel': 3,
                  'desire for new experience':2,
                  'cost':1,            
                  'hipster points':3,
                  'cuisine':2,
                  'vegetarian':1 }
          ,'Harry': {'willingness to travel': 5,
                  'desire for new experience':5,
                  'cost':5,                  
                  'hipster points':5,
                  'cuisine':5,
                  'vegetarian':5 }
          ,'Allen': {'willingness to travel': 2,
                  'desire for new experience':1,
                  'cost':1,            
                  'hipster points':2,
                  'cuisine':1,
                  'vegetarian':2}
          ,'Harsh': {'willingness to travel': 3,
                  'desire for new experience':4,
                  'cost':3,                  
                  'hipster points':3,
                  'cuisine':3,
                  'vegetarian':3 }
          ,'Shiv': {'willingness to travel': 4,
                  'desire for new experience':4,
                  'cost':1,            
                  'hipster points':4,
                  'cuisine':4,
                  'vegetarian':4 }
          ,'Jose': {'willingness to travel': 2,
                  'desire for new experience':2,
                  'cost':5,                  
                  'hipster points':2,
                  'cuisine':2,
                  'vegetarian':3 }
          ,'Banjamin': {'willingness to travel': 3,
                  'desire for new experience':2,
                  'cost':1,            
                  'hipster points':3,
                  'cuisine':2,
                  'vegetarian':4 }
          ,'Felix': {'willingness to travel': 5,
                  'desire for new experience':5,
                  'cost':5,                  
                  'hipster points':2,
                  'cuisine':5,
                  'vegetarian':0 }
          ,'Michael': {'willingness to travel': 3,
                  'desire for new experience':3,
                  'cost':4,            
                  'hipster points':5,
                  'cuisine':5,
                  'vegetarian':5 }
          ,'Angela': {'willingness to travel': 2,
                  'desire for new experience':3,
                  'cost':4,                  
                  'hipster points':1,
                  'cuisine':5,
                  'vegetarian':5 }
                }          

# Transform the user data into a matrix ( M_people). Keep track of column and row IDs.   

# Next you collected data from an internet website. You got the following information.

restaurants  = {'La Cafe Michi':{'distance' : 1,
                        'novelty' :5,
                        'cost': 1,
                        'average rating': 5,
                        'cuisine':4,
                        'vegetarian':3}
                ,'SushiOne':{'distance' : 2,
                        'novelty' :3,
                        'cost': 1,
                        'average rating': 2,
                        'cuisine':2,
                        'vegetarian':0}
                ,'Fire Hot':{'distance' : 2,
                        'novelty' :4,
                        'cost': 3,
                        'average rating': 5,
                        'cuisine':4,
                        'vegetarian':3}
                ,'Jameh':{'distance' : 4,
                        'novelty' :3,
                        'cost': 3,
                        'average rating': 3,
                        'cuisine':4,
                        'vegetarian':0}
                ,'Kinton':{'distance' : 3,
                        'novelty' :2,
                        'cost': 2,
                        'average rating': 3,
                        'cuisine':3,
                        'vegetarian':1}
                ,'Potato Patio':{'distance' : 1,
                        'novelty' :3,
                        'cost': 2,
                        'average rating': 4,
                        'cuisine':4,
                        'vegetarian':1}
                ,'Popeyes':{'distance' : 5,
                        'novelty' :5,
                        'cost': 5,
                        'average rating': 3,
                        'cuisine':2,
                        'vegetarian':0}
                ,'Max Vein':{'distance' : 1,
                        'novelty' :2,
                        'cost': 1,
                        'average rating': 4,
                        'cuisine':5,
                        'vegetarian':0}
                ,'Dragon Pearl':{'distance' : 3,
                        'novelty' :2,
                        'cost': 2,
                        'average rating': 2,
                        'cuisine':2,
                        'vegetarian':1}
                ,'Bamboo Forest':{'distance' : 2,
                        'novelty' :4,
                        'cost': 1,
                        'average rating': 4,
                        'cuisine':4,
                        'vegetarian':4}}









import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

#1
peop_nam=list(people)
People_col=list(people[peop_nam[1]])
M_people = np.ones((len(people),len(People_col)))
for i, p in enumerate(people):
	M_people[i,] = np.array(list(people[p].values()))
print(M_people)
#2
rest_nam=list(restaurants)
rest_col=list(restaurants[rest_nam[1]])
M_restaurants = np.ones((len(restaurants),len(rest_col)))
for i, p in enumerate(restaurants):
	M_restaurants[i,] = np.array(list(restaurants[p].values()))
print(M_restaurants)
#3
sample=np.dot(M_people[peop_nam.index('Harry')],M_restaurants.T)
print(sample)
#Q4
M_usr_x_rest=np.dot(M_people,M_restaurants.T)
print(M_usr_x_rest)
#Aij is jth restaurant's mark correspond to ith person
#Q5
score= M_usr_x_rest.sum(axis=0)
print("Best resturant is", rest_nam[np.argmax(score)])
#Each entry is the score for one resturant among the group.
#Q6
df=pd.DataFrame(M_usr_x_rest)
a = np.argsort(df.iloc[:, :].to_numpy(), axis=1)
n, m = a.shape
b = np.empty_like(a)
c, d = np.mgrid[:n, :m]
b[c, a] = m - d
df.iloc[:, :] = b
M_usr_x_rest_rank=df.copy()
print(M_usr_x_rest_rank)
score1=(M_usr_x_rest_rank.sum(axis=0)).tolist()
print("Best resturant is", rest_nam[score1.index(min(score1))])
#7
#Scoring method takes in all range and skewness of each indicidual's entries and would have problem with outliers within the data inbound to the model.
#Ranking method would removes the skewness and range and ignore the maginitude totally.
#In real problems, individual can over effecting the first model and the second model waives the efficient magnitude difference.
#8
# As the model is biased by outlier, one easy way is to perform log transform.
score_log  = [np.log(M_usr_x_rest[x,:]).sum() for x in range(M_usr_x_rest.shape[0])]
print("Best resturant is", rest_nam[np.argmax(score_log)])
#9
#Based on the first model last ranking we can tell Harry is problematic as all entries are as capped 5 and that would make any resturant without vegetarian lose a lot.
print(rest_nam[np.argmax(score)] ," scores",M_usr_x_rest[rest_nam.index(rest_nam[np.argmax(score)]),:]," Final Method 1-Scores: ",score[rest_nam.index(rest_nam[np.argmax(score)])])
print(rest_nam[score1.index(min(score1))], " Ranks",M_usr_x_rest[rest_nam.index(rest_nam[score1.index(min(score1))]),:]," Final Rank Method 2-Rank mark: ",score1[score1.index(min(score1))])
#10 
rest_best_max_score = M_usr_x_rest[np.argmax(score)].max()
people_dissatisfaction =  abs(M_usr_x_rest[np.argmax(score)]   - rest_best_max_score)
for x in reversed(np.argsort(people_dissatisfaction) ) :
    print(peop_nam[x],"disattisfaction with model 1:",people_dissatisfaction[x])
rest2_best_max_score = M_usr_x_rest[np.argmax(score),:].max()
people_dissatisfaction2 =  abs(M_usr_x_rest[np.argmax(score),:]   - rest2_best_max_score)
for x in reversed(np.argsort(people_dissatisfaction2) ) :
    print(peop_nam[x],"disattisfaction with model 2:",people_dissatisfaction2[x])
#11
km = KMeans(n_clusters=2, random_state=0,n_init=10,max_iter=300).fit(X = M_usr_x_rest.T)
index=[]
for i in range(2):
    data=np.dot(M_restaurants,M_people[km.labels_ == i].T)   
    index[i]=np.where(km.labels_==i)

#12
#As Boss is paying means hu=igher budget while in most general case(Or I would fire him(Just kidding)), the weight of cost should be waived or reduced. Per consideration that we want to spend wisely making them all 3 would be doable.
M_people_Boss=M_people.copy()
M_people_Boss[:,People_col.index('cost')]=2
M_usr_x_rest_Boss=np.dot(M_people_Boss,M_restaurants.T)
score_Boss= M_usr_x_rest_Boss.sum(axis=0)
print("Best resturant is", rest_nam[np.argmax(score_Boss)])
#13
#Result of optimal nor rank array would not be able to recreate the weight matrix the only approach would be M_usr_x_rest, so lety's assume we have that
new_team_x_rest =  M_usr_x_rest.copy()
new_team_matrix= np.linalg.lstsq(M_restaurants,new_team_x_rest,rcond=-1)[0].T
for i,p in enumerate(peop_nam): print(p,"weights:",new_team_matrix[i,:])
