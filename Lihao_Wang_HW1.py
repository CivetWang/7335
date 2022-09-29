# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 10:34:56 2022

@author: Harry Wang
"""
import math
#List:
append() #appends an element to the end of the list ,Example:list.append(element)
extend() #adds all the elements to the end of the list,Example:list.extend(elements_list)
index() #returns the position at the first occurrence of the specified value,Example:list.index(element)
index(value, integer)#returns the position at the first occurrence of the specified value starting from integer index to the end,Example:list.index(element,integer)
insert(position)  #inserts an element to the list at the specified index ,Example:list.insert(index, element)
remove() #removes the first matching element, Example:list.remove(element)
pop() #removes the item at the given index from the list and returns the removed item, Example:list.pop(element_index)
count() #the number of elements with the specified value,Example:list.count(element)
reverse() #reverses the elements of the list by index, Example: list.reverse()
sort() # sorts the items of a list by given order way, Example:list.sort() ## sorting the list in ascending order
[1]+[1] #add an list of element 1 at the end of the first list of element 1
[2]*2 # Duplicate the list of element container:{2} for 2 times
[1,2][1:] # Get elements from index 1 to the end index for set {1,2}
[x for x in [2,3]] # Get elements in set x
[x for x in [1,2] if x ==1] # Get where element in x equal to 1
[y*2 for x in [[1,2],[3,4]] for y in x] # Get all element in x to the power of 2

A = [1] # Assign A as the list contains element container {1}



#Tuple:
count() # the number of elements with the specified value,Example:tuple.count(element)
index() # returns the index of the specified element in the tuple, Example:tuple.index(element)
build a dictionary from tuples # use dict() like tuple=((element,""element_2),...) to build dictionary 
unpack tuples # extrace tuple value into  by ((element,""element_2),...)=tuple

#Dicts:
a_dict = {'I hate':'you', 'You should': 'leave'} # create a dictionary to collect 'I hate'as'you' and 'You should'as 'leave'
keys()#returns a view object. The view object contains the keys of the dictionary,Example:print(a_dict.keys()) returns dict_keys(['I hate','You should'])
items()#returns a view object. The view object contains the key-value pairs of the dictionary,Example:print(a_dict.keys()) returns dict_keys([('I hate','You'),('You should','leave')])
values()#returns a view object. The view object contains the values of the dictionary,Example:print(a_dict.keys()) returns dict_keys(['You''leave'])
has_key()#returns true if a given key is available in the dictionary,Example:a.dict.has_kay('I hate') is True
‘never’ in a_dict # returns False 
del a_dict['me'] #won't delete anything 
a_dict.clear() #removes all items from the dictionary,Example: a_dict.clear() makes the dictionary item empty.

#Ok enough by me do the rest on your own!
#use dir() to get built in functions***
Sets:
    add()#adds a given element to a set if it's not there,Example: set.add(element)
	clear()#removes all items from the set,Example: set.clear()
	copy()# returns a shallow copy of the set, Example: set2 = set1.copy() 
	difference()#eturns a set that is the difference between two sets, Example:set_A.difference(set_B) mean (set_A - set_B)
	discard()#removes the specified item from the set,Example:set.discard(value)
	intersection()#returns a set that contains the similarity between two or more sets,Example:set.intersection(set1,...)
	issubset()#returns the logical result 
	pop()#removes the item at the given index from the set and returns the removed set, Example:set.pop(element_index)
	remove() #removes the first matching element, Example:set.remove(element)
	union() #eturns a set that contains all items from the original set, and all items from the specified set,Example :set.union(set1,...)
	update()#updates the current set, by adding items from another set,Example:set.update(set)
#Strings:
	capitalize()#returns a copy of the original string and converts the first character of the string to a capital letter
	casefold()#converts all characters of the string into lowercase letters and returns a new string
	center()#will center align the string
	count()#the number of characters within the string
	encode()#returns an encoded version of the given string
	find()#returns the index of first occurrence of the substring
	partition()#searches for a specified string
	replace()#replace a specified string
	split()#split a string at given place or character
	title()#returns a string where the first character in every word is upper case
	zfill() #adds zero at the beginning of the string till specifiled length
     
from collections import Counter
	List = [1,1,2,3,4,5,3,2,3,4,2,1,2,3]
   print(Counter(List))
from itertools import * #(Bonus: this one is optional, but recommended)
    natuals = count(1)
    ns = takewhile(lambda x: x <= 10, natuals)
    list(ns)
    
    
flower_orders=['W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B',
 'W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B',
 'W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B',
 'W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R','W/R',
 'W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R',
 'W/R','W/R','W/R','W/R','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y',
 'R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','W/R/V','W/R/V','W/R/V',
 'W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/N/R/V',
 'W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V',
 'W/N/R/V','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y',
 'W/R/B/Y','B/Y','B/Y','B/Y','B/Y','B/Y','R/B/Y','R/B/Y','R/B/Y',
 'R/B/Y','R/B/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y',
 'W/N/R/B/V/Y','W/N/R/B/V/Y','W/G','W/G','W/G','W/G','R/Y','R/Y',
 'R/Y','R/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','W/R/B/V',
 'W/R/B/V','W/R/B/V','W/R/B/V','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y',
 'W/N/R/V/Y','N/R/Y','N/R/Y','N/R/Y','W/V/O','W/V/O','W/V/O',
 'W/N/R/Y','W/N/R/Y','W/N/R/Y','R/B/V/Y','R/B/V/Y','R/B/V/Y',
 'W/R/V/Y','W/R/V/Y','W/R/V/Y','W/R/B/V/Y','W/R/B/V/Y','W/R/B/V/Y',
 'W/N/R/B/Y','W/N/R/B/Y','W/N/R/B/Y','R/G','R/G','B/V/Y','B/V/Y',
 'N/B/Y','N/B/Y','W/B/Y','W/B/Y','W/N/B','W/N/B','W/N/R','W/N/R',
 'W/N/B/Y','W/N/B/Y','W/B/V/Y','W/B/V/Y','W/N/R/B/V/Y/G/M',
 'W/N/R/B/V/Y/G/M','B/R','N/R','V/Y','V','N/R/V','N/V/Y','R/B/O',
 'W/B/V','W/V/Y','W/N/R/B','W/N/R/O','W/N/R/G','W/N/V/Y','W/N/Y/M',
 'N/R/B/Y','N/B/V/Y','R/V/Y/O','W/B/V/M','W/B/V/O','N/R/B/Y/M',
 'N/R/V/O/M','W/N/R/Y/G','N/R/B/V/Y','W/R/B/V/Y/P','W/N/R/B/Y/G',
 'W/N/R/B/V/O/M','W/N/R/B/V/Y/M','W/N/B/V/Y/G/M','W/N/B/V/V/Y/P']
#1
col=[sub.replace("/", "") for sub in flower_orders]
#2
W_true=[s for s in col if "W" in s]
print(len(W_true))
#3
col_full=('/'.join(flower_orders)).replace("/", "")
all_freq = {}
for i in col_full:
    if i in all_freq:
        all_freq[i] += 1
    else:
        all_freq[i] = 1
print(all_freq)
import matplotlib.pyplot as plt
import pandas as pd
plt.bar(all_freq.keys(), all_freq.values(), color='g')
#4&5
col_two=(list(permutations(all_freq.keys(), 2)))
col_three=(list(permutations(all_freq.keys(), 3)))



---
dead_men_tell_tales = ['Four score and seven years ago our fathers brought forth on this',
'continent a new nation, conceived in liberty and dedicated to the',
'proposition that all men are created equal. Now we are engaged in',
'a great civil war, testing whether that nation or any nation so',
'conceived and so dedicated can long endure. We are met on a great',
'battlefield of that war. We have come to dedicate a portion of',
'that field as a final resting-place for those who here gave their',
'lives that that nation might live. It is altogether fitting and',
'proper that we should do this. But in a larger sense, we cannot',
'dedicate, we cannot consecrate, we cannot hallow this ground.',
'The brave men, living and dead who struggled here have consecrated',
'it far above our poor power to add or detract. The world will',
'little note nor long remember what we say here, but it can never',
'forget what they did here. It is for us the living rather to be',
'dedicated here to the unfinished work which they who fought here',
'have thus far so nobly advanced. It is rather for us to be here',
'dedicated to the great task remaining before us--that from these',
'honored dead we take increased devotion to that cause for which',
'they gave the last full measure of devotion--that we here highly',
'resolve that these dead shall not have died in vain, that this',
'nation under God shall have a new birth of freedom, and that',
'government of the people, by the people, for the people shall',
'not perish from the earth.']
#1
tales_full=(' '.join(dead_men_tell_tales))
#2 
tales_full=tales_full.replace(" ", "")
#3
all_freq = {}
for i in tales_full.lower():
    if i in all_freq:
        all_freq[i] += 1
    else:
        all_freq[i] = 1
print ("Count of all characters in tales is :\n "+ str(all_freq))
#4&5
def window(seq, n=2):
    """Sliding window width n from seq.  From old itertools recipes."""
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
pairs = pd.DataFrame(window(tales_full), columns=['state1', 'state2'])
counts = pairs.groupby('state1')['state2'].value_counts()
probs = (counts / counts.sum()).unstack()
print(probs)
#6

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(probs)
ax.set_xticklabels(all_freq.keys())
ax.set_yticklabels(all_freq.keys())