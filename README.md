# Minhash-LSH
Implementation of Minhash and Locality Sensitive Hashing algorithms.

This project was part of the course 'Algorithms for Big Data' MYE047 for the spring semester of 2020.
It was developed in Python 3.8.2 and requires only matplotlib to be able to print all the statistical plots. 
The two files, *ratings.csv*  and *ratings_100users.csv*, were used as testing. 
The script will accept any other file in the same format as the two files mentioned.

## Execution
Make sure you have installed Python 3.8.2 or later and matplotlib 3.2.1 or later.
In order to execute the script through CLI just type:
```python
python3 item_similarities.py ratings.csv
```
where `ratings.csv` type your file or one of the two given. If the input file is ommitted as a command line argument 
then the script will search for the `ratings_100users.csv` file as a fallback. The algorithm will output 4 `.csv` files 
containing the signature matrix and all the dictionaries that are being created to store the dataset as a visualisation method 
of the inner workings. Additionally, the script will print two plots, one for the MinHash and one for the LSH experimentation.

## Experimentation
The experimenation for both algorithms happens sequentially starting with minhash and moving on to the lsh. The global experimentation
variables are:  
* **NO_OF_HASH_FUNCTIONS**: The number of hash functions, or signatures, a movie will have in the signature matrix.  
* **NO_OF_CONSIDERED_MOVIES**: The number of the first <X> movie ids of the dataset. If for example that number is 20, then the first 20 movie
ids (ascending) will be considered. These movies will form pairs with each other and give us our ground truth. Then the same movie pairs
will be considered when testing the algorithms against the ground truth results.  
* **ACCEPTANCE_LEVEL**: The level for two movies to be considered similar.
* **n_values**: A list that contains the number of signatures the minhash algorithm has to take into account in each iteration. For example,
[5, 10, 15, 20, 25, 30, 35, 40] means minhash will consider in its first iteration calculations the first 5 signatures out of the *NO_OF_HASH_FUNCTIONS*
available and so on. *The script does not check if the numbers given are less or equal to the *NO_OF_HASH_FUNCTIONS* **yet**.  
* **n**: Number of signatures the lsh algorithm has to take into account. (Mush n = b * r)  
* **b_values**: A list that contains the number of bands that are to be created in each iteration of the lsh algorithm.  
* **r_values**: A list that contains the number of signatures each band will have. Must n = b * r and of course, len(b_values) = len(r_values).  


### Min Hash
#### With *ratings_100users.csv*
The signature matrix will be created with 40 signatures per movie and the first 20 movie ids will be considered for the sampling. 
This means the movies with ids (1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21) will form pairs with each other and set the 
ground truth with the acceptance level being 0.25. The default **n_values** are: [5, 10, 15, 20, 25, 30, 35, 40]. Example plot is the following:  

![alt text](https://github.com/paschalishom/Minhash-LSH/blob/master/images/ratings_100users_lsh.png "ratings_100users.csv minhash")

#### With *ratings.csv*
Testing values should be set as:  
1. **NO_OF_HASH_FUNCTIONS**= 100  
2. **NO_OF_CONSIDERED_MOVIES**= 100  
3. **ACCEPTANCE_LEVEL**= 0.25  
4. **n_values**= [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]  
Resulting plot example:  

![alt text](https://github.com/paschalishom/Minhash-LSH/blob/master/images/ratings_100users_minhash_100considered.png "ratings.csv minhash")

### LSH
#### With *ratings_100users.csv*
Testing values should be set as:  
1. **NO_OF_HASH_FUNCTIONS**= 40  
2. **NO_OF_CONSIDERED_MOVIES**= 20  
3. **ACCEPTANCE_LEVEL**= 0.25  
4. **n** = 40  
5. **b_values** = [20, 10, 8, 5, 4, 2]  
6. **r_values** = [2, 4, 5, 8, 10, 20]  
Example plot:  

![alt text](https://github.com/paschalishom/Minhash-LSH/blob/master/images/ratings_100users_lsh.png "ratings_100users.csv lsh")

#### With *ratings.csv*
Testing values should be set as:  
1. **NO_OF_HASH_FUNCTIONS**= 100  
2. **NO_OF_CONSIDERED_MOVIES**= 100  
3. **ACCEPTANCE_LEVEL**= 0.25  
4. **n** = 40  
5. **b_values** = [20, 10, 8, 5, 4, 2]  
6. **r_values** = [2, 4, 5, 8, 10, 20]  
Example plot:  

![alt text](https://github.com/paschalishom/Minhash-LSH/blob/master/images/ratings_lsh_100considered_100hf.png "ratings.csv lsh")
