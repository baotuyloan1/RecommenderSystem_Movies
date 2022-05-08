#!D:\VKU\Python\python.exe
print("Content-Type: text/html\n")
from unicodedata import category
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import json

movies_df = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python',encoding='iso-8859-1')

ratings_df = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python')

movies_df.columns = ['MovieID', 'Title', 'Genres']
ratings_df.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']


movies_df['List Index'] = movies_df.index


merged_df = movies_df.merge(ratings_df, on='MovieID')


merged_df = merged_df.drop('Timestamp', axis=1).drop('Title', axis=1).drop('Genres', axis=1)

user_Group = merged_df.groupby('UserID')

"""
Định dạng dữ liệu đàu vào cho RBM
Lưu trữ các xếp hạn của người đã chuẩn hóa vào 1 danh sách trX
"""

# Tổng số người dùng cho việc train
amountOfUsedUsers = 1000

# Tạo danh sách trX
trX = []

# For each user in the group
#index, row
for userID, curUser in user_Group:

    #tạo 1 mảng lưu trữ xếp hạng của tất cả bộ phim
    temp = [0]*len(movies_df)

    # Lặp tất cả các phim của mỗi người
    for num, movie in curUser.iterrows():
        temp[movie['List Index']] = movie['Rating']/5.0
    trX.append(temp)
    if amountOfUsedUsers == 0:
        break
    amountOfUsedUsers -= 1

# Đặt thông số cho mô hình
#quy tắc chọn hiddenUnits
#Số lượng hiddenUnits phải nằm giữa kích thước của visibleUnits và kích thước của lớp đầu ra.
#Số lượng hiddenUnits phải nhỏ hơn hai lần kích thước của visibleUnits
hiddenUnits = 50
visibleUnits = len(movies_df)

#định nghĩa 1 hàm nhưng không biết trước giá trị đầu vào thì dùng placeholder
vb = tf.placeholder(tf.float32, [visibleUnits])  
hb = tf.placeholder(tf.float32, [hiddenUnits])  
W = tf.placeholder(tf.float32, [visibleUnits, hiddenUnits])  # Weights that connect the hidden and visible layers

#  Xử lý trước dữ liệu đầu vào
v0 = tf.placeholder("float", [None, visibleUnits])
#Activation functions là những hàm phi tuyến được áp dụng vào đầu ra của các nơ-ron trong tầng ẩn của một mô hình mạng, và được sử dụng làm input data cho tầng tiếp theo.
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation (2)| hàm kích hoạt phi tuyến
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))  # Gibb's Sampling




# Xây dựng lại dữ liệu đầu vào đã được xử lý trước (các chức năng kích hoạt Sigmoid và ReLU được sử dụng)
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  # Hidden layer activation
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)
# print(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
""" Set RBM Training Parameters """

# Learning rate
alpha = 1.0

# Create the gradients
w_pos_grad = tf.matmul(tf.transpose(v0), h0) # Set positive gradients
w_neg_grad = tf.matmul(tf.transpose(v1), h1) # Set negative gradients

# Calculate the Contrastive Divergence to maximize

CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

# Create methods to update the weights and biases
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

# Set the error function, here we use Mean Absolute Error Function
err = v0 - v1
err_sum = tf.reduce_mean(err*err)

""" Initialize our Variables with Zeroes using Numpy Library """


cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)


cur_vb = np.zeros([visibleUnits], np.float32)

# Current hidden unit biases
cur_hb = np.zeros([hiddenUnits], np.float32)

# Previous weight
prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

# Previous visible unit biases
prv_vb = np.zeros([visibleUnits], np.float32)

# Previous hidden unit biases
prv_hb = np.zeros([hiddenUnits], np.float32)
#Một phiên cho phép thực hiện các đồ thị hoặc một phần của đồ thị. 
# Nó phân bổ tài nguyên (trên một hoặc nhiều máy) cho việc đó và giữ các giá trị thực tế 
# của các kết quả và biến trung gian.
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #khởi tạo tất cả các biến

print("Trainning...")
# Train RBM with 15 Epochs, with Each Epoch using 10 batches with size 100, After training print out the error by epoch
epochs = 15
batchsize = 100
errors = []
for i in range(epochs):
    for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        batch = trX[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
    errors.append(sess.run(err_sum, feed_dict={v0: trX, W: cur_w, vb: cur_vb, hb: cur_hb}))
    # print(errors[-1])


"""
Recommendation System :-

- We can now predict movies that an arbitrarily selected user might like. 
- This can be accomplished by feeding in the user's watched movie preferences into the RBM and then reconstructing the 
  input. 
- The values that the RBM gives us will attempt to estimate the user's preferences for movies that he hasn't watched 
  based on the preferences of the users that the RBM was trained on.
"""

# Select the input User
inputUser = [trX[50]]

# Feeding in the User and Reconstructing the input
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb) #Energy based Model
feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})


# List the 20 most recommended movies for our mock user by sorting it by their scores given by our model.
scored_movies_df_50 = movies_df
scored_movies_df_50["Recommendation Score"] = rec[0]




# Find the mock user's UserID from the data
# print(merged_df.iloc[50])  # Result you get is UserID 150

# Find all movies the mock user has watched before
movies_df_50 = merged_df[merged_df['UserID'] == 150]
# print(movies_df_50.head())

""" Merge all movies that our mock users has watched with predicted scores based on his historical data: """

# Merging movies_df with ratings_df by MovieID
merged_df_50 = scored_movies_df_50.merge(movies_df_50, on='MovieID', how='outer')

# Dropping unnecessary columns
merged_df_50 = merged_df_50.drop('List Index_y', axis=1).drop('UserID', axis=1)

# Sort and take a look at first 20 rows
top_20 = merged_df_50.sort_values(['Recommendation Score'], ascending=False).head(50)




class Rating():
    def __init__(self, id, name, category, score, rating):
        self.id = id
        self.name = name
        self.category = category
        self.score = score
        self.rating = rating

    def to_dict(self):
        return  {'id': self.id,
                               'name': self.name,
                               'category': self.category,
                               'score':self.score,
                               'rating':self.rating}

recommenderMovies =[];
for index in top_20.index:
    id = int(top_20['MovieID'][index])
    title = top_20['Title'][index]
    category = top_20['Genres'][index]
    score = str(top_20['Recommendation Score'][index])
    rating = str(top_20['Rating'][index])
    movie = Rating(id,title,category, score, rating)
    recommenderMovies.append(movie)






json_string = json.dumps([data.to_dict() for data in recommenderMovies])
# json_string = object_schema.dumps(recommenderMovies, many=True)
# results.sort(key=lambda obj: obj["rating"])

# jsdata = Rating.toJSON({"results": results})
with open('json_data.json', 'w') as outfile:
    outfile.write(json_string)
    print("Train Successfull")

# print(trX)



    
""" There are some movies the user has not watched and has high score based on our model. So, we can recommend them. """

