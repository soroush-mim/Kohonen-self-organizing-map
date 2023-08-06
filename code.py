import numpy as np
import os
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix

def create_data_set(): 
    """ Load the Yale Faces data set, extract the faces on the images and generate labels for each image.
        
        Returns: Train and validation samples with their labels. The training samples are flattened arrays 
        of size  (243*320) , the labels are one-hot-encoded values for each category
    """
    images_path = [ os.path.join("yalefaces", item)  for item in  os.listdir("yalefaces") ]
    image_data = []
    image_labels = []
    
    for im_path in images_path:
        im = io.imread(im_path , as_gray = True)
        image_data.append(np.array(im, dtype='uint8'))
        
        label = os.path.split(im_path)[1].split(".")[1]
       
        image_labels.append(label)

    X_ = np.array(image_data).astype('float32')
    enc = LabelBinarizer()
    y_ = enc.fit_transform(image_labels)
    #print(enc.classes_)
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=.7, random_state = 42)
    X_val , X_test , y_val , y_test = train_test_split(X_test, y_test, train_size=2/3, random_state = 42)
    X_test = X_test/255.0
    X_train = X_train/255.0
    X_val = X_val/255.0
    return (X_train).reshape((X_train.shape[0],X_train.shape[1]*X_train.shape[2])),\
            (X_test).reshape((X_test.shape[0],X_test.shape[1]*X_test.shape[2]))\
            ,X_val.reshape((X_val.shape[0],X_val.shape[1]*X_val.shape[2])), y_train, y_test , y_val , enc.classes_
    


class SOM:
    def __init__(self ,map_size , lr , sigma , decay):
        #map size = (w,h,f)
        self.map = np.random.random(map_size) 
        #ind matrix is a matrix that holds index of each cell
        self.ind_matrix = np.zeros((self.map.shape[0] , self.map.shape[1] , 2))
        #mask matrix is a matrix that for each neuron holds a 
        #mask that is corrosponding to gaussian neighborhood for given sigma
        self.mask = np.zeros((map_size[0] ,map_size[1] ,map_size[0] ,map_size[1]))
        self.lr = lr
        self.sigma = sigma
        self.decay = decay
        #initilizing ind matrix
        for i in range(map_size[0]):
            for j in range(map_size[1]):
                self.ind_matrix[i,j] = [i,j]
        #initializing mask
        for i in range(map_size[0]):
            for j in range(map_size[1]):
                self.mask[i][j] = self.get_N_mask(np.array([i,j]) , sigma)

                
    def get_winner(self,x):
        dists = np.linalg.norm(self.map - x , axis = 2)
        winner = np.unravel_index(np.argmin(dists), dists.shape)
        return np.array(winner)
        
    def get_N_mask(self ,winner , sigma):
        mask = np.linalg.norm(self.ind_matrix - winner , axis = 2)
        mask = mask**2
        mask = -(mask/(2*sigma**2))
        mask = np.exp(mask)
        return mask
    
    def train(self,X_train  , y, itr_num , error_t = 10**-10):
        Js = []
        k = -1
        sigma = self.sigma
        beta = self.lr
        for i in range(itr_num):
            if i%10==0:
                k+=1
            prev_map = np.copy(self.map)
            shuffle_ind = np.random.randint(low = 0 , high = len(X_train) , size =len(X_train))
            sigma_change = 1/(self.decay**(2*k))
            beta = self.lr * self.decay**k #(1 - (i+1)/itr_num)
            sigma = self.sigma* self.decay**k
            for j in range(len(X_train)):
                x = X_train[shuffle_ind[j]]
                winner = self.get_winner(x)
                mask = np.power(self.mask[winner[0] , winner[1]] , sigma_change)
                self.map = self.map + beta*( mask[:,:,None] * (x - self.map))
            
            Js.append(np.linalg.norm(prev_map - self.map))
            
            if Js[-1]<error_t:
                return Js
            print(i)
            print('beta: ' , beta)
            print('sigma: ' , sigma)
            #print('purity: ', self.purity(X_train , y))


        return Js
    
    def u_matrix(self):
        u_matrix = np.zeros((self.map.shape[0],self.map.shape[1]))
        
        for i in range(u_matrix.shape[0]):
            for j in range(u_matrix.shape[1]):
                t = 0
                n = 0
                for k in [1,0,-1]:
                    for l in [1,0,-1]:
                        if (i+k) in range(u_matrix.shape[0]) and (j+l) in range(u_matrix.shape[1]):
                            t+= np.linalg.norm(self.map[i+k][j+l]-self.map[i][j])
                            n+=1
                u_matrix[i][j] = t/(n-1)
        return u_matrix
    def purity(self , X , y):
        #y is one hot
        map_size = self.map.shape
        count = np.zeros((map_size[0] , map_size[1] , y.shape[1]))
        for i in range(len(X)):
            x = X[i]
            winner = self.get_winner(x)
            count[winner[0] , winner[1]] += y[i]

        t = np.max(count , axis = 2)
        #print(t)

        purity = np.sum(t) / len(X)
        return purity

    def visualize(self,X,y,  colors , label_names):
        
        w_x , w_y = zip(*[self.get_winner(d) for d in X])
        w_x = np.array(w_x)
        w_y = np.array(w_y)
        target = np.argmax(y , axis = 1)
        ax = plt.figure(figsize=(8, 8))
        plt.pcolor(self.u_matrix(), cmap='bone_r', alpha=.5)
        plt.colorbar()

        for c in np.unique(target):
            idx_target = target==c
            plt.scatter(w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                        w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8, 
                        s=30, color=colors[c], label=label_names[c] )

        plt.legend(loc='upper left',bbox_to_anchor=(1.2, 1.0))
        plt.grid()
        plt.show()
    
    def extract_feature(self , X):
        features = np.zeros((len(X) , self.map.shape[0],self.map.shape[1]))
        for i in range(len(X)):
            x = X[i]
            features[i] = 1 / (1 + np.linalg.norm(self.map - x , axis = 2))
        return features
#creating datasets
X_train , X_test, X_val, y_train , y_test , y_val  , label_names= create_data_set()
#creating and training SOM
som = SOM((8,8,77760) , lr = .02 ,sigma = 3. , decay = 0.95)
j = som.train(X_train=X_train , y = y_train, itr_num=500)
print('purity on train set: ' ,som.purity(X_train , y_train) )
print('purity on test set: ' , som.purity(X_test , y_test))
#plotting J
plt.plot(j)
plt.xlabel('itr')
plt.ylabel('norm2 distance')
plt.clf()

#visualizing SOM

colors = colors = np.array(
         [[1.,0.,0.,1.],               # center  red
          [1., 1., 0. , 1.],           #glasses yellow
          [0.,1.,0.,1.],               #happy green
          [0.,0.,1.,.9],               #leftlight blue
          [.9,0.,0.,.5],               #no glass red 
          [.7,0.,0.,1.],               #normal red
          [.5,0.,.5,1.],               #right light purple
          [1., 0.,1.,.8],              #sad pink
          [0., 0., 0. , 1.],           #sleepy black
          [0.,.8,0.,.8],               #suprized green
          [1., 1., 1. , 1.]])          #wink white
som.visualize(X_train , y_train , colors , label_names)
#model with clean features
model1 = tf.keras.models.Sequential([
                                  layers.Dense(3000, activation='relu' ),    
                                  layers.Dense(500, activation='relu'),
                                  layers.Dense(100, activation='relu'),
                                  layers.Dense(11 , activation='softmax')    
])
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
callback = tf.keras.callbacks.EarlyStopping(patience=30)
model1.compile(optimizer=opt,loss ='sparse_categorical_crossentropy' ,metrics=['accuracy'])
history = model1.fit(X_train, np.argmax(y_train , axis = 1), epochs=1000, validation_data=(X_val, np.argmax(y_val , axis = 1)) ,
                    callbacks=[callback] , verbose = 1)


model1.evaluate(X_test,np.argmax(y_test , axis = 1) , verbose = 1)

cm = confusion_matrix(np.argmax(y_test , axis = 1), np.argmax(model1.predict(X_test) , axis = 1))


plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.clf()
cm = confusion_matrix(np.argmax(y_train , axis = 1), np.argmax(model1.predict(X_train) , axis = 1))


plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.clf()

#model with extracted features

model2 = tf.keras.models.Sequential([
                                  tf.keras.layers.Flatten(),
                                  layers.Dense(3000, activation='relu' ),   
                                  layers.Dense(500, activation='relu'),
                                  layers.Dense(100, activation='relu'),
                                  layers.Dense(11 , activation='softmax')    
])
opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
callback = tf.keras.callbacks.EarlyStopping(patience=200)
model2.compile(optimizer=opt,loss ='sparse_categorical_crossentropy' ,metrics=['accuracy'])
history2 = model2.fit(som.extract_feature(X_train), np.argmax(y_train , axis = 1), epochs=200, validation_data=(som.extract_feature(X_val), np.argmax(y_val , axis = 1)) ,
                    callbacks=[callback] , verbose = 1)

model2.evaluate(som.extract_feature(X_test),np.argmax(y_test , axis = 1) , verbose = 1)

cm = confusion_matrix(np.argmax(y_train , axis = 1), np.argmax(model2.predict(som.extract_feature(X_train)) , axis = 1))


plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.clf()

cm = confusion_matrix(np.argmax(y_test , axis = 1), np.argmax(model2.predict(som.extract_feature(X_test)) , axis = 1))


plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
