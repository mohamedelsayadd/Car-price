# 1 - import packges : 
import numpy as np
import pandas as pd
import sklearn as sk

# 2 - reed the file : 
car = pd.read_csv('Desktop\quikr_car.csv')
car = pd.DataFrame(car)

# 3 - get some informaition about the data :
# print(car.head(5))
# print(car.shape)
# print(car.columns)
# print(car.dtypes)
# print(car.describe(exclude=[np.number]))
# print(car.info())
# print(car.isnull().sum())
# print(car['year'].unique())
# print(car['Price'].unique())
# print(car['kms_driven'].unique())
# print(car['fuel_type'].unique())

backup=car.copy()

# 3 - Data Preprocessing :
car.dropna(inplace=True)
# print(car.isnull().sum())

car=car[car['year'].str.isnumeric()]
car['year']=car['year'].astype(int)
# print(car.dtypes)

car=car[car['Price']!='Ask For Price']
car['Price']=car['Price'].str.replace(',','')
car['Price'] = car['Price'].astype(int)
# print(car['Price'].unique())

car['kms_driven']=car['kms_driven'].str.replace(' kms','')
car['kms_driven']=car['kms_driven'].str.replace(',','')
car=car[car['kms_driven'].str.isnumeric()]
car['kms_driven'] = car['kms_driven'].astype(int)
# print(car['kms_driven'].unique())

car['name'] = car['name'].str.split().str[:3].str.join(' ')
# print(car['fuel_type'].unique())

# print(car.head(5))

car = car.reset_index(drop=True)

# print(car.describe())
car=car[car['Price']<6e6]
car = car.reset_index(drop=True)

# print(car.describe(exclude=[np.number]))

# 4 - split the data :
x=car[['name','company','year','kms_driven','fuel_type']]
y=car['Price']

# 5- train the data :
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1 , random_state=302)

# 6 - import the packges for models : 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# 7 - fit the model : 
ohe = OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])
column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']), remainder='passthrough')
lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
# print(y_pred)
# print(r2_score(y_test,y_pred))

#8 - Evaluate classification model performance :
scores=[]
for i in range(1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    scores.append(r2_score(y_test,y_pred))
    # print(scores)
# print(np.argmax(scores))
# print(scores[np.argmax(scores)])
# print(pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5))))

# 9 - save the model :
import pickle
pickle.dump(pipe,open('car_price.pkl','wb'))
model = pickle.load(open( 'car_price.pkl', 'rb' ))
print(model.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5))))