import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

# Uncomment the below lines based on which rating has to be predicted

df = pd.read_csv('overallRating_data.csv')
# df = pd.read_csv('cleanliness_data.csv')
# df = pd.read_csv('neighbourhood_data.csv')
# df = pd.read_csv('accuracy_data.csv')
# df = pd.read_csv('checkin_data.csv')
# df = pd.read_csv('communication_data.csv')
# df = pd.read_csv('value_data.csv')


df["host_is_superhost"].replace({'t':1,'f':0},inplace = True)
df["has_availability"].replace({'t':1,'f':0},inplace = True)
df["instant_bookable"].replace({'t':1,'f':0},inplace = True)
df["host_has_profile_pic"].replace({'t':1,'f':0},inplace = True)
df["host_identity_verified"].replace({'t':1,'f':0},inplace = True)
df["amenities"] = [len(item.split(",")) for item in df["amenities"]]


# Uncomment the below lines based on which rating has to be predicted

review_to_predict = 'review_scores_rating'
# review_to_predict = 'review_scores_accuracy'
# review_to_predict = 'review_scores_cleanliness'
# review_to_predict = 'review_scores_checkin'
# review_to_predict = 'review_scores_communication'
# review_to_predict = 'review_scores_location'
# review_to_predict = 'review_scores_value'

df = df[[  'host_is_superhost', 'host_listings_count',
       'host_total_listings_count',
       'host_has_profile_pic', 'host_identity_verified', 'latitude', 'longitude', 'accommodates', 'bedrooms', 'beds',
       'amenities','number_of_reviews',
       'number_of_reviews_ltm', 'number_of_reviews_l30d', 'instant_bookable',
       'calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms',
       'calculated_host_listings_count_shared_rooms', 'sentiment_score',review_to_predict]]


X = df.iloc[:, 0:20]
y = df.iloc[:,-1:]


# Split the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
mean_arr = []
std_arr = []
mse = []
kvalues = [1,3,5,7,9,11,13,15]
for k in kvalues:
    model = KNeighborsRegressor(n_neighbors=k).fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=5,scoring='neg_mean_squared_error')
    mean_arr.append(abs(np.array(scores).mean()))
    std_arr.append(abs(np.array(scores).std()))

plt.errorbar(kvalues, mean_arr, yerr=std_arr)      
plt.xlabel("Number of k neighbours")
plt.ylabel("Mean Squared Error")
plt.title("Mean Squared Error vs number of k neighbours ")
plt.legend(["Mean Squared Error"],loc="upper right")
plt.show()  



from sklearn.metrics import mean_squared_error
model = KNeighborsRegressor(n_neighbors=7).fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(mse) 



from sklearn.ensemble import RandomForestRegressor
maxFeatures = [1,5,10,20]
nEstimators = [10,20,50,75,100,150,200,250]
for p in maxFeatures:
    mean_array = []
    std_array = []
    for c in nEstimators:
        model = RandomForestRegressor(n_estimators=c, max_features=p)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
        mean_array.append(abs(np.array(scores).mean()))
        std_array.append(abs(np.array(scores).std()))

    plt.errorbar(
            nEstimators, mean_array, yerr=std_array, label="MaxFeatures = {0}".format(p)
        )
plt.xlabel("nEstimators Value")
plt.ylabel("Mean Square Error")
plt.title("Mean Square Error vs nEstimators for different values of MaxFeaures")

plt.legend(loc="upper right")
plt.show()


model = RandomForestRegressor(n_estimators=10, max_features=10).fit(X_train,y_train)
ypred = model.predict(X_test)
mse = mean_squared_error(y_test, ypred)
print(mse) 



from sklearn.dummy import DummyRegressor
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)
ypred = dummy_regr.predict(X_test)
mse = mean_squared_error(y_test, ypred)
print(mse) 


importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(model.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)
importances[:7].plot.bar()
plt.show()