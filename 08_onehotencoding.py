from sklearn.preprocessing import OneHotEncoder
import pandas as pd

data_le = pd.DataFrame([             
    {'place':'大安區', 'price':90},
    {'place':'信義區', 'price':100},
    {'place':'南港區', 'price':80},
    {'place':'中正區', 'price':80}
])
onehotencoder = OneHotEncoder()
data_str_ohe=onehotencoder.fit_transform(data_le[['place']]).toarray()
df = pd.DataFrame(data_str_ohe)

print(df)
##################
dum_df = pd.get_dummies(data_le, columns=["place"], prefix=["Type_is"] )


print(dum_df)