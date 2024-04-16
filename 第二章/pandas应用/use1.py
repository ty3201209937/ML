import numpy as np
import pandas as pd
data = {"name":["Tom","Mike","Mary","Bob","Alice","Betty","Helen","David","John","Jack"],"age":[18,11,14,22,14,17,15,16,18,19],
        "city":["beijing","shanghai","shenzhen","hubei","guangzhou","nanjing","jinan","wuhan","chengdu","dazou"]}
user = pd.DataFrame(data)
user["sex"] = "male"
print(user)
user.to_csv('user.cvs')
print(user.describe())
print(user.name)
print(user.loc[2])
print(user.head(2))
print(user.tail(2))
print(user.age.max())
print(user.age.cumsum())
print(user.sort_values(by='age'))
user.loc[2,'age']==22
print(user)

