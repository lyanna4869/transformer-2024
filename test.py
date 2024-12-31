import pandas as pd

# raw_data = {'src': ["Go.", "Run!"], 'trg': ["Va !", "Cours !"]}
# df = pd.DataFrame(raw_data, columns=["src", "trg"], dtype=str)
# print(df)

data = [['Alice', 25], 
        ['Bob', 30], 
        ['Charlie', 35]]
df1 = pd.DataFrame(data,index=['a','b','c'] ,columns=['Name', 'Age'])
print(df1)
                 