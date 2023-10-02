import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import r2_score

df = pd.read_csv("Equity_Apartments_Data.csv")

print(df.info(), df.describe())


```python
#Renaming Erroneously Titled Variable

df.rename(columns = {'Estiamted_Vacancy':'Estimated_Vacancy'}, inplace = True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Price</th>
      <th>Beds</th>
      <th>Baths</th>
      <th>sq.ft</th>
      <th>Floor</th>
      <th>Move_in_date</th>
      <th>building_id</th>
      <th>unit_id</th>
      <th>URL</th>
      <th>...</th>
      <th>Fireplace</th>
      <th>City_Skyline</th>
      <th>Kitchen_Island</th>
      <th>Stainless_Appliances</th>
      <th>Renovated</th>
      <th>Office_Space</th>
      <th>Days_Till_Available</th>
      <th>Day_of_the_week_recorded</th>
      <th>Unique_ID</th>
      <th>Estimated_Vacancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>432</td>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>9/11/21</td>
      <td>160</td>
      <td>32A \r\n</td>
      <td>https://www.equityapartments.com/new-york-city...</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>56.0</td>
      <td>Wednesday</td>
      <td>16032A\r\n160RiversideBoulevardApartments</td>
      <td>0.057143</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2579</td>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>9/11/21</td>
      <td>160</td>
      <td>32A \r\n</td>
      <td>https://www.equityapartments.com/new-york-city...</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>57.0</td>
      <td>Monday</td>
      <td>16032A\r\n160RiversideBoulevardApartments</td>
      <td>0.057143</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5354</td>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>9/11/21</td>
      <td>160</td>
      <td>32A \r\n</td>
      <td>https://www.equityapartments.com/new-york-city...</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>58.0</td>
      <td>Friday</td>
      <td>16032A\r\n160RiversideBoulevardApartments</td>
      <td>0.059341</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8402</td>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>9/11/21</td>
      <td>160</td>
      <td>32A \r\n</td>
      <td>https://www.equityapartments.com/new-york-city...</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>59.0</td>
      <td>Sunday</td>
      <td>16032A\r\n160RiversideBoulevardApartments</td>
      <td>0.063736</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11467</td>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>9/11/21</td>
      <td>160</td>
      <td>32A \r\n</td>
      <td>https://www.equityapartments.com/new-york-city...</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>60.0</td>
      <td>Saturday</td>
      <td>16032A\r\n160RiversideBoulevardApartments</td>
      <td>0.068132</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>




```python
df.isnull().sum()
```




    id                             0
    Price                          0
    Beds                           0
    Baths                          0
    sq.ft                          0
    Floor                          0
    Move_in_date                 788
    building_id                  852
    unit_id                      852
    URL                            0
    Day_Recorded                   0
    Amenity                     2491
    Apartment Name                 0
    Address                        0
    City                           0
    Units                          0
    Northern_Exposure           2491
    Southern_Exposure           2491
    Eastern_Exposure            2491
    Western_Exposure            2491
    Balcony                     2491
    Walk_In_Closet              2491
    Fireplace                   2491
    City_Skyline                2491
    Kitchen_Island              2491
    Stainless_Appliances        2491
    Renovated                   2491
    Office_Space                2491
    Days_Till_Available          788
    Day_of_the_week_recorded       0
    Unique_ID                    852
    Estimated_Vacancy              0
    dtype: int64




```python
df['building_unit_id'] = df['building_id'] + df['unit_id']

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Price</th>
      <th>Beds</th>
      <th>Baths</th>
      <th>sq.ft</th>
      <th>Floor</th>
      <th>Move_in_date</th>
      <th>building_id</th>
      <th>unit_id</th>
      <th>URL</th>
      <th>...</th>
      <th>City_Skyline</th>
      <th>Kitchen_Island</th>
      <th>Stainless_Appliances</th>
      <th>Renovated</th>
      <th>Office_Space</th>
      <th>Days_Till_Available</th>
      <th>Day_of_the_week_recorded</th>
      <th>Unique_ID</th>
      <th>Estimated_Vacancy</th>
      <th>building_unit_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>432</td>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>9/11/21</td>
      <td>160</td>
      <td>32A \r\n</td>
      <td>https://www.equityapartments.com/new-york-city...</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>56.0</td>
      <td>Wednesday</td>
      <td>16032A\r\n160RiversideBoulevardApartments</td>
      <td>0.057143</td>
      <td>16032A \r\n</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2579</td>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>9/11/21</td>
      <td>160</td>
      <td>32A \r\n</td>
      <td>https://www.equityapartments.com/new-york-city...</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>57.0</td>
      <td>Monday</td>
      <td>16032A\r\n160RiversideBoulevardApartments</td>
      <td>0.057143</td>
      <td>16032A \r\n</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5354</td>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>9/11/21</td>
      <td>160</td>
      <td>32A \r\n</td>
      <td>https://www.equityapartments.com/new-york-city...</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>58.0</td>
      <td>Friday</td>
      <td>16032A\r\n160RiversideBoulevardApartments</td>
      <td>0.059341</td>
      <td>16032A \r\n</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8402</td>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>9/11/21</td>
      <td>160</td>
      <td>32A \r\n</td>
      <td>https://www.equityapartments.com/new-york-city...</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>59.0</td>
      <td>Sunday</td>
      <td>16032A\r\n160RiversideBoulevardApartments</td>
      <td>0.063736</td>
      <td>16032A \r\n</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11467</td>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>9/11/21</td>
      <td>160</td>
      <td>32A \r\n</td>
      <td>https://www.equityapartments.com/new-york-city...</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>60.0</td>
      <td>Saturday</td>
      <td>16032A\r\n160RiversideBoulevardApartments</td>
      <td>0.068132</td>
      <td>16032A \r\n</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
df.drop(['id', 'Move_in_date', 'building_id', 'unit_id', 'URL', 'Day_Recorded', 'Amenity', 'Apartment Name',
         'Address', 'Northern_Exposure', 'Southern_Exposure', 'Eastern_Exposure', 'Western_Exposure',
         'Days_Till_Available', 'Day_of_the_week_recorded', 'Unique_ID', 'Stainless_Appliances'], axis=1, inplace=True)

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
      <th>Beds</th>
      <th>Baths</th>
      <th>sq.ft</th>
      <th>Floor</th>
      <th>City</th>
      <th>Units</th>
      <th>Balcony</th>
      <th>Walk_In_Closet</th>
      <th>Fireplace</th>
      <th>City_Skyline</th>
      <th>Kitchen_Island</th>
      <th>Renovated</th>
      <th>Office_Space</th>
      <th>Estimated_Vacancy</th>
      <th>building_unit_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>New York City</td>
      <td>455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.057143</td>
      <td>16032A \r\n</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>New York City</td>
      <td>455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.057143</td>
      <td>16032A \r\n</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>New York City</td>
      <td>455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.059341</td>
      <td>16032A \r\n</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>New York City</td>
      <td>455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.063736</td>
      <td>16032A \r\n</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>New York City</td>
      <td>455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.068132</td>
      <td>16032A \r\n</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop_duplicates(subset='building_unit_id', keep='first', inplace=True, ignore_index=False)

print(df.info())

df.head()

# Dropping the duplicate rows cut out about 94% of the data; since most units did not see a price change over the
# duration of the recording, this was a necessary step in the analysis.
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4046 entries, 0 to 62808
    Data columns (total 16 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Price              4046 non-null   int64  
     1   Beds               4046 non-null   int64  
     2   Baths              4046 non-null   float64
     3   sq.ft              4046 non-null   int64  
     4   Floor              4046 non-null   int64  
     5   City               4046 non-null   object 
     6   Units              4046 non-null   int64  
     7   Balcony            3871 non-null   float64
     8   Walk_In_Closet     3871 non-null   float64
     9   Fireplace          3871 non-null   float64
     10  City_Skyline       3871 non-null   float64
     11  Kitchen_Island     3871 non-null   float64
     12  Renovated          3871 non-null   float64
     13  Office_Space       3871 non-null   float64
     14  Estimated_Vacancy  4046 non-null   float64
     15  building_unit_id   4045 non-null   object 
    dtypes: float64(9), int64(5), object(2)
    memory usage: 537.4+ KB
    None





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
      <th>Beds</th>
      <th>Baths</th>
      <th>sq.ft</th>
      <th>Floor</th>
      <th>City</th>
      <th>Units</th>
      <th>Balcony</th>
      <th>Walk_In_Closet</th>
      <th>Fireplace</th>
      <th>City_Skyline</th>
      <th>Kitchen_Island</th>
      <th>Renovated</th>
      <th>Office_Space</th>
      <th>Estimated_Vacancy</th>
      <th>building_unit_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12091</td>
      <td>3</td>
      <td>3.0</td>
      <td>1797</td>
      <td>32</td>
      <td>New York City</td>
      <td>455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.057143</td>
      <td>16032A \r\n</td>
    </tr>
    <tr>
      <th>8</th>
      <td>11964</td>
      <td>3</td>
      <td>2.0</td>
      <td>1917</td>
      <td>34</td>
      <td>New York City</td>
      <td>455</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.074725</td>
      <td>16034D \r\n</td>
    </tr>
    <tr>
      <th>13</th>
      <td>10760</td>
      <td>2</td>
      <td>2.0</td>
      <td>1115</td>
      <td>21</td>
      <td>New York City</td>
      <td>269</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.022305</td>
      <td>121G \r\n</td>
    </tr>
    <tr>
      <th>31</th>
      <td>9890</td>
      <td>2</td>
      <td>2.0</td>
      <td>1463</td>
      <td>17</td>
      <td>New York City</td>
      <td>269</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.022305</td>
      <td>117D \r\n</td>
    </tr>
    <tr>
      <th>35</th>
      <td>9828</td>
      <td>3</td>
      <td>2.5</td>
      <td>1403</td>
      <td>5</td>
      <td>New York City</td>
      <td>163</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.030675</td>
      <td>105H \r\n</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Price'].plot(kind='box')
```




    <Axes: >




    
![png](output_7_1.png)
    



```python
df['Beds'].plot(kind='box')
```




    <Axes: >




    
![png](output_8_1.png)
    



```python
sns.boxplot(x=df['Beds'], y=df['Price'])
plt.show()
```


    
![png](output_9_0.png)
    



```python
df['Baths'].plot(kind='box')
```




    <Axes: >




    
![png](output_10_1.png)
    



```python
sns.boxplot(x=df['Baths'], y=df['Price'])
plt.show()
```


    
![png](output_11_0.png)
    



```python
df['Floor'].plot(kind='hist')
plt.xlabel("Floor")
```




    Text(0.5, 0, 'Floor')




    
![png](output_12_1.png)
    



```python
g = sns.boxplot(x=df['City'], y=df['sq.ft'])
sns.set(rc={'figure.figsize':(20, 6)})
sns.set(font_scale=1.5)
fmy='0.5g'
plt.show()
```


    
![png](output_13_0.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 5))
ax.scatter(df["Floor"], df["Price"])
ax.set_xlabel("Floor")
ax.set_ylabel("Price")
sns.set(font_scale=1.5)
fmy='0.3g'
sns.regplot(x=df['Floor'], y=df['Price'], ci=False, line_kws={'color':'red'})
plt.show()
```


    
![png](output_14_0.png)
    



```python
fig, ax = plt.subplots(figsize=(15, 5))
ax.scatter(df["Price"], df["Baths"])
ax.set_xlabel("Price")
ax.set_ylabel("Baths")
sns.set(font_scale=1.5)
fmy='0.3g'
plt.show()

# Same as above; I'd like to show the price distribution (using a boxplot) for each number of baths
```


    
![png](output_15_0.png)
    



```python
fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(df["Price"], df["sq.ft"])
ax.set_xlabel("Price")
ax.set_ylabel("sq.ft")

sns.regplot(x=df['Price'], y=df['sq.ft'], ci=False, line_kws={'color':'red'})
sns.set(font_scale=1.2)
fmy='0.3g'

ax.annotate("r-squared = {0.3f}").format(r2_score(df['Price'], df['sq.ft']))
plt.show()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[34], line 10
          7 sns.set(font_scale=1.2)
          8 fmy='0.3g'
    ---> 10 ax.annotate("r-squared = {0.3f}").format(r2_score(df['Price'], df['sq.ft']))
         11 plt.show()


    TypeError: Axes.annotate() missing 1 required positional argument: 'xy'



    
![png](output_16_1.png)
    



```python
q1 = df.quantile(q=0.25, axis=0)
q3 = df.quantile(q=0.75, axis=0)
iqr = q3 - q1
print(iqr)
```

    Price                1244.000000
    Beds                    1.000000
    Baths                   1.000000
    sq.ft                 389.000000
    Floor                   5.000000
    Units                 209.000000
    Balcony                 0.000000
    Walk_In_Closet          0.000000
    Fireplace               0.000000
    City_Skyline            0.000000
    Kitchen_Island          0.000000
    Renovated               0.000000
    Office_Space            0.000000
    Estimated_Vacancy       0.033497
    dtype: float64


    /var/folders/_g/6hsv1grs35ndmt6n2_kk44480000gn/T/ipykernel_22519/469988821.py:1: FutureWarning: The default value of numeric_only in DataFrame.quantile is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      q1 = df.quantile(q=0.25, axis=0)
    /var/folders/_g/6hsv1grs35ndmt6n2_kk44480000gn/T/ipykernel_22519/469988821.py:2: FutureWarning: The default value of numeric_only in DataFrame.quantile is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      q3 = df.quantile(q=0.75, axis=0)



```python
lower_limit = q1 - (1.5 * iqr)
upper_limit = q3 + (1.5 * iqr)

outliers = df[((df < lower_limit) | (df > upper_limit)).any(axis=1)]

print(outliers.info())

print(df.info(), df.describe())

# There are roughly 700 outliers in each column. I will not remove the outliers here, since I believe them to be 
# true outliers and want to present an accurate depiction of the data.
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2472 entries, 0 to 62808
    Data columns (total 16 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Price              2472 non-null   int64  
     1   Beds               2472 non-null   int64  
     2   Baths              2472 non-null   float64
     3   sq.ft              2472 non-null   int64  
     4   Floor              2472 non-null   int64  
     5   City               2472 non-null   object 
     6   Units              2472 non-null   int64  
     7   Balcony            2458 non-null   float64
     8   Walk_In_Closet     2458 non-null   float64
     9   Fireplace          2458 non-null   float64
     10  City_Skyline       2458 non-null   float64
     11  Kitchen_Island     2458 non-null   float64
     12  Renovated          2458 non-null   float64
     13  Office_Space       2458 non-null   float64
     14  Estimated_Vacancy  2472 non-null   float64
     15  building_unit_id   2471 non-null   object 
    dtypes: float64(9), int64(5), object(2)
    memory usage: 328.3+ KB
    None
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4046 entries, 0 to 62808
    Data columns (total 16 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   Price              4046 non-null   int64  
     1   Beds               4046 non-null   int64  
     2   Baths              4046 non-null   float64
     3   sq.ft              4046 non-null   int64  
     4   Floor              4046 non-null   int64  
     5   City               4046 non-null   object 
     6   Units              4046 non-null   int64  
     7   Balcony            3871 non-null   float64
     8   Walk_In_Closet     3871 non-null   float64
     9   Fireplace          3871 non-null   float64
     10  City_Skyline       3871 non-null   float64
     11  Kitchen_Island     3871 non-null   float64
     12  Renovated          3871 non-null   float64
     13  Office_Space       3871 non-null   float64
     14  Estimated_Vacancy  4046 non-null   float64
     15  building_unit_id   4045 non-null   object 
    dtypes: float64(9), int64(5), object(2)
    memory usage: 537.4+ KB
    None               Price         Beds        Baths        sq.ft        Floor  \
    count   4046.000000  4046.000000  4046.000000  4046.000000  4046.000000   
    mean    3127.308947     1.352447     1.394464   865.478744     6.241226   
    std     1067.242297     0.720083     0.496822   252.468369     7.594751   
    min        0.000000     0.000000     1.000000   210.000000     1.000000   
    25%     2370.000000     1.000000     1.000000   676.000000     2.000000   
    50%     2920.000000     1.000000     1.000000   812.000000     3.000000   
    75%     3614.000000     2.000000     2.000000  1065.000000     7.000000   
    max    12091.000000     3.000000     3.000000  1917.000000   100.000000   
    
                 Units      Balcony  Walk_In_Closet    Fireplace  City_Skyline  \
    count  4046.000000  3871.000000     3871.000000  3871.000000   3871.000000   
    mean    352.304004     0.226556        0.125291     0.094808      0.019892   
    std     149.560163     0.418657        0.331091     0.292987      0.139646   
    min      18.000000     0.000000        0.000000     0.000000      0.000000   
    25%     241.000000     0.000000        0.000000     0.000000      0.000000   
    50%     322.000000     0.000000        0.000000     0.000000      0.000000   
    75%     450.000000     0.000000        0.000000     0.000000      0.000000   
    max     761.000000     1.000000        1.000000     1.000000      1.000000   
    
           Kitchen_Island    Renovated  Office_Space  Estimated_Vacancy  
    count     3871.000000  3871.000000   3871.000000        4046.000000  
    mean         0.039525     0.204598      0.020666           0.075054  
    std          0.194865     0.403460      0.142284           0.102248  
    min          0.000000     0.000000      0.000000           0.002252  
    25%          0.000000     0.000000      0.000000           0.037092  
    50%          0.000000     0.000000      0.000000           0.054422  
    75%          0.000000     0.000000      0.000000           0.070588  
    max          1.000000     1.000000      1.000000           0.675325  


    /var/folders/_g/6hsv1grs35ndmt6n2_kk44480000gn/T/ipykernel_22519/4174770730.py:4: FutureWarning: Automatic reindexing on DataFrame vs Series comparisons is deprecated and will raise ValueError in a future version. Do `left, right = left.align(right, axis=1, copy=False)` before e.g. `left == right`
      outliers = df[((df < lower_limit) | (df > upper_limit)).any(axis=1)]



```python
df.drop(df[df['Price'] == 0].index, inplace = True)

df.drop(df[df['Floor'] == 100].index, inplace = True)

print(df.head(4000).sort_values('Floor', ascending=False))
```

           Price  Beds  Baths  sq.ft  Floor           City  Units  Balcony  \
    983     6042     1    1.0    620     49  New York City    302      NaN   
    29560   2869     0    1.0    404     49  New York City    491      0.0   
    5191    4450     2    2.0   1005     48  New York City    480      0.0   
    6032    4320     2    2.0    965     48  New York City    480      0.0   
    68      9230     2    2.0   1113     47  New York City    302      0.0   
    ...      ...   ...    ...    ...    ...            ...    ...      ...   
    26251   2977     2    2.0   1050      1  Orange County    344      0.0   
    26246   2977     2    2.0   1151      1  Inland Empire    467      0.0   
    51687   2166     1    1.0    770      1    Los Angeles    272      0.0   
    26199   2979     2    2.0   1050      1  Orange County    344      0.0   
    23768   3060     2    2.0   1020      1  San Francisco    410      0.0   
    
           Walk_In_Closet  Fireplace  City_Skyline  Kitchen_Island  Renovated  \
    983               NaN        NaN           NaN             NaN        NaN   
    29560             0.0        0.0           0.0             0.0        0.0   
    5191              0.0        0.0           0.0             0.0        0.0   
    6032              0.0        0.0           0.0             0.0        0.0   
    68                0.0        0.0           0.0             0.0        0.0   
    ...               ...        ...           ...             ...        ...   
    26251             0.0        0.0           0.0             0.0        0.0   
    26246             0.0        0.0           0.0             1.0        1.0   
    51687             0.0        0.0           0.0             0.0        0.0   
    26199             0.0        0.0           0.0             0.0        0.0   
    23768             0.0        0.0           0.0             0.0        0.0   
    
           Office_Space  Estimated_Vacancy building_unit_id  
    983             NaN           0.029801        149I \r\n  
    29560           0.0           0.048880        149B \r\n  
    5191            0.0           0.062500            14804  
    6032            0.0           0.095833            14805  
    68              0.0           0.029801        147D \r\n  
    ...             ...                ...              ...  
    26251           0.0           0.020349         3B \r\ne  
    26246           0.0           0.038544            44111  
    51687           0.0           0.047794            E112E  
    26199           0.0           0.014535        21C \r\ne  
    23768           0.0           0.104878         333 \r\n  
    
    [4000 rows x 16 columns]



```python
df.drop(df[df['Price'] == 0].index, inplace = True)
df_corr = df.corr()

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(df_corr, annot=True, cmap='Greens')
sns.set(font_scale=1.0)
fmt='.5g'

# The four factors having the greatest influence on price (with the exception of factors not included in the analysis,
# such as location, competition, state of the market, etc.) are # of beds, # of baths, square footage, and floor
# location. Square footage has the largest effect on price, with an r value of 0.49 (a medium effect).
```

    /var/folders/_g/6hsv1grs35ndmt6n2_kk44480000gn/T/ipykernel_22519/1555599166.py:2: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      df_corr = df.corr()



    
![png](output_20_1.png)
    



```python
print(df_corr)
```

                          Price      Beds     Baths     sq.ft     Floor     Units  \
    Price              1.000000  0.424900  0.435701  0.491176  0.420901 -0.155281   
    Beds               0.424900  1.000000  0.819859  0.860964 -0.169664 -0.080715   
    Baths              0.435701  0.819859  1.000000  0.815526 -0.074590 -0.080876   
    sq.ft              0.491176  0.860964  0.815526  1.000000 -0.065429  0.006199   
    Floor              0.420901 -0.169664 -0.074590 -0.065429  1.000000  0.088855   
    Units             -0.155281 -0.080715 -0.080876  0.006199  0.088855  1.000000   
    Balcony           -0.017465  0.039516  0.034032  0.093094 -0.057590  0.124342   
    Walk_In_Closet     0.185860  0.035739  0.037584  0.063507  0.080772  0.019018   
    Fireplace         -0.087587  0.164055  0.140412  0.129982 -0.150430  0.036644   
    City_Skyline       0.016151  0.000304  0.019975 -0.006234  0.052254 -0.020139   
    Kitchen_Island    -0.008293  0.038169  0.032369  0.045427 -0.038473  0.039569   
    Renovated         -0.100206  0.037780  0.004853 -0.031841 -0.124881  0.046269   
    Office_Space       0.042194  0.030534  0.048201  0.141523 -0.035606  0.016735   
    Estimated_Vacancy  0.230773  0.095423  0.091584  0.129670  0.186678 -0.213801   
    
                        Balcony  Walk_In_Closet  Fireplace  City_Skyline  \
    Price             -0.017465        0.185860  -0.087587      0.016151   
    Beds               0.039516        0.035739   0.164055      0.000304   
    Baths              0.034032        0.037584   0.140412      0.019975   
    sq.ft              0.093094        0.063507   0.129982     -0.006234   
    Floor             -0.057590        0.080772  -0.150430      0.052254   
    Units              0.124342        0.019018   0.036644     -0.020139   
    Balcony            1.000000        0.090380  -0.014953     -0.006341   
    Walk_In_Closet     0.090380        1.000000  -0.021092      0.018838   
    Fireplace         -0.014953       -0.021092   1.000000      0.010712   
    City_Skyline      -0.006341        0.018838   0.010712      1.000000   
    Kitchen_Island     0.051839        0.079615   0.033887      0.009069   
    Renovated         -0.021920       -0.005958   0.115590      0.014851   
    Office_Space       0.064608        0.027409   0.008749     -0.007701   
    Estimated_Vacancy  0.000404        0.211524  -0.057580     -0.004900   
    
                       Kitchen_Island  Renovated  Office_Space  Estimated_Vacancy  
    Price                   -0.008293  -0.100206      0.042194           0.230773  
    Beds                     0.038169   0.037780      0.030534           0.095423  
    Baths                    0.032369   0.004853      0.048201           0.091584  
    sq.ft                    0.045427  -0.031841      0.141523           0.129670  
    Floor                   -0.038473  -0.124881     -0.035606           0.186678  
    Units                    0.039569   0.046269      0.016735          -0.213801  
    Balcony                  0.051839  -0.021920      0.064608           0.000404  
    Walk_In_Closet           0.079615  -0.005958      0.027409           0.211524  
    Fireplace                0.033887   0.115590      0.008749          -0.057580  
    City_Skyline             0.009069   0.014851     -0.007701          -0.004900  
    Kitchen_Island           1.000000   0.012097      0.035755          -0.051045  
    Renovated                0.012097   1.000000     -0.051211          -0.098645  
    Office_Space             0.035755  -0.051211      1.000000          -0.030725  
    Estimated_Vacancy       -0.051045  -0.098645     -0.030725           1.000000  



```python
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.dropna(inplace=True)
```


```python
df_sorted = df_corr.sort_values('Price', ascending=False)

df_sorted['Price'].plot(kind='bar', title = "Correlated Bar Plot")
```




    <Axes: title={'center': 'Correlated Bar Plot'}>




    
![png](output_23_1.png)
    



```python
df.isna().sum()
```




    Price                0
    Beds                 0
    Baths                0
    sq.ft                0
    Floor                0
    City                 0
    Units                0
    Balcony              0
    Walk_In_Closet       0
    Fireplace            0
    City_Skyline         0
    Kitchen_Island       0
    Renovated            0
    Office_Space         0
    Estimated_Vacancy    0
    building_unit_id     0
    dtype: int64




```python
ind_var = df[['Beds', 'sq.ft', 'Floor', 'Units', 'Balcony','Walk_In_Closet',
              'Fireplace']]

dep_var = df[['Price']]

ind_var = sm.add_constant(ind_var)
regression_model = sm.OLS(dep_var, ind_var).fit()

print(regression_model.summary())


# Three variables here (Kitchen Island, Renovated, Estimated vacancy) have p-values greater
# than 0.05. We cannot surmise that these variables have a significant effect on the price of
# a unit.
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  Price   R-squared:                       0.521
    Model:                            OLS   Adj. R-squared:                  0.520
    Method:                 Least Squares   F-statistic:                     599.5
    Date:                Tue, 01 Aug 2023   Prob (F-statistic):               0.00
    Time:                        10:45:14   Log-Likelihood:                -31075.
    No. Observations:                3869   AIC:                         6.217e+04
    Df Residuals:                    3861   BIC:                         6.222e+04
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    ==================================================================================
                         coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------
    const           1377.5423     55.881     24.651      0.000    1267.983    1487.102
    Beds             246.6750     34.003      7.254      0.000     180.009     313.341
    sq.ft              1.6829      0.096     17.598      0.000       1.495       1.870
    Floor             66.4251      1.661     39.984      0.000      63.168      69.682
    Units             -1.3017      0.081    -15.978      0.000      -1.461      -1.142
    Balcony          -59.5795     29.256     -2.036      0.042    -116.938      -2.221
    Walk_In_Closet   394.0139     36.577     10.772      0.000     322.301     465.727
    Fireplace       -317.6365     41.905     -7.580      0.000    -399.794    -235.479
    ==============================================================================
    Omnibus:                     1181.759   Durbin-Watson:                   0.961
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             6238.607
    Skew:                           1.360   Prob(JB):                         0.00
    Kurtosis:                       8.594   Cond. No.                     4.66e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 4.66e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
reg_coeff = regression_model.params.sort_values(ascending=False)
reg_coeff.plot(kind='bar', title="Regressive Coefficients for Price")

```




    <Axes: title={'center': 'Regressive Coefficients for Price'}>




    
![png](output_26_1.png)
    



```python
df.to_csv('Updated_Apartments.csv')
```


```python

```


```python

```
