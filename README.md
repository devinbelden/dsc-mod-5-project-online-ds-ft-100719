
## Final Project Submission

Please fill out:
* Student name: Devin Belden
* Student pace: full time
* Scheduled project review date/time: 	Fri Feb 7, 2020 1:30pm – 2:30pm (MST)
* Instructor name: James Irving, Ph.D.
* Blog post URL: TBD

# Business Case

For this project, we attempt to use existing data to predict the winners of chess matches, given such variables as the type of opening, game length, the ratings of the players, etc. This analysis, therefore, is for those coaches and professionals whom wish to have deeper insight into the patterns and behaviors behind wins and losses, such that those behaviors might be encouraged or mitigated as needed. 

The possible endings, underneath the column `winner`, are White, Black, and Draw; due to having three target classes, random guessing leads to an overall accuracy rating of 33%. We should then pick models that have an overall accuracy rating that is higher than this, and we should be sure to take into account model runtime as well.

# Importing, Exploration, and Preprocessing

First, we import relevant packages, as well as the dataset.


```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import (train_test_split, GridSearchCV, 
                                     RandomizedSearchCV)

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (precision_score, recall_score, 
                             accuracy_score, f1_score)

from imblearn.over_sampling import ADASYN,SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier,RandomForestClassifier,
                              BaggingClassifier,GradientBoostingClassifier,
                              ExtraTreesClassifier)

from xgboost import XGBClassifier,XGBRFClassifier
from sklearn.svm import SVC
from sklearn.tree import export_graphviz
from IPython.display import Image  
from pydotplus import graph_from_dot_data

pd.options.display.float_format = '{:.2f}'.format

df = pd.read_csv('games.csv')
df.head()
```

    Using TensorFlow backend.
    




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
      <th>rated</th>
      <th>created_at</th>
      <th>last_move_at</th>
      <th>turns</th>
      <th>victory_status</th>
      <th>winner</th>
      <th>increment_code</th>
      <th>white_id</th>
      <th>white_rating</th>
      <th>black_id</th>
      <th>black_rating</th>
      <th>moves</th>
      <th>opening_eco</th>
      <th>opening_name</th>
      <th>opening_ply</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TZJHLljE</td>
      <td>False</td>
      <td>1504210000000.00</td>
      <td>1504210000000.00</td>
      <td>13</td>
      <td>outoftime</td>
      <td>white</td>
      <td>15+2</td>
      <td>bourgris</td>
      <td>1500</td>
      <td>a-00</td>
      <td>1191</td>
      <td>d4 d5 c4 c6 cxd5 e6 dxe6 fxe6 Nf3 Bb4+ Nc3 Ba5...</td>
      <td>D10</td>
      <td>Slav Defense: Exchange Variation</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>l1NXvwaE</td>
      <td>True</td>
      <td>1504130000000.00</td>
      <td>1504130000000.00</td>
      <td>16</td>
      <td>resign</td>
      <td>black</td>
      <td>5+10</td>
      <td>a-00</td>
      <td>1322</td>
      <td>skinnerua</td>
      <td>1261</td>
      <td>d4 Nc6 e4 e5 f4 f6 dxe5 fxe5 fxe5 Nxe5 Qd4 Nc6...</td>
      <td>B00</td>
      <td>Nimzowitsch Defense: Kennedy Variation</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mIICvQHh</td>
      <td>True</td>
      <td>1504130000000.00</td>
      <td>1504130000000.00</td>
      <td>61</td>
      <td>mate</td>
      <td>white</td>
      <td>5+10</td>
      <td>ischia</td>
      <td>1496</td>
      <td>a-00</td>
      <td>1500</td>
      <td>e4 e5 d3 d6 Be3 c6 Be2 b5 Nd2 a5 a4 c5 axb5 Nc...</td>
      <td>C20</td>
      <td>King's Pawn Game: Leonardis Variation</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>kWKvrqYL</td>
      <td>True</td>
      <td>1504110000000.00</td>
      <td>1504110000000.00</td>
      <td>61</td>
      <td>mate</td>
      <td>white</td>
      <td>20+0</td>
      <td>daniamurashov</td>
      <td>1439</td>
      <td>adivanov2009</td>
      <td>1454</td>
      <td>d4 d5 Nf3 Bf5 Nc3 Nf6 Bf4 Ng4 e3 Nc6 Be2 Qd7 O...</td>
      <td>D02</td>
      <td>Queen's Pawn Game: Zukertort Variation</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9tXo1AUZ</td>
      <td>True</td>
      <td>1504030000000.00</td>
      <td>1504030000000.00</td>
      <td>95</td>
      <td>mate</td>
      <td>white</td>
      <td>30+3</td>
      <td>nik221107</td>
      <td>1523</td>
      <td>adivanov2009</td>
      <td>1469</td>
      <td>e4 e5 Nf3 d6 d4 Nc6 d5 Nb4 a3 Na6 Nc3 Be7 b4 N...</td>
      <td>C41</td>
      <td>Philidor Defense</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



To reiterate (or perhaps as proof), the target column contains three unique values, and the data is whole and complete.


```python
df.winner.unique()
```




    array(['white', 'black', 'draw'], dtype=object)




```python
df.isna().sum()
```




    id                0
    rated             0
    created_at        0
    last_move_at      0
    turns             0
    victory_status    0
    winner            0
    increment_code    0
    white_id          0
    white_rating      0
    black_id          0
    black_rating      0
    moves             0
    opening_eco       0
    opening_name      0
    opening_ply       0
    dtype: int64



The range of rankings of players in this dataset is very large, going from fairly low-ranked games (~800 ranking) to very high-level games (>2700). For reference, the lowest possible ranking is 100, and the highest possible ranking is, in theory, 3000. The current highest-ranked player in the world, Magnus Carlsen, is ranked 2845. 


```python
print(df['white_rating'].min(), df['white_rating'].max())
print(df['black_rating'].min(), df['black_rating'].max())
```

    784 2700
    789 2723
    

## Dropping Features

As there is sufficient information contained in the columns regarding the opening moves, and due to the task of processing the data within `moves`, the decision was made to remove that column entirely, opting to use the opening of the match as any indication of moves taken within the game.

Additionally, there is next to no information within the `created_at` and `last_move_at` columns, as the source of the data truncated the last (and arguably most important) 7 digits of the time values. Those columns should be removed as well.

The column `victory_status` will be dropped as well, as failing to do so will introduce major bias to our analysis, e.g. "We knew the status of `winner` would be 'draw', as the entry in `victory_status` also said 'draw'". 

Finally, the `id` should be dropped, as it also offers no valuable information.


```python
df.drop(['moves','created_at','last_move_at','id','victory_status'], axis=1, inplace=True)
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
      <th>rated</th>
      <th>turns</th>
      <th>winner</th>
      <th>increment_code</th>
      <th>white_id</th>
      <th>white_rating</th>
      <th>black_id</th>
      <th>black_rating</th>
      <th>opening_eco</th>
      <th>opening_name</th>
      <th>opening_ply</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>13</td>
      <td>white</td>
      <td>15+2</td>
      <td>bourgris</td>
      <td>1500</td>
      <td>a-00</td>
      <td>1191</td>
      <td>D10</td>
      <td>Slav Defense: Exchange Variation</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>16</td>
      <td>black</td>
      <td>5+10</td>
      <td>a-00</td>
      <td>1322</td>
      <td>skinnerua</td>
      <td>1261</td>
      <td>B00</td>
      <td>Nimzowitsch Defense: Kennedy Variation</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>61</td>
      <td>white</td>
      <td>5+10</td>
      <td>ischia</td>
      <td>1496</td>
      <td>a-00</td>
      <td>1500</td>
      <td>C20</td>
      <td>King's Pawn Game: Leonardis Variation</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>61</td>
      <td>white</td>
      <td>20+0</td>
      <td>daniamurashov</td>
      <td>1439</td>
      <td>adivanov2009</td>
      <td>1454</td>
      <td>D02</td>
      <td>Queen's Pawn Game: Zukertort Variation</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>95</td>
      <td>white</td>
      <td>30+3</td>
      <td>nik221107</td>
      <td>1523</td>
      <td>adivanov2009</td>
      <td>1469</td>
      <td>C41</td>
      <td>Philidor Defense</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20058 entries, 0 to 20057
    Data columns (total 11 columns):
    rated             20058 non-null bool
    turns             20058 non-null int64
    winner            20058 non-null object
    increment_code    20058 non-null object
    white_id          20058 non-null object
    white_rating      20058 non-null int64
    black_id          20058 non-null object
    black_rating      20058 non-null int64
    opening_eco       20058 non-null object
    opening_name      20058 non-null object
    opening_ply       20058 non-null int64
    dtypes: bool(1), int64(4), object(6)
    memory usage: 1.5+ MB
    

## How Many Features Are There?

Next, we should get a feel for the dimensionality of our data, should we choose to one-hot encode it as is. 


```python
total_cols = 0
for col in ['opening_eco','opening_name','opening_ply','white_id','black_id','increment_code']:
    total_cols += df[col].nunique()

total_cols
```




    21034



One-hot encoding our categorical data as is would result in an increase in dimensionality of 21,034. Clearly we must pare this down to not only keep our interpretability high, but to keep our computational costs manageable. Additionally, given the former, let's hold off on using Principal Component Analysis until we've further explored the data.

Instead, let's try dropping a few columns that play no clear role in the classification models we'll use down the road.


```python
df.drop(['white_id','black_id','opening_name'], axis=1, inplace=True)
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
      <th>rated</th>
      <th>turns</th>
      <th>winner</th>
      <th>increment_code</th>
      <th>white_rating</th>
      <th>black_rating</th>
      <th>opening_eco</th>
      <th>opening_ply</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>13</td>
      <td>white</td>
      <td>15+2</td>
      <td>1500</td>
      <td>1191</td>
      <td>D10</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>16</td>
      <td>black</td>
      <td>5+10</td>
      <td>1322</td>
      <td>1261</td>
      <td>B00</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>61</td>
      <td>white</td>
      <td>5+10</td>
      <td>1496</td>
      <td>1500</td>
      <td>C20</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>61</td>
      <td>white</td>
      <td>20+0</td>
      <td>1439</td>
      <td>1454</td>
      <td>D02</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>95</td>
      <td>white</td>
      <td>30+3</td>
      <td>1523</td>
      <td>1469</td>
      <td>C41</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Next we'll bin our `opening_eco` column by the letter category.


```python
df['eco_category'] = df.apply(lambda row: row['opening_eco'][0], axis=1)
df.drop('opening_eco', axis=1, inplace=True)
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
      <th>rated</th>
      <th>turns</th>
      <th>winner</th>
      <th>increment_code</th>
      <th>white_rating</th>
      <th>black_rating</th>
      <th>opening_ply</th>
      <th>eco_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>13</td>
      <td>white</td>
      <td>15+2</td>
      <td>1500</td>
      <td>1191</td>
      <td>5</td>
      <td>D</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>16</td>
      <td>black</td>
      <td>5+10</td>
      <td>1322</td>
      <td>1261</td>
      <td>4</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>61</td>
      <td>white</td>
      <td>5+10</td>
      <td>1496</td>
      <td>1500</td>
      <td>3</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>61</td>
      <td>white</td>
      <td>20+0</td>
      <td>1439</td>
      <td>1454</td>
      <td>3</td>
      <td>D</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>95</td>
      <td>white</td>
      <td>30+3</td>
      <td>1523</td>
      <td>1469</td>
      <td>5</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



While we're at it, we should map our target column to numerical values. We'll do this by order of decreasing appearance within the dataset.


```python
df['winner'].value_counts(normalize=True)
```




    white   0.50
    black   0.45
    draw    0.05
    Name: winner, dtype: float64




```python
winner_map = {
    "white": 0,
    "black": 1,
    "draw": 2
    }

df['winner'] = df['winner'].map(winner_map)
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
      <th>rated</th>
      <th>turns</th>
      <th>winner</th>
      <th>increment_code</th>
      <th>white_rating</th>
      <th>black_rating</th>
      <th>opening_ply</th>
      <th>eco_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>13</td>
      <td>0</td>
      <td>15+2</td>
      <td>1500</td>
      <td>1191</td>
      <td>5</td>
      <td>D</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>16</td>
      <td>1</td>
      <td>5+10</td>
      <td>1322</td>
      <td>1261</td>
      <td>4</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>61</td>
      <td>0</td>
      <td>5+10</td>
      <td>1496</td>
      <td>1500</td>
      <td>3</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>61</td>
      <td>0</td>
      <td>20+0</td>
      <td>1439</td>
      <td>1454</td>
      <td>3</td>
      <td>D</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>95</td>
      <td>0</td>
      <td>30+3</td>
      <td>1523</td>
      <td>1469</td>
      <td>5</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>



## Class Imbalance

While we're here, let's take a moment to understand our class imbalance issue. Random guessing, as stated before, gives an overall accuracy of 33%. "Weighted guessing"--that is, putting 50% of our guesses as White, 45% of our guesses as Black, and 5% of our guesses as Draw--gives an overall accuracy of 45.5%. Further still, simply guessing every game as ending in White victory would net us 50%. We should then instead look for models that surpass this accuracy. 

## More Data Cleanup and One-Hot Encoding

Let's take care of the rest of the categorical column preprocessing before we one-hot encode them. We'll change `rated` from boolean to binary, and  change `increment_code` to display the starting time on the player clocks. After that, we'll be ready to encode our columns.


```python
rated_map = {False: 0, True: 1}

df['rated'] = df['rated'].map(rated_map)
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
      <th>rated</th>
      <th>turns</th>
      <th>winner</th>
      <th>increment_code</th>
      <th>white_rating</th>
      <th>black_rating</th>
      <th>opening_ply</th>
      <th>eco_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>15+2</td>
      <td>1500</td>
      <td>1191</td>
      <td>5</td>
      <td>D</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>5+10</td>
      <td>1322</td>
      <td>1261</td>
      <td>4</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>61</td>
      <td>0</td>
      <td>5+10</td>
      <td>1496</td>
      <td>1500</td>
      <td>3</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>61</td>
      <td>0</td>
      <td>20+0</td>
      <td>1439</td>
      <td>1454</td>
      <td>3</td>
      <td>D</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>95</td>
      <td>0</td>
      <td>30+3</td>
      <td>1523</td>
      <td>1469</td>
      <td>5</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['game_time'] = df.apply(lambda row: row['increment_code'].split('+')[0], axis=1)
df['game_time'] = df['game_time'].astype('int64')
df.drop('increment_code', axis=1, inplace=True)
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
      <th>rated</th>
      <th>turns</th>
      <th>winner</th>
      <th>white_rating</th>
      <th>black_rating</th>
      <th>opening_ply</th>
      <th>eco_category</th>
      <th>game_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>1500</td>
      <td>1191</td>
      <td>5</td>
      <td>D</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>1322</td>
      <td>1261</td>
      <td>4</td>
      <td>B</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>61</td>
      <td>0</td>
      <td>1496</td>
      <td>1500</td>
      <td>3</td>
      <td>C</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>61</td>
      <td>0</td>
      <td>1439</td>
      <td>1454</td>
      <td>3</td>
      <td>D</td>
      <td>20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>95</td>
      <td>0</td>
      <td>1523</td>
      <td>1469</td>
      <td>5</td>
      <td>C</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20058 entries, 0 to 20057
    Data columns (total 8 columns):
    rated           20058 non-null int64
    turns           20058 non-null int64
    winner          20058 non-null int64
    white_rating    20058 non-null int64
    black_rating    20058 non-null int64
    opening_ply     20058 non-null int64
    eco_category    20058 non-null object
    game_time       20058 non-null int64
    dtypes: int64(7), object(1)
    memory usage: 1.2+ MB
    


```python
df = pd.get_dummies(df, columns=['eco_category'])

df
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
      <th>rated</th>
      <th>turns</th>
      <th>winner</th>
      <th>white_rating</th>
      <th>black_rating</th>
      <th>opening_ply</th>
      <th>game_time</th>
      <th>eco_category_A</th>
      <th>eco_category_B</th>
      <th>eco_category_C</th>
      <th>eco_category_D</th>
      <th>eco_category_E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>1500</td>
      <td>1191</td>
      <td>5</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>16</td>
      <td>1</td>
      <td>1322</td>
      <td>1261</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>61</td>
      <td>0</td>
      <td>1496</td>
      <td>1500</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>61</td>
      <td>0</td>
      <td>1439</td>
      <td>1454</td>
      <td>3</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>95</td>
      <td>0</td>
      <td>1523</td>
      <td>1469</td>
      <td>5</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20053</th>
      <td>1</td>
      <td>24</td>
      <td>0</td>
      <td>1691</td>
      <td>1220</td>
      <td>2</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20054</th>
      <td>1</td>
      <td>82</td>
      <td>1</td>
      <td>1233</td>
      <td>1196</td>
      <td>2</td>
      <td>10</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20055</th>
      <td>1</td>
      <td>35</td>
      <td>0</td>
      <td>1219</td>
      <td>1286</td>
      <td>3</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20056</th>
      <td>1</td>
      <td>109</td>
      <td>0</td>
      <td>1360</td>
      <td>1227</td>
      <td>4</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20057</th>
      <td>1</td>
      <td>78</td>
      <td>1</td>
      <td>1235</td>
      <td>1339</td>
      <td>3</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>20058 rows × 12 columns</p>
</div>



# Preliminary Modeling

Now that we've got our data largely in the form we want, we can begin our modeling. We'll first define some functions that will help us train and evaluate the models.


```python
y = df['winner']
X = df.drop('winner', axis=1)

def train_test(df=df, drop_cols=['winner']):
    
    """Takes in a DataFrame and any columns to drop from training
    data as inputs. Returns a train-test split of that data."""
    
    y = df['winner']
    X = df.drop(drop_cols, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    return X_train, X_test, y_train, y_test


def scale(df=df, drop_cols=['winner']):
    
    """Takes in a DataFrame and any columns to drop from training data
    as inputs. Returns a Standard Scaled train-test split."""
    
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test(df=df, drop_cols=drop_cols)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) 
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def smote(df=df, scaled=True, verbose=False, drop_cols=['winner']):
    
    """Takes in a DataFrame and any columns to drop from training data
    as inputs. Returns an oversampled train-test split. Optionally, will
    Standard Scale the data as well."""
    
    smote=SMOTE()
    
    if scaled:
        X_train, X_test, y_train, y_test = scale(df=df, drop_cols=drop_cols)
        X_train, y_train = smote.fit_sample(X_train, y_train)
        
    else:
        X_train, X_test, y_train, y_test = train_test(df=df, drop_cols=drop_cols)
        X_train, y_train = smote.fit_sample(X_train, y_train)
        
    if verbose:
        print(pd.Series(y_train).value_counts())
        print(pd.Series(y_test).value_counts())
        
    return X_train, X_test, y_train, y_test
    
    
def print_metrics(labels, preds):
    
    """Takes test labels and predicted labels as inputs.
    Returns precision, recall, accuracy, and f1 scores."""
    
    print(f"Precision Score: {precision_score(labels, preds, average=None)}")
    print(f"Recall Score: {recall_score(labels, preds, average=None)}")
    print(f"Accuracy Score: {accuracy_score(labels, preds)}")
    print(f"F1 Score: {f1_score(labels, preds, average=None)}")
    
    
def plot_importance(model,top_n=20,figsize=(10,10), drop_cols=['winner']):
    
    """Returns a plot of features ranked by their importance, listed from
    most to least important."""
    
    df_importance = pd.Series(model.feature_importances_,
                              index=df.drop(drop_cols, axis=1).columns)
    df_importance.sort_values(ascending=True).tail(top_n).plot(
        kind='barh',figsize=figsize)
    return df_importance
```

## Vanilla KNN


```python
X_train, X_test, y_train, y_test = scale()

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

test_preds = clf.predict(X_test)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
```

    0.7174100910722595
    0.5794616151545364
    


```python
import seaborn as sns



plot_confusion_matrix(clf, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd');
```


![png](output_32_0.png)


### KNN With SMOTE


```python
X_train, X_test, y_train, y_test = smote(scaled=True, verbose=True)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

test_preds = clf.predict(X_test)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
```

    2    7472
    1    7472
    0    7472
    dtype: int64
    0    2529
    1    2233
    2     253
    Name: winner, dtype: int64
    0.7863579586009993
    0.5136590229312064
    


```python
plot_confusion_matrix(clf, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a39632a898>




![png](output_35_1.png)


With SMOTE, we've gained a much higher degree of homogeneity at the cost of overall accuracy and overfitting. Let's try a different model.

## Vanilla Decision Tree


```python
tree = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = smote(scaled=True, verbose=True)

tree.fit(X_train, y_train)

test_preds = tree.predict(X_test)

print(tree.score(X_train, y_train))
print(tree.score(X_test, y_test))
```

    2    7514
    1    7514
    0    7514
    dtype: int64
    0    2487
    1    2281
    2     247
    Name: winner, dtype: int64
    1.0
    0.5509471585244268
    


```python
plot_confusion_matrix(tree, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a3965750b8>




![png](output_39_1.png)



```python
plot_importance(tree.fit(X_train, y_train));
```


![png](output_40_0.png)


### Low Depth Decision Tree


```python
tree = DecisionTreeClassifier(max_depth=10)
X_train, X_test, y_train, y_test = smote(scaled=True, verbose=True)

tree.fit(X_train, y_train)

test_preds = tree.predict(X_test)

print(tree.score(X_train, y_train))
print(tree.score(X_test, y_test))

plot_confusion_matrix(tree, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```

    2    7515
    1    7515
    0    7515
    dtype: int64
    0    2486
    1    2274
    2     255
    Name: winner, dtype: int64
    0.6703925482368596
    0.5308075772681954
    




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a3969b5080>




![png](output_42_2.png)



```python
plot_importance(tree.fit(X_train, y_train));
```


![png](output_43_0.png)


## Vanilla Bagging


```python
X_train, X_test, y_train, y_test = smote(scaled=True, verbose=True)

bag = BaggingClassifier(n_estimators=100)
bag.fit(X_train, y_train)
print(bag.score(X_train, y_train))
print(bag.score(X_test, y_test))
```

    2    7538
    1    7538
    0    7538
    dtype: int64
    0    2463
    1    2315
    2     237
    Name: winner, dtype: int64
    1.0
    0.6017946161515454
    


```python
plot_confusion_matrix(bag, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a39696bb00>




![png](output_46_1.png)


## Vanilla Random Forest


```python
# bootstrap aggregation is an improvement over KNN. Let's try RF
X_train, X_test, y_train, y_test = smote(scaled=False, verbose=True)

rf = RandomForestClassifier()

rf.fit(X_train, y_train)
print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))
```

    2    7468
    1    7468
    0    7468
    dtype: int64
    0    2533
    1    2251
    2     231
    Name: winner, dtype: int64
    1.0
    0.6406779661016949
    


```python
plot_confusion_matrix(rf, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a3969ff438>




![png](output_49_1.png)



```python
plot_importance(rf.fit(X_train, y_train));
```


![png](output_50_0.png)


### Low Depth Random Forest


```python
rf = RandomForestClassifier(max_depth=10)

rf.fit(X_train, y_train)
print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))
```

    0.783252990537404
    0.6161515453639083
    


```python
plot_confusion_matrix(rf, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a396ea58d0>




![png](output_53_1.png)



```python
plot_importance(rf.fit(X_train, y_train));
```


![png](output_54_0.png)


## Low Depth Extra Trees


```python
X_train, X_test, y_train, y_test = smote(scaled=False, verbose=True)

et = ExtraTreesClassifier(n_estimators=100, max_depth=10)
et.fit(X_train, y_train)

print(et.score(X_train, y_train))
print(et.score(X_test, y_test))
```

    2    7525
    1    7525
    0    7525
    dtype: int64
    0    2476
    1    2297
    2     242
    Name: winner, dtype: int64
    0.7035215946843854
    0.6001994017946162
    


```python
plot_confusion_matrix(et, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a396f52da0>




![png](output_57_1.png)



```python
plot_importance(et.fit(X_train, y_train));
```


![png](output_58_0.png)


## Support Vector Machines


```python
X_train, X_test, y_train, y_test = smote(scaled=True, verbose=True)

clf = SVC(gamma='auto', C=10)
clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

plot_confusion_matrix(clf, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```

    2    7532
    1    7532
    0    7532
    dtype: int64
    0    2469
    1    2289
    2     257
    Name: winner, dtype: int64
    0.6234289254735351
    0.5371884346959123
    




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a3968724a8>




![png](output_60_2.png)



```python
kernels = ['linear','poly','sigmoid']

for kernel in kernels:

    clf = SVC(gamma='auto', kernel=kernel)
    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))
    plot_confusion_matrix(clf, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```

    0.5500973623650204
    0.5234297108673978
    0.5755885997521685
    0.5274177467597209
    0.3971941936625952
    0.39740777666999005
    


![png](output_61_1.png)



![png](output_61_2.png)



![png](output_61_3.png)


# Ensemble Methods

## AdaBoost With Low Depth Random Forest


```python
X_train, X_test, y_train, y_test = smote(scaled=True, verbose=True)

ada = AdaBoostClassifier(RandomForestClassifier(max_depth=10), learning_rate=0.1)
ada.fit(X_train, y_train)
print(ada.score(X_train, y_train))
print(ada.score(X_test, y_test))

plot_confusion_matrix(ada, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```

    2    7547
    1    7547
    0    7547
    dtype: int64
    0    2454
    1    2321
    2     240
    Name: winner, dtype: int64
    0.8938209443045801
    0.6568295114656032
    




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a3979d5e80>




![png](output_64_2.png)


## Max Depth Gradient Boost


```python
X_train, X_test, y_train, y_test = smote(scaled=True, verbose=True)

grad = GradientBoostingClassifier(max_depth=None)
grad.fit(X_train, y_train)
print(grad.score(X_train, y_train))
print(grad.score(X_test, y_test))

plot_confusion_matrix(grad, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```

    2    7457
    1    7457
    0    7457
    dtype: int64
    0    2544
    1    2223
    2     248
    Name: winner, dtype: int64
    1.0
    0.6536390827517448
    




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a397d40cc0>




![png](output_66_2.png)


### Low Depth Gradient Boost


```python
X_train, X_test, y_train, y_test = smote(scaled=True, verbose=True)

grad = GradientBoostingClassifier(max_depth=10)
grad.fit(X_train, y_train)
print(grad.score(X_train, y_train))
print(grad.score(X_test, y_test))

plot_confusion_matrix(grad, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```

    2    7481
    1    7481
    0    7481
    dtype: int64
    0    2520
    1    2267
    2     228
    Name: winner, dtype: int64
    0.9860090005792452
    0.8103688933200399
    




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a397d660f0>




![png](output_68_2.png)


### Low Depth Gradient Boost with Gridsearch

So far, the Gradient Boosting algorithm, while still quite unable to accurately predict 'Draw' endings, has produced the highest degree of overall test accuracy, albeit somewhat overfitting to the training data. After looking at more ensemble methods, it would be worth pursuing hyperparameter tuning for this model, in order to either increase the generalizability, or to increase test accuracy for 'Draw' outcomes. In discussing which parameters to tune, and which value choices to give the gridsearch algorithm, it is worth mentioning that, as the dataset contains over 20,000 rows, the samples required to create a new split or leaf in our gradient boosted random forest should be increased. Additionally, the maximum tree depth should be limited so as to prevent our previous case of overfitting. Finally, the learning rate will be tweaked in the aim of attaining a higher degree of granularity in the gradient descent.


```python
X_train, X_test, y_train, y_test = smote(scaled=True, verbose=True)

go_ahead = input("Cell will take several minutes to run. Do you wish to run this cell (y/n)? ")

if go_ahead == 'y':

    grad = GradientBoostingClassifier()
    grid = {'learning_rate': [0.1,0.01],
            'max_depth': [3,5,10],
            'min_samples_split':[20,30,50],
            'min_samples_leaf':[10,30]}

    gridsearch = GridSearchCV(grad, param_grid=grid, cv=5)

    grad_cv = gridsearch.fit(X_train, y_train)
    
    best_params = grad_cv.best_params_

    print(best_params)
    
else:
    
    best_params = {'learning_rate': 0.1, 
                   'max_depth': 10, 
                   'min_samples_leaf': 10, 
                   'min_samples_split': 20}
    
    print(best_params)
```

    2    7450
    1    7450
    0    7450
    dtype: int64
    0    2551
    1    2239
    2     225
    Name: winner, dtype: int64
    Cell will take several minutes to run. Do you wish to run this cell (y/n)? n
    {'learning_rate': 0.1, 'max_depth': 10, 'min_samples_leaf': 10, 'min_samples_split': 20}
    


```python
grad = GradientBoostingClassifier(**best_params)
grad.fit(X_train, y_train)
print(grad.score(X_train, y_train))
print(grad.score(X_test, y_test))
```

    0.959910514541387
    0.8111665004985045
    


```python
sns.set_style(style='ticks')
plot_confusion_matrix(grad, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a39689d6d8>




![png](output_72_1.png)



```python
sns.set_style(style='darkgrid')
plot_importance(grad.fit(X_train, y_train))
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
plt.xticks(ticks=[0,.05,.1,.15,.2,.25,.3,.35,.4], 
           labels=['0%','5%','10%','15%','20%','25%','30%','35%','40%'])

plt.yticks(ticks=range(11), 
           labels=['Opening Type E','Opening Type D','Opening Type A','Opening Type C',
                   'Opening Type B','Ranked','Opening Length','Black Rating','White Rating',
                  'Game Time','Turn Count'])
plt.show()
```


![png](output_73_0.png)


A gridsearch of hyperparameters has resulted in an accuracy increase of less than 1%. A bit underwhelming, especially given the time it took the gridsearch to run, but the level of overfit has gone down somewhat. Regardless, let's try a few other prediction methods.


```python
plt.figure(figsize=(10,10))
sns.barplot(y=df['turns'], x=df['winner'])
sns.set_context('poster')
plt.xlabel("Winner")
plt.xticks(ticks=range(3), labels=['White','Black','Draw'], rotation=45)
plt.ylabel("Average Turn Count Per Game")
plt.title("Average Turn Count Per Outcome")
```




    Text(0.5, 1.0, 'Average Turn Count Per Outcome')




![png](output_75_1.png)



```python
plt.figure(figsize=(10,10))
sns.barplot(y=df['game_time'], x=df['winner'])
sns.set_context('poster')
plt.xlabel("Winner")
plt.xticks(ticks=range(3), labels=['White','Black','Draw'], rotation=45)
plt.ylabel("Average Starting Time Per Game")
plt.title("Average Starting Game Time Per Outcome")
```




    Text(0.5, 1.0, 'Average Starting Game Time Per Outcome')




![png](output_76_1.png)


## Vanilla XGBoost


```python
X_train, X_test, y_train, y_test = smote(scaled=True, verbose=True)

xgb_rf = XGBRFClassifier()
xgb_rf.fit(X_train, y_train)
print(xgb_rf.score(X_train, y_train))
print(xgb_rf.score(X_test,y_test))
```

    2    7521
    1    7521
    0    7521
    dtype: int64
    0    2480
    1    2276
    2     259
    Name: winner, dtype: int64
    0.5661924389487214
    0.5226321036889332
    


```python

plot_confusion_matrix(xgb_rf, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a3979d5c50>




![png](output_79_1.png)


### High Depth XGBoost

While a max depth of 10 is low for a regular decision tree, it is rather high for an XGBoost algorithm. We'll use that depth here.


```python
X_train, X_test, y_train, y_test = smote(scaled=True, verbose=True)

xgb_rf = XGBRFClassifier(max_depth=10)
xgb_rf.fit(X_train, y_train)
print(xgb_rf.score(X_train, y_train))
print(xgb_rf.score(X_test,y_test))
```

    2    7512
    1    7512
    0    7512
    dtype: int64
    0    2489
    1    2270
    2     256
    Name: winner, dtype: int64
    0.7663294284700035
    0.6071784646061814
    


```python
plot_confusion_matrix(xgb_rf, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a398401668>




![png](output_82_1.png)



```python
plot_importance(xgb_rf.fit(X_train, y_train))
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
```




    Text(0, 0.5, 'Feature Name')




![png](output_83_1.png)


## Random Forest With Gridsearch


```python
X_train, X_test, y_train, y_test = smote(scaled=True, verbose=True)
```

    2    7494
    1    7494
    0    7494
    dtype: int64
    0    2507
    1    2281
    2     227
    Name: winner, dtype: int64
    


```python
go_ahead = input("Cell will take several minutes to run. Do you wish to run this cell (y/n)? ")

if go_ahead == 'y':

    rf_clf = RandomForestClassifier()
    grid = {'max_depth': [3,5,10,None],
            'criterion': ['gini','entropy'],
            'min_samples_split':[20,30,50],
            'min_samples_leaf':[10,30]}

    gridsearch = GridSearchCV(rf_clf, param_grid=grid, cv=5)

    forest_cv = gridsearch.fit(X_train, y_train)
    
    best_params = forest_cv.best_params_

    print(best_params)
    
else:
    
    best_params = {'criterion': 'gini', 
                   'max_depth': 10, 
                   'max_features': 3, 
                   'min_samples_leaf': 1, 
                   'min_samples_split': 2}
    
    print(best_params)
```

    Cell will take several minutes to run. Do you wish to run this cell (y/n)? n
    {'criterion': 'gini', 'max_depth': 10, 'max_features': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}
    


```python
forest = RandomForestClassifier(**best_params)
forest.fit(X_train, y_train)

print(forest.score(X_train, y_train))
print(forest.score(X_test, y_test))
```

    0.7369006316164043
    0.5744765702891326
    


```python
plot_confusion_matrix(forest, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a397943320>




![png](output_88_1.png)



```python
sns.set_context('talk')
plot_importance(forest.fit(X_train, y_train))
plt.xlabel("Feature Importance")
plt.ylabel("Feature Name")
```




    Text(0, 0.5, 'Feature Name')




![png](output_89_1.png)


### Single Tree With Gridsearch Parameters


```python
X_train, X_test, y_train, y_test = smote(scaled=True, verbose=True)

tree = DecisionTreeClassifier(**best_params)

tree.fit(X_train, y_train)

print(tree.score(X_train, y_train))
print(tree.score(X_test, y_test))
```

    2    7504
    1    7504
    0    7504
    dtype: int64
    0    2497
    1    2289
    2     229
    Name: winner, dtype: int64
    0.6094083155650319
    0.47357926221335994
    


```python
dot_data = export_graphviz(tree, out_file=None, 
                           feature_names=X.columns,  
                           class_names=y.unique().astype('str'), 
                           filled=True, rounded=True, special_characters=True,
                           rotate=True)

graph = graph_from_dot_data(dot_data)  

Image(graph.create_png())
```

    dot: graph is too large for cairo-renderer bitmaps. Scaling by 0.681907 to fit
    
    




![png](output_92_1.png)




```python
plot_confusion_matrix(tree, X_test, y_test, normalize='true', display_labels=['White','Black','Draw'], cmap='OrRd')
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x2a3c20278d0>




![png](output_93_1.png)


# Which Model is Best?

After running what seems like a dozen or more models, the next task is to choose the best one. Judging by raw accuracy, this seems to be the Gradient Boosted Random Forest; an overall accuracy of over 80% is extremely impressive, and no other model appears to even come close to this value. While its success of predicting draws is not something to write home about, this can most likely be remedied by procuring more data about draws. 

# Conclusion and Recommendations

Using a Gradient Boosted Random Forest model allows us to predict the endings of chess matches with over 80% accuracy, nearly twice as accurate as weighted guessing, and nearly 2.5 times as accurate as random guessing. Additionally, the model considers the turn count and the starting game time to account for 58% of its prediction. Draws are somewhat difficult to predict, perhaps owing to their relative rarity compared to either player winning outright, but further exploration of stalemate games might remedy this as well.

In discussing the results of the Gradient Boosted Random Forest model's predictions, two possible recommendations for players and/or coaches would be to point out that longer game times generally lead to Draws, and higher move counts generally favor Black, with the highest move counts also encouraging Draws. For example, when playing as Black, aiming to extend the game's turn count will give a statistical advantage.
