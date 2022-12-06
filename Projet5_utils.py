import numpy as np
import pandas as pd
import os
from datetime import datetime, date


def diff_lists(L1,L2):
    """Returns the elements that are in a list L1 but not in another list L2, and vice versa"""
    
    diff_12 = list(set(L1) - set(L2))  #in L1 but not in L2
    diff_21 = list(set(L2) - set(L1))  #in L2 but not in L1
    
    return diff_12, diff_21


def eta_squared_ANOVA(x,y):
    """Computes the eta squared value between a categorical variable x and a continuous variable y by ANOVA"""
    
    moyenne_y = y.mean()
    classes = []
    missing=[]
    for classe in x.dropna().unique():
        #print('CLASSE : ', classe)
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
        
    return round(SCE/SCT, 2)


def groupby_agg(df, group, column, func):
    """Groups a column 'column' from dataframe 'df' by 'group', and applies the function 'func'"""
    
    d = df.groupby(group).agg({column:func})
    df = df.drop(column, axis=1)
    df = pd.merge(df, d, on=group)
    return df



def convert_date(df):
    """ Converts a date to a number of days between said date and a reference d0, which is the latest date in the dataframe"""
    
    date_format = '%Y-%m-%d %H:%M:%S'
    d0_format = datetime.strptime(df['order_purchase_timestamp'].max(), date_format)  # Reference
    d0 = date(d0_format.year, d0_format.month, d0_format.day)
    list_days = []

    for i in df['order_purchase_timestamp']:
        d_format = datetime.strptime(i, date_format)
        d = date(d_format.year, d_format.month, d_format.day)
        delta = d0 - d
        list_days.append(delta.days)

    df = df.assign(days_since_order=list_days)
    
    return df



def transform_set(df):
    """Transforms data (imputer and scaler) to prepare for the clustering algorithms"""
    
    ### Define continuous and categorical features
    index = df.index  # to keep the customer_unique_id as the index
    list_cont = [
        'days_since_order',
        'number_of_orders',
        'average_review_score',
        'average_price',
        'number_of_items'
    ]
    list_cat = [i for i in df.columns.tolist() if i not in list_cont]
    
    ### Categorical variables
    df_cat = df[list_cat].values
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    std_cat = StandardScaler(with_mean=False)  #Scaler
    df_cat_scaled = pd.DataFrame(std_cat.fit_transform(df_cat))
    df_cat_scaled.columns = list_cat

    ### Continuous variables
    df_cont = df[list_cont].values
    
    imputer_cont = SimpleImputer(missing_values=np.nan, strategy='mean')
    std_cont = StandardScaler()

    imputer_cont.fit(df_cont)
    df_cont = imputer_cont.transform(df_cont)
    df_cont_scaled = pd.DataFrame(std_cont.fit_transform(df_cont))
    df_cont_scaled.columns = list_cont
    
    ### Aggregate    
    df = pd.concat([df_cont_scaled, df_cat_scaled], axis=1)
    df.index = index  # Set the customer_unique_id as the index
    
    return df
    
    
    
    
    
    
    
    
    

def clean_engineer_set(PATH, duration):
    """Concatenates, cleans and engineers dataframe to have the correct format for the tranformation of the data set. See the notebook on cleaning, engineering and exploration for more details, as it follows the same process (but condensed).
    
    Tranforms then data to prepare for algorithms"""
    
        
    limit_date = 700 - duration  #We consider only the 700 last days

    df_customers = pd.read_csv(os.path.join(PATH, "olist_customers_dataset.csv"))
    df_items = pd.read_csv(os.path.join(PATH, "olist_order_items_dataset.csv"))
    df_reviews = pd.read_csv(os.path.join(PATH, "olist_order_reviews_dataset.csv"))
    df_orders = pd.read_csv(os.path.join(PATH, "olist_orders_dataset.csv"))
    df_products = pd.read_csv(os.path.join(PATH, "olist_products_dataset.csv"))

        ### df_orders

    df_orders = df_orders.drop(df_orders.loc[df_orders['order_status']!='delivered'].index, axis=0) #drop non delivered orders
    df_orders = convert_date(df_orders)  # Convert date format into days passed between order and reference date

    orders_relevant_features = ['order_id', 'customer_id', 'days_since_order'] # drop irrelevant features
    df_orders = df_orders[orders_relevant_features]

    df_orders = df_orders.drop(df_orders.loc[df_orders['days_since_order']<limit_date].index, axis=0)


        ### merge with df_customers

    df1 = pd.merge(df_customers, df_orders, on='customer_id', how='right')

    df1_non_relevant_features = [
        'customer_zip_code_prefix',
        'customer_city',
        'customer_state'        
    ]
    df1 = df1.drop(df1_non_relevant_features, axis=1)

    orders_count = df1['order_id'].groupby(df1['customer_unique_id']).transform('count')  #number of orders per customer
    df1 = df1.assign(number_of_orders=orders_count)

        ### df_reviews

    df_reviews = df_reviews.drop_duplicates(subset=['order_id'], keep='last') # drop duplicated reviews (on same order)

    reviews_relevant_features = ['order_id', 'review_score'] # drop irrelevant features
    df_reviews = df_reviews[reviews_relevant_features]

        ### merge with df1

    df2 = pd.merge(df1, df_reviews, on='order_id', how='left')

    average_review = df2.groupby('customer_unique_id').review_score.agg(average_review_score = 'mean') # compute average reviews
    df2 = pd.merge(df2, average_review, on='customer_unique_id')
    df2 = df2.drop('review_score', axis=1)

        ### df_items and df_products

    df_products = df_products[['product_id', 'product_category_name']] # keep only relevant variables
    df_items = pd.merge(df_items, df_products, on='product_id', how='left') # add product category to df_items

    items_relevant_features = ['order_id', 'price', 'product_category_name'] # delete irrelevant features in df_items
    df_items = df_items[items_relevant_features]

    average_price = df_items.groupby('order_id').price.agg(average_price='mean') # compute average price per order
    df_items = pd.merge(df_items, average_price, on='order_id')
    df_items = df_items.drop('price', axis=1)

    items_count = df_items['order_id'].groupby(df_items['order_id']).transform('count') # Number of items bought per order
    df_items = df_items.assign(number_of_items=items_count)

    L = []  # Set all categories where the count of products is too small to 'Other'

    for i in df_items['product_category_name'].unique():
        if (df_items['product_category_name']==i).sum() <= 500:
            L.append(i)

    df_items.loc[df_items['product_category_name'].isin(L), 'product_category_name'] = 'Other'

    y = pd.get_dummies(df_items.product_category_name, prefix='Cat') # OHE on product category
    df_items = pd.concat([df_items, y], axis=1)
    df_items = df_items.drop('product_category_name', axis=1)

        ### Merge with df2

    df3 = pd.merge(df2, df_items, on='order_id', how='left')

        ### Engineering
    list_cat = [i for i in df3.columns if 'Cat_' in i]  #Get list of columns corresponding to a category

    # We compute here the sum of products in each category per order and add a new column to the df containing the number of items

    for i in list_cat:
        df3_new = df3.groupby('order_id').agg({i:'sum'})
        df3 = df3.drop(i, axis=1)
        df3 = pd.merge(df3, df3_new, on='order_id', how='left')

    df = df3.drop_duplicates(subset=['order_id'], keep='first') # keep only 1 row per order

        ### Group by client

    list_to_min = ['days_since_order']
    list_to_average = ['average_review_score', 'average_price']
    list_to_leave = ['customer_unique_id', 'number_of_orders']

    list_to_sum = [i for i in df.columns.tolist() if i not in list_to_min 
                   and i not in list_to_average 
                   and i not in list_to_leave]

    for i in list_to_min:
        df = groupby_agg(df, 'customer_unique_id', i, 'min')

    for i in list_to_average:
        df = groupby_agg(df, 'customer_unique_id', i, 'mean')

    for i in list_to_sum:
        df = groupby_agg(df, 'customer_unique_id', i, 'sum')

    df = df.drop_duplicates(subset=['customer_unique_id'], keep='first')
    
    df = df.set_index('customer_unique_id') #Set the customer_unique_id as the index
    #print(df.head())
    
    df = df.drop([
        'order_id',
        'customer_id'
    ], axis=1)
    
    
    
        ### Transform data
        
    df = transform_set(df)
    #display(df)
        
    return df
