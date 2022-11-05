
# Sorting of Products and Reviews

"""
In this article, I will cover the issue of Sorting of Products and Reviews. For easy tracking and jumping to topics
I am adding the table of contents. (You can go directly to that section by clicking on the title. Added html link.)
For sorting, I will apply bayesian, wilson lower bound and hybrid methods.
I will also touch on other sorting related issues, but only give information. I will apply them in detail in my other articles.
"""

"""
CONTENTS
1.  Data Story                                  [line 34 to 75] 
2.  Libraries & Pandas option settings          [line 76 to 101] 
3.  Data Preparation                            [line 102 to 287] 
4.  Wilson Lower Bound                          [line 288 to 317] 
5.  Product Scores via Wilson Lower Bound       [line 318 to 369] 
6.  Rewiews Scores via Wilson Lower Bound       [line 370 to 386] 
7.  Bayesian Approximation                      [line 387 to 426] 
8.  Product Scores via Bayesian Approximation   [line 427 to 438]                
9.  Rewiew Scores via Bayesian Approximation    [line 439 to 447] 
10. Hybrid Approximations                       [line 448 to 455] 
11. Sorting Product via Hybrid Approximations   [line 456 to 551] 
12. Sorting Reviews via Hybrid Approximations   [line 552 to 645] 
13. Results                                     [line 646 to 650] 
14. Results of Products                         [line 651 to 676] 
15. Results of Reviews                          [line 676 to 690] 
16. Basic sorting algorithms                    [line 691 to 700] 
17. Advance sorting algorithms                  [line 701 to 711] 
"""

########################################################################
########################################################################
# 1. Data Story
########################################################################
########################################################################
"""
The data consists of data scraped from the website of an India-based e-commerce company called Flipkart.

ProductUrl           : url/link of products 
productTitle         : name of products     
productPrice         : price of products
averageRating        : average rating of products
reviewTitle          : review title of products
reviewDescription    : content of reviews
reviewAuthor         : name of reviewers
reviewAt             : date of review (month/year)
reviewLikes          : number of likes of the review
reviewDislikes       : number of dislikes of the review
certifiedBuyer       : certified or uncertified customer's review
reviewerLocation     : reviewer's location (city)
fiveStarRatingCount  : total number of 5 star ratings
fourStarRatingCount  : total number of 4 star ratings
threeStarRatingCount : total number of 3 star ratings
twoStarRatingCount   : total number of 2 star ratings
oneStarRatingCount   : total number of 1 star ratings
reviewImages         : review's image
scrapedAt            : scraping date of data
uniqId               : Unique id of each review

# Variables to add later

wlb_products         : Product Scores via Wilson Lower Bound
wlb_reviews          : Rewiews Scores via Wilson Lower Bound
bay_products         : Products Scores via Bayesian Approximation
bay_reviews          : Rewiews Scores via Bayesian Approximation
certifiedBuyerRandom : Randomly generated certified and uncertified customers
reviewTotal          : total number of reviews
hybrid_product_score : Hybrid Scores of Products 
hybrid_reviews_score : Hybrid Scores of Reviews


Note: This dataset looks like a piece of a very large dataset.
If you access Flipkart's entire dataset, you just need to read the data and run the codes.
"""
########################################################################
########################################################################
# 2. Libraries & Pandas option settings
########################################################################
########################################################################

import numpy as np
import pandas as pd
import math
import random
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

def set_option(max_columns       : int = 50,
               max_rows          : int = 50,
               expand_frame_repr : bool = True,
               float_format      : int = False):
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.expand_frame_repr', expand_frame_repr)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

# pd.options.mode.chained_assignment = None

set_option(None,None,False)

########################################################################
########################################################################
# 3. Data Preparation
########################################################################
########################################################################

# I am also adding the link of the data in case there is a problem with the data file.
data_ = pd.read_csv("flipkart_reviews_large_dataset_sample.csv")
data_link = pd.read_csv('https://query.data.world/s/uv3uhvbs5j4l5fjijhwa5x4cuvxf4x')

data = data_.copy()

data.head()

data.shape

data.size

data.info()

data.describe().T

data.isna().sum()

data.isnull().sum()

data.dtypes

# This function is returns missing data, dtypes of columns, columns number, value counts

def data_info(data):
    index   = []
    missing = []
    dtypes  = []
    count   = []
    _data_    = {'Missing Values': missing, 'Data Type': dtypes, 'Value Counts': count,
               "Columns Counts": range(1, len(data.columns) + 1)}

    for i in data.columns:
        missing.append(data[i].isnull().sum())
        index.append(data[i].name)
        dtypes.append(data[i].dtype)
        count.append(data[i].count())

    table = pd.DataFrame(data=_data_, index=index)

    print("Data Frame Shape: {0}".format(data.shape))
    print(data.index)

    return table

data_info(data)

# Dropping out variables that not related

"""
This are:
ProductUrl
reviewImages
scrapedAt
"""
# I'm keeping the scraping date, may be need it.

data["scrapedAt"][0]

scrapedate = pd.to_datetime(data["scrapedAt"][0], dayfirst=True)

scrapedate

data.drop(["reviewImages","ProductUrl","scrapedAt"], axis=1, inplace=True)

# Changing expressions such as months ago, days ago inside of ReviewAt variable which the review was added date.

data["reviewAt"].value_counts()

rev_map = {r'1 month ago'   : 'Jan, 2022',
           r'2 months ago'  : 'Dec, 2021',
           r'3 months ago'  : 'Nov, 2021',
           r'4 months ago'  : 'Oct, 2021',
           r'5 months ago'  : 'Sep, 2021',
           r'6 months ago'  : 'Aug, 2021',
           r'7 months ago'  : 'Jul, 2021',
           r'8 months ago'  : 'Jun, 2021',
           r'9 months ago'  : 'May, 2021',
           r'10 months ago' : 'Apr, 2021',
           r'11 months ago' : 'Mar, 2021',
           r'14 days ago'   : 'Feb, 2022'}

data['reviewAt'] = data['reviewAt'].replace(rev_map, regex=True)


# There is only 1 null value in the reviewTitle variable, I will assign a value to it.
# Getting SettingWithCopyWarning but still assigns. Just ignore it.

data.isnull().sum()

data[data["reviewTitle"].isnull()]

data["reviewTitle"].iloc[672] = "reviewTitle"

data.groupby("productTitle").agg({"averageRating" : "mean"}).sort_values("averageRating", ascending=False)

data[data["productTitle"] == 'OPPO F11 Pro (Thunder Black, 64 GB)']

data[data["productTitle"] == 'OPPO F11 Pro (Thunder Black, 64 GB)']

data[data["productTitle"] == 'APPLE MD827ZM/B Wired Headset']

# Removing commas from StarRatingCount columns and convert object type to int.

data[[i for i in data.columns if "StarRatingCount" in i]].head()

data[[i for i in data.columns if "StarRatingCount" in i]].dtypes

data[[i for i in data.columns if "StarRatingCount" in i]] = data[[i for i in data.columns if "StarRatingCount" in i]].applymap(lambda x: x.replace(",","")).astype("int")

# Removing commas and Indian currency (rupee) symbols from productPrice columns and convert object type to int.
# Getting FutureWarning just ignore it.

data[["productPrice"]].head()

data[["productPrice"]].dtypes

data['productPrice'] = data['productPrice'].str.replace('₹|,', "").astype(int)

data.head()

data.info()

# The commas of the StarRatingCount variables were confusing a little bit.
# When you remove the commas, it is necessary to check whether the average of the this numbers is equal to the average.

def avarage_rating_calculator_five(stars:int and list) -> float:

    """
     Averages out of 5 stars.

    :param stars: The total value of each star. Please enter values from 5th to 1st star. Otherwise, the results may be wrong.
    :return: float , average rating out of 5 stars

    :example:

    avarage_rating_calculator_five([220147, 83936, 30026,10267,21199])
    # 4.289926827600356

    """

    if len(stars) != 5:
        raise ValueError ("The parameter of stars must contain only 5 integers!")
    return (5*stars[0]  + 4*stars[1]  + 3*stars[2]  + 2*stars[3]  + 1*stars[4] ) / (stars[0] +stars[1] +stars[2] +stars[3] +stars[4])

data.iloc[0]

"""
data.iloc[0]

averageRating                                          4.300
fiveStarRatingCount                                   220147
fourStarRatingCount                                    83936
threeStarRatingCount                                   30026
twoStarRatingCount                                     10267
oneStarRatingCount                                     21199
"""

avarage_rating_calculator_five([220147, 83936, 30026,10267,21199])
# 4.289926827600356 -> i.e. 4.3


data.iloc[10]

"""
data.iloc[10]

averageRating                                          4.000
fiveStarRatingCount                                      385
fourStarRatingCount                                      195
threeStarRatingCount                                     100
twoStarRatingCount                                       53
oneStarRatingCount                                       61

"""

avarage_rating_calculator_five([385, 195, 100,53,61])
# 3.994962216624685 -> i.e. 4.0
data.head()

########################################################################
########################################################################
# 4. Wilson Lower Bound
########################################################################
########################################################################
"""
Wilson Confidence Interval considers binomial distribution for score calculation 
i.e. it considers only positive and negative ratings. 
If your product is rated on a 5 scale rating, 
then we can convert ratings {1–3} into negative and {4,5} to positive rating and 
can calculate Wilson score.
"""
########################################################################
########################################################################

def wilson_lower_bound(pos, n, confidence=0.95):
    """
    Function to provide lower bound of wilson score

    :param pos: No of positive ratings
    :param n: Total number of ratings
    :param confidence: Confidence interval, by default is 95 %
    :return: float , Wilson Lower bound score
    """
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

########################################################################
########################################################################
# 5. Product Scores via Wilson Lower Bound
########################################################################
########################################################################
"""
pos = sum of 4-5 stars , 
i.e. data["fiveStarRatingCount"] + data["fourStarRatingCount"]

n = sum of 1-2-3-4-5 stars ,
i.e. data["fiveStarRatingCount"] + data["fourStarRatingCount"]+ data["threeStarRatingCount"]+ data["twoStarRatingCount"]+ data["oneStarRatingCount"]
"""
########################################################################
########################################################################

data["wlb_products"] = data.apply(lambda x: wilson_lower_bound((x["fiveStarRatingCount"] + x["fourStarRatingCount"]),
                                                      (x["fiveStarRatingCount"]  +
                                                       x["fourStarRatingCount"]  +
                                                       x["threeStarRatingCount"] +
                                                       x["twoStarRatingCount"]   +
                                                       x["oneStarRatingCount"])), axis=1)

(data["fiveStarRatingCount"] + data["fourStarRatingCount"]).head()
#304083

220147 + 83936
#304083

385 + 195
#580

(data["fiveStarRatingCount"] +
 data["fourStarRatingCount"] +
 data["threeStarRatingCount"] +
 data["twoStarRatingCount"] +
 data["oneStarRatingCount"]).head()

#365575

220147 + 83936 + 30026 + 10267 + 21199

#365575

385 + 195 + 100 + 53 + 61
#794

wilson_lower_bound(304083,365575)
# 0.8305777449643907

wilson_lower_bound(580,794)
# 0.6985602692257357

########################################################################
########################################################################
# 6. Rewiew Scores via Wilson Lower Bound
########################################################################
########################################################################
"""
pos = number of positive ratings 
i.e. reviewLikes
n = Total number review rating
i.e. reviewLikes + reviewDislikes
"""
########################################################################
########################################################################

data["wlb_reviews"] = data.apply(lambda x: wilson_lower_bound(x["reviewLikes"],
                                                              x["reviewLikes"] + x["reviewDislikes"]), axis=1)

########################################################################
########################################################################
# 7. Bayesian Approximation
########################################################################
########################################################################
"""
Bayesian Approximation provides a way to give a score to a product when they are rated on the K star scale.

Bayesian Approximation for K scale rating
where s_k = k i.e. 1 point, 2 points, ….N point Scale

N = total ratings with n_k ratings for k scale

The above expression provides the lower bound of a normal approximation to a Bayesian credible interval for the average rating. 
For more mathematical details please check [4].
"""

def bayesian_rating(n, confidence=0.95):
    """
    Function to calculate wilson score for N star rating system.
    :param n: Array having count of star ratings where ith index represent the votes for that category i.e. [3, 5, 6, 7, 10]
    here, there are 3 votes for 1-star rating, similarly 5 votes for 2-star rating.
    :param confidence: Confidence interval
    :return: Score
    """
    if sum(n)==0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k+1)*(n[k]+1)/(N+K)
        second_part += (k+1)*(k+1)*(n[k]+1)/(N+K)
    score = first_part - z * math.sqrt((second_part - first_part*first_part)/(N+K+1))
    return score

# bayesian_rating_products([0, 0, 0, 0, 1]) = 2.2290

########################################################################
########################################################################
# 8. Products Scores via Bayesian Approximation
########################################################################
########################################################################

data["bay_products"] = data.apply(lambda x: bayesian_rating(x[["oneStarRatingCount",
                                                               "twoStarRatingCount",
                                                               "threeStarRatingCount",
                                                               "fourStarRatingCount",
                                                               "fiveStarRatingCount"]]), axis=1)

########################################################################
########################################################################
# 9. Rewiew Scores via Bayesian Approximation
########################################################################
########################################################################

data["bay_rewiews"] = data.apply(lambda x: bayesian_rating_products(x[["reviewLikes",
                                                                       "reviewDislikes"]]), axis=1)

########################################################################
########################################################################
# 10. Hybrid Approximations
########################################################################
########################################################################
"""
I will try to develop Hybrid Approximations for this dataset.
"""
########################################################################
########################################################################
# 11. Sorting Products via Hybrid Approximations
########################################################################
########################################################################
"""
rewiewTotal 
averageRating
bayesian_rating_products

Note: May be we can add purchase count but we don't have it for this dataset. 
If you have it in your own dataset just scale and weight it.
"""

# Get the total number of reviews
# There are multiple product titles, groupby to the productTitle, aggregate to the uniqId and get count.
# and creating new data frame
data_rewiew_total = data.groupby("productTitle").agg({"uniqId": lambda x: x.count()})

# merging this new data frame to orijinal dataframe
data = pd.merge(data,data_rewiew_total,
                left_on='productTitle',
                right_index=True)

# Renaming new columns as reviewTotal
# and uniqId also changed to uniqId_x, fixing that too
data.rename(columns={'uniqId_y': 'reviewTotal',
                     'uniqId_x': 'uniqId'}, inplace=True)

# averageRating & bar_score_products consists of float values between 1 and 5. But reviewTotal is between 1 and 10 so I just scale it.
# The following function returns the weighted score of Total Number of Reviews and Average Rating of Products and Bayesian Score of Products.

def hybrid_score(dataframe,
                 bayesian_n      : str and list,
                 review_total_w  : int,
                 averageRating_w : int,
                 bay_w           : int,
                 averageRating_c : str,
                 scaled_c        : str) -> float:

    """
    :param dataframe        : the dataframe
    :param bayesian_n       : bayesian rating n parameter , str and list
    :param review_total_w   : weight of total review      , int
    :param averageRating_w  : weight of avarage rating    , int
    :param bay_w            : weight of bayesian rating   , int
    :param averageRating_c  : column of Average Rating
    :param scaled_c         : column of scale variable
    :return                 : hybrid score, float
    """

    def bayesian_rating(n, confidence=0.95):

        """
        Function to calculate wilson score for N star rating system.
        :param n: Array having count of star ratings where ith index represent the votes for that category i.e. [3, 5, 6, 7, 10]
        here, there are 3 votes for 1-star rating, similarly 5 votes for 2-star rating.
        :param confidence: Confidence interval
        :return: Score
        """

        if sum(n) == 0:
            return 0
        K = len(n)
        z = st.norm.ppf(1 - (1 - confidence) / 2)
        N = sum(n)
        first_part = 0.0
        second_part = 0.0
        for k, n_k in enumerate(n):
            first_part += (k + 1) * (n[k] + 1) / (N + K)
            second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
        score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))

        return score

    bay_score = dataframe.apply(lambda x: bayesian_rating(x[bayesian_n]), axis=1)

    dataframe[[scaled_c+"_scaled"]] = MinMaxScaler(feature_range=(1, 5)).fit(dataframe[[scaled_c]]).transform(dataframe[[scaled_c]])

    reviewTotal_w   = dataframe[scaled_c+"_scaled"] * review_total_w / 100
    averageRating_w = dataframe[averageRating_c] * averageRating_w / 100
    bay_score_w     = bay_score*bay_w/100

    return reviewTotal_w + averageRating_w + bay_score_w/100

data["hybrid_product_score"] = hybrid_score(data,["oneStarRatingCount",
                                                  "twoStarRatingCount",
                                                  "threeStarRatingCount",
                                                  "fourStarRatingCount",
                                                  "fiveStarRatingCount"],
                                            40,
                                            40,
                                            20,
                                            "averageRating",
                                            "reviewTotal")

########################################################################
########################################################################
# 12. Sorting Reviews via Hybrid Approximations
########################################################################
########################################################################
"""
wlb + certified customer

Here I am doing an operation to the certified customer variable. All reviwers are certified customers(True), 
but to fit the story, I will randomly assign half of the new variable as uncertified customer.

btw may be we can add total purchase but in this dataset dont has it. 
Can't get the total purchase, certified Buyers gives whether or not commenters are certified customers.
We should have variable such as invoice etc.There may be people who purchase and did not reviewed. Or reviewing without purchasing.
"""
########################################################################
########################################################################

# Generating numbers from 1's and 0's
cb = np.zeros(data.shape[0], dtype=int)
cb[:int(data.shape[0]/2)] = 1
np.random.shuffle(cb)

data["certifiedBuyerRandom"] = cb

data.groupby("productTitle").agg({"certifiedBuyerRandom": ["sum","count"]})

data.head()

data["certifiedBuyerRandom"].sum()

(3*60)/100 + (7*40)/100

data_cert_rev = data.groupby("productTitle")["certifiedBuyerRandom"].agg(["sum","count"])

data = pd.merge(data,data_cert_rev,
                 left_on='productTitle',
                 right_index=True)

data.rename(columns={"sum"   : "certifiedBuyerRandom_yes",
                     "count" : "certifiedBuyerRandom_total"}, inplace=True)

def hybrid_score_wlb_cb(data,
                        cb_wlb_w   : int,
                        r_wlb_w    : int,
                        cb_yes     : str,
                        cb_total   : str,
                        rw_like    : str,
                        rw_dislike : str
                        ) -> float:
    """
        returning hybrid score of reviews

    :param data       : dataframe                                       , dataframe
    :param cb_wlb_w   : certified buyer wilson lower bound score weight , int
    :param r_wlb      : review wilson lower bound score weight          , int
    :param cb_yes     : variable of certified yes                       , str
    :param cb_total   : variable of certified buyer total               , str
    :param rw_like    : variable of review like                         , str
    :param rw_dislike : variable of review dislike                      , str
    :return           : hybrid score                                    , float
    """

    def wilson_lower_bound(pos, n, confidence=0.95):
        """
        Function to provide lower bound of wilson score

        :param pos: No of positive ratings
        :param n: Total number of ratings
        :param confidence: Confidence interval, by default is 95 %
        :return: float , Wilson Lower bound score
        """
        if n == 0:
            return 0
        z = st.norm.ppf(1 - (1 - confidence) / 2)
        phat = 1.0 * pos / n
        return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

    certifiedBuyer_wlb_score = data.apply(lambda x: wilson_lower_bound(x[cb_yes],
                                        x[cb_total]), axis=1)

    rewiew_wlb = data.apply(lambda x: wilson_lower_bound(x[rw_like],
                                                         x[rw_like] + x[rw_dislike]), axis=1)
    return certifiedBuyer_wlb_score*cb_wlb_w/100 + rewiew_wlb*r_wlb_w/100

data["hybrid_review_score"] = hybrid_score_wlb_cb(data,
                                                  55, 45,
                                                  "certifiedBuyerRandom_yes",
                                                  "certifiedBuyerRandom_total",
                                                  "reviewLikes",
                                                  "reviewDislikes")



########################################################################
########################################################################
# 13. Results
########################################################################
########################################################################
# 14. Results of Products
########################################################################
########################################################################
"""
rewiew Total 
averageRating
bayesian_rating_products
hybrid

Note: May be we can purchase count but we dont have it for this dataset
if you have it in your own dataset just scaling and weighting 
"""

data.head()

data.groupby("productTitle").agg({**dict.fromkeys(["averageRating" ,
                                                   "reviewTotal" ,
                                                   "hybrid_product_score",
                                                   "bay_score_products",
                                                   "wlb_products",
                                                   "fiveStarRatingCount",
                                                   "fourStarRatingCount",
                                                   "threeStarRatingCount",
                                                   "twoStarRatingCount",
                                                   "oneStarRatingCount"], 'mean')}).sort_values("hybrid_product_score",ascending=False)

########################################################################
########################################################################
# 15. Results of Reviews
########################################################################
########################################################################
data.head()

data.groupby("productTitle").agg({**dict.fromkeys(["hybrid_review_score" ,
                                                   "wlb_reviews",
                                                   "certifiedBuyerRandom_yes",
                                                   "bay_score_rewiews","reviewTotal"], 'mean'),
                                  **dict.fromkeys(["reviewLikes" ,
                                                   "reviewDislikes" ,
                                                   ], 'sum')}).sort_values("hybrid_review_score",ascending=False)

########################################################################
########################################################################
# 16. Basic sorting & ranking algorithms
########################################################################
########################################################################

# https://en.wikipedia.org/wiki/Sorting_algorithm

########################################################################
########################################################################
# 16. Advance sorting & ranking algorithms
########################################################################
########################################################################

# https://vitalflux.com/ranking-algorithms-types-concepts-examples/
# https://en.wikipedia.org/wiki/Ranking_(information_retrieval)

########################################################################
########################################################################
