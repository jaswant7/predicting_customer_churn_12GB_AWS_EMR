
import pyspark
from pyspark.sql import SparkSession, Window

from pyspark.sql.functions import count, col, udf, desc, max as Fmax, lag, struct, date_add, sum as Fsum,                         datediff, date_trunc, row_number, when, coalesce, avg as Favg
from pyspark.sql.types import IntegerType, DateType

import datetime

from pyspark.ml.classification import LogisticRegression, GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# import matplotlib.pyplot as plt
# import seaborn as sns

import pandas as pd

def compare_date_cols(x,y):
        '''
        Compares x to y. Returns 1 if different
        '''
        if x != y:
            return 0
        else:
            return 1
def label_churn(x):
        '''
        INPUT
        x: Page
        
        OUTPUT
        Returns 1 if an instance of Churn, else returns 0
        '''
        if x=='Cancellation Confirmation':
            return 1
        elif x=='Downgrade':
            return 1
        else:
            return 0

if __name__=='__main__':
    # Creating a Spark Session
    spark = SparkSession.builder.appName("Sparkify").getOrCreate()

    event_data = "s3n://udacity-dsnd/sparkify/mini_sparkify_event_data.json"
    df = spark.read.json(event_data)
    # df = spark.read.json('mini_sparkify_event_data.json')
    # df.printSchema()


    #total rows
    # df.count()
    
    # total rows after dropping null userId
    df = df.where(col('userId')!='')
    # df.count()


    # df.select('location').dropDuplicates().take(10)


    #we can extract state name and put into another column name 'state'
    # print(len(df.columns))
    get_state = udf(lambda x:x[-2:])
    df = df.withColumn('state',get_state(col('location')))
    # print(len(df.columns))


    # df.select('page').dropDuplicates().show()


    #unique users by state
    # df.filter(col('page')=='NextSong').dropDuplicates(['userId']).groupBy('state').agg(count('userId').alias('count')).sort(desc('count')).show(10)


    #timestamp feature
    # df.select('ts').show(10)
    # df.take(1)


    # Defining user define functions to get hour, day, month, and year
    get_hour = udf(lambda x: datetime.datetime.fromtimestamp(x/1000).hour,IntegerType())
    get_day = udf(lambda x: datetime.datetime.fromtimestamp(x/1000).day,IntegerType())
    get_month = udf(lambda x: datetime.datetime.fromtimestamp(x/1000).month,IntegerType())
    get_year = udf(lambda x: datetime.datetime.fromtimestamp(x/1000).year,IntegerType())


    df = df.withColumn('hour',get_hour(col('ts'))).withColumn('day',get_day(col('ts'))).withColumn('month',get_month(col('ts'))).withColumn('year',get_year(col('ts')))

    # print("len of col",len(df.columns))


    # Transferring the above date analysis onto a Pandas DF
    # df_pd = df.filter(col('page')=='NextSong').select(['hour','day','month','userId']).toPandas()
    # df_pd.head()


    #plot for users acc to hour, day and month
    # plt.figure(figsize=(16,4))

    # plt.subplot(131)
    # sns.countplot(x='hour',data=df_pd)
    # plt.title('Events by Hour')

    # plt.subplot(132)
    # sns.countplot(x='day',data=df_pd)
    # plt.title('Events by Day')

    # plt.subplot(133)
    # sns.countplot(x='month',data=df_pd)
    # plt.title('Events by Month')

    # plt.tight_layout()


    # Creating a column containing 1 if the event was a "NextSong" page visit or 0 otherwise
    listen_flag = udf(lambda x: 1 if x=='NextSong' else 0, IntegerType())
    df = df.withColumn('listen_flag',listen_flag('page'))
    # df.take(1)


    # Also creating a feature with the PySpark DateType() just in case
    get_date = udf(lambda x: datetime.datetime.fromtimestamp(x/1000),DateType())
    df = df.withColumn('date',get_date(col('ts')))
    # df.take(1)


    # Creating a second table where I will create this feature, then join it back to the main table later
    df_listen_day = df.select(['userId','date','listen_flag']).groupBy(['userId','date']).agg(Fmax('listen_flag').alias('listen_flag')).sort(['userId','date'])
    # df_listen_day.show(10)


    # Defining a window partitioned by User and ordered by date
    window = Window.partitionBy('userId').orderBy(col('date'))

    # Using the above defined window and a lag function to create a previous day column
    df_listen_day = df_listen_day.withColumn('prev_day',lag(col('date')).over(window))
    # df_listen_day.show()


    # Creating a udf to compare one date to another
    
    date_group = udf(compare_date_cols, IntegerType())

    # Creating another window partitioned by userId and ordered by date
    windowval = (Window.partitionBy('userId').orderBy('date')
                 .rangeBetween(Window.unboundedPreceding, 0))#this function helps in cumulative sum

    df_listen_day = df_listen_day.withColumn('date_group',date_group(col('date'), date_add(col('prev_day'),1))\
                                        # The above line checks if current day and previous day +1 day are equivalent
                                            # If They are equivalent (i.e. consecutive days), return 1
                                )\
                            .withColumn('days_consec_listen',Fsum('date_group').over(windowval))\
                            .select(['userId','date','days_consec_listen'])
                                        # The above lines calculate a running total summing consecutive listens
    # df_listen_day.show()


    # Joining this intermediary table back into the original DataFrame
    df = df.join(other=df_listen_day,on=['userId','date'],how='left')
    # df.show()

    # Re-stating the window
    windowval = Window.partitionBy('userId').orderBy('date')

    # Calculate difference (via datediff) between current date and previous date (taken with lag), and filling na's with 0
    df_last_listen = df_listen_day.withColumn('days_since_last_listen',
                                                datediff(col('date'),lag(col('date')).over(windowval))) \
                                .fillna(0,subset=['days_since_last_listen']) \
                                .select(['userId','date','days_since_last_listen'])


    # Joining back results
    df = df.join(df_last_listen,on=['userId','date'],how='left')
    # df.show(2)


    # Creating udf's to flag whenever a user visits each particular page
    thU_flag = udf(lambda x: 1 if x=='Thumbs Up' else 0, IntegerType())
    thD_flag = udf(lambda x: 1 if x=='Thumbs Down' else 0, IntegerType())
    err_flag = udf(lambda x: 1 if x=='Error' else 0, IntegerType())
    addP_flag = udf(lambda x: 1 if x=='Add to Playlist' else 0, IntegerType())
    addF_flag = udf(lambda x: 1 if x=='Add Friend' else 0, IntegerType())


    # Creating the flag columns
    df = df.withColumn('thU_flag',thU_flag('page')).withColumn('thD_flag',thD_flag('page')).withColumn('err_flag',err_flag('page')).withColumn('addP_flag',addP_flag('page')).withColumn('addF_flag',addF_flag('page'))


    #defining churn when page visit to 'Cancellation Confirmation' or 'Downgrade'
    

    # Creating udf
    udf_label_churn = udf(label_churn, IntegerType())
    # Creating column
    df = df.withColumn('Churn',udf_label_churn(col('page')))
    # print("len of col",len(df.columns))


    df_listens_user = df.groupBy('userId').agg(Fmax(col('days_since_last_listen')).alias('most_days_since_last_listen'),
                     Fmax(col('days_consec_listen')).alias('most_days_consec_listen'),
                     Fsum(col('listen_flag')).alias('total_listens'),
                     Fsum(col('thU_flag')).alias('total_thumbsU'),
                     Fsum(col('thD_flag')).alias('total_thumbsD'),
                     Fsum(col('err_flag')).alias('total_err'),
                     Fsum(col('addP_flag')).alias('total_add_pl'),
                     Fsum(col('addF_flag')).alias('total_add_fr')
                    )
    # df_listens_user.show(5)

    #to see how each user behave in one session average
    df_sess = df.select(['userId','sessionId','listen_flag','thU_flag','thD_flag','err_flag','addP_flag','addF_flag'])
    				.groupBy(['userId','sessionId'])             
    				.agg(Fsum(col('listen_flag')).alias('sess_listens'),
                     Fsum(col('thU_flag')).alias('sess_thU'),
                     Fsum(col('thD_flag')).alias('sess_thD'),
                     Fsum(col('err_flag')).alias('sess_err'),
                     Fsum(col('addP_flag')).alias('sess_addP'),
                     Fsum(col('addF_flag')).alias('sess_addF'))
    # df_sess.show()


    df_sess_agg = df_sess.groupBy('userId').agg(Favg(col('sess_listens')).alias('avg_sess_listens'),
                        Favg(col('sess_thU')).alias('avg_sess_thU'),
                        Favg(col('sess_thD')).alias('avg_sess_thD'),
                        Favg(col('sess_err')).alias('avg_sess_err'),
                        Favg(col('sess_addP')).alias('avg_sess_addP'),
                        Favg(col('sess_addF')).alias('avg_sess_addF'))
    # df_sess_agg.show()


    dfUserMatrix = df.groupBy('userId').agg(Fmax(col('gender')).alias('gender')
                                                 ,Fmax(col('churn')).alias('churn'))
    # dfUserMatrix.show(5)


    dfUserMatrix = dfUserMatrix.join(df_listens_user,['userId']).join(df_sess_agg,['userId'])
    # dfUserMatrix.count()


    # Indexing gender to turn a categorical feature into a binary feature
    gender_indexer = StringIndexer(inputCol='gender',outputCol='gender_indexed')
    fitted_gender_indexer = gender_indexer.fit(dfUserMatrix)
    dfModel = fitted_gender_indexer.transform(dfUserMatrix)


    # dfModel.printSchema()


    # Defining the that we want to vectorize in a list
    features = [col for col in dfModel.columns if col not in ('userId','gender','churn')]


    # Vectorizing the features
    assembler = VectorAssembler(inputCols=features,
                                outputCol='features')
    dfModelVec = assembler.transform(dfModel)


    dfModelVec = dfModelVec.select(col('features'),col('Churn').alias('label'))


    # Scaling to mean 0 and unit std dev
    scaler = StandardScaler(inputCol='features', outputCol='features_scaled', withMean=True, withStd=True)
    #not working
    # scalerModel = scaler.fit(dfModelVec)

    # dfModelVecScaled = scalerModel.transform(dfModelVec)

    # dfMain = dfModelVecScaled.select(col('features_scaled').alias('features'),col('label'))

    #as scaler is not woring 
    dfMain = dfModelVec
    # Train/Test split - 80% train and 20% test
    df_train, df_test = dfMain.randomSplit([0.8,0.2], seed=42)


    # def train_eval(model,df_train=df_train, df_test=df_test):
    #     '''
    #     Used to train and evaluate a SparkML model based on accuracy and f-1 score
        
    #     INPUT
    #     model: ML Model to train
    #     df_train: DataFrame with data
        
    #     OUTPUT
    #     None
    #     '''
    #     print(f'Training {model}...')
    #     # Instantiating Evaluators
    #     acc_evaluator = MulticlassClassificationEvaluator(metricName='accuracy')
    #     f1_evaluator = MulticlassClassificationEvaluator(metricName='f1')
        
    #     # Training and predicting with model
    #     modelFitted = model.fit(df_train)
    #     results = modelFitted.transform(df_test)
        
    #     # Calculating metrics
    #     acc = acc_evaluator.evaluate(results)
    #     f1 = f1_evaluator.evaluate(results)
        
    #     print(f'{str(model):<35s}Accuracy: {acc:<4.2%} F-1 Score: {f1:<4.3f}')


    gbt = GBTClassifier()


    # train_eval(gbt)


    # Going for a very small grid because of compute time
    paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth,[3]).addGrid(gbt.maxBins,[16]).build()

    crossVal = CrossValidator(estimator=gbt,
                              estimatorParamMaps=paramGrid,
                              evaluator=MulticlassClassificationEvaluator(),
                              numFolds=2,
                              seed=42,
                              parallelism=2)


    cvModel = crossVal.fit(df_train)


    # Now evaluating on the test set
    predictions = cvModel.transform(df_test)


    # Re-evaluating metrics using the resulting model
    acc_eval = MulticlassClassificationEvaluator(metricName='accuracy')
    f1_eval = MulticlassClassificationEvaluator(metricName='f1')


    # Calculating metrics
    acc = acc_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)
    print(f'Accuracy: {acc:<4.2%} F-1 Score: {f1:<4.3f}')
    spark.stop()


