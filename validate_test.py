import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import  roc_auc_score
def runXGB(train_X, train_y, test_X, test_X1, test_y=None, feature_names=None, seed_val=0,depth =8):
        params = {}
        params["objective"] = "reg:linear"
        params['eval_metric'] = 'rmse'
        params["eta"] = .01 #0.00334
        params["min_child_weight"] = 1
        params["subsample"] = 0.7
        params["colsample_bytree"] = 0.3
        params["silent"] = 1
        params["max_depth"] = depth
        params["seed"] = seed_val
        #params["max_delta_step"] = 2
        #params["gamma"] = 0.5
        num_rounds = 500 #2500

        plst = list(params.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)

        if test_y is not None:
                xgtest = xgb.DMatrix(test_X, label=test_y)
                watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
                model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds= 300)
        else:
                xgtest = xgb.DMatrix(test_X)
                xgtest1 = xgb.DMatrix(test_X1)
                model = xgb.train(plst, xgtrain, 4000)

        if feature_names:
                        create_feature_map(feature_names)
                        model.dump_model('xgbmodel.txt', 'xgb.fmap', with_stats=True)
                        importance = model.get_fscore(fmap='xgb.fmap')
                        importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
                        imp_df = pd.DataFrame(importance, columns=['feature','fscore'])
                        imp_df['fscore'] = imp_df['fscore'] / imp_df['fscore'].sum()
                        imp_df.to_csv("imp_feat.txt", index=False)

        pred_test_y = model.predict(xgtest)
        pred_test_y1 = model.predict(xgtest1)
        if test_y is not None:
                loss = rmsle(np.expm1(test_y), np.expm1(pred_test_y))
        	return loss
	else:
		return pred_test_y,pred_test_y1


def rmsle(h, y): 
    """
    Compute the Root Mean Squared Log Error for hypthesis h and targets y

    Args:
        h - numpy array containing predictions with shape (n_samples, n_targets)
        y - numpy array containing targets with shape (n_samples, n_targets)
    """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())
def func1(x):
    try: 
        if pd.isnull(x):
            return 0
        else:
            return len(x)
                    
    except:
        return len(x)
def func(x):
    try: 
        if pd.isnull(x):
            return -1
        else:
            return sum(x)                
    except:
        return sum(x)

def validation(train,test,test1):
    weather = pd.read_csv('weather/weather_ready_to_use.csv')
    weather['visit_date'] = pd.to_datetime(weather['visit_date'],format= '%Y-%m-%d')
    air_reserve = pd.read_csv('air_reserve.csv')
    hpg_reserve = pd.read_csv('hpg_reserve.csv')
    hpg_store_info = pd.read_csv('hpg_store_info.csv')
    air_store_info = pd.read_csv('air_store_info.csv')
    air_tf = list(air_store_info.apply(lambda x:'%s %s' % (x['air_area_name'],x['air_genre_name']),axis=1))
    tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
    tfv.fit(air_tf)

    air_visit_data = train.copy()
    sample_submission = test.copy()
    sample_submission1 = test1.copy()

    sample_submission['visit_date'] = pd.to_datetime(sample_submission['visit_date'],format= '%Y-%m-%d')
    air_visit_data = air_visit_data.merge(air_store_info,how = 'left',on= 'air_store_id')
    sample_submission = sample_submission.merge(air_store_info,how = 'left',on= 'air_store_id')
    air_visit_data['visit_date'] = pd.to_datetime(air_visit_data['visit_date'],format= '%Y-%m-%d')

    sample_submission1['air_store_id'] = sample_submission1['id'].apply(lambda x: x.split('_')[0]+str('_') +x.split('_')[1])
    sample_submission1['visit_date'] = sample_submission1['id'].apply(lambda x: x.split('_')[2])
    sample_submission1['visit_date'] = pd.to_datetime(sample_submission1['visit_date'],format= '%Y-%m-%d %H:%M:%S')
    sample_submission1 = sample_submission1.merge(air_store_info,how = 'left',on= 'air_store_id')



    air_visit_data.shape,sample_submission.shape
    air_reserve['visit_datetime'] = pd.to_datetime(air_reserve['visit_datetime'],format= '%Y-%m-%d %H:%M:%S')
    air_reserve['visit_date'] = air_reserve.visit_datetime.apply(lambda x: str(x).split(' ')[0])
    air_reserve['visit_date'] = pd.to_datetime(air_reserve['visit_date'],format= '%Y-%m-%d %H:%M:%S')
    hpg_reserve['visit_datetime'] = pd.to_datetime(hpg_reserve['visit_datetime'],format= '%Y-%m-%d %H:%M:%S')
    hpg_reserve['visit_date'] = hpg_reserve.visit_datetime.apply(lambda x: str(x).split(' ')[0])
    hpg_reserve['visit_date'] = pd.to_datetime(hpg_reserve['visit_date'],format= '%Y-%m-%d %H:%M:%S')




    for i in ['reserve_datetime','reserve_visitors']:
        k = air_reserve[[i,'visit_date','air_store_id']].groupby(['visit_date','air_store_id'])[i].apply(lambda x: x.tolist()).reset_index()
        name = i + 'list'
        if i == 'reserve_datetime':
            k1  = k.copy()
        else:
            k1[name] = k[i].copy()
            
    air_visit_data = air_visit_data.merge(k1,on=['air_store_id','visit_date'],how = 'left')
    sample_submission = sample_submission.merge(k1,on=['air_store_id','visit_date'],how = 'left')
    sample_submission1 = sample_submission1.merge(k1,on=['air_store_id','visit_date'],how = 'left')
    
    air_visit_data['visit_date_month'] =air_visit_data.visit_date.dt.month
    air_visit_data['visit_date_dayofw'] =air_visit_data.visit_date.dt.dayofweek
    air_visit_data['visit_date_year'] =air_visit_data.visit_date.dt.year
    air_visit_data['visit_date_dayofm'] =air_visit_data.visit_date.dt.day
    sample_submission['visit_date_month'] =sample_submission.visit_date.dt.month
    sample_submission['visit_date_dayofw'] =sample_submission.visit_date.dt.dayofweek
    sample_submission['visit_date_year'] =sample_submission.visit_date.dt.year
    sample_submission['visit_date_dayofm'] =sample_submission.visit_date.dt.day

    sample_submission1['visit_date_month'] =sample_submission1.visit_date.dt.month
    sample_submission1['visit_date_dayofw'] =sample_submission1.visit_date.dt.dayofweek
    sample_submission1['visit_date_year'] =sample_submission1.visit_date.dt.year
    sample_submission1['visit_date_dayofm'] =sample_submission1.visit_date.dt.day


    
    air_visit_data['total_reserve']= air_visit_data['reserve_visitorslist'].apply(func)
    air_visit_data['numb_total_reserve'] = air_visit_data['reserve_visitorslist'].apply(func1)
    sample_submission['total_reserve']= sample_submission['reserve_visitorslist'].apply(func)
    sample_submission['numb_total_reserve'] = sample_submission['reserve_visitorslist'].apply(func1)
    sample_submission1['total_reserve']= sample_submission1['reserve_visitorslist'].apply(func)
    sample_submission1['numb_total_reserve'] = sample_submission1['reserve_visitorslist'].apply(func1)


    
    k = [i for i in air_visit_data.columns if i in sample_submission.columns]
    train = air_visit_data.copy()
    test = sample_submission.copy()
    test1 = sample_submission1.copy()
    
    train1 = train.loc[(train.visit_date_year>=2017)].copy()
    k1 = train1[['visitors','air_store_id']].groupby('air_store_id').agg('mean').reset_index()
    k1.columns = ['air_store_id','mean_visitors']
    k2 = train1[['visitors','air_store_id']].groupby('air_store_id').agg('median').reset_index()
    k2.columns = ['air_store_id','median_visitors']
    k3 = train[['visitors','air_store_id','visit_date_month']].groupby(['air_store_id','visit_date_month']).agg('mean').reset_index()
    k3.columns = ['air_store_id','visit_date_month','mean_visitors1']
    k4 = train[['visitors','visit_date_month']].groupby(['visit_date_month']).agg('mean').reset_index()
    k4.columns = ['visit_date_month','mean_visitors2']
    k5 = train1[['visitors','air_store_id','visit_date_dayofw']].groupby(['air_store_id','visit_date_dayofw']).agg('mean').reset_index()
    k5.columns = ['air_store_id','visit_date_dayofw','mean_visitors3']
    k6 = train1[['visitors','visit_date_dayofw']].groupby(['visit_date_dayofw']).agg('mean').reset_index()
    k6.columns = ['visit_date_dayofw','mean_visitors4']
    k7 = train[['visitors','visit_date_month']].groupby(['visit_date_month']).agg('median').reset_index()
    k7.columns = ['visit_date_month','median_visitors2']
    k8 = train1[['visitors','air_store_id','visit_date_dayofw']].groupby(['air_store_id','visit_date_dayofw']).agg('median').reset_index()
    k8.columns = ['air_store_id','visit_date_dayofw','median_visitors3']
    k9 = train1[['visitors','visit_date_dayofw']].groupby(['visit_date_dayofw']).agg('median').reset_index()
    k9.columns = ['visit_date_dayofw','median_visitors4']
    k10 = train[['visitors','air_store_id']].groupby('air_store_id').agg('mean').reset_index()
    k10.columns = ['air_store_id','mean_visitors_f']
    k11 = train[['visitors','air_store_id','visit_date_dayofw']].groupby(['air_store_id','visit_date_dayofw']).agg('mean').reset_index()
    k11.columns = ['air_store_id','visit_date_dayofw','mean_visitors3_f']
    k12 = train[['visitors','visit_date_dayofw']].groupby(['visit_date_dayofw']).agg('mean').reset_index()
    k12.columns = ['visit_date_dayofw','mean_visitors4_f']
    train = air_visit_data.copy()
    test = sample_submission.copy()
    test1 = sample_submission1.copy()

    
    y = train.visitors.values
    print (test1.columns)
    train = train[k]
    test = test[k]
    test1 = test1[k]
    
    train = train.merge(k1,on='air_store_id',how='left')
    test = test.merge(k1,on='air_store_id',how='left')
    test1 = test1.merge(k1,on='air_store_id',how='left')
    
    train = train.merge(k2,on='air_store_id',how='left')
    test = test.merge(k2,on='air_store_id',how='left')
    test1 = test1.merge(k2,on='air_store_id',how='left')
    
    train = train.merge(k3,on=['air_store_id','visit_date_month'],how='left')
    test = test.merge(k3,on= ['air_store_id','visit_date_month'],how='left')
    test1 = test1.merge(k3,on= ['air_store_id','visit_date_month'],how='left')
    
    train = train.merge(k4,on=['visit_date_month'],how='left')
    test = test.merge(k4,on= ['visit_date_month'],how='left')
    test1 = test1.merge(k4,on= ['visit_date_month'],how='left')
    
    train = train.merge(k5,on=['air_store_id','visit_date_dayofw'],how='left')
    test = test.merge(k5,on= ['air_store_id','visit_date_dayofw'],how='left')
    test1 = test1.merge(k5,on= ['air_store_id','visit_date_dayofw'],how='left')
    
    train = train.merge(k6,on=['visit_date_dayofw'],how='left')
    test = test.merge(k6,on= ['visit_date_dayofw'],how='left')
    test1 = test1.merge(k6,on= ['visit_date_dayofw'],how='left')

    
    train = train.merge(k7,on=['visit_date_month'],how='left')
    test = test.merge(k7,on= ['visit_date_month'],how='left')
    test1 = test1.merge(k7,on= ['visit_date_month'],how='left')
    
    train = train.merge(k8,on=['air_store_id','visit_date_dayofw'],how='left')
    test = test.merge(k8,on= ['air_store_id','visit_date_dayofw'],how='left')
    test1 = test1.merge(k8,on= ['air_store_id','visit_date_dayofw'],how='left')
    
    train = train.merge(k9,on=['visit_date_dayofw'],how='left')
    test = test.merge(k9,on= ['visit_date_dayofw'],how='left')
    test1 = test1.merge(k9,on= ['visit_date_dayofw'],how='left')
    
    train = train.merge(k10,on=['air_store_id'],how='left')
    test = test.merge(k10,on= ['air_store_id'],how='left')
    test1 = test1.merge(k10,on= ['air_store_id'],how='left')

    
    train = train.merge(k11,on=['air_store_id','visit_date_dayofw'],how='left')
    test = test.merge(k11,on= ['air_store_id','visit_date_dayofw'],how='left')
    test1 = test1.merge(k11,on= ['air_store_id','visit_date_dayofw'],how='left')
    
    train = train.merge(k12,on=['visit_date_dayofw'],how='left')
    test = test.merge(k12,on= ['visit_date_dayofw'],how='left')
    test1 = test1.merge(k12,on= ['visit_date_dayofw'],how='left')
    
    date_info = pd.read_csv('date_info.csv')
    date_info['calendar_date']  = pd.to_datetime(date_info['calendar_date'],format= '%Y-%m-%d')
    date_info.rename(columns = {'calendar_date':'visit_date'},inplace = True)
    wkend_holidays = date_info.apply((lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)
    date_info.loc[wkend_holidays, 'holiday_flg'] = 0
    date_info['weight'] = ((date_info.index + 1.0) / len(date_info)) ** 5.0
    train = train.merge(date_info,on='visit_date',how='left')
    test = test.merge(date_info,on='visit_date',how='left')
    test1 = test1.merge(date_info,on='visit_date',how='left')
    relation = pd.read_csv('store_id_relation.csv')
    relation['both'] = 1
    
    train = train.merge(relation,how='left',on='air_store_id')
    test = test.merge(relation,how='left',on='air_store_id')
    test1 = test1.merge(relation,how='left',on='air_store_id')
    
    train = train.merge(hpg_store_info,how='left',on='hpg_store_id')
    test = test.merge(hpg_store_info,how='left',on='hpg_store_id')
    test1 = test1.merge(hpg_store_info,how='left',on='hpg_store_id')
    
    train = train.merge(weather,on=['air_store_id','visit_date'],how='left')
    test = test.merge(weather,on= ['air_store_id','visit_date'],how='left')
    test1 = test1.merge(weather,on= ['air_store_id','visit_date'],how='left')
    
    train_tf = list(train.apply(lambda x:'%s %s' % (x['air_area_name'],x['air_genre_name']),axis=1))
    test_tf = list(test.apply(lambda x:'%s %s' % (x['air_area_name'],x['air_genre_name']),axis=1))
    test1_tf = list(test1.apply(lambda x:'%s %s' % (x['air_area_name'],x['air_genre_name']),axis=1))
    
    train_tf_vec =  tfv.transform(train_tf) 
    test_tf_vec = tfv.transform(test_tf)
    test1_tf_vec = tfv.transform(test1_tf)


    svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
    svd.fit(train_tf_vec)
    train_tf_vec = svd.transform(train_tf_vec)
    test_tf_vec = svd.transform(test_tf_vec)
    test1_tf_vec = svd.transform(test1_tf_vec)



    train_tf_vec = pd.DataFrame(train_tf_vec)
    test_tf_vec = pd.DataFrame(test_tf_vec)
    test1_tf_vec = pd.DataFrame(test1_tf_vec)

    train = train.drop(['hpg_area_name','hpg_genre_name','reserve_visitorslist','reserve_datetime','visit_date'],axis =1)
    test = test.drop(['hpg_area_name','hpg_genre_name','reserve_visitorslist','reserve_datetime','visit_date'],axis =1)
    test1 = test1.drop(['hpg_area_name','hpg_genre_name','reserve_visitorslist','reserve_datetime','visit_date'],axis =1)
    
    
    from sklearn import ensemble, preprocessing
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    train.replace(np.nan,-1,inplace=True)
    test.replace(np.nan,-1,inplace=True)
    text_columns = []
    for f in train.columns:
        if train[f].dtype == 'object':  
            text_columns.append(f)            
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values)+list(test1[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))
            test1[f] = lbl.transform(list(test1[f].values)) 
    train = pd.concat((train,train_tf_vec),axis=1)
    test = pd.concat((test,test_tf_vec),axis=1)
    test1 = pd.concat((test1,test1_tf_vec),axis=1)
    
    train.replace(np.nan,-1,inplace=True)
    test.replace(np.nan,-1,inplace=True)
    test1.replace(np.nan,-1,inplace=True)
    
    y1 = np.log1p(y+1)
    pred1,pred2 = runXGB(train,y1, test,test1,depth = 8)
    p= np.expm1(pred1)-1
    p[p<0] = 0
    p1= np.expm1(pred2)-1
    p1[p1<0] = 0
    
    return p,p1

if __name__ == "__main__":
    met = []
    train = pd.read_csv('air_visit_data.csv')
    test = pd.read_csv('sample_submission.csv')
    train['visit_date'] = pd.to_datetime(train['visit_date'],format= '%Y-%m-%d')
    k = train.loc[train.visit_date > pd.to_datetime('2017-04-01',format= '%Y-%m-%d'),'visit_date'].values
    t = len(np.unique(k))
    print t
    for i in range(0,t):
        print i
        o = pd.to_datetime('2017-04-01',format= '%Y-%m-%d') + pd.DateOffset(i)
        ind1 = train.loc[train.visit_date <= o].index
        ind2 = train.loc[train.visit_date > o].index
        X_train = train.iloc[ind1]
        X_test = train.iloc[ind2]
        test_y = X_test.visitors.values
        X_test = X_test.drop('visitors',axis=1)
        pred_test_y,pred_test_y1 = validation(X_train.copy(),X_test.copy(),test.copy())
        e = rmsle(test_y, pred_test_y)
        test[str(i)] = pred_test_y1
        print e
        met.append(e)
    print ("mean for whole",np.mean(met))
    test.to_csv('21_day_op.csv',index = False)
        

"""
23
0
1.2425517037
1
1.27272414297
2
0.632099078805
3
0.578458950094
4
0.530093638095
5
0.511066750225
6
0.500868601162
7
0.49396321709
8
0.483424149459
9
0.484540962603
10
0.483221479301
11
0.47933288407
12
0.472528277756
13
0.468189920585
14
0.466281831745
15
0.467318196308
16
0.469043437379
17
0.467006140899
18
0.46204094153
19
0.446112252412
20
0.440478157546
21
0.417091892638
22
0.443950912361
('mean for whole', 0.55271250081449952)
"""






