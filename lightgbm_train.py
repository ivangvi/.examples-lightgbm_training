'''
- Fetches optimization results from lightgbm_optimize.py and trains a list of models, very similar in fashion to lightgbm_train.
- It also takes T_COUNT and F_COUNT columns (which should be passed through) to determine weights, and applies them to the model to have correctly adjusted probabilites when data input is a sample.
'''

# Nota técnica LGBM: es importante que la LISTA DE VARIABLES sobre la cual se hace x_train (values) sea siempre la misma (train-deploy) ya que de ahí sale el ORDEN de las columnas, que en la DF de base es indistinto si la lista de variables tiene el orden correcto.

import pandas as pd
import numpy as np
import lightgbm as lgb
import datetime
import math
import itertools
import sys
import pkg_resources
import time
import os

# ---- FUNCIONES

def tdelta_store(secdelta):
    dd = [math.floor(secdelta / (24*60*60)),'d']
    hh = [math.floor(secdelta / (60*60) - 24*dd[0]),'h']
    mm = [math.floor(secdelta / 60 - 24*60*dd[0] - 60*hh[0]),'m']
    ss = [math.floor(secdelta - 24*60*60*dd[0] - 60*60*hh[0] - 60*mm[0]),'s']
    
    tdelta = ''
    for x in (dd,hh,mm,ss):
        x[0] = str(x[0])
        if len(x[0]) == 1:
            x[0] = '0' + x[0]
        tdelta = tdelta + x[0] + x[1] + '-'
    tdelta = tdelta [:-1]
    return tdelta

def periods_lister (period,periods):
    hora = str(datetime.datetime.time(datetime.datetime.now().replace(microsecond=0)))
    print(hora + ' - Executing PERIODS_LISTER , PERIOD = ' + str(period) + ', PERIODS = ' + str(periods) + '..')  
    periods_list = []
    date = datetime.date(int('20'+str(period)[0:2]),int(str(period)[2:4]),1)
    for x in range(periods):
        if x == 0:
            dxdate = date      
        else:
            dxdate = dxdate - datetime.timedelta(days=1)
            dxdate = dxdate.replace(day=1)
        if len(str(dxdate.month)) == 1:
            dxperiod_month = '0' + str(dxdate.month)
        else:
            dxperiod_month = str(dxdate.month)
        periods_list.insert(x, str(dxdate.year)[2:4] + dxperiod_month)
    print('Returned PERIODS_LIST = ' + str(periods_list) + '..')
    return periods_list

def vars_remover_m (df,matches):
    print ('Trying to drop specific matching vars..')
    for x in matches:           
        try:
            df = df.drop(x, axis = 1)
            print ('DROPPED: ' + x)
        except:
            print ('FAILED:  ' + x)
    return df  

def vars_remover_s(df,starting):
    for x in starting:
        print ('Trying to drop vars starting with: < ' + x + ' > ..')
        lista = list(df.filter(regex = '^' + x))
        df    = df[df.columns.drop(lista)]
        
        print ('Matched exactly ' + str(len(lista)) + ' vars.')
        if len(lista) > 5:
            print ('Showing first 5 dropped:')
            lista = lista[0:5]     
        else:
            print ('Dropped all these vars..')
        print("\n".join(lista))        
    return df

def vars_remover_c_e (df,contain,exclude): # DROP VARS CONTANING ALL contain (list) STR, BUT NOT CONTAINING ALL exclude (list) STR
    print ('Trying to drop vars contaning all strings in ' + ''.join(contain) + 'but not containing strings in ' + ''.join(exclude) + '..')
    lists_c = []  
    for x in contain:   
        lista = list(df.filter(regex = x))
        lists_c.append(lista)

    lists_e = []  
    for x in exclude:   
        lista = list(df.filter(regex = x))
        lists_e.append(lista)
        
    for x in range(len(lists_c)):
        if x == range(len(lists_c))[-1]:
            continue
        if x == 0:
            lista_c_union = list(set(lists_c[x]).intersection(lists_c[x + 1]))
        else:
            lista_c_union = list(set(lista_c_union).intersection(lists_c[x + 1]))
    
    print ('Vars found CONTAINING:')
    for x in lista_c_union:
        print (x)

    for x in range(len(lists_e)):
        if range(len(lists_e))[0] == range(len(lists_e))[-1]:
            lista_e_union = lists_e[0]
        if x == range(len(lists_e))[-1]:
            continue
        if x == 0:
            lista_e_union = list(set(lists_e[x]).intersection(lists_e[x + 1]))
        else:
            lista_e_union = list(set(lista_e_union).intersection(lists_e[x + 1]))
            
    print ('Vars found EXCLUDING:')
    for x in lista_e_union:
        print (x)

    lista_exclude_final = list(set(lista_c_union) - set(lista_e_union))
    
    print ('Vars REMOVING:')
    for x in lista_exclude_final:
        print (x)
        
    df = df[df.columns.drop(lista_exclude_final)]
    return df

def decorrelator (df,threshold):
    hora = str(datetime.datetime.time(datetime.datetime.now().replace(microsecond=0)))
    print('Running DECORRELATOR at: ' + hora + '..')                
    
    tmp = df.drop('TARGET', axis=1)
    ori = len(tmp.columns)
    tmp = tmp.loc[:,df.apply(pd.Series.nunique) != 1].corr()
    uni = ori - len(tmp.columns)
    tmp = tmp.corr()
    vars_list = []
    for i in tmp.columns:
        tmp.loc[i,i] = 0
    tmp = tmp.where(np.triu(np.ones(tmp.shape)).astype(np.bool))
    for i in tmp.columns:
        fijo = 0
        for j in tmp.columns:
            if not math.isnan(tmp.loc[j,i]):
                if abs(tmp.loc[j,i]) > threshold:
                    fijo = 1
        if fijo==0:
            vars_list.append(i)
    fin = len(vars_list)    
    cor = ori - uni - fin
    del tmp
    
    hora = str(datetime.datetime.time(datetime.datetime.now().replace(microsecond=0)))
    print('Finished DECORRELATOR at: ' + hora + '..') 
    print('Originally ' + str(ori) + ' vars.')
    print('Removed    ' + str(ori - fin) + ' vars.')
    print('Removed    ' + str(uni) + ' vars for NO UNIQUES.')
    print('Removed    ' + str(cor) + ' vars for ' + str(threshold) + ' + CORRELATION.')     
    return vars_list

def target_sampler_rows_ratio (df,rows,ratio): # la df final tiene como máximo ROWS. El RATIO es cuantos F hay como mínimo por cada T. 
    df_t = df[(df["TARGET"]==1)].reset_index(drop = True)
    if (rows - len(df_t)) / len(df_t) < ratio:
        df_t = df_t.sample(n = math.floor(rows / (ratio + 1))) # check the math, da bien -- (t + f = rows) ^ (ratio * t = f) => ratio * t = rows - t ...

    df_f = df[(df["TARGET"]==0)].sample(n = rows - len(df_t)).reset_index(drop = True) 
    df   = pd.concat([df_t,df_f]).reset_index(drop = True) 

    print('Finished TARGET_SAMPLER_ROWS_RATIO..')
    print('T   Count: ' + str(len(df_t)))
    print('F   Count: ' + str(len(df_f)))
    print('T/R Ratio: ' + str(round(len(df_t)/len(df),2)))
    print('F/T Ratio: ' + str(round(len(df_f)/float(len(df_t)),2)))
    del df_t,df_f
    return df

def lgb_train_creator (df,variables): # espera una DF con TARGET y una lista de VARIABLES predictivas, rellena con -30 por defecto los NA
    df        = df.fillna(-30)
    y_train   = df['TARGET'].values
    x_train   = df[variables].values
    lgb_train = lgb.Dataset(x_train, y_train)
    return lgb_train

def optimizer(lgb_train,model,leaves_vec,mcw_vec,csbt_vec,rate,xv_folds):   
    hora = str(datetime.datetime.time(datetime.datetime.now().replace(microsecond = 0)))
    print('Optimizing ' + model + ' at: ' + hora + '..')
           
    results = pd.DataFrame(columns=['leaves','mcw','csbt','rate','rounds','auc','auc_dev','auc-dev'])

    iter = 0
    for leaves,mcw,csbt in itertools.product(range(len(leaves_vec)),range(len(mcw_vec)),range(len(csbt_vec))):
        params = {
                    'boosting_type'   : 'gbdt',
                    'objective'       : 'binary',
                    'metric'          : 'auc',
                    'num_leaves'      : leaves_vec[leaves],
                    'learning_rate'   : rate,
                    'colsample_bytree': csbt_vec[csbt],
                    'min_child_weight': mcw_vec[mcw],
                    'verbose'         : 0,
                }
    
        hora = str(datetime.datetime.time(datetime.datetime.now().replace(microsecond=0)))
        comp = '{:.0%}'.format((iter + 1) / float(len(leaves_vec)*len(mcw_vec)*len(csbt_vec))) # tira % de completado ya en str
        print('Iteration ' + str(iter + 1) + '/' + str(len(leaves_vec)*len(mcw_vec)*len(csbt_vec)) + ' running at: ' + hora + '.. ' + comp + ' done!')
        
        gbm_cv = lgb.cv(params,
                        lgb_train,
                        num_boost_round = 10000,      
                        nfold = xv_folds,
                        early_stopping_rounds = 30,
                        stratified = True,
                        )
        iter = iter + 1
        results.loc[iter,'leaves']  = leaves_vec[leaves]
        results.loc[iter,'mcw']     = mcw_vec[mcw]
        results.loc[iter,'csbt']    = csbt_vec[csbt]
        results.loc[iter,'rate']    = rate
        results.loc[iter,'rounds']  = len(gbm_cv['auc-mean'])
        results.loc[iter,'auc']     = gbm_cv['auc-mean'][-1]
        results.loc[iter,'auc_dev'] = gbm_cv['auc-stdv'][-1]
        results.loc[iter,'auc-dev'] = gbm_cv['auc-mean'][-1] - gbm_cv['auc-stdv'][-1]
        hora = str(datetime.datetime.time(datetime.datetime.now().replace(microsecond=0)))

    print('Optimizer done at: ' + hora + ', printing results..')
    print('Best iteration:')
    print(results.sort_values(['auc'], ascending=[False]).head(1))  
    print('Worst iteration:')
    print(results.sort_values(['auc'], ascending=[True]).head(1))  
    print('Spread best/worst: ' + str(results['auc'].max() - results['auc'].min()))
    return results

def vars_selector(lgb_train,pars,vars_list):
    hora = str(datetime.datetime.time(datetime.datetime.now().replace(microsecond=0)))
    print('Running VARS_SELECTOR, training single model to select variables at: ' + hora + '..')
    
    params = {
        'boosting_type'   : 'gbdt',
        'objective'       : 'binary',
        'metric'          : 'auc',
        'learning_rate'   : pars.loc[0,'rate'],
        'num_leaves'      : pars.loc[0,'leaves'],
        'min_child_weight': pars.loc[0,'mcw'],
        'colsample_bytree': pars.loc[0,'csbt'],
        'verbose'         : 0, 
            }
    
    model  = lgb.train(params,
                      lgb_train,
                      num_boost_round = pars.loc[0,'rounds'],
                      feature_name    = vars_list
                      )
    
    features_gain  = pd.Series(model.feature_importance(importance_type="gain"),name='gain')
    features_names = pd.Series(model.feature_name(),name='name')
    importance  = pd.concat([features_names,features_gain],axis =1)
    importance  = importance.sort_values(['gain'],ascending=False)
    variables_1 = importance.loc[importance['gain']>0]
    vars_used   = variables_1['name'].tolist()
    vars_used   = [str(i).strip() for i in vars_used]
    del model,importance,variables_1,features_gain,features_names
    
    hora = str(datetime.datetime.time(datetime.datetime.now().replace(microsecond=0)))
    print('Finished VARS_SELECTOR at: ' + hora + '..') 
    print('Originally    ' + str(len(vars_list)) + ' vars.')
    print('Ended up with ' + str(len(vars_used)) + ' vars: ' + '{:.1%}'.format(len(vars_used) / float(len(vars_list))) + '.')
    return vars_used

def trainer_gini_rounds(lgb_train,pars,variables,folds): # este trainer sale con el modelo, el xv gini y las rondas
    params = {
        'boosting_type'   : 'gbdt',
        'objective'       : 'binary',
        'metric'          : 'auc',
        'learning_rate'   : pars.loc[0,'rate'],
        'num_leaves'      : pars.loc[0,'leaves'],
        'min_child_weight': pars.loc[0,'mcw'],
        'colsample_bytree': pars.loc[0,'csbt'],
        'verbose'         : 0, 
    }

    hora = str(datetime.datetime.time(datetime.datetime.now().replace(microsecond=0)))
    print('Starting XV at: ' + hora + '.')
    gbm_cv = lgb.cv(params,
                    lgb_train,
                    num_boost_round = 10000,      
                    nfold = xv_folds,
                    early_stopping_rounds = 30,
                    stratified = True,
                    )
    rounds = len(gbm_cv['auc-mean'])
    gini   = 2 * gbm_cv['auc-mean'][-1] - 1

    hora = str(datetime.datetime.time(datetime.datetime.now().replace(microsecond=0)))
    print('Starting TRAINING at: ' + hora + '.')
    lgb_model = lgb.train(params,
                          lgb_train,
                          num_boost_round = rounds,
                          feature_name    = variables
                          )       

    return lgb_model, gini, rounds

def check_model():
    # pars vars y gini diferencias entre este mes y el anterior. incremento decremento, anticipar el error.
    print("----------- Check " + model + " -----------")
    print("Gini Actual: ")
    gini_actual = pd.read_csv(direc + 'GINI/' + model + '_' + str(periods_list[0]) + '.txt')['0'][0]
    print(str(round(100*gini_actual,2))+"%")
    print("Gini Anterior: ")
    try:
        gini_antes  = pd.read_csv(direc + 'GINI/' + model + '_' + str(periods_list[1]) + '.txt')['0'][0]
    except:
        print('Anterior es -2, -1 NOT FOUND!')
        gini_antes  = pd.read_csv(direc + 'GINI/' + model + '_' + str(periods_list[2]) + '.txt')['0'][0]
    print(str(round(100*gini_antes,2))+"%")
    gini_var = 100*(gini_actual-gini_antes)/gini_antes
    print("Variacion porcentual en GINI: " + str(round(gini_var,2)) + "%")
    print("--------------------------------------------")
    print("Variables - Top 10: ")
    vars_actual = pd.read_csv(direc + 'VARS/' + model + '_' + str(periods_list[0]) + '.txt')['0'].head(10)
    vars_actual = vars_actual.rename("Actual")
    try:
        vars_antes  = pd.read_csv(direc + 'VARS/' + model + '_' + str(periods_list[1]) + '.txt')['0'].head(10)
    except:
        print('Anterior es -2, -1 NOT FOUND!')
        vars_antes  = pd.read_csv(direc + 'VARS/' + model + '_' + str(periods_list[2]) + '.txt')['0'].head(10)
    vars_antes = vars_antes.rename("Antes")
    print(pd.concat([vars_actual,vars_antes],axis=1))
    print("--------------------------------------------")
    print("Parametros: ")
    pars_actual = pd.read_csv(direc + 'PARS/' + model + '_' + str(periods_list[0]) + '.txt')
    pars_actual.loc[0,'Unnamed: 0'] = 'Actual'
    try:
        pars_antes  = pd.read_csv(direc + 'PARS/' + model + '_' + str(periods_list[1]) + '.txt')
    except:
        print('Anterior es -2, -1 NOT FOUND!')
        pars_antes  = pd.read_csv(direc + 'PARS/' + model + '_' + str(periods_list[2]) + '.txt')
    pars_antes.loc[0,'Unnamed: 0'] = 'Antes'
    print(pars_actual.append(pars_antes))
    print("----------- End Check Model -----------")

# ---- RUN
    
start_time = time.time()

print("Starting up DEVOPTIMIZER MM7 by dataEvo for MMINER SR.")
period  = int(input("Enter model final period (YYMM): "))
periods = int(input("Enter amount of periods to load in total (12): "))
rows_train = int(input("Enter amount of rows for train DF (120K): "))

models  = (
    'FBA',
    'PAQUETE',
    'PFUSD',
    'PFARS',
    'PP',
    'TC',
    'SC',
    'SEG_AP',
    'SEG_AUTO',
    'SEG_DES',
    'SEG_VIDA',
    'SEG_VIVI',
    )

direc           = 'Z:/SR.CRM/BASES/MMINER 7/'
xv_folds        = 5
ratio_f_t       = 2

periods_list = periods_lister(period,periods)

print('Models to run: ' + format(models) + '..')
print('Periods list: ' + format(periods_list) + '..')
not_trained = 'Models not trained: '

for model in models:
    model_start_time = time.time()
    hora = str(datetime.datetime.time(datetime.datetime.now().replace(microsecond=0)))
    # DIRECS
    direc_periods = direc + 'XINOUT/XINOUT_'     + model + '_'
    direc_results = direc + 'OPTIMIZER_RESULTS/' + model + '_' + str(period) + '.txt'
    direc_pars    = direc + 'PARS/'              + model + '_' + str(period) + '.txt'  
    direc_vars    = direc + 'VARS/'              + model + '_' + str(period) + '.txt'
    direc_model   = direc + 'MODELS/'            + model + '_' + str(period) + '.txt'
    direc_gini    = direc + 'GINI/'              + model + '_' + str(period) + '.txt'
    direc_meta    = direc + 'METAMODEL/'         + model + '_' + str(period) + '_model_meta.txt'

    if os.path.isfile(direc_meta):
        print(model + ' MODEL META FILE FOUND! NOT TRAINING!')
        not_trained = not_trained + model + ' - '
        continue

    vars_final = pd.read_csv(direc_vars)['0'].tolist()
    pars       = pd.read_csv(direc_pars)
                
    # LOAD DE TRAIN DF FINAL
    rows_per_period = math.floor(rows_train / periods)
    for count,i in enumerate(periods_list):
        hora = str(datetime.datetime.time(datetime.datetime.now().replace(microsecond=0)))
        print('Loading for TRAINER, ' + model.upper() + ' period ' + str(i) + ' at: ' + hora + '..')
        df = pd.read_csv(direc_periods + str(i) + '.csv', sep = ';', decimal = ',', usecols = vars_final + ['TARGET','T_COUNT','F_COUNT']) 
        df = target_sampler_rows_ratio(df,rows_per_period,ratio_f_t)

        if count == 0:
            df_final = df
            T_SUM = df.loc[0,'T_COUNT']
            F_SUM = df.loc[0,'F_COUNT']
        else:
            df_final = df_final.append(df)       
            T_SUM = df.loc[0,'T_COUNT'] + T_SUM
            F_SUM = df.loc[0,'F_COUNT'] + F_SUM
        del df  
        
    lgb_train = lgb_train_creator(df_final,vars_final)
    # WEIGHT ASSIGN - podría ser mini función o estar incorporado en lgb_train_creator, diciendo que ignore si weights = 1.
    weight_t = float(T_SUM)/float(df_final[df_final['TARGET']==1].count()['TARGET'])
    weight_f = float(F_SUM)/float(df_final[df_final['TARGET']==0].count()['TARGET'])    
    weights = np.where(df_final['TARGET']==1,weight_t,weight_f)
    lgb_train.set_weight(weights)
    del df_final

    lgb_model, gini, rounds = trainer_gini_rounds(lgb_train,pars,vars_final,xv_folds) 
    lgb_model.save_model(direc_model)
    gini = pd.Series(gini) 
    pd.DataFrame(gini).to_csv(direc_gini)
      
    meta = pd.DataFrame(index = [0])
    meta['N_PERIODS'] = periods
    meta['ROWS_TRAIN']= rows_train
    meta['RATIO_F_T'] = ratio_f_t
    meta['XV_GINI']   = gini
    meta['XV_FOLDS']  = xv_folds
    meta['N_VARS']    = len(vars_final)
    
    meta['P_TREES']   = rounds
    meta['P_RATE']    = pars.loc[0,'rate']
    meta['P_LEAVES']  = pars.loc[0,'leaves']
    meta['P_MCW']     = pars.loc[0,'mcw']
    meta['P_COL']     = pars.loc[0,'csbt']
    
    meta['VER_LGB']   = pkg_resources.get_distribution("lightgbm").version
    meta['VER_PD']    = pd.__version__
    meta['VER_PY']    = sys.version

    meta['RUNTIME']   = tdelta_store(time.time() - model_start_time)

    meta.to_csv(direc_meta, index = False)
    
for model in models:
    check_model()

print(not_trained)
runtime = tdelta_store(time.time() - start_time)
print("All done! Have a cookie.")
print('Time to run process: ' + runtime)