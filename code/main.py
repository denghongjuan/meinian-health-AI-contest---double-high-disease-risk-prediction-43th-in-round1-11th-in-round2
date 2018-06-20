import time
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import re
import sys  
import requests
import time
import itertools
import shutil
import os

DIR_DATA = '../data/'
DATA_PART1 = '../data/meinian_round1_data_part1_20180408.txt'
DATA_PART2 = '../data/meinian_round1_data_part2_20180408.txt'
DATA_TRAIN_Y = '../data/meinian_round1_train_20180408.csv'
DATA_TEST_Y = '../data/meinian_round1_test_b_20180505.csv'
CACHE_X_RAW = '../data/cache/tmp.csv'
CACHE_X_NUM_TEXT = '../data/cache/train_num_text.csv'
CACHE_X_TEXT = '../data/cache/train_text.csv'
CACHE_LABLE = '../data/cache/lable.csv'
CACHE_X_ALL = '../data/cache/train_all.csv'
CACHE_X_LAST = '../data/cache/train_final.csv'


NUM_FOLDS=5

def merge_table(df):
    df['field_results'] = df['field_results'].astype(str)
    if df.shape[0] > 1:
        merge_df = " ".join(list(df['field_results']))
    else:
        merge_df = df['field_results'].values[0]
    return merge_df

def merge_raw_data():
    if os.path.isfile(CACHE_X_RAW):
        print('Load X_row from cache!')
        return pd.read_csv(CACHE_X_RAW,index_col=0,low_memory=False)
    else:
        part_1 = pd.read_csv(DATA_PART1,sep='$')
        part_2 = pd.read_csv(DATA_PART2,sep='$')
        part_1_2 = pd.concat([part_1,part_2])
        part_1_2 = pd.DataFrame(part_1_2).sort_values('vid').reset_index(drop=True)
        is_happen = part_1_2.groupby(['vid','table_id']).size().reset_index()

        # 重塑index用来去重
        is_happen['new_index'] = is_happen['vid'] + '_' + is_happen['table_id']
        is_happen_new = is_happen[is_happen[0]>1]['new_index']
        part_1_2['new_index'] = part_1_2['vid'] + '_' + part_1_2['table_id']
        unique_part = part_1_2[part_1_2['new_index'].isin(list(is_happen_new))]
        unique_part = unique_part.sort_values(['vid','table_id'])
        no_unique_part = part_1_2[~part_1_2['new_index'].isin(list(is_happen_new))]
        part_1_2_not_unique = unique_part.groupby(['vid','table_id']).apply(merge_table).reset_index()
        part_1_2_not_unique.rename(columns={0:'field_results'},inplace=True)
        tmp = pd.concat([part_1_2_not_unique,no_unique_part[['vid','table_id','field_results']]])
        # # 行列转换shiftdim
        tmp = tmp.pivot(index='vid',values='field_results',columns='table_id')
        tmp.to_csv(CACHE_X_RAW)
        print('X_raw has been saved {}'.format(CACHE_X_RAW))
        return tmp

def strQ2B(ustring):
    """全角转半角"""
    if isinstance(ustring, float) or isinstance(ustring, int) or ustring is None:
        return ustring
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换            
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def change_norom(df_in):
    df=df_in.copy()
    df.loc[:,'2406']=df['2406'].replace("-90",np.nan)
    df.loc[:,'2403']=df['2403'].replace("0",np.nan)
    df.loc[:,'2403']=df['2403'].replace("4.3",43)
    df.loc[:,'2403']=df['2403'].replace("5.3",53)
    df.loc[:,'2404']=df['2404'].replace("0",np.nan)
    df.loc[:,'2405']=df['2405'].replace("0",np.nan)
    df.loc[:,:] = df.replace('nan',np.nan)
    return df

##转化成flaot
def data_clean(df):
    fl_col = []
    str_col = []
    for c in df.columns:
        try:
            df.loc[:,c] = df[c].astype('float32')
            fl_col.append(c)
        except:
            str_col.append(c)
    return fl_col, str_col
    
##区分数值和文本列
def split_cols(tmp):
    df=tmp.copy()
    digit = re.compile(r"\d+\.?\d?")
    h = re.compile(u"[\u4E00-\u9FA5]+")
    cols_num,cols_num_text,cols_text,matched=[],[],[],[]
    
    for col in df.columns:
        if df[col].dtype=="object":
            s = "\t".join(df[col].astype(str).tolist())
            num = digit.findall(s)
            num2 = h.findall(s)
            if len(num2)>len(num):
                cols_text.append(col)
                #pass
            elif len(num) > 1000:
                matched.append(col)
        else:matched.append(col)
    df_num_all = df[matched]
    df_num_all = change_norom(df_num_all)
    df_num_cols = data_clean(df_num_all)
    cols_num = df_num_cols[0]
    cols_num_text = df_num_cols[1]
    return cols_num,cols_num_text,cols_text,df

#白细胞数目
def datCl_HP(data_input1_df,colname_temp):
    if colname_temp in data_input1_df.columns:
        col_temp = data_input1_df[colname_temp]
        new_value_empty = [( info1 == 'nan') for info1 in col_temp]
        new_value_0 = []
        for i in range(len(new_value_empty)) :
            new_value_0.append( not new_value_empty[i] )
        new_value_1 = []
        sum_1 = 0
        count_1 = 0
        for i in col_temp[new_value_0]:
            subNum_1 = re.findall(r"\d+\.?\d*",str(i))
            if len(subNum_1) > 1:
                subNum_2 = sum([float(temp) for temp in subNum_1])/len(subNum_1)
            elif len(subNum_1) == 1:
                subNum_2 = float(subNum_1[0])
            elif (("-" in str(i)) | ("未见" in str(i)) | ("未检出"  in str(i)) | ("阴性"  in str(i)) | ("偶见"  in str(i))| ("少许"  in str(i))):
                subNum_2 = 0
            elif (("阳性"  in str(i)) | ("++"  in str(i)) | ("Ⅱ"  in str(i))):
                subNum_2 = 10
            elif ("满视野" in str(i)):
                subNum_2 = 100
            else:
                subNum_2 = 'nan'
            new_value_1.append(subNum_2)           
        data_input1_df.loc[new_value_0,colname_temp] = new_value_1
    return data_input1_df

##心跳次数
def datCl_HeaRhy(data_input1_df,colname_temp):
    if colname_temp in data_input1_df.columns:
        col_temp = data_input1_df[colname_temp]
        new_value_empty = [((info1 == 'nan') | (info1 == '0') ) for info1 in col_temp]
        new_value_0 = []
        for i in range(len(new_value_empty)) :
            new_value_0.append( not new_value_empty[i] )
        new_value_1 = []
        sum_1 = 0
        count_1 = 0
        for i in col_temp[new_value_0]:
            subNum_1 = re.findall(r"\d+\.?\d*",str(i))
            if len(subNum_1) > 1:
                if(("<60次/分"  in str(i)) | (">100次/分" in str(i))):
                    subNum_m2 = float(subNum_1[1])
                else:    
                    subNum_m2 = sum([float(temp) for temp in subNum_1])/len(subNum_1)
            elif len(subNum_1) == 1:
                if("<60次/分"  in str(i)):
                    subNum_m2=55
                elif(">100次/分" in str(i)):
                    subNum_m2=105
                else:
                    subNum_m2 = float(subNum_1[0])
            elif len(subNum_1) == 0:
                if ("心动过缓"  in str(i)):
                    subNum_m2 = 55
                elif ("心动过速"  in str(i)):
                    subNum_m2 = 105
                else:
                    subNum_m2 = 80                  
            new_value_1.append(subNum_m2)
        data_input1_df.loc[new_value_0,colname_temp] = new_value_1
    return data_input1_df

###模式替换成捕获值函数
def re_place(x,pattern):
    match=pattern.search(x)
    if match:
        x = match.group(1)
    return x

###模式替换成捕获值平均值函数
def re_place3(x,pattern):
    match=pattern.search(x)
    if match:
        x=(float(match.group(1))+float(match.group(2)))/2
    return x

def replace_some_letter(s):
    s = s.apply(strQ2B)
    s = s.astype(str)
    s = s.str.replace("\.\.",".").str.replace(",",".")
    s[s.str.contains('0\.\-3|\.45\.21|\.3\.700|\.0\.04|1、7\.71|2\.792\.20|0\.00\-25\.00|16\.7\.07|\+|\*', na=False)] = np.nan
    s[s.str.contains('未见|阴性|少量', na=False)] = 0
    s = s.str.extract('([-+]?\d*\.\d+|\d+)', expand=False).astype(float)
    return s

##数值文本列处理
def process_num_col(df_num_text):
    if os.path.isfile(CACHE_X_NUM_TEXT):
        print('Load processed num text cols from cache!')
        return pd.read_csv(CACHE_X_NUM_TEXT,index_col=0)
        
    print('process num text cols')
    df=df_num_text.copy()
    df['0424'].fillna='nan'
    df['300036'].fillna='nan'
    df=datCl_HeaRhy(df,'0424')
    df=datCl_HP(df,'300036')   
    df[df=='nan'] = np.nan
    df[df=='-'] = 0
    
    z = df.values
    for line in z:
        for num in range(len(line)):
            elem = line[num]
            try:
                float(elem)
            except:
                elem=re_place(str(elem),re.compile(r'(\d+.\d+?)\s+弃查'))
                elem=re_place(str(elem),re.compile(r'(\d+.\d+?)\s+未查'))
                elem=re_place(str(elem),re.compile(r'未查\s+(\d+.\d+?)'))
                elem=re_place(str(elem),re.compile(r'弃查\s+(\d+.\d+?)')) 
                elem=re_place(str(elem),re.compile(r'\d+\-\d+\s+(\d+\.\d+|\d+)'))
                elem=re_place3(str(elem),re.compile(r'(\d+\.\d+|\d+)\-(\d+\.\d+|\d+)'))
                elem=re_place3(str(elem),re.compile(r'(\d+\.\d+|\d+)\s+(\d+\.\d+|\d+)'))
                line[num]=elem
    df = pd.DataFrame(z,index=df.index,columns=df.columns)
    for col in df.columns:
        df[col] = replace_some_letter(df[col])
    df.to_csv(CACHE_X_NUM_TEXT)
    return df

############ 文本部分数据处理func ################################
##心率归类
def datCl_Arhy(data_input1_df,colname_temp):
    if colname_temp in data_input1_df.columns:

        col_temp = data_input1_df[colname_temp]
        new_value_empty = [((str(info1) == "NaN") | (("详见纸质报告" in str(info1)))) for info1 in col_temp]
       
        new_value_1_1 = [("右房异常" in str(info1)) for info1 in col_temp ]
        new_value_1_2 = [("右心房负荷过重" in str(info1)) for info1 in col_temp ]
        new_value_1_3 = [("右心房肥大" in str(info1)) for info1 in col_temp ]
        new_value_1_4 = [("左房异常" in str(info1)) for info1 in col_temp ]
        new_value_1_5 = [("左心房负荷过重" in str(info1)) for info1 in col_temp ]
        new_value_1_6 = [("左心房肥大" in str(info1)) for info1 in col_temp ]
        new_value_1_7 = [("右室高电压" in str(info1)) for info1 in col_temp ]
        new_value_1_8 = [(("右心室肥大" in str(info1)) | ("右心室肥厚" in str(info1))) for info1 in col_temp ]
        new_value_1_9 = [("左室高电压" in str(info1)) for info1 in col_temp ]
        new_value_1_10 = [(("左心室肥大" in str(info1)) | ("左心室肥厚" in str(info1))) for info1 in col_temp ]
        data_input1_df['avh']=0
        data_input1_df.loc[new_value_1_1,'avh']  = 1
        data_input1_df.loc[new_value_1_2,'avh']  = 2
        data_input1_df.loc[new_value_1_3,'avh']  = 3
        data_input1_df.loc[new_value_1_4,'avh']  = 4
        data_input1_df.loc[new_value_1_5,'avh']  = 5
        data_input1_df.loc[new_value_1_6,'avh']  = 6
        data_input1_df.loc[new_value_1_7,'avh']  = 7
        data_input1_df.loc[new_value_1_8,'avh']  = 8
        data_input1_df.loc[new_value_1_9,'avh']  = 9
        data_input1_df.loc[new_value_1_10,'avh']  = 10
        data_input1_df.loc[new_value_empty,'avh']  = np.nan
        
        
        new_value_2_1 = [(("供血不足" in str(info1))| ("心肌供血不足" in str(info1)) | ("疑似心肌缺血"  in str(info1))) for info1 in col_temp ]
        new_value_2_2 = [(("心肌缺血" in str(info1))  & ("疑似心肌缺血" not in str(info1))) for info1 in col_temp ]
        new_value_2_3 = [("心肌梗塞" in str(info1)) for info1 in col_temp ]
        new_value_2_4 = [("心肌梗死" in str(info1)) for info1 in col_temp ]
        new_value_2_5 = [(("早期复极综合征" in str(info1))|("过早复极" in str(info1))) for info1 in col_temp ]
        data_input1_df['myocardium']=0
        data_input1_df.loc[new_value_2_1,'myocardium']  = 1
        data_input1_df.loc[new_value_2_2,'myocardium']  = 2
        data_input1_df.loc[new_value_2_3,'myocardium']  = 3
        data_input1_df.loc[new_value_2_4,'myocardium']  = 4
        data_input1_df.loc[new_value_2_5,'myocardium']  = 5
        data_input1_df.loc[new_value_empty,'myocardium']  = np.nan
        
        new_value_3_1 = [(("正常心电图" in str(info1))  | ("心电图未发现明显异常" in str(info1))) for info1 in col_temp ]
        new_value_3_2 = [(("大致正常心电图" in str(info1)) | ("窦房结" in str(info1))) for info1 in col_temp ]
        new_value_3_3 = [("窦性心律" in str(info1)) for info1 in col_temp ]
        new_value_3_5 = [("肢体导联低电压" in str(info1)) for info1 in col_temp ]
        new_value_3_6 = [(("冠状窦性心律" in str(info1))|("冠状静脉窦节律" in str(info1))|("冠状静脉窦心律" in str(info1))) for info1 in col_temp ]
        new_value_3_7 = [("窦性心律不齐" in str(info1)) for info1 in col_temp ]
        new_value_3_8 = [(("窦性心动过缓" in str(info1)) & ("正常心电图" not in str(info1))) for info1 in col_temp ]
        new_value_3_9 = [(("窦性心动过速" in str(info1)) & ("正常心电图" not in str(info1))) for info1 in col_temp ]
        new_value_3_10 = [("心电轴右偏" in str(info1)) for info1 in col_temp ]
        new_value_3_11 = [("心电轴左偏" in str(info1)) for info1 in col_temp ]
        new_value_3_12 = [(("交界性心律" in str(info1)) | ("交界性心动过速"  in str(info1))) for info1 in col_temp ]
        new_value_3_13 = [("窦性停搏" in str(info1)) for info1 in col_temp ]
        new_value_3_14 = [(("病窦综合征" in str(info1))| ("3S综合征" in str(info1))| ("SISIISIII综合征" in str(info1))) for info1 in col_temp ]
        new_value_3_15 = [("左房心律" in str(info1)) for info1 in col_temp ]        
        new_value_3_16 = [(("偶发房性早搏" in str(info1)) | ("偶发室性早搏" in str(info1))) for info1 in col_temp ]
        new_value_3_17 = [(("间位性室性早搏" in str(info1)) | ("交界性早搏"  in str(info1)) | ("房性早搏" in str(info1)) | ("室性早搏" in str(info1)) & ("偶发房性早搏" not in str(info1)) & ("偶发室性早搏" not in str(info1))) for info1 in col_temp ]
        new_value_3_18 = [(("频发房性早搏" in str(info1)) | ("频发室性早搏" in str(info1))) for info1 in col_temp ]
        new_value_3_19 = [("房扑" in str(info1)) for info1 in col_temp ]
        new_value_3_20 = [(("房颤" in str(info1)) | ("心房纤颤" in str(info1))) for info1 in col_temp ]
        new_value_3_21 = [(("起搏器心律" in str(info1))| ("起搏心律" in str(info1))) for info1 in col_temp ]
        new_value_3_4 = []
        for i in range(len(new_value_3_1)) :
            new_value_3_4.append( ( not new_value_3_1[i] ) and ( not new_value_3_2[i] ) and ( not new_value_3_3[i] ) and ( not new_value_3_5[i] ) and ( not new_value_3_6[i] )  and ( not new_value_3_6[i] )  and ( not new_value_3_7[i] )  and ( not new_value_3_8[i] )  and ( not new_value_3_9[i] )  and ( not new_value_3_10[i] )  and ( not new_value_3_11[i] )  and ( not new_value_3_12[i] )  and ( not new_value_3_13[i] )  and ( not new_value_3_14[i] )  and ( not new_value_3_15[i] )  and ( not new_value_3_16[i] )  and ( not new_value_3_17[i] )  and ( not new_value_3_18[i] )  and ( not new_value_3_19[i] )  and ( not new_value_3_20[i] )  and ( not new_value_3_21[i] ) and ( not new_value_empty[i] ))       

        data_input1_df['heart_rate']=0
        data_input1_df.loc[new_value_3_3,'heart_rate']  = 3
        data_input1_df.loc[new_value_3_1,'heart_rate']  = 1
        data_input1_df.loc[new_value_3_2,'heart_rate']  = 2       
        data_input1_df.loc[new_value_3_4,'heart_rate']  = 4
        data_input1_df.loc[new_value_3_5,'heart_rate']  = 5
        data_input1_df.loc[new_value_3_6,'heart_rate']  = 6
        data_input1_df.loc[new_value_3_7,'heart_rate']  = 7
        data_input1_df.loc[new_value_3_8,'heart_rate']  = 8
        data_input1_df.loc[new_value_3_9,'heart_rate']  = 9
        data_input1_df.loc[new_value_3_10,'heart_rate']  = 10
        data_input1_df.loc[new_value_3_11,'heart_rate']  = 11
        data_input1_df.loc[new_value_3_12,'heart_rate']  = 12
        data_input1_df.loc[new_value_3_13,'heart_rate']  = 13       
        data_input1_df.loc[new_value_3_14,'heart_rate']  = 14
        data_input1_df.loc[new_value_3_15,'heart_rate']  = 15
        data_input1_df.loc[new_value_3_16,'heart_rate']  = 16
        data_input1_df.loc[new_value_3_17,'heart_rate']  = 17
        data_input1_df.loc[new_value_3_18,'heart_rate']  = 18
        data_input1_df.loc[new_value_3_19,'heart_rate']  = 19
        data_input1_df.loc[new_value_3_20,'heart_rate']  = 20
        data_input1_df.loc[new_value_empty,'heart_rate']  = np.nan
           
        new_value_4_1 = [(("左前分支传导阻滞" in str(info1)) | (("左后分支传导阻滞" in str(info1)))) for info1 in col_temp ]
        new_value_4_2 = [("完全性左束支传导阻滞" in str(info1)) for info1 in col_temp ]
        new_value_4_3 = [("不完全性左束支传导阻滞" in str(info1)) for info1 in col_temp ]
        new_value_4_4 = [("完全性右束支传导阻滞" in str(info1)) for info1 in col_temp ]
        new_value_4_5 = [("不完全性右束支传导阻滞" in str(info1)) for info1 in col_temp ]
        data_input1_df['lbbb']=0
        data_input1_df.loc[new_value_4_1,'lbbb']  = 1
        data_input1_df.loc[new_value_4_2,'lbbb']  = 2
        data_input1_df.loc[new_value_4_3,'lbbb']  = 3
        data_input1_df.loc[new_value_4_4,'lbbb']  = 4
        data_input1_df.loc[new_value_4_5,'lbbb']  = 5
        data_input1_df.loc[new_value_empty,'lbbb']  = np.nan
        
        new_value_5_1 = [(("Ⅰ度房室传导阻滞" in str(info1)) | ("I度房室传导阻滞" in str(info1))) for info1 in col_temp ]
        new_value_5_2 = [(("Ⅱ度Ⅰ型房室传导阻滞" in str(info1)) | ("Ⅱ度房室传导阻滞" in str(info1))) for info1 in col_temp ]
        new_value_5_3 = [("Ⅱ度Ⅱ型窦房传导阻滞" in str(info1)) for info1 in col_temp ]
        new_value_5_4 = [(("Ⅲ度房室传导阻滞" in str(info1))) for info1 in col_temp ]
        new_value_5_5 = [(("非典型预激LG" in str(info1)) | ("w-p-w综合 征" in str(info1)) | ("预激综合征" in str(info1)) | ("心室预激" in str(info1)) | ("短P-R间期综合征" in str(info1)) ) for info1 in col_temp ]
        data_input1_df['atrioventricular_block']=0
        data_input1_df.loc[new_value_5_1,'atrioventricular_block']  = 1
        data_input1_df.loc[new_value_5_2,'atrioventricular_block']  = 2
        data_input1_df.loc[new_value_5_3,'atrioventricular_block']  = 3
        data_input1_df.loc[new_value_5_4,'atrioventricular_block']  = 4
        data_input1_df.loc[new_value_5_5,'atrioventricular_block']  = 5
        data_input1_df.loc[new_value_empty,'atrioventricular_block']  = np.nan
        
        data_input1_df['ecg']  = data_input1_df['avh']+data_input1_df['myocardium']+data_input1_df['lbbb']+data_input1_df['atrioventricular_block']
        data_input1_df['lbbb_atrioventricular']  = data_input1_df['lbbb']+data_input1_df['atrioventricular_block']
    return data_input1_df


##健康状态分类
def datCl_Healt(data_input1_df,colname_temp):

    if colname_temp in data_input1_df.columns:
        col_temp = data_input1_df[colname_temp]

        new_value_empty = [str(info1) == "NaN"  for info1 in col_temp]
        new_value_1 = [(("健康" in str(info1)) | ("正" in str(info1)) | ("阴" in str(info1)) | ("-" in str(info1)) | ("未见异常" in str(info1)) ) &
                       (("+" not in str(info1)) & ("肥" not in str(info1)) & ("亚" not in str(info1))) for info1 in col_temp ]
        new_value_2 = [(("疾病" in str(info1)) | ("+" in str(info1)) | ("阳" in str(info1)) ) & ("-" not in str(info1)) for info1 in col_temp ]
        new_value_3 = [("亚" in str(info1)) for info1 in col_temp ]
        new_value_confused = []
        for i in range(len(new_value_1)) :
            new_value_confused.append( ( not new_value_1[i] ) and ( not new_value_2[i] ) and ( not new_value_3[i] ) and ( not new_value_empty[i] ))

        data_input1_df.loc[new_value_1,colname_temp]  = 0
        data_input1_df.loc[new_value_2,colname_temp]  = 1
        data_input1_df.loc[new_value_3,colname_temp]  = 2
        data_input1_df.loc[new_value_confused,colname_temp]  = np.nan

    return data_input1_df

###阴阳性结果分类
def datCl_Degree(data_input1_df,colname_temp):

    if colname_temp in data_input1_df.columns:
        col_temp = data_input1_df[colname_temp]

        new_value_empty = [str(info1) == "NaN"   for info1 in col_temp]
        new_value_1 = [((str(info1).count('I') == 1) | (str(info1).count('Ⅰ') == 1) | (str(info1).count('i') == 1) ) &
                       ((str(info1).count('V') < 1) & (str(info1).count('v') < 1) )  for info1 in col_temp ]
        new_value_2 = [((str(info1).count('I') == 2) | (str(info1).count('Ⅰ') == 2) | (str(info1).count('i') == 2) |
                        (str(info1).count('Ⅱ') == 1) | (str(info1).count('II') == 1)| (str(info1).count('ii') == 1)) &
                       ((str(info1).count('V') < 1) & (str(info1).count('v') < 1) )  for info1 in col_temp ]
        new_value_3 = [((str(info1).count('I') == 3) | (str(info1).count('Ⅰ') == 3) | (str(info1).count('i') == 3) |
                        (str(info1).count('Ⅲ') == 1) | (str(info1).count('III') == 1)| (str(info1).count('iii') == 1)) &
                       ((str(info1).count('V') < 1) & (str(info1).count('v') < 1) )  for info1 in col_temp ]
        new_value_4 = [(("Ⅳ" in str(info1)) |
                        ("Ⅰv" in str(info1)) | ("ⅠV" in str(info1)) |
                        ("Iv" in str(info1)) | ("IV" in str(info1)) | ("iv" in str(info1)) | ("iV" in str(info1)) ) for info1 in col_temp ]

        new_value_2_1 = [(("normal" in str(info1).lower()) | ("未" in str(info1)) | ("正" in str(info1)) | ("阴" in str(info1)) | ("-" in str(info1)) | ("未见" in str(info1)) ) & ("+" not in str(info1)) for info1 in col_temp ]
        new_value_2_2 = [(("+" in str(info1)) | ("阳" in str(info1)) )  for info1 in col_temp ]


        new_value_confused = []
        for i in range(len(new_value_1)) :
            new_value_confused.append( ( not new_value_1[i] ) and ( not new_value_2[i] ) and ( not new_value_3[i] ) and ( not new_value_4[i] ) and
                                      ( not new_value_2_1[i] ) and ( not new_value_2_2[i] ) and ( not new_value_empty[i] ))

        data_input1_df.loc[new_value_1,colname_temp]  = 1
        data_input1_df.loc[new_value_2,colname_temp]  = 2
        data_input1_df.loc[new_value_3,colname_temp]  = 3
        data_input1_df.loc[new_value_4,colname_temp]  = 4
        data_input1_df.loc[new_value_2_1,colname_temp]  = 0
        data_input1_df.loc[new_value_2_2,colname_temp]  = 1
        data_input1_df.loc[new_value_confused,colname_temp]  = np.nan

    return data_input1_df

####阴阳性分等级3194
def datCl_Degree2(data_input1_df,colname_temp):
    if colname_temp in data_input1_df.columns:
        col_temp = data_input1_df[colname_temp]
        new_value_empty = [str(info1) == "NaN"   for info1 in col_temp]
        new_value_1 = [((str(info1)=="0-1")| (str(info1)=="0个/LP")| (str(info1)=="0-2")| (str(info1)=="0-2")| (str(info1)=="0-1/HP")| (str(info1)=="1-3/HP")| (str(info1)=="0-1/lp")| (str(info1)=="0")| (str(info1)=="1")| (str(info1)=="2")| (str(info1)=="3")|("偶见" in str(info1))| ("未见" in str(info1))| ("无" in str(info1))| ("未检出" in str(info1)) | ("Normal" in str(info1))| ("正常" in str(info1))|(str(info1)=="4") | ("-" in str(info1)) | ("阴性" in str(info1)) )  for info1 in col_temp ]
        new_value_2 = [(("1-3/HP" in str(info1))|("3-4/HP" in str(info1))|("+-" in str(info1))|("查见" in str(info1))|("少见" in str(info1))|("少数" in str(info1)))   for info1 in col_temp ]
        new_value_3 = [(("+" in str(info1))|("检出" in str(info1))|("检到" in str(info1))|("阳性" in str(info1)))   for info1 in col_temp ]
        new_value_4 = [(("２＋" in str(info1))|("+2" in str(info1))|("2+" in str(info1))|("++" in str(info1))|("透明" in str(info1)))  for info1 in col_temp ]
        new_value_5 = [(("+3" in str(info1))|("3+" in str(info1))|("+++" in str(info1))|("颗粒管型" in str(info1)) | (str(info1)=="565"))   for info1 in col_temp ]
        new_value_6 = [(("+4" in str(info1))|("4+" in str(info1))|("++++" in str(info1)))   for info1 in col_temp ]
        new_value_7 = [(("见刮片" in str(info1))|("未做" in str(info1))|("见TCT" in str(info1)))   for info1 in col_temp ]
        new_value_confused = []
        for i in range(len(new_value_1)) :
            new_value_confused.append( ( not new_value_1[i] ) and ( not new_value_2[i] ) and ( not new_value_3[i] ) and ( not new_value_4[i] ) and
                                      ( not new_value_5[i] ) and ( not new_value_6[i] )and ( not new_value_7[i] ) and ( not new_value_empty[i] ))
        data_input1_df.loc[new_value_1,colname_temp]  = 0
        data_input1_df.loc[new_value_2,colname_temp]  = 1
        data_input1_df.loc[new_value_3,colname_temp]  = 2
        data_input1_df.loc[new_value_4,colname_temp]  = 3
        data_input1_df.loc[new_value_5,colname_temp]  = 4
        data_input1_df.loc[new_value_6,colname_temp]  = 5
        data_input1_df.loc[new_value_confused,colname_temp]  = 0
        data_input1_df.loc[new_value_empty,colname_temp]  = np.nan
        data_input1_df.loc[new_value_7,colname_temp]  = np.nan
    return data_input1_df
     
        
def datCl_Bone(data_input1_df,colname_temp):

    if colname_temp in data_input1_df.columns:
        col_temp = data_input1_df[colname_temp]

        new_value_empty = [((str(info1) == "NaN" ) | ("详见" in str(info1))) for info1 in col_temp]
        new_value_1 = [("正常" in str(info1)) for info1 in col_temp ]
        new_value_2 = [("轻度骨量减少" in str(info1)) for info1 in col_temp ]
        new_value_3 = [(("骨量减少" in str(info1)) & ("轻度骨量减少" not in str(info1))) for info1 in col_temp ]
        new_value_4 = [("中度骨量减少" in str(info1)) for info1 in col_temp ]
        new_value_5 = [("重度骨量减少" in str(info1)) for info1 in col_temp ]
        new_value_6 = [(("疏松" in str(info1)) | ("骨质减少" in str(info1)) & ("骨量减少" not in str(info1))) for info1 in col_temp ]
        new_value_7 = [("严重骨质疏松" in str(info1)) for info1 in col_temp ]
        new_value_8 = [(("骨密度降低" in str(info1)) & ("骨量减少" not in str(info1))) for info1 in col_temp ]       
        data_input1_df.loc[new_value_1,colname_temp]  = 0
        data_input1_df.loc[new_value_2,colname_temp]  = 1
        data_input1_df.loc[new_value_3,colname_temp]  = 2
        data_input1_df.loc[new_value_4,colname_temp]  = 3
        data_input1_df.loc[new_value_5,colname_temp]  = 4
        data_input1_df.loc[new_value_6,colname_temp]  = 5
        data_input1_df.loc[new_value_7,colname_temp]  = 6
        data_input1_df.loc[new_value_8,colname_temp]  = 7
        new_value_confused = []
        for i in range(len(new_value_1)) :
            new_value_confused.append( ( not new_value_1[i] ) and ( not new_value_2[i] ) and ( not new_value_3[i] ) and ( not new_value_4[i] ) and
                                      ( not new_value_5[i] ) and ( not new_value_6[i] ) and ( not new_value_empty[i] ))
        data_input1_df.loc[new_value_empty,colname_temp]  = np.nan
        data_input1_df.loc[new_value_confused,colname_temp]  = 0

    return data_input1_df

##病史
def datCl_disease(data_input1_df,colname_temp):
    data_input1_df['hbp']=0
    data_input1_df['hyperlipidemia']=0
    if colname_temp in data_input1_df.columns:
        col_temp =data_input1_df[colname_temp]
        new_value_empty = [("NaN" in str(info1)) for info1 in col_temp]
        new_value_1 = [(("糖尿病" in str(info1)) | ("血糖偏高"  in str(info1)) | (("血糖"  in str(info1)) & ("均偏高"  in str(info1)))) for info1 in col_temp ]
        new_value_2 = [("脑供血不足" in str(info1)) for info1 in col_temp ]
        new_value_3 = [("脑梗塞" in str(info1)) for info1 in col_temp ]
        new_value_4 = [(("脑瘤" in str(info1)) | ("脑血管"  in str(info1)) | ("脑溢血"  in str(info1)) | ("脑血栓"  in str(info1))) for info1 in col_temp ]
        new_value_5 = [(("血压偏高" in str(info1)) | (("血压"  in str(info1)) & ("均偏高"  in str(info1)))) for info1 in col_temp ]       
        new_value_6 = [("冠心病" in str(info1)) for info1 in col_temp ]
        new_value_7 = [("高血压" in str(info1)) for info1 in col_temp ]
        new_value_8 = [(("高血压" in str(info1)) & ("冠心病" in str(info1))) for info1 in col_temp ]
        new_value_9 = [(("高血压" in str(info1)) & ("脑血栓" in str(info1))) for info1 in col_temp ]
        new_value_10 = [(("高血压" in str(info1)) & ("脑梗塞" in str(info1))) for info1 in col_temp ]
        new_value_11 = [(("高血压" in str(info1)) & ("脑溢血" in str(info1))) for info1 in col_temp ]
        
        new_value_11 = [("肾" in str(info1)) for info1 in col_temp ]
        new_value_12 = [("甲状腺" in str(info1)) for info1 in col_temp ]
        new_value_13 = [(("心脏杂音" in str(info1)) |("心动过缓" in str(info1)) |("心动过速" in str(info1)) |("心肌炎" in str(info1)) | ("心肌供血" in str(info1)) | ("心律不齐" in str(info1)) | ("早搏" in str(info1))) for info1 in col_temp ]
        new_value_14 = [(("心肌梗塞" in str(info1))|("心脏病" in str(info1))| ("瓣膜病变史二尖瓣" in str(info1)) | ("房颤" in str(info1))) for info1 in col_temp ]
        new_value_15 = [("甲肝史" in str(info1)) for info1 in col_temp ]
        new_value_16 = [(("肺癌" in str(info1))|("结肠癌" in str(info1))) for info1 in col_temp ]
        new_value_17 = [("脂肪肝" in str(info1)) for info1 in col_temp ] 
        new_value_18 = [(("肝部分切除" in str(info1)) |  ("肝硬化史" in str(info1)) |  ("肝肿瘤" in str(info1))) for info1 in col_temp ]
        new_value_19 = [("冠心病" in str(info1)) for info1 in col_temp ]
        new_value_20 = [(("冠状动脉支架植入" in str(info1))|("冠状动脉搭桥" in str(info1))) for info1 in col_temp ]
        new_value_21 = [("心脏起搏器" in str(info1)) for info1 in col_temp ]
        new_value_22 = [(("血脂偏高"  in str(info1)) | (("血脂"  in str(info1)) & ("均偏高"  in str(info1)))) for info1 in col_temp ]              
        new_value_23 = [("高血脂" in str(info1)) for info1 in col_temp ]
        new_value_24 = [(("高血脂" in str(info1)) & ("冠心病" in str(info1))) for info1 in col_temp ]
                
        data_input1_df.loc[new_value_1,'hbp']  =1
        data_input1_df.loc[new_value_2,'hbp']  =2
        data_input1_df.loc[new_value_3,'hbp']  =3
        data_input1_df.loc[new_value_4,'hbp']  =4
        data_input1_df.loc[new_value_5,'hbp']  =5
        data_input1_df.loc[new_value_6,'hbp']  =6
        data_input1_df.loc[new_value_7,'hbp']  =7
        data_input1_df.loc[new_value_8,'hbp']  =8
        data_input1_df.loc[new_value_9,'hbp']  =9
        data_input1_df.loc[new_value_10,'hbp']  =10
        
        data_input1_df.loc[new_value_11,'hyperlipidemia']  =1
        data_input1_df.loc[new_value_12,'hyperlipidemia']  =2
        data_input1_df.loc[new_value_13,'hyperlipidemia']  =3
        data_input1_df.loc[new_value_14,'hyperlipidemia']  =4
        data_input1_df.loc[new_value_15,'hyperlipidemia']  =5
        data_input1_df.loc[new_value_16,'hyperlipidemia']  =6
        data_input1_df.loc[new_value_17,'hyperlipidemia']  =7
        data_input1_df.loc[new_value_18,'hyperlipidemia']  =8
        data_input1_df.loc[new_value_19,'hyperlipidemia']  =9
        data_input1_df.loc[new_value_20,'hyperlipidemia']  =10
        data_input1_df.loc[new_value_21,'hyperlipidemia']  =11
        data_input1_df.loc[new_value_22,'hyperlipidemia']  =12
        data_input1_df.loc[new_value_23,'hyperlipidemia']  =13
        data_input1_df.loc[new_value_24,'hyperlipidemia']  =14
        
        data_input1_df.loc[new_value_empty,'hbp']  = np.nan
        data_input1_df.loc[new_value_empty,'hyperlipidemia']  = np.nan
    return data_input1_df

##脂肪肝 、甲状腺、息肉  102     
def datCl_FLD(data_input1_df,colname_temp):
    data_input1_df['fld']=0
    data_input1_df['thyroid']=0
    data_input1_df['gps']=0
    data_input1_df['cc']=0
    data_input1_df['renal_cyst']=0
    data_input1_df['kidney_stone']=0
    data_input1_df['sponge_kidney']=0
    data_input1_df['liver_cyst']=0
    data_input1_df['liver_calcification']=0
    data_input1_df['hch']=0
    if colname_temp in data_input1_df.columns:
        col_temp =data_input1_df[colname_temp]
        new_value_empty = [str(info1) == "NaN"  for info1 in col_temp]
        new_value_1 = [("脂肪肝（轻度）" in str(info1)) for info1 in col_temp ]
        new_value_2 = [("脂肪肝（中度）" in str(info1)) for info1 in col_temp ]
        new_value_3 = [("脂肪肝（重度）" in str(info1)) for info1 in col_temp ]
        new_value_4 = [(("甲状腺" in str(info1)) & ("结节" in str(info1))) for info1 in col_temp ]
        new_value_5 = [("胆囊息肉" in str(info1)) for info1 in col_temp ]
        new_value_6 = [("胆囊结石" in str(info1)) for info1 in col_temp ]
        new_value_6_2 = [("胆囊炎" in str(info1)) for info1 in col_temp ]
        new_value_6_3 = [("胆固醇结晶" in str(info1)) for info1 in col_temp ]
        new_value_7 = [("肾囊肿" in str(info1)) for info1 in col_temp ]
        new_value_8 = [("肾结石" in str(info1)) for info1 in col_temp ]
        new_value_8_1 = [("肾结晶" in str(info1)) for info1 in col_temp ]
        new_value_8_2 = [("海绵肾" in str(info1)) for info1 in col_temp ]
        new_value_9 = [("肝囊肿" in str(info1)) for info1 in col_temp ]
        new_value_10 = [("肝血管瘤" in str(info1)) for info1 in col_temp ]
        new_value_11 = [("肝内钙化" in str(info1)) for info1 in col_temp ]
        data_input1_df.loc[new_value_1,'fld']  =1
        data_input1_df.loc[new_value_2,'fld']  =2
        data_input1_df.loc[new_value_3,'fld']  =3
        data_input1_df.loc[new_value_4,'thyroid']  =1
        data_input1_df.loc[new_value_5,'gps']  =1
        data_input1_df.loc[new_value_6,'gps']  =2
        data_input1_df.loc[new_value_6_2,'gps']  =3
        data_input1_df.loc[new_value_6_3,'cc']  =1
        data_input1_df.loc[new_value_7,'renal_cyst']  =1
        data_input1_df.loc[new_value_8,'kidney_stone']  =2
        data_input1_df.loc[new_value_8_1,'kidney_stone']  =1
        data_input1_df.loc[new_value_8_2,'sponge_kidney']  =1
        data_input1_df.loc[new_value_9,'liver_cyst']  =1
        data_input1_df.loc[new_value_10,'hch']  =1
        data_input1_df.loc[new_value_11,'liver_calcification']  =1
        data_input1_df.loc[new_value_empty,'fld']  =np.nan
        data_input1_df.loc[new_value_empty,'thyroid']  =np.nan
        data_input1_df.loc[new_value_empty,'gps']  =np.nan
        data_input1_df.loc[new_value_empty,'cc']  =np.nan
        data_input1_df.loc[new_value_empty,'renal_cyst']  =np.nan
        data_input1_df.loc[new_value_empty,'kidney_stone']  =np.nan
        data_input1_df.loc[new_value_empty,'sponge_kidney']  =np.nan
        data_input1_df.loc[new_value_empty,'liver_cyst']  =np.nan
        data_input1_df.loc[new_value_empty,'hch']  =np.nan
        data_input1_df.loc[new_value_empty,'liver_calcification']  =np.nan
    return data_input1_df

###动脉硬化102 
def datCl_FLD2(data_input1_df,colname_temp):
    data_input1_df['carotid']=0
    data_input1_df['arteriosclerosis_102']=0
    data_input1_df['aortic_stenosis']=0
    data_input1_df['aortic_insufficiency']=0
    data_input1_df['aortic_regurgitation']=0
    data_input1_df['tricuspid_regurgitation']=0
    data_input1_df['MR_VTI']=0
    data_input1_df['pulmonary_regurgitaion']=0
    
    if colname_temp in data_input1_df.columns:
        
        col_temp =data_input1_df[colname_temp]
        new_value_empty = [str(info1) == "NaN"  for info1 in col_temp]
        new_value_1_1 = [(("颈动脉硬化" in str(info1))|("颈总动脉硬化" in str(info1))|("颈动脉粥样硬化" in str(info1))) for info1 in col_temp ]
        new_value_1_2 = [("颈总动脉斑块" in str(info1)) for info1 in col_temp ]
        new_value_1_3 = [(("颈总动脉硬化伴斑块" in str(info1))|("颈动脉粥样硬化多发斑块" in str(info1))|("颈动脉粥样硬化伴斑块" in str(info1))) for info1 in col_temp ]
        new_value_2_1 = [("主动脉轻度硬化" in str(info1)) for info1 in col_temp ]
        new_value_2_2 = [("主动脉中度硬化" in str(info1)) for info1 in col_temp ]
        new_value_2_3 = [("主动脉硬化" in str(info1)) for info1 in col_temp ]
        new_value_2_4 = [("主动脉钙化" in str(info1)) for info1 in col_temp ]
        
        data_input1_df.loc[new_value_1_1,'carotid']  =1
        data_input1_df.loc[new_value_1_2,'carotid']  =2
        data_input1_df.loc[new_value_1_3,'carotid']  =3
        data_input1_df.loc[new_value_empty,'carotid']  =np.nan
        
        data_input1_df.loc[new_value_2_1,'arteriosclerosis_102']  =1
        data_input1_df.loc[new_value_2_2,'arteriosclerosis_102']  =2
        data_input1_df.loc[new_value_2_3,'arteriosclerosis_102']  =3
        data_input1_df.loc[new_value_2_4,'arteriosclerosis_102']  =4
        data_input1_df.loc[new_value_empty,'arteriosclerosis_102']  =np.nan
        
        new_value_3_1 = [("主动脉瓣狭窄(轻度)" in str(info1)) for info1 in col_temp ]
        new_value_3_2 = [("主动脉瓣狭窄(轻-中度)" in str(info1)) for info1 in col_temp ]
        new_value_3_3 = [("主动脉瓣狭窄(中度)" in str(info1)) for info1 in col_temp ]
        new_value_3_4 = [("主动脉瓣狭窄(中-重度)" in str(info1)) for info1 in col_temp ]
        new_value_3_5 = [("主动脉瓣狭窄(重度)" in str(info1)) for info1 in col_temp ]       
        
        data_input1_df.loc[new_value_3_1,'aortic_stenosis']  =1
        data_input1_df.loc[new_value_3_2,'aortic_stenosis']  =2
        data_input1_df.loc[new_value_3_3,'aortic_stenosis']  =3
        data_input1_df.loc[new_value_3_4,'aortic_stenosis']  =4
        data_input1_df.loc[new_value_3_5,'aortic_stenosis']  =5
        data_input1_df.loc[new_value_empty,'aortic_stenosis']  =np.nan
                
        new_value_4_1 = [("主动脉瓣关闭不全(轻度)" in str(info1)) for info1 in col_temp ]
        new_value_4_2 = [("主动脉瓣关闭不全(轻-中度)" in str(info1)) for info1 in col_temp ]
        new_value_4_3 = [("主动脉瓣关闭不全(中度)" in str(info1)) for info1 in col_temp ]
        new_value_4_4 = [("主动脉瓣关闭不全(中-重度)" in str(info1)) for info1 in col_temp ]
        new_value_4_5 = [("主动脉瓣关闭不全(重度)" in str(info1)) for info1 in col_temp ]
        
        data_input1_df.loc[new_value_4_1,'aortic_insufficiency']  =1
        data_input1_df.loc[new_value_4_2,'aortic_insufficiency']  =2
        data_input1_df.loc[new_value_4_3,'aortic_insufficiency']  =3
        data_input1_df.loc[new_value_4_4,'aortic_insufficiency']  =4
        data_input1_df.loc[new_value_4_5,'aortic_insufficiency']  =5
        data_input1_df.loc[new_value_empty,'aortic_insufficiency']  =np.nan
        
        new_value_5_1 = [("主动脉瓣少量反流" in str(info1)) for info1 in col_temp ]
        new_value_5_2 = [("主动脉瓣反流(轻度)" in str(info1)) for info1 in col_temp ]
        new_value_5_3 = [("主动脉瓣反流(轻-中度)" in str(info1)) for info1 in col_temp ]
        new_value_5_4 = [("主动脉瓣反流(中度)" in str(info1)) for info1 in col_temp ]
        new_value_5_5 = [("主动脉瓣反流(中-重度)" in str(info1)) for info1 in col_temp ]
        new_value_5_6 = [("主动脉瓣反流(重度)" in str(info1)) for info1 in col_temp ]
        
        data_input1_df.loc[new_value_5_1,'aortic_regurgitation']  =1
        data_input1_df.loc[new_value_5_2,'aortic_regurgitation']  =2
        data_input1_df.loc[new_value_5_3,'aortic_regurgitation']  =3
        data_input1_df.loc[new_value_5_4,'aortic_regurgitation']  =4
        data_input1_df.loc[new_value_5_5,'aortic_regurgitation']  =5
        data_input1_df.loc[new_value_5_6,'aortic_regurgitation']  =6
        data_input1_df.loc[new_value_empty,'aortic_regurgitation']  =np.nan
        
        new_value_6_1 = [("三尖瓣少量反流" in str(info1)) for info1 in col_temp ]
        new_value_6_2 = [("三尖瓣反流(轻度)" in str(info1)) for info1 in col_temp ]
        new_value_6_3 = [("三尖瓣反流(轻-中度)" in str(info1)) for info1 in col_temp ]
        new_value_6_4 = [("三尖瓣反流(中度)" in str(info1)) for info1 in col_temp ]
        new_value_6_5 = [("三尖瓣反流(中-重度)" in str(info1)) for info1 in col_temp ]
        new_value_6_6 = [("三尖瓣反流(重度)" in str(info1)) for info1 in col_temp ]
        
        data_input1_df.loc[new_value_6_1,'tricuspid_regurgitation']  =1
        data_input1_df.loc[new_value_6_2,'tricuspid_regurgitation']  =2
        data_input1_df.loc[new_value_6_3,'tricuspid_regurgitation']  =3
        data_input1_df.loc[new_value_6_4,'tricuspid_regurgitation']  =4
        data_input1_df.loc[new_value_6_5,'tricuspid_regurgitation']  =5
        data_input1_df.loc[new_value_6_6,'tricuspid_regurgitation']  =6
        data_input1_df.loc[new_value_empty,'tricuspid_regurgitation']  =np.nan
        
        new_value_7_1 = [("二尖瓣少量反流" in str(info1)) for info1 in col_temp ]
        new_value_7_2 = [("二尖瓣反流(轻度)" in str(info1)) for info1 in col_temp ]
        new_value_7_3 = [("二尖瓣反流(轻-中度)" in str(info1)) for info1 in col_temp ]
        new_value_7_4 = [("二尖瓣反流(中度)" in str(info1)) for info1 in col_temp ]
        new_value_7_5 = [("二尖瓣反流(中-重度)" in str(info1)) for info1 in col_temp ]
        new_value_7_6 = [("二尖瓣反流(重度)" in str(info1)) for info1 in col_temp ]
        
        data_input1_df.loc[new_value_7_1,'MR_VTI']  =1
        data_input1_df.loc[new_value_7_2,'MR_VTI']  =2
        data_input1_df.loc[new_value_7_3,'MR_VTI']  =3
        data_input1_df.loc[new_value_7_4,'MR_VTI']  =4
        data_input1_df.loc[new_value_7_5,'MR_VTI']  =5
        data_input1_df.loc[new_value_7_6,'MR_VTI']  =6
        data_input1_df.loc[new_value_empty,'MR_VTI']  =np.nan
        
        new_value_8_1 = [("肺动脉瓣少量反流" in str(info1)) for info1 in col_temp ]
        new_value_8_2 = [("肺动脉瓣反流(轻度)" in str(info1)) for info1 in col_temp ]
        new_value_8_3 = [("肺动脉瓣反流(轻-中度)" in str(info1)) for info1 in col_temp ]
        new_value_8_4 = [("肺动脉瓣反流(中度)" in str(info1)) for info1 in col_temp ]
        new_value_8_5 = [("肺动脉瓣反流(中-重度)" in str(info1)) for info1 in col_temp ]
        new_value_8_6 = [("肺动脉瓣反流(重度)" in str(info1)) for info1 in col_temp ]
        
        data_input1_df.loc[new_value_8_1,'pulmonary_regurgitaion']  =1
        data_input1_df.loc[new_value_8_2,'pulmonary_regurgitaion']  =2
        data_input1_df.loc[new_value_8_3,'pulmonary_regurgitaion']  =3
        data_input1_df.loc[new_value_8_4,'pulmonary_regurgitaion']  =4
        data_input1_df.loc[new_value_8_5,'pulmonary_regurgitaion']  =5
        data_input1_df.loc[new_value_8_6,'pulmonary_regurgitaion']  =6
        data_input1_df.loc[new_value_empty,'pulmonary_regurgitaion']  =np.nan
    return data_input1_df    
    
## CT,第一列A201，第二列A202
def datCl_CT(data_input1_df,colname_temp1,colname_temp2):
    data_input1_df['abi']=0
    data_input1_df['ao_a202']=0
    data_input1_df['coronary_a202']=0
    if colname_temp1 in data_input1_df.columns:        
        col_temp =data_input1_df[colname_temp1]
        new_value_empty = [str(info1) == "NaN"  for info1 in col_temp]
        new_value_1 = [("结石" in str(info1)) for info1 in col_temp ]
        new_value_2 = [("低密度影" in str(info1)) for info1 in col_temp ]
        new_value_3 = [("高密度影" in str(info1)) for info1 in col_temp ]
        new_value_confused = []
        for i in range(len(new_value_1)) :
            new_value_confused.append( ( not new_value_1[i] ) and ( not new_value_2[i] ) and ( not new_value_3[i] ) and ( not new_value_empty[i] ))
        data_input1_df.loc[new_value_1,colname_temp1]  = 1
        data_input1_df.loc[new_value_2,colname_temp1]  = 2
        data_input1_df.loc[new_value_3,colname_temp1]  = 3
        data_input1_df.loc[new_value_confused,colname_temp1]  = 4       
    if colname_temp2 in data_input1_df.columns:        
        col_temp =data_input1_df[colname_temp2]
        new_value_empty = [str(info1) == "NaN"  for info1 in col_temp]
        new_value_1 = [("脑梗塞" in str(info1)) for info1 in col_temp ]
        
        new_value_2_1 = [(("主动脉硬化" in str(info1))|("主动脉及冠状动脉硬化" in str(info1))) for info1 in col_temp ]
        new_value_2_2 = [(("主动脉钙化" in str(info1))|("主动脉少许钙化" in str(info1))|("主动脉弓钙化" in str(info1))|("主动脉钙化" in str(info1))) for info1 in col_temp ]
        new_value_2_3 = [(("主动脉壁及冠状动脉局部钙化" in str(info1)) | ("主动脉管壁钙化" in str(info1)) | ("主动脉、头臂干及冠状动脉管壁钙化" in str(info1))|("主动脉及冠状动脉管壁钙化" in str(info1))|("主动脉、冠状动脉管壁钙化" in str(info1))) for info1 in col_temp ]        
        
        new_value_3_1 = [(("冠状动脉硬化" in str(info1)) |("主动脉及冠状动脉硬化" in str(info1))) for info1 in col_temp ]               
        new_value_3_2 = [(("冠状动脉局部钙化" in str(info1)) |("主动脉少许钙化" in str(info1))|("冠状动脉钙化" in str(info1))) for info1 in col_temp ]
        new_value_3_3 = [(("主动脉壁及冠状动脉局部钙化" in str(info1)) | ("主动脉、头臂干及冠状动脉管壁钙化" in str(info1))|("主动脉及冠状动脉管壁钙化" in str(info1))|("主动脉、冠状动脉管壁钙化" in str(info1))|("冠状动脉管壁钙化" in str(info1))) for info1 in col_temp ]
        
        data_input1_df.loc[new_value_1,'abi']  = 1
        data_input1_df.loc[new_value_2_1,'ao_a202']  = 1
        data_input1_df.loc[new_value_2_2,'ao_a202']  = 2
        data_input1_df.loc[new_value_2_3,'ao_a202']  = 3
        data_input1_df.loc[new_value_3_1,'coronary_a202']  = 1
        data_input1_df.loc[new_value_3_2,'coronary_a202']  = 2
        data_input1_df.loc[new_value_3_3,'coronary_a202']  = 3
        data_input1_df.loc[new_value_empty,'abi']  = np.nan
        data_input1_df.loc[new_value_empty,'ao_a202']  = np.nan
        data_input1_df.loc[new_value_empty,'coronary_a202']  = np.nan
        data_input1_df['ct_head']  = data_input1_df['abi']+data_input1_df['ao_a202']+data_input1_df['coronary_a202']
    return data_input1_df    
        
###B超回声判断 肝、肾   113 114 115 116 117+118
def datCl_echo(data_input1_df,colname_temp):
    
    if colname_temp in data_input1_df.columns:
        col_temp =data_input1_df[colname_temp]
        new_value_empty = [str(info1) == "NaN"  for info1 in col_temp]
        new_value_1 = [("高回声" in str(info1)) for info1 in col_temp ]
        new_value_2 = [("中等回声" in str(info1)) for info1 in col_temp ]
        new_value_3 = [("低回声" in str(info1)) for info1 in col_temp ]
        new_value_4 = [("无回声" in str(info1)) for info1 in col_temp ]
        new_value_confused = []
        for i in range(len(new_value_1)) :
            new_value_confused.append( ( not new_value_1[i] ) and ( not new_value_empty[i] ))
        data_input1_df.loc[new_value_1,colname_temp]  = 1
        data_input1_df.loc[new_value_2,colname_temp]  = 2
        data_input1_df.loc[new_value_3,colname_temp]  = 3
        data_input1_df.loc[new_value_4,colname_temp]  = 4
        data_input1_df.loc[new_value_confused,colname_temp]  = 0
    return data_input1_df

###心房血流 101
def datCl_blood(data_input1_df,colname_temp):
    data_input1_df['bffs']=0   
    if colname_temp in data_input1_df.columns:
        col_temp =data_input1_df[colname_temp]
        new_value_empty = [str(info1) == "NaN"  for info1 in col_temp]
        new_value_1 = [(("A峰>E峰" in str(info1)) | ("A峰>E" in str(info1)) | ("A>E" in str(info1)) | ("E峰<A峰" in str(info1)) | ("E<A峰" in str(info1)) | ("E<A" in str(info1))) for info1 in col_temp ]
        new_value_2 = [(("E峰>A峰" in str(info1)) | ("E>A峰" in str(info1)) | ("E>A" in str(info1))) for info1 in col_temp ]
        new_value_confused = []
        for i in range(len(new_value_1)) :
            new_value_confused.append( ( not new_value_1[i] ) and ( not new_value_empty[i] ))
        data_input1_df.loc[new_value_1,'bffs']  = 2
        data_input1_df.loc[new_value_2,'bffs']  = 1
        data_input1_df.loc[new_value_confused,'bffs']  = 0
        data_input1_df.loc[new_value_empty,'bffs']  = np.nan
    return data_input1_df

###血流速度 1402
def datCl_CBFV(data_input1_df,colname_temp):
    data_input1_df['BA_CBFV']=0
    data_input1_df['MCA_CBFV']=0
    if colname_temp in data_input1_df.columns:
        col_temp =data_input1_df[colname_temp]
        new_value_empty = [str(info1) == "NaN"  for info1 in col_temp]
        new_value_1 = [("基底动脉血流速度减慢" in str(info1)) for info1 in col_temp ]
        new_value_2 = [("基底动脉血流速度略减慢" in str(info1)) for info1 in col_temp ]
        new_value_3 = [("基底动脉血流速度略增快" in str(info1)) for info1 in col_temp ]
        new_value_4 = [("基底动脉血流速度增快" in str(info1)) for info1 in col_temp ]
        new_value_5 = [("大脑中动脉血流速度减慢" in str(info1)) for info1 in col_temp ]
        new_value_6 = [("大脑中动脉血流速度略减慢" in str(info1)) for info1 in col_temp ]
        new_value_7 = [("大脑中动脉血流速度略增快" in str(info1)) for info1 in col_temp ]
        new_value_8 = [(("大脑中动脉血流速度增快" in str(info1))|("双侧大脑中动脉、颈内动脉末端、大脑前动脉血流速度增快" in str(info1))) for info1 in col_temp ]
        new_value_confused1 = []
        new_value_confused2 = []
        for i in range(len(new_value_1)) :
            new_value_confused1.append( ( not new_value_1[i] ) and ( not new_value_2[i] ) and ( not new_value_3[i] ) and ( not new_value_4[i] ) and ( not new_value_empty[i] ))
        data_input1_df.loc[new_value_1,'BA_CBFV']  = 1
        data_input1_df.loc[new_value_2,'BA_CBFV']  = 2
        data_input1_df.loc[new_value_3,'BA_CBFV']  = 3
        data_input1_df.loc[new_value_4,'BA_CBFV']  = 4
        data_input1_df.loc[new_value_empty,'BA_CBFV']  = np.nan
        
        for i in range(len(new_value_5)) :
            new_value_confused2.append( ( not new_value_5[i] ) and ( not new_value_6[i] ) and ( not new_value_7[i] ) and ( not new_value_8[i] ) and ( not new_value_empty[i] ))       
        data_input1_df.loc[new_value_5,'MCA_CBFV']  = 1
        data_input1_df.loc[new_value_6,'MCA_CBFV']  = 2
        data_input1_df.loc[new_value_7,'MCA_CBFV']  = 3
        data_input1_df.loc[new_value_8,'MCA_CBFV']  = 4
        data_input1_df.loc[new_value_empty,'MCA_CBFV']  = np.nan
        
    return data_input1_df

###血管弹性 4001
def datCl_Vasoactivity(data_input1_df,colname_temp):
    data_input1_df['vasoactivity']=0
    data_input1_df['arteriosclerosis']=0
    if colname_temp in data_input1_df.columns:
        col_temp =data_input1_df[colname_temp]
        new_value_empty = [str(info1) == "NaN"  for info1 in col_temp]
        new_value_1 = [(("血管弹性度正常" in str(info1)) | ("血管弹性良好" in str(info1))) for info1 in col_temp ]
        new_value_2 = [("有血管弹性减弱趋势" in str(info1)) for info1 in col_temp ]
        new_value_3 = [("临界血管弹性减弱" in str(info1)) for info1 in col_temp ]
        new_value_4 = [("血管弹性轻度减弱" in str(info1)) for info1 in col_temp ]
        new_value_5 = [("血管弹性中度减弱" in str(info1)) for info1 in col_temp ]
        new_value_6 = [("血管弹性重度减弱" in str(info1)) for info1 in col_temp ]
        new_value_7 = [("血管钙化可能" in str(info1)) for info1 in col_temp ]
        
        new_value_8 = [("动脉硬化可能" in str(info1)) for info1 in col_temp ]
        new_value_9 = [("动脉轻度硬化可能" in str(info1)) for info1 in col_temp ]
        new_value_10 = [("动脉硬化" in str(info1)) for info1 in col_temp ]
        new_value_11 = [("动脉重度硬化" in str(info1)) for info1 in col_temp ]
        

        data_input1_df.loc[new_value_1,'vasoactivity']  = 0
        data_input1_df.loc[new_value_2,'vasoactivity']  = 1
        data_input1_df.loc[new_value_3,'vasoactivity']  = 2
        data_input1_df.loc[new_value_4,'vasoactivity']  = 3
        data_input1_df.loc[new_value_5,'vasoactivity']  = 4
        data_input1_df.loc[new_value_6,'vasoactivity']  = 5
        data_input1_df.loc[new_value_7,'vasoactivity']  = 6
        data_input1_df.loc[new_value_empty,'vasoactivity']  = np.nan
        
        data_input1_df.loc[new_value_10,'arteriosclerosis']  = 3
        data_input1_df.loc[new_value_8,'arteriosclerosis']  = 2
        data_input1_df.loc[new_value_9,'arteriosclerosis']  = 1
        data_input1_df.loc[new_value_11,'arteriosclerosis']  = 4
        data_input1_df.loc[new_value_empty,'arteriosclerosis']  = np.nan
    return data_input1_df


###核磁共振A301 A302
def datCl_MRI(data_input1_df,colname_temp1,colname_temp2):
    data_input1_df['T1WI']=0
    data_input1_df['T2WI']=0
    data_input1_df['FLAIR']=0
    data_input1_df['leukodystrophy']=0  
    if colname_temp1 in data_input1_df.columns:        
        col_temp =data_input1_df[colname_temp1]
        new_value_empty = [str(info1) == "NaN"  for info1 in col_temp]
        new_value_1 = [("T1WI高信号" in str(info1)) for info1 in col_temp ]
        new_value_2 = [(("T1WI等信号" in str(info1)) | ("T1WI呈等信号" in str(info1))) for info1 in col_temp ]
        new_value_3 = [(("T1WI等低信号" in str(info1)) |( "T1WI呈等或低信号" in str(info1))) for info1 in col_temp ]
        new_value_4 = [(("T1WI为略低信号" in str(info1)) | ("T1WI略低信号" in str(info1))) for info1 in col_temp ]
        new_value_5 = [(("T1WI低信号" in str(info1)) | ("T1WI呈低信号" in str(info1))|("T1WI囊样低信号" in str(info1))) for info1 in col_temp ]       
        new_value_6 = [("T2WI呈稍高信号" in str(info1)) for info1 in col_temp ]        
        new_value_7 = [(("T2W及FLAIR高信号" in str(info1))|("T2WI、FLAIR呈高信号" in str(info1))|("T2WI呈高信号" in str(info1))) for info1 in col_temp ]
        new_value_8 = [(("T2WI呈明亮高信号" in str(info1))|("T2WI呈明显高信号" in str(info1))) for info1 in col_temp ]
        
        new_value_9 = [(("FLAIR像上呈高信号" in str(info1))|("FLAIR下呈椭圆形略高信号" in str(info1))|("FLAIR序列中上述部分病灶呈高信号" in str(info1))|("FLAIR高信号" in str(info1))|("FLAIR呈高信号" in str(info1))|("FLAIR呈略高信号" in str(info1))|("FLAIR序列呈高信号" in str(info1))|("FLAIR序列呈稍高信号" in str(info1))) for info1 in col_temp ]
        data_input1_df.loc[new_value_1,'T1WI']  = 1
        data_input1_df.loc[new_value_2,'T1WI']  = 2
        data_input1_df.loc[new_value_3,'T1WI']  = 3
        data_input1_df.loc[new_value_4,'T1WI']  = 4
        data_input1_df.loc[new_value_5,'T1WI']  = 5
        
        data_input1_df.loc[new_value_6,'T2WI']  = 1
        data_input1_df.loc[new_value_7,'T2WI']  = 2
        data_input1_df.loc[new_value_8,'T2WI']  = 3
        
        data_input1_df.loc[new_value_9,'FLAIR']  = 1
        data_input1_df.loc[new_value_empty,'T1WI']  = np.nan
        data_input1_df.loc[new_value_empty,'T2WI']  = np.nan
        data_input1_df.loc[new_value_empty,'FLAIR']  = np.nan
        
    if colname_temp2 in data_input1_df.columns:        
        col_temp =data_input1_df[colname_temp2]
        new_value_empty = [str(info1) == "NaN"  for info1 in col_temp]
        new_value_1 = [("脑白质变性" in str(info1)) for info1 in col_temp ]
        data_input1_df.loc[new_value_1,'leukodystrophy']  = 1
        data_input1_df.loc[new_value_empty,'leukodystrophy']  = np.nan
    return data_input1_df

##30018 30019
def datCl_num(data_input1_df,colname_temp,value):
    if colname_temp in data_input1_df.columns:        
        col_temp =data_input1_df[colname_temp]
        new_value_empty = [str(info1) == "NaN"  for info1 in col_temp]
        new_value_1 = [(("阴性" in str(info1))| ("-" in str(info1)) | ("<5" in str(info1))| ("<20" in str(info1)))  for info1 in col_temp ]
        new_value_2 = [(("+" in str(info1))& ("-" not in str(info1)))  for info1 in col_temp ]
        new_value_confused = []
        for i in range(len(new_value_1)) :
            new_value_confused.append( ( not new_value_1[i] ) and ( not new_value_2[i] )  and ( not new_value_empty[i] ))
        data_input1_df.loc[new_value_1,colname_temp]  = 0
        data_input1_df.loc[new_value_2,colname_temp]  = value+1
        data_input1_df.loc[new_value_confused,colname_temp]  = 0
        data_input1_df.loc[new_value_empty,colname_temp]  = np.nan
    return data_input1_df    

##男女判断
def datCl_sex(data_input1_df,colname_temp,colw,colm):
    data_input1_df['sex']="None"
    if colname_temp in data_input1_df.columns:
        col_temp = data_input1_df[colname_temp]
        new_value_1 = [(("子宫" in str(info1)) | ("左附件 "  in str(info1)) | ("乳腺小叶"  in str(info1)) ) for info1 in col_temp ]
        new_value_2 = [(("前列腺" in str(info1)) ) for info1 in col_temp ]
        data_input1_df.loc[new_value_1,'sex']  =0
        data_input1_df.loc[new_value_2,'sex']  =1
    if colw in data_input1_df.columns:
        col_temp = data_input1_df[colw]
        new_value_1 = [(("子宫" in str(info1)) | ("左附件 "  in str(info1)) | ("乳腺小叶"  in str(info1)) ) for info1 in col_temp ]
        data_input1_df.loc[new_value_1,'sex']  =0
    if colm in data_input1_df.columns:
        col_temp = data_input1_df[colm]
        new_value_1 = [(("前列腺" in str(info1)) ) for info1 in col_temp ]
        data_input1_df.loc[new_value_1,'sex']  =1
    return data_input1_df
            
####glsycogen A705
def datCl_Glsycogen(data_input1_df,colname_temp):
    data_input1_df['glsycogen']=0
    if colname_temp in data_input1_df.columns:
        col_temp =data_input1_df[colname_temp]
        new_value_empty = [str(info1) == "NaN"  for info1 in col_temp]
        new_value_1 = [("脂肪肝倾向" in str(info1)) for info1 in col_temp ]
        new_value_2 = [("脂肪衰减值大于240" in str(info1)) for info1 in col_temp ]
        new_value_3 = [("肝脏硬度值偏高" in str(info1)) for info1 in col_temp ]       
        new_value_4 = [(("脂肪肝倾向" in str(info1)) & ("脂肪衰减值大于240" in str(info1))) for info1 in col_temp ]
        new_value_5 = [(("脂肪肝倾向" in str(info1)) & ("肝脏硬度值偏高" in str(info1))) for info1 in col_temp ]
        new_value_6 = [(("脂肪衰减值大于240" in str(info1)) & ("肝脏硬度值偏高" in str(info1))) for info1 in col_temp ]       
        data_input1_df.loc[new_value_1,'glsycogen']  = 1
        data_input1_df.loc[new_value_2,'glsycogen']  = 2
        data_input1_df.loc[new_value_3,'glsycogen']  = 3
        data_input1_df.loc[new_value_4,'glsycogen']  = 4
        data_input1_df.loc[new_value_5,'glsycogen']  = 5
        data_input1_df.loc[new_value_6,'glsycogen']  = 6
        data_input1_df.loc[new_value_empty,'glsycogen']  = np.nan
    return data_input1_df

def missing(data):
    missing = data.isnull().sum()
    missing.sort_values(inplace=True,ascending=True)
    types = data[missing.index].dtypes
    percent = (data[missing.index].isnull().sum()/data[missing.index].isnull().count()).sort_values(ascending=True)
    missing_data = pd.concat([missing, percent,types], axis=1, keys=['Total', 'Percent','Types'])
    missing_data.sort_values('Total',ascending=True,inplace=True)
    return missing_data
    
def process_text_col(df_text):
    if os.path.isfile(CACHE_X_TEXT):
        print('Load processed text cols from cache!')
        return pd.read_csv(CACHE_X_TEXT,index_col=0)
    print('process text cols')
    df=df_text.copy()
    missing_test=missing(df)
    features = df.drop(missing_test[missing_test['Percent']>0.999].index,1)
    features=features.fillna('NaN')
    ###病史提取
    features['bingshi']=features['0409']+features['0434']
    features['kidney']=features['0117']+features['0118']
    ## CT,第一列A201，第二列A202
    features=datCl_CT(features,'A201','A202')
    ####B超结论
    features=datCl_FLD(features,'0102')
    features=datCl_FLD2(features,'0102')
    ###核磁共振A301 A302
    features=datCl_MRI(features,'A301','A302')
    ####区分男女
    features['性别信息']=features['0101']+features['0102']
    features=datCl_sex(features,'性别信息','0121','0120')
    
    colname_used_all = ["0113","0114","0115","0116","kidney","bingshi","A705","0101","1402","4001","100010","1001","2228","2229","2230","2231","2233","2302","30007","3190","3191","3192","3194","3195","3196","3197","3198","3430","3485","3486","3730","3601"]
    rule_dict = {
            "0113":"datCl_echo",
            "0114":"datCl_echo",
            "0115":"datCl_echo",
            "0116":"datCl_echo",
            "kidney":"datCl_echo",
            "bingshi":"datCl_disease",
            "100010":"datCl_Degree",
            "1001":"datCl_Arhy",
            "2228":"datCl_Degree2",
            "2229":"datCl_Degree2",
            "2230":"datCl_Degree2",
            "2231":"datCl_Degree2",
            "2233":"datCl_Degree2",
            "2302":"datCl_Healt",
            "30007":"datCl_Degree",
            "3190":"datCl_Degree2",
            "3191":"datCl_Degree2",
            "3192":"datCl_Degree2",
            "3194":"datCl_Degree2",
            "3195":"datCl_Degree2",
            "3196":"datCl_Degree2",
            "3197":"datCl_Degree2",
            "3198":"datCl_Degree2",
            "3430":"datCl_Degree2",
            "3485":"datCl_Degree2",
            "3486":"datCl_Degree2",
            "3730":"datCl_Degree2",
            "3601":"datCl_Bone",
            "4001":"datCl_Vasoactivity",
            "1402":"datCl_CBFV",
            "0101":"datCl_blood",
            "A705":"datCl_Glsycogen"
            }
    for colname_temp in colname_used_all:
        rule_used = rule_dict[colname_temp]
        if rule_used == "datCl_Arhy":
            features = datCl_Arhy(features,colname_temp)
        elif rule_used == "datCl_HeaRhy":
            features = datCl_HeaRhy(features,colname_temp)
        elif rule_used == "datCl_Degree":
            features = datCl_Degree(features,colname_temp)
        elif rule_used == "datCl_Degree2":
            features = datCl_Degree2(features,colname_temp)
        elif rule_used == "datCl_Healt":
            features = datCl_Healt(features,colname_temp)
        elif rule_used == "datCl_Bone":
            features = datCl_Bone(features,colname_temp)
        elif rule_used == "datCl_Vasoactivity":
            features = datCl_Vasoactivity(features,colname_temp)
        elif rule_used == "datCl_CBFV":
            features = datCl_CBFV(features,colname_temp)
        elif rule_used == "datCl_blood":
            features = datCl_blood(features,colname_temp)
        elif rule_used == "datCl_Glsycogen":
            features = datCl_Glsycogen(features,colname_temp)
        elif rule_used == "datCl_disease":
            features = datCl_disease(features,colname_temp)
        elif rule_used == "datCl_echo":
            features = datCl_echo(features,colname_temp)
            
    features=features.replace('None',np.nan)
    features=features.replace('NaN',np.nan)
    ##310008 310009
    features=datCl_num(features,'300018',20)
    features=datCl_num(features,'300019',5)
    ##提取数值列
    features_index=data_clean(features)
    features=features[features_index[0]]
    features.to_csv(CACHE_X_TEXT)
    return features

######lable 处理
def process_lable():
    if os.path.isfile(CACHE_LABLE):
        print('Load lable from cache!')
        return pd.read_csv(CACHE_LABLE,index_col=0)
    else:
        lable=pd.read_csv(DATA_TRAIN_Y,index_col=0,encoding="gbk")
        ###去除lable异常值 
        lable=lable.drop(lable[lable['收缩压']=='弃查'].index)
        lable=lable.drop(lable[lable['收缩压']=='未查'].index)
        lable=lable.drop(lable[lable['收缩压']==0].index)
        lable=lable.drop(lable[lable['舒张压']==100164].index)
        lable=lable.drop(lable[lable['舒张压']==974].index)
        lable=lable.drop(lable[lable['舒张压']==0].index)
        lable=lable.drop(lable[lable['血清甘油三酯']=='2.2.8'].index)
        lable=lable.drop(lable[lable['血清甘油三酯']=='> 11.00'].index)
        lable=lable.drop(lable[lable['血清甘油三酯']=='> 6.22'].index)
        lable['血清甘油三酯']=lable['血清甘油三酯'].replace('7.75轻度乳糜','7.75')
        lable['血清甘油三酯']=lable['血清甘油三酯'].str.replace("+","")
        lable.columns=['SBP','DBP','TG','HDL','LDL']
        lable=lable.astype(np.float32)
        lable[lable<0]=0
        lable=np.log1p(lable)
        lable[lable<0]=0
        lable.to_csv(CACHE_LABLE)
        return lable
    
def merge_all_features(df_num,df_num_text,df_text):
    print('merge all features')
    df=pd.concat((df_num,df_num_text,df_text), axis=1)
    df=df.astype(np.float32)
    df['hwr']=df['2403']/df['2404']
    for i in df.columns:
        if(df[i].max()>100.0):
            df[i]=np.log1p(df[i])
    df.to_csv(CACHE_X_ALL)
    return df

###创造新特征
#####包含检测结果80%异常的特征数目统计
def abnormal_num1(df,lable,col,value,cut):
    cols=[]
    for i in df.columns:
        indexs=df[df[i].notnull()].index
        filterid=(indexs).intersection(lable.index)
        lable2=lable.loc[filterid,col]
        lable3=lable2[lable2>value]
        if(len(lable3)/len(lable2)>=cut):
            cols.append(i)
    return cols

###小的值是异常值高密度脂蛋白        
def abnormal_num2(df,lable,col,value,cut):
    cols=[]
    for i in df.columns:
        indexs=df[df[i].notnull()].index
        filterid=(indexs).intersection(lable.index)
        lable2=lable.loc[filterid,col]
        lable3=lable2[lable2<value]
        if(len(lable3)/len(lable2)>=cut):
            cols.append(i)
    return cols

#####统计高于中位值3倍或低于中位值均值三倍的异常值数目
def abnormal_value1(df,lable,col,name,value,cor):
    cols=cor.columns[:-24]
    df[name]=0
    for i in cols:       
        med=df[i].median()
        if cor.loc[col,i] >0 :           
            index1=df[df[i]>3*med].index
            filterid=(index1).intersection(lable.index)
            if(len(filterid)>0):
                lable2=lable.loc[filterid,col]
                lable3=lable2[lable2>value]
                if(len(lable3)/len(lable2)>=0.4):
                    df.loc[index1,name]=df.loc[index1,name]+1
        if cor.loc[col,i] <0 :           
            index1=df[df[i]<med/3].index
            filterid=(index1).intersection(lable.index)
            if(len(filterid)>0):
                lable2=lable.loc[filterid,col]
                lable3=lable2[lable2>value]
                if(len(lable3)/len(lable2)>=0.4):
                    df.loc[index1,name]=df.loc[index1,name]+1
    return df[name]

def abnormal_value2(df,lable,col,name,value,cor):
    cols=cor.columns[:-24]
    df[name]=0
    for i in cols:       
        med=df[i].median()
        if cor.loc[col,i] >0 :           
            index1=df[df[i]>2*med].index
            ##行名取交集
            filterid=(index1).intersection(lable.index)
            if(len(filterid)>0):
                lable2=lable.loc[filterid,col]
                lable3=lable2[lable2<value]
                if(len(lable3)/len(lable2)>=0.4):
                   df.loc[index1,name]=df.loc[index1,name]+1
        if cor.loc[col,i] <0 :           
            index1=df[df[i]<med/2].index
            filterid=(index1).intersection(lable.index)
            if(len(filterid)>0):
                lable2=lable.loc[filterid,col]
                lable3=lable2[lable2<value]
                if(len(lable3)/len(lable2)>=0.4):
                    df.loc[index1,name]=df.loc[index1,name]+1
    return df[name]
    
def get_new_features(df_all,df_raw,df_y):
    if os.path.isfile(CACHE_X_LAST):
        print('Load feayures from cache!')
        return pd.read_csv(CACHE_X_LAST)
    else:
        print('get new features')
        df=df_all.copy()
        tmp=df_raw.copy()
        lable0=df_y.copy()
        train_tmp=tmp.loc[lable0.index]
        missing_test2=missing(train_tmp)
        lable2=np.exp(lable0)-1
    
        ##利用缺失值数目生成新特征
        tmp = tmp.drop(missing_test2[missing_test2['Percent']==1].index,1)
        tmp2=tmp.drop(missing_test2[missing_test2['Percent']<0.9].index,1)
        tmp3=tmp.drop(missing_test2[missing_test2['Percent']<0.99].index,1)
        tmp4=tmp.drop(missing_test2[missing_test2['Percent']<0.5].index,1)
        df['miss_all']=tmp.notnull().sum(axis=1)
        df['miss_90']=tmp2.notnull().sum(axis=1)
        df['miss_99']=tmp3.notnull().sum(axis=1)
        df['miss_50']=tmp4.notnull().sum(axis=1)
    
        ##统计异常列生成新特征
        cols_SBP=abnormal_num1(tmp,lable2,'SBP',140,0.6)
        cols_DBP=abnormal_num1(tmp,lable2,'DBP',90,0.6)
        cols_TG=abnormal_num1(tmp,lable2,'TG',2.25,0.6)
        cols_HDL=abnormal_num2(tmp,lable2,'HDL',0.88,0.2)
        cols_LDL=abnormal_num1(tmp,lable2,'LDL',3.37,0.6)
        df['SBP_abn']=tmp[cols_SBP].notnull().sum(axis=1)
        df['DBP_abn']=tmp[cols_DBP].notnull().sum(axis=1)
        df['TG_abn']=tmp[cols_TG].notnull().sum(axis=1)
        df['HDL_abn']=tmp[cols_HDL].notnull().sum(axis=1)
        df['LDL_abn']=tmp[cols_LDL].notnull().sum(axis=1)

        cols_SBP=abnormal_num1(tmp,lable2,'SBP',140,0.5)
        cols_DBP=abnormal_num1(tmp,lable2,'DBP',90,0.5)
        cols_TG=abnormal_num1(tmp,lable2,'TG',2.25,0.1)
        cols_HDL=abnormal_num2(tmp,lable2,'HDL',0.88,0.5)
        cols_LDL=abnormal_num1(tmp,lable2,'LDL',3.37,0.5)
        df['SBP_abn2']=tmp[cols_SBP].notnull().sum(axis=1)
        df['DBP_abn2']=tmp[cols_DBP].notnull().sum(axis=1)
        df['TG_abn2']=tmp[cols_TG].notnull().sum(axis=1)
        df['HDL_abn2']=tmp[cols_HDL].notnull().sum(axis=1)
        df['LDL_abn2']=tmp[cols_LDL].notnull().sum(axis=1)

        cols_SBP=abnormal_num1(tmp,lable2,'SBP',140,0.4)
        cols_DBP=abnormal_num1(tmp,lable2,'DBP',90,0.4)
        cols_TG=abnormal_num1(tmp,lable2,'TG',2.25,0.06)
        cols_HDL=abnormal_num2(tmp,lable2,'HDL',0.88,0.4)
        cols_LDL=abnormal_num1(tmp,lable2,'LDL',3.37,0.4)
        df['SBP_abn3']=tmp[cols_SBP].notnull().sum(axis=1)
        df['DBP_abn3']=tmp[cols_DBP].notnull().sum(axis=1)
        df['TG_abn3']=tmp[cols_TG].notnull().sum(axis=1)
        df['HDL_abn3']=tmp[cols_HDL].notnull().sum(axis=1)
        df['LDL_abn3']=tmp[cols_LDL].notnull().sum(axis=1)
    
        ####计算相关性
        x_train=df.loc[lable.index]
        merge_data = pd.merge(x_train, lable2, left_index=True,right_index=True,how='outer')
        corrmat = merge_data.corr()
        df['SBP_abnvalue']=abnormal_value1(df,lable2,'SBP','SBP_abnvalue',140,corrmat)
        df['DBP_abnvalue']=abnormal_value1(df,lable2,'DBP','DBP_abnvalue',90,corrmat)
        df['TG_abnvalue']=abnormal_value1(df,lable2,'TG','TG_abnvalue',2.25,corrmat)
        df['LDL_abnvalue']=abnormal_value1(df,lable2,'LDL','LDL_abnvalue',3.37,corrmat)
        df['HDL_abnvalue']=abnormal_value2(df,lable2,'HDL','HDL_abnvalue',0.88,corrmat)
        df=pd.concat((df[df.columns[-24:]],df[df.columns[:-24]]), axis=1)
        df.to_csv(CACHE_X_LAST)
        return df
    
###lightgbm
import lightgbm as lgb
from sklearn.model_selection import KFold

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'mse',
    'metric': 'l2',
    'num_leaves': 80,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'bagging_seed': 100,
    'min_child_weight':10,
    'num_all_rounds': 3000,
    'num_early_stop': 50,
    'verbose': 0
}


class lgb_model():
    def __init__(self,params):
        self.params = params
        
    def train(self, X_train, y_train, X_val, y_val):
        dtrain, dval = lgb.Dataset(X_train, y_train), lgb.Dataset(X_val, y_val)
        num_all_rounds = self.params['num_all_rounds']
        num_early_stop = self.params['num_early_stop']
        self.model = lgb.train(self.params, dtrain, num_boost_round=num_all_rounds,
                                   valid_sets=dval, early_stopping_rounds=num_early_stop,
                                   verbose_eval=False)
        print('best iteration is {}'.format(self.model.best_iteration))
        return (self.model,
                    self.model.best_score['valid_0']['l2'])
    def predict(self,X_test):
        dtest = X_test
        num_it = self.model.best_iteration
        return self.model.predict(dtest, num_iteration=num_it)
        
def kfolds_train_predict(X,Y,test):
    x_train=X
    y_train=Y
    scores=[]
    pred_y=pd.DataFrame()
    counts=1
    folder = KFold(n_splits=NUM_FOLDS, random_state=100, shuffle=True)
    for train_index,test_index in folder.split(x_train):
        print('begin Fold{}'.format(counts))
        X_train,X_test=x_train.iloc[train_index],x_train.iloc[test_index]
        Y_train,Y_test=y_train.iloc[train_index],y_train.iloc[test_index]
        model=lgb_model(params)
        mod_best,score_best=model.train(X_train,Y_train,X_test,Y_test)
        pred_t = mod_best.predict(test)
        scores.append(score_best)
        pred_y = pd.concat((pred_y,pd.DataFrame(pred_t.T)), axis=1, ignore_index=True)
        counts+=1
    print ('The score of cv: Mean={}; std={}'.format(np.mean(scores), np.std(scores)))
    pred_y_log  = pred_y.mean(axis=1)
    pred_y_last = np.exp(pred_y_log)-1
    return pred_y_last,np.mean(scores)
    
    
if __name__ == '__main__':
    #shutil.rmtree('../data/cache', ignore_errors=True)
    #os.makedirs('../data/cache')
    
    tmp=merge_raw_data()
    cols_num,cols_num_text,cols_text,tmp=split_cols(tmp)
    print ('num cols is {}'.format(len(cols_num)) )
    print ('num text cols is {}'.format(len(cols_num_text)) )
    print ('text is {}'.format(len(cols_text)) )
    df_num,df_num_text,df_text=tmp[cols_num],tmp[cols_num_text],tmp[cols_text]
    df_num_text2=process_num_col(df_num_text)
    df_text2=process_text_col(df_text)
    df_all=merge_all_features(df_num,df_num_text2,df_text2)
    lable=process_lable()
    df_final=get_new_features(df_all,tmp,lable)
    
    test=pd.read_csv(DATA_TEST_Y,index_col=0,encoding="gbk")
    x_train=df_all.loc[lable.index]
    x_test=df_all.loc[test.index]
    
    ###模型预测
    print ('******************* SBP ********************' )
    result_SBP,score_SBP=kfolds_train_predict(x_train, lable['SBP'],x_test)

    print ('******************* DBP ********************' )
    result_DBP,score_DBP=kfolds_train_predict(x_train, lable['DBP'],x_test)

    print ('******************* TG ********************' )
    result_TG,score_TG=kfolds_train_predict(x_train, lable['TG'],x_test)

    print ('******************* HDL ********************' )
    result_HDL,score_HDL=kfolds_train_predict(x_train, lable['HDL'],x_test)

    print ('******************* LDL ********************' )
    result_LDL,score_LDL=kfolds_train_predict(x_train, lable['LDL'],x_test)
    
    print ('The mean og all score is : {}'.format(np.mean([score_SBP,score_DBP,score_TG,score_HDL,score_LDL])))

    result_all = pd.concat((pd.DataFrame(result_SBP.T),pd.DataFrame(result_DBP.T),pd.DataFrame(result_TG.T),pd.DataFrame(result_HDL.T),pd.DataFrame(result_LDL.T)), axis=1, ignore_index=True)
    result_all=result_all.round(3)
    result_all.index=x_test.index
    result_all.to_csv(DIR_DATA+"prey_test_result.csv",header=False)
    
    print('Done!')
    
    