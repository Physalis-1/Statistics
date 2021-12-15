import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import t
import seaborn as sn
from sklearn.linear_model import LinearRegression
from IPython.core.display import display, HTML
display(HTML("<style>div.output_scroll { height: 44em; }</style>"))
std=0
p = 0.95
# 10%  или менее, но не 0=>10
# 90  или более, но не 100%=>90
# Облаков нет.=>0
# 20–30%.=>25
# 40%.=>40
# 50%.=>50
# 60%.=>60
# 70 – 80%.=>75
# 100%.=>100
def korrel(df):
    global p
    df_T = df['T'].dropna().values
    df_P = df['Po'].dropna().values
    otklonenie_T = []
    otklonenie_P = []
    multiply = []
    for j in range (0, len(df_P)):
        otklonenie_P.append(math.pow(df_P[j]-df['Po'].dropna().mean(),2))
        otklonenie_T.append(math.pow(df_T[j]-df['T'].dropna().mean(),2))
        multiply.append((df_T[j]-df_T.mean())*(df_P[j]-df_P.mean()))
    rxy=sum(multiply)/math.sqrt(sum(otklonenie_P)*sum(otklonenie_T))
    print('значение коэффициента корреляции Пирсона ',rxy)
    t_nab=rxy*math.sqrt(len(df['Po'].dropna()) - 2)/(math.sqrt(1-rxy*rxy))
    st = len(df['Po'].dropna()) - 2
    t_krit=t.ppf(p, st)
    print('t критерий наблюдаемый',t_nab)
    print('t критерий по таблице ',t_krit)
    if t_nab>t_krit:
        print('критерий наблюдаемый больше табличного, полученное значение коэффициента корреляции признается значимым')
    else:
        print('критерий наблюдаемый меньше табличного, полученное значение коэффициента корреляции не признается значимым')

    xs = df['T'].dropna()
    ys = df['Po'].dropna()
    pd.DataFrame(np.array([xs, ys]).T).plot.scatter(0, 1, s=12, grid=True)
    plt.xlabel('Темпераура')
    plt.ylabel('Давление')
    plt.show()

# тут
def cloud(df):
    for i in range(0, 12):
        cloud = []
        for j in range(0, len(df[i]['T'])):
            t1 = df[i].reset_index(drop=True)
            t = t1['N'].iloc[j]
            if t == '10%  или менее, но не 0':
                cloud.append(10)
            elif t == '90  или более, но не 100%':
                cloud.append(90)
            elif t == 'Облаков нет.':
                cloud.append(0)
            elif t == '20–30%.':
                cloud.append(25)
            elif t == '40%.':
                cloud.append(40)
            elif t == '50%.':
                cloud.append(50)
            elif t == '60%.':
                cloud.append(60)
            elif t == '70 – 80%.':
                cloud.append(75)
            elif t == '100%.':
                cloud.append(100)
            elif t == '0%.':
                cloud.append(0)
            else:
                # неба не видно туман
                cloud.append(t)
        df[i].insert(loc=5, column='cloud', value=cloud)
    return df

# тут
def wind(df):
    for i in range(0, 12):
        wind = []
        for j in range(0, len(df[i]['T'])):
            t1 = df[i].reset_index(drop=True)
            t = t1['DD'].iloc[j]
            if t == 'Ветер, дующий с севера':
                wind.append(360)
            elif t == 'Ветер, дующий с северо-северо-востока':
                wind.append(22.5)
            elif t == 'Ветер, дующий с северо-востока':
                wind.append(45)
            elif t == 'Ветер, дующий с востоко-северо-востока':
                wind.append(67.5)
            elif t == 'Ветер, дующий с востока':
                wind.append(90)
            elif t == 'Ветер, дующий с востоко-юго-востока':
                wind.append(112.5)
            elif t == 'Ветер, дующий с юго-востока':
                wind.append(135)
            elif t == 'Ветер, дующий с юго-юго-востока':
                wind.append(157.5)
            elif t == 'Ветер, дующий с юга':
                wind.append(180)
            elif t == 'Ветер, дующий с юго-юго-запада':
                wind.append(202.5)
            elif t == 'Ветер, дующий с юго-запада':
                wind.append(225)
            elif t == 'Ветер, дующий с западо-юго-запада':
                wind.append(247.5)
            elif t == 'Ветер, дующий с запада':
                wind.append(270)
            elif t == 'Ветер, дующий с западо-северо-запада':
                wind.append(292.5)
            elif t == 'Ветер, дующий с северо-запада':
                wind.append(315)
            elif t == 'Ветер, дующий с северо-северо-запада':
                wind.append(337.5)
            elif t == 'Штиль, безветрие':
                wind.append(0)
            else:
                wind.append(t)
        df[i].insert(loc=5, column='wind', value=wind)
        # print(df[i].columns)
    return df
#
def read_update():
    cf = pd.read_csv('7.csv', encoding='utf-8', sep=';', skiprows=6, comment='g', index_col=False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    # print(cf)
    t = cf['Местное время в По'].str.split(' ', expand=True)
    date = t[0].str.split('.', expand=True)
    time = t[1]
    cf.insert(loc=0, column='date', value=date[0])
    cf.insert(loc=1, column='mounth', value=date[1])
    cf.insert(loc=2, column='year', value=date[2])
    cf.insert(loc=3, column='time', value=time)
    df = []
    df.append(cf[cf['mounth'] == '01'])
    df.append(cf[cf['mounth'] == '02'])
    df.append(cf[cf['mounth'] == '03'])
    df.append(cf[cf['mounth'] == '04'])
    df.append(cf[cf['mounth'] == '05'])
    df.append(cf[cf['mounth'] == '06'])
    df.append(cf[cf['mounth'] == '07'])
    df.append(cf[cf['mounth'] == '08'])
    df.append(cf[cf['mounth'] == '09'])
    df.append(cf[cf['mounth'] == '10'])
    df.append(cf[cf['mounth'] == '11'])
    df.append(cf[cf['mounth'] == '12'])
    return df
#
def confidence_interval (df):
    global p,std
    st = len(df) - 1
    print('критичекое значение при  уровне значимости ', p * 100, '% и ', st, ' степени(ях) свободы', t.ppf(p, st))
    L=df.mean()-t.ppf(p, st)*std/math.sqrt(len(df))
    R=df.mean()+t.ppf(p, st)*std/math.sqrt(len(df))
    print('доверительный интервал для среднего',L,'  ',R)

def print_stat(df):
    global std
    print(df.max())
    print(df.min())
    print('размах', df.max() - df.min())
    print('среднее', df.mean())
    print('медианное', df.median())
    std=df.std()*pow(len(df),0.5)/pow(len(df)-1,0.5)
    print('отклонение',std)
    confidence_interval(df)
    print('коэффициент вариации ', std / df.mean() * 100, '%')

def day_night(df):
    df_mounth_day = []
    df_mounth_night = []
    for i in range(0, 12):
        df_night = df[i].query('time in ["21:00","00:00","03:00","06:00"]')
        df_day = df[i].query('time in ["09:00","12:00","15:00","18:00"]')
        df_mounth_day.append(df_day)
        df_mounth_night.append(df_night)
    return [df_mounth_day, df_mounth_night]

def gistogramms (df_day,df_night,name,column_name):
    plt.hist(df_day[column_name].dropna(),bins=150, alpha=0.5, label='day')
    plt.hist(df_night[column_name].dropna(),bins=150,  alpha=0.5, label='night')
    plt.legend(loc='upper right')
    plt.xlabel(name)
    plt.ylabel('Частота')
    plt.show()

def stat(df_day, df_night):
    for i in range(0, 12):
        print()
        print('//////////////////////////////////////////////////////////////////////////////////')
        print('номер месяца ', i+1)
        print('\nТемпература день')
        print_stat(df_day[i]['T'].dropna())
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
        print('Температура ночь')
        print_stat(df_night[i]['T'].dropna())
        gistogramms(df_day[i],df_night[i],'Температура','T')
        print('\nВлажность день')
        print_stat(df_day[i]['U'].dropna())
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
        print('Влажность ночь')
        print_stat(df_night[i]['U'].dropna())
        gistogramms(df_day[i], df_night[i], 'Влажность', 'U')
        print('\nДавление день')
        print_stat(df_day[i]['Po'].dropna())
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
        print('Давление ночь')
        print_stat(df_night[i]['Po'].dropna())
        gistogramms(df_day[i], df_night[i], 'Давление', 'Po')
        print('\nОблачность день')
        print_stat(df_day[i]['cloud'].dropna())
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
        print('Облачность ночь')
        print_stat(df_night[i]['cloud'].dropna())
        gistogramms(df_day[i], df_night[i], 'Облачность', 'cloud')
        print('\nВетер день')
        print_stat(df_day[i]['wind'].dropna())
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ')
        print('Ветер ночь')
        print_stat(df_night[i]['wind'].dropna())
        gistogramms(df_day[i], df_night[i], 'Ветер', 'wind')
        print('корреляция T/Po')
        print('день')
        korrel(df_day[i])
        print('ночь')
        korrel(df_night[i])
        correl_matrix(df_day[i], i, 'дневное время')
        correl_matrix(df_night[i], i, 'ночное время')

def regres(matr,dataset):
    col = ['T', 'Po', 'U', 'cloud', 'wind']
    for i in range(0,len(col)):
        x_y_cp=0
        t1 = matr[col[i]].reset_index(drop=True)
        for j in range(i, len(matr[col[i]])):
            t = float(t1.iloc[j])
            if (t>=0.7 and t<1)  or (t<=-0.7 and t >-1):
                X = dataset[[col[i]]]
                y = dataset[[col[j]]]
                for u in range(0, len(X)):
                    t = X.iloc[u].values[0]*y.iloc[u].values[0]
                    x_y_cp=t+x_y_cp
                regressor = LinearRegression()
                regressor.fit(X, y)
                plt.scatter(X, y, color='red')
                plt.plot(X, regressor.predict(X), color='blue')
                plt.title(col[i]+' and '+col[j])
                plt.xlabel(col[i])
                plt.ylabel(col[j])
                plt.show()
                m_X = dataset[[col[i]]].mean().values[0]
                m_y = dataset[[col[j]]].mean().values[0]
                r_v=(x_y_cp/len(X)-m_X*m_y)/(dataset[[col[i]]].std().values[0]*dataset[[col[j]]].std().values[0])
                print('x='+col[i]+'  y='+col[j])
                a=r_v*(dataset[[col[j]]].std().values[0])/(dataset[[col[i]]].std().values[0])
                b=(-1)*r_v*(dataset[[col[j]]].std().values[0])/(dataset[[col[i]]].std().values[0])*m_X+m_y
                if b>0:
                    print("уравнение линейной регрессии")
                    print("y="+str(a)+'x+'+str(b))
                else:
                    print("уравнение линейной регрессии")
                    print("y=" + str(a) + 'x' + str(b))


def correl_matrix(df, i, name):
    df = copy.deepcopy(df[['T', 'Po', 'U', 'cloud', 'wind']].dropna())
    numeric_col = ['T', 'Po', 'U', 'cloud', 'wind']
    corr_matrix = df.loc[:, numeric_col].corr().fillna(0)
    corr_matrix.loc['T', 'T'] = 1
    corr_matrix.loc['Po', 'Po'] = 1
    corr_matrix.loc['U', 'U'] = 1
    corr_matrix.loc['cloud', 'cloud'] = 1
    corr_matrix.loc['wind', 'wind'] = 1
    stroka = 'Корреляционная матрица за ' + name + ' ' + str(i + 1) + ' меяца'
    sn.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, cmap='BrBG').set_title(stroka, fontdict={'fontsize': 10},pad=16)
    plt.show()
    regres(copy.deepcopy(corr_matrix),copy.deepcopy(df))

df_mounth = read_update()
df_mounth_day_night = day_night(df_mounth)
df_mounth_day = df_mounth_day_night[0]
df_mounth_night = df_mounth_day_night[1]
wind_day = wind(df_mounth_day)
wind_night = wind(df_mounth_night)
cloud_night = cloud(df_mounth_night)
cloud_day = cloud(df_mounth_day)
stat(cloud_day,cloud_night)
