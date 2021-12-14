import numpy as np
from numpy.random import seed
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb

sns.set(rc={'figure.figsize': (15, 12)})
sns.set(style='darkgrid', context='notebook', palette='deep')

# 导入训练集和测试集
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data['Set'] = "Train"
test_data['Set'] = "Test"
# 将测试集房价置为-1
test_data['SalePrice'] = -1
# 合并训练集与测试集
total_data = train_data.append(test_data)
total_data.reset_index(inplace=True)
# 统计空值，并画出条形图
total_data[total_data.columns[total_data.isna().sum() > 0]].isna().sum().sort_values().plot.bar()
plt.savefig(r'nulldata.png')
plt.show()

# 用合适的值填充缺失值  比如说：MSZoning:标识销售的一般分区类，可以用出现次数最多的数据填充
# 而Alley:胡同类型，null表示为没有胡同，即NO，不能用出现次数最多的Reg替换
# 使用inplace=True修改原来的total_data，
# for i in total_data.columns:
#     print(total_data[i].value_counts().head(2))

total_data['MSZoning'].fillna("RL", inplace=True)
total_data['Alley'].fillna("NO", inplace=True)
total_data['Utilities'].fillna('AllPub', inplace=True)
total_data['Exterior1st'].fillna("VinylSd", inplace=True)
total_data['Exterior2nd'].fillna("VinylSd", inplace=True)
total_data['MasVnrArea'].fillna(0., inplace=True)
total_data['BsmtCond'].fillna("No", inplace=True)
total_data['BsmtExposure'].fillna("NB", inplace=True)
total_data['BsmtFinType1'].fillna("NB", inplace=True)
total_data['BsmtFinType2'].fillna("NB", inplace=True)
total_data['BsmtFinSF1'].fillna(0., inplace=True)
total_data['BsmtFinSF2'].fillna(0., inplace=True)
total_data['BsmtUnfSF'].fillna(0., inplace=True)
total_data['TotalBsmtSF'].fillna(0., inplace=True)
total_data['Electrical'].fillna("SBrkr", inplace=True)
total_data['BsmtFullBath'].fillna(0., inplace=True)
total_data['BsmtHalfBath'].fillna(0., inplace=True)
total_data['KitchenQual'].fillna("TA", inplace=True)
total_data['Functional'].fillna('Typ', inplace=True)
total_data['FireplaceQu'].fillna("No", inplace=True)
total_data['GarageType'].fillna("No", inplace=True)
total_data['GarageYrBlt'].fillna(0, inplace=True)
total_data['GarageFinish'].fillna("No", inplace=True)
total_data['GarageCars'].fillna(0, inplace=True)
total_data['GarageArea'].fillna(0, inplace=True)
total_data['GarageQual'].fillna("No", inplace=True)
total_data['GarageCond'].fillna("No", inplace=True)
total_data['PoolQC'].fillna("No", inplace=True)
total_data['Fence'].fillna("No", inplace=True)
total_data['MiscFeature'].fillna("No", inplace=True)
total_data['SaleType'].fillna("Con", inplace=True)
total_data['SaleCondition'].fillna("Normal", inplace=True)
# 对数据进行分组，并进行广播以中位数填充缺失值
total_data['LotFrontage'] = \
    total_data.groupby(['Neighborhood', 'Street'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))

# 转换数据类型
total_data['BsmtFullBath'].replace(3.0, 2.0, inplace=True)
total_data['BsmtFullBath'] = total_data['BsmtFullBath'].astype('int')
total_data['BsmtHalfBath'] = total_data['BsmtHalfBath'].astype('int')
total_data['KitchenAbvGr'] = pd.cut(total_data['KitchenAbvGr'], 2)
total_data['KitchenAbvGr'] = total_data['KitchenAbvGr'].astype('category').cat.rename_categories([0, 1])
total_data['TotRmsAbvGrd'] = total_data['TotRmsAbvGrd'].apply(lambda row: 4 if row < 5 else 10)
total_data['TotRmsAbvGrd'] = total_data['TotRmsAbvGrd'].apply(lambda row: 2 if row >= 2 else row)
total_data['TotRmsAbvGrd'] = total_data['TotRmsAbvGrd'].astype('int')
total_data['GarageAgeCat'] = total_data['GarageYrBlt'].apply(lambda row: 'recent' if row >= 2000 else 'old')
total_data['GarageCars'] = total_data['GarageCars'].astype('int')

# 给每个标记做映射,写映射函数
marks = {"No": -1, "Po": 0, 'Fa': 1, "TA": 2, 'Gd': 3, 'Ex': 4}


def mark_change(mark):
    return marks[mark]


# 分组运算，将标记的类型转换成数字
total_data['ExterQual'] = total_data['ExterQual'].apply(mark_change)
total_data['ExterCond'] = total_data['ExterCond'].apply(mark_change)
total_data['HeatingQC'] = total_data['HeatingQC'].apply(mark_change)
total_data['KitchenQual'] = total_data['KitchenQual'].apply(mark_change)
total_data['FireplaceQu'] = total_data['FireplaceQu'].apply(mark_change)
total_data['GarageQual'] = total_data['GarageQual'].apply(mark_change)
total_data['GarageCond'] = total_data['GarageCond'].apply(mark_change)
total_data['PoolQC'] = total_data['PoolQC'].apply(mark_change)

# 将数据再处理，得到聚合数据
# 地下室总面积
total_data['BsmtFinSF'] = total_data['BsmtFinSF1'] + total_data['BsmtFinSF2']
# 门廊
total_data['Porch'] = total_data['ScreenPorch'] +\
                      total_data['EnclosedPorch'] + total_data['OpenPorchSF'] + total_data['WoodDeckSF']
# 总面积
total_data['Total_surface'] = total_data['TotalBsmtSF'] + total_data['1stFlrSF'] + total_data['2ndFlrSF']
# 建成年份
total_data['Age'] = total_data['YrSold'] - total_data['YearBuilt']
# 改建
total_data['RemodAge'] = total_data['YrSold'] - total_data['YearRemodAdd']
# 车库年份
total_data['GarageAge'] = total_data['YrSold'] - total_data['GarageYrBlt']
# 总体的状况
total_data['Overall'] = (total_data['OverallCond'] * total_data['OverallQual'])
# 外部的整体状况
total_data['External_Overall'] = total_data['ExterCond'] * total_data['ExterQual']
# 绘制双变量分布图
sns.jointplot(data=total_data[total_data.Set == "Train"], x="SalePrice", y="Total_surface", kind='reg')
plt.savefig(r'Bivariate_distribution.png')
plt.show()
# 该删除的列
will_drop_data = ['BsmtFinSF1', 'BsmtFinSF2', 'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
             'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', '1stFlrSF', '2ndFlrSF', 'BsmtUnfSF',
             'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd']

# 对数字列名称进行排序
digital = sorted(
    ['LotFrontage', 'MasVnrArea', 'BsmtFinSF', 'GrLivArea', 'GarageArea', 'Porch', 'Total_surface', 'Age', 'RemodAge',
     'OverallQual', 'GarageCars', 'LotArea', 'ExterQual', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
     'GarageCond', 'Overall', 'External_Overall'])

# 有类别的列
categorical = sorted(
    ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'LandSlope', 'Neighborhood',
     'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'Foundation', 'BsmtQual',
     'BsmtCond', 'BsmtExposure', 'Heating', 'CentralAir', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
     'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType',
     'GarageFinish', 'PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'LotConfig', 'GarageAgeCat',
     'Utilities', 'OverallCond', 'ExterCond', 'PoolQC'])

# 删除对应列
total_data.drop(will_drop_data, inplace=True, axis=1)
# 记为分类列
total_data[categorical] = total_data[categorical].astype('category')
# 转换应该是数字的列 变为浮点型
total_data[digital] = total_data[digital].astype('float')
# 去掉训练集中地面起居面积平方英尺4000以上的数据
a = total_data[total_data.Set == "Train"][total_data[total_data.Set == "Train"]['GrLivArea'] > 4000].index
total_data = total_data.drop(a)


# 绘制散点图函数
def scatterplot(x, y, **kwargs):
    sns.regplot(x=x, y=y)
    _ = plt.xticks(rotation=90)

# 使用pandas.melt  进行行转列
# 参考 https://blog.csdn.net/maymay_/article/details/105349956
frame = pd.melt(total_data[total_data.Set == "Train"], id_vars=['SalePrice'], value_vars=digital)
# 绘制多图网格
grid = sns.FacetGrid(frame, col="variable", col_wrap=4, sharex=False, sharey=True, height=5)
# 绘制散点图
grid = grid.map(scatterplot, "value", "SalePrice")
plt.savefig(r'scatterplot.png')
plt.show()

# 标准化，归一化
Normalized = StandardScaler()
total_data[digital] = Normalized.fit_transform(total_data[digital])


# 绘制箱型图函数
def boxplot(x, y, **kwargs):
    sns.boxplot(x=x, y=y)
    _ = plt.xticks(rotation=90)

# 绘制箱型图
frame = pd.melt(total_data[total_data.Set == "Train"], id_vars=['SalePrice'], value_vars=categorical)
grid = sns.FacetGrid(frame, col="variable", col_wrap=4, sharex=False, sharey=True, height=5)
grid = grid.map(boxplot, "value", "SalePrice")
plt.savefig(r'boxplot.png')
plt.show()

# 在画布中创建子图
fig, ax = plt.subplots(figsize=(18, 18))
# 带有热图的相关性子图
grid = sns.heatmap(total_data[total_data.Set == 'Train']
                   [[*digital, 'SalePrice']].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.savefig(r'Correlation.png')
plt.show()

# 找出条件中的特殊值，并建立特征dataframe
all_condition = total_data[['Condition1', 'Condition2']]
condition_cats = ["Condition_" +
                  i for i in set([*all_condition.Condition1.unique(), *all_condition.Condition2.unique()])]
COND_FRAME = pd.DataFrame(columns=condition_cats, index=total_data.index).fillna(0)
for i in all_condition.index:
    cs = set(all_condition.loc[i, ['Condition1', 'Condition2']].values)
    for c in cs:
        COND_FRAME.loc[i]["Condition_" + c] = 1

# 将条件特征值dataframe加入总数据集并删除原特征列，之后的的都是一样的操作
total_data = total_data.join(COND_FRAME)
total_data.drop(['Condition1', 'Condition2'], axis=1, inplace=True)

frame = pd.melt(total_data[total_data.Set == "Train"], id_vars=['SalePrice'],
            value_vars=total_data[total_data.Set == 'Train'][total_data.filter(like='Condition_').columns])
grid = sns.FacetGrid(frame, col="variable", col_wrap=5, sharex=False, sharey=True, height=5)
grid = grid.map(scatterplot, "value", "SalePrice")
# 绘制销售分类SaleCondition的数据图
plt.savefig(r'SaleCondition.png')
plt.show()

ext = total_data[['Exterior1st', 'Exterior2nd']]
ext_cats = ["Ext_" + s for s in set([*ext.Exterior1st.unique(), *ext.Exterior2nd.unique()])]
EXT_FRAME = pd.DataFrame(columns=ext_cats, index=total_data.index).fillna(0)
for i in ext.index:
    cs = set(ext.loc[i, ['Exterior1st', 'Exterior2nd']].values)
    for c in cs:
        EXT_FRAME.loc[i]["Ext_" + c] = 1

total_data = total_data.join(EXT_FRAME)
total_data.drop(['Exterior1st', 'Exterior2nd'], axis=1, inplace=True)

frame = pd.melt(total_data[total_data.Set == "Train"], id_vars=['SalePrice'], value_vars=total_data[total_data.Set == 'Train'][total_data.filter(like='Ext_').columns])
grid = sns.FacetGrid(frame, col="variable", col_wrap=5, sharex=False, sharey=True, height=5)
grid = grid.map(scatterplot, "value", "SalePrice")
# 绘制外观Exterior的分类数据图
plt.savefig(r'Exterior.png')
plt.show()

bf = total_data[['BsmtFinType1', 'BsmtFinType2']]
bf_cats = ["BF_" + s for s in set([*bf.BsmtFinType1.unique(), *bf.BsmtFinType2.unique()])]
BF_FRAME = pd.DataFrame(columns=bf_cats, index=total_data.index).fillna(0)

for i in bf.index:
    cs = set(bf.loc[i, ['BsmtFinType1', 'BsmtFinType2']].values)
    for c in cs:
        BF_FRAME.loc[i]["BF_" + c] = 1

total_data = total_data.join(BF_FRAME)
total_data.drop(['BsmtFinType1', 'BsmtFinType2'], axis=1, inplace=True)

frame = pd.melt(total_data[total_data.Set == "Train"], id_vars=['SalePrice'],
            value_vars=total_data[total_data.Set == 'Train'][total_data.filter(like="BF_").columns])
grid = sns.FacetGrid(frame, col="variable", col_wrap=5, sharex=False, sharey=True, height=5)
grid = grid.map(scatterplot, "value", "SalePrice")
# 绘制地下室装修面积评级BsmtFinType的分类数据图
plt.savefig(r'BsmtFinType.png')
plt.show()

# 将原数据编码，并且类型为categorical的也会被转换
total_data = pd.get_dummies(total_data, columns=categorical)
# 新的数据集
train_data = total_data[total_data.Set == 'Train']
test_data = total_data[total_data.Set == 'Test']
# 房屋Id
HouseIds = test_data.Id.to_list()
# 测试数据删除不需要的列
test_data = test_data.drop(['Id', 'Set', "SalePrice", 'index'], axis=1)
# 划分
y = train_data['SalePrice']
X = train_data.drop(['SalePrice', 'Id', 'Set', 'index'], axis=1)
# 该函数可按照用户设定的比例，随机将样本集合划分为训练集 和测试集，并返回划分好的训练集和测试集数据。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)

# 使用随机种子，使用LGBM算法
np.random.seed(13)
model_lgbm = lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05, n_estimators=720, max_bin=55,
                              bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319, feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6,
                              min_sum_hessian_in_leaf=11).fit(X_train, y_train)

features = {}
# 用feature_importances_来做特征筛选
for feature, importance in zip(X_train.columns, model_lgbm.feature_importances_):
    features[feature] = importance

importances = pd.DataFrame({"LGBM": features})
importances.sort_values("LGBM", ascending=False, inplace=True)
importances[:20].plot.bar()
# 前20个重要特征
plt.savefig(r'importances.png')
plt.show()
# 核脊回归
KRR_model = KernelRidge(alpha=0.1, degree=1, kernel='polynomial').fit(X_train, y_train)

# scikit-learn中score用训练好的模型在测试集上进行评分
# 根据kaggle上的提示输出预测值的对数与观察到的销售价格的对数之间的均方根误差 （RMSE）评估提交
print("LGBM:")
print("train_score:", model_lgbm.score(X_train, y_train))
print("test_score:", model_lgbm.score(X_test, y_test))
print("train RMSE:", np.sqrt(mean_squared_error(np.log(y_train), np.log(model_lgbm.predict(X_train)))))
print("test RMSE:", np.sqrt(mean_squared_error(np.log(y_test), np.log(model_lgbm.predict(X_test)))))
print("------------------------------")
print("KRR:")
print("train_score:", KRR_model.score(X_train, y_train))
print("test_score:", KRR_model.score(X_test, y_test))
print("train RMSE:", np.sqrt(mean_squared_error(np.log(y_train), np.log(KRR_model.predict(X_train)))))
print("test RMSE:", np.sqrt(mean_squared_error(np.log(y_test), np.log(KRR_model.predict(X_test)))))

# 预测结果
LGBM_y = model_lgbm.predict(X)
KRR_y = KRR_model.predict(X)

# 绘制LGBM的实际与预测的散点图
plt.scatter(LGBM_y, y, alpha=0.8, color='b')
plt.title("LGBM")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(r'LGBM_scatterplot.png')
plt.show()

# 绘制KRR的实际与预测的散点图
plt.scatter(KRR_y, y, alpha=0.8, color='b')
plt.title("KRR")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(r'KRR_scatterplot.png')
plt.show()

# 将LGBM和KRR预测的值合并取平均值
# 输出取平均后的评估值
# 绘制KRR的实际与预测的散点图
ALL_train_data = pd.DataFrame({"Id": train_data.Id, "y": y, 'LGBM': LGBM_y, "KRR": KRR_y})
ALL_train_data['Voting'] = ALL_train_data[["LGBM", "KRR"]].mean(axis=1)
ALL_train_data['Voting_tree'] = ALL_train_data[["LGBM"]].mean(axis=1)
ALL_train_data.head(10)

print("------------------------------")
print("Voting RMSE:", np.sqrt(mean_squared_error(np.log(y), np.log(ALL_train_data['Voting']))))
print("Voting selection RMSE:", np.sqrt(mean_squared_error(np.log(y), np.log(ALL_train_data['Voting_tree']))))

plt.scatter(ALL_train_data['Voting'], y, alpha=0.8, color='r', label='Voting')
plt.title("Voting")
plt.legend()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(r'Voting_scatterplot.png')
plt.show()

y_predict_lgb = model_lgbm.predict(test_data)
y_predict_krr = KRR_model.predict(test_data)
All = pd.DataFrame({"Id": HouseIds, 'LGBM': y_predict_lgb, "KRR": y_predict_krr})
All['Voting'] = All[['LGBM', 'KRR']].mean(axis=1)

# 将三个预测结果与真实值放在同一个散点图里面对比
disp = 150
fig, ax = plt.subplots(figsize=(30, 10))
for col in All.columns[1:].tolist():
    plt.scatter(x=All[:disp].Id, y=All[:disp][col], alpha=0.8)
    plt.legend(All.columns[1:].tolist())
plt.savefig(r'compare.png')
plt.show()

output = pd.DataFrame({"Id": HouseIds, "SalePrice": All['Voting']})
output.to_csv('submission.csv', index=False)
