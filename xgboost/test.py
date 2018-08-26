# -*-coding:utf-8-*-
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.grid_search import GridSearchCV
import seaborn as sns
from pandas.core.frame import DataFrame
import warnings
import argparse
warnings.filterwarnings('ignore')
parse = argparse.ArgumentParser(description='  ')
parse.add_argument('--path', help='path', default='data.csv', type=str)
parse.add_argument('--silent', help='silent', default=1, type=int)
parse.add_argument('--eta', help='eta', default=0.1, type=float)
parse.add_argument('--colsample_bytree', help='colsample_bytree',
                   default=0.5, type=float)
parse.add_argument('--subsample', help='subsample', default=0.5, type=float)
parse.add_argument('--max_depth', help='max_depth', default=5, type=int)
parse.add_argument('--min_child_weight', help='min_child_weight',
                   default=3, type=int)
parse.add_argument('--num_boost_round', help='num_boost_round',
                   default=100, type=int)
parse = parse.parse_args()


def main():
    # 读取数据
    train = pd.read_csv(parse.path)
    # 提取自变量字段（该特征均为连续数据特征）
    all_features = [x for x in train.columns
                    if x not in ['企业名称', '评估结果', '评估结果N', '评级展望',
                                 '评级展望N', 'SUM评估', 'SUM展望']]
    # 重新定义数据特征 提取连续数据特征
    num_features = [x for x in train.select_dtypes(exclude=['object']).columns
                    if x not in ['企业名称', '评估结果', '评估结果N',
                                 '评级展望', '评级展望N', 'SUM评估', 'SUM展望']]
    print "连续特征总数:", len(num_features)
    print "全部特征总数:", len(all_features)

    # 划分自变量和因变量
    train_x = train[all_features]
    # train_y1 = train['评估结果N']
    # train_y2 = train['评级展望N']
    train_y3 = train['SUM评估']
    # train_y4 = train['SUM展望']

    train_y = train_y3

    # 建立第一个基础模型
    # mean_absolute_error(np.exp(y), np.exp(yhat))
    def xg_eval_mae(yhat, dtrain):
        y = dtrain.get_label()
        return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))

    dtrain = xgb.DMatrix(train_x, train['SUM评估'])
    # 定义原始参数
    xgb_params = {'seed': 0, 'eta': parse.eta,
                  'colsample_bytree': parse.colsample_bytree,
                  'silent': parse.silent, 'subsample': parse.subsample,
                  'objective': 'reg:linear',
                  'max_depth': parse.max_depth,
                  'min_child_weight': parse.min_child_weight}

    # xgboost参数设置
    bst_cv1 = xgb.cv(xgb_params, dtrain, num_boost_round=parse.num_boost_round,
                     nfold=3, seed=0,
                     feval=xg_eval_mae, maximize=False,
                     early_stopping_rounds=10)
    print 'CV score:', bst_cv1.iloc[-1, :]['test-mae-mean']

    class XGBoostRegressor(object):
        def __init__(self, **kwargs):
            self.params = kwargs
            if 'num_boost_round' in self.params:
                self.num_boost_round = self.params['num_boost_round']
            self.params.update(
                {'silent': parse.silent, 'objective': 'reg:linear', 'seed': 0})

        def fit(self, x_train, y_train):
            dtrain = xgb.DMatrix(x_train, y_train)
            self.bst = xgb.train(params=self.params, dtrain=dtrain,
                                 num_boost_round=self.num_boost_round,
                                 feval=xg_eval_mae, maximize=False)

        def predict(self, x_pred):
            dpred = xgb.DMatrix(x_pred)
            return self.bst.predict(dpred)

        def kfold(self, x_train, y_train, nfold=5):
            dtrain = xgb.DMatrix(x_train, y_train)
            cv_rounds = xgb.cv(params=self.params, dtrain=dtrain,
                               num_boost_round=self.num_boost_round,
                               nfold=nfold, feval=xg_eval_mae,
                               maximize=False, early_stopping_rounds=10)
            return cv_rounds.iloc[-1, :]

        def plot_feature_importances(self):
            feat_imp = pd.Series(self.bst.get_fscore()
                                 ).sort_values(ascending=False)
            feat_imp.plot(title='Feature Importances')
            plt.ylabel('Feature Importance Score')

        def get_params(self, deep=True):
            return self.params

        def set_params(self, **params):
            self.params.update(params)
            return self

    def mae_score(y_true, y_pred):
        return mean_absolute_error(np.exp(y_true), np.exp(y_pred))
    mae_scorer = make_scorer(mae_score, greater_is_better=False)

    # Step 1: 选择一组初始参数作为基础参考
    bst = XGBoostRegressor(eta=parse.eta,
                           colsample_bytree=parse.colsample_bytree,
                           subsample=parse.subsample,
                           max_depth=parse.max_depth,
                           min_child_weight=parse.min_child_weight,
                           num_boost_round=parse.num_boost_round)

    bst.kfold(train_x, train_y, nfold=5)

    # Step 2: 树的深度与节点权重
    xgb_param_grid = {'max_depth': list(
        range(4, 9)), 'min_child_weight': list((1, 3, 6))}
    print "max_depth参数选择:" + str(xgb_param_grid['max_depth'])
    print "min_child_weight参数选择:" + str(xgb_param_grid['min_child_weight'])
    grid = GridSearchCV(XGBoostRegressor(eta=parse.eta,
                                         num_boost_round=parse.num_boost_round,
                                         colsample_bytree=parse.colsample_bytree,
                                         subsample=parse.subsample),
                        param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)
    grid.fit(train_x, train_y.values)
    print "--------------------------------------"
    for i in grid.grid_scores_:
        print i
    print grid.best_params_, '\n', grid.best_score_
    max_depth = grid.best_params_.setdefault('max_depth')
    min_child_weight = grid.best_params_.setdefault('min_child_weight')
    print "--------------------------------------"
    print "最优max_depth : " + str(max_depth)
    print "最优min_child_weight : " + str(min_child_weight)

    # Step 3: 调节 gamma去降低过拟合风险
    xgb_param_grid = {'gamma': [0.1 * i for i in range(0, 5)]}
    grid = GridSearchCV(XGBoostRegressor(eta=parse.eta,
                                         num_boost_round=parse.num_boost_round,
                                         max_depth=max_depth,
                                         min_child_weight=min_child_weight,
                                         colsample_bytree=parse.colsample_bytree,
                                         subsample=parse.subsample),
                        param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)
    grid.fit(train_x, train_y.values)
    print "--------------------------------------"
    for i in grid.grid_scores_:
        print i
    print grid.best_params_, '\n', grid.best_score_
    gamma = grid.best_params_.setdefault('gamma')
    print "--------------------------------------"
    print "最优gamma : " + str(gamma)

    # Step 4: 调节样本采样方式 subsample 和 colsample_bytree
    xgb_param_grid = {'subsample': [
        0.1 * i for i in range(6, 9)],
        'colsample_bytree': [0.1 * i for i in range(6, 9)]}
    grid = GridSearchCV(XGBoostRegressor(eta=parse.eta, gamma=gamma,
                                         num_boost_round=parse.num_boost_round,
                                         max_depth=max_depth,
                                         min_child_weight=min_child_weight),
                        param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)
    grid.fit(train_x, train_y.values)
    print "--------------------------------------"
    for i in grid.grid_scores_:
        print i
    print grid.best_params_, '\n', grid.best_score_
    subsample = grid.best_params_.setdefault('subsample')
    colsample_bytree = grid.best_params_.setdefault('colsample_bytree')
    print "--------------------------------------"
    print "最优subsample : " + str(subsample)
    print "最优colsample_bytree : " + str(colsample_bytree)

    # Step 5: 学习eta调节
    xgb_param_grid = {'eta': [0.5, 0.4, 0.3,
                              0.2, 0.1, 0.075, 0.05, 0.04, 0.03]}
    grid = GridSearchCV(XGBoostRegressor(num_boost_round=parse.num_boost_round,
                                         gamma=gamma,
                                         max_depth=max_depth,
                                         min_child_weight=min_child_weight,
                                         colsample_bytree=colsample_bytree,
                                         subsample=subsample),
                        param_grid=xgb_param_grid, cv=5, scoring=mae_scorer)
    grid.fit(train_x, train_y.values)
    print "--------------------------------------"
    for i in grid.grid_scores_:
        print i
    print grid.best_params_, '\n', grid.best_score_
    eta = grid.best_params_.setdefault('eta')
    print "--------------------------------------"
    print "最优eta : " + str(eta)

    # Step6: 树个数调节和学习率
    # 绘图所使用函数
    def convert_grid_scores(scores):
        _params = []
        _params_mae = []
        for i in scores:
            _params.append(i[0].values())
            _params_mae.append(i[1])
        # params = np.array(_params)
        grid_res = np.column_stack((_params, _params_mae))
        return [grid_res[:, i] for i in range(grid_res.shape[1])]

    list_ = [100 * i for i in range(1, 10)]
    xgb_param_grid = {'eta': [0.5, 0.4, 0.3,
                              0.2, 0.1, 0.075, 0.05, 0.04, 0.03]}
    # list_tree = []
    list_score = []

    def num_eta(num, xgb_param_grid):
        xgb_param_grid = xgb_param_grid
        grid = GridSearchCV(XGBoostRegressor(num_boost_round=num, gamma=gamma,
                                             max_depth=max_depth,
                                             min_child_weight=min_child_weight,
                                             colsample_bytree=colsample_bytree,
                                             subsample=subsample),
                            param_grid=xgb_param_grid,
                            cv=5, scoring=mae_scorer)
        grid.fit(train_x, train_y.values)
        print "num_boost_round数量：" + str(num)
        print grid.best_params_, '\n', grid.best_score_
        _, scores = convert_grid_scores(grid.grid_scores_)
        score = scores.tolist()
        return score
    for i in list_:
        print i
        score = num_eta(i, xgb_param_grid)
        list_score.append(score)

    data = DataFrame(list_score, index=['tree"{}"'.format(i) for i in list_],
                     columns=['eta"{}"'.format(i)
                              for i in xgb_param_grid["eta"]])

    for i in range(len(data))[0:1]:
        for j in data.columns[0:1]:
            a = data[j][i]
    for i in range(len(data)):
        for j in data.columns:
            if a <= data[j][i]:
                a = data[j][i]
                b = data.index[i]
                c = j

    num_boost_round = int(b.split("\"")[1])
    eta = float(c.split("\"")[1])
    print "最优tree:" + str(num_boost_round)
    print "最优eta:" + str(eta)

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.heatmap(data,
                     annot=True,
                     fmt='.2f',
                     )
    ax.set_title('Tree and Eta')

    # Final XGBoost model
    bst = XGBoostRegressor(num_boost_round=num_boost_round,
                           eta=eta,
                           gamma=gamma,
                           max_depth=max_depth,
                           min_child_weight=min_child_weight,
                           colsample_bytree=colsample_bytree,
                           subsample=subsample)
    cv = bst.kfold(train_x, train_y, nfold=5)
    print cv
    print "num_boost_round:" + str(num_boost_round)
    print "eta:" + str(eta)
    print "gamma:" + str(gamma)
    print "max_depth:" + str(max_depth)
    print "min_child_weight:" + str(min_child_weight)
    print "colsample_bytree:" + str(colsample_bytree)
    print "subsample:" + str(subsample)

    bst.fit(train_x, train_y)
    bst.predict(train_x)
    print len(bst.predict(train_x))
    df = pd.DataFrame(bst.predict(train_x))
    df.to_csv("result.csv")


if __name__ == "__main__":
    main()
