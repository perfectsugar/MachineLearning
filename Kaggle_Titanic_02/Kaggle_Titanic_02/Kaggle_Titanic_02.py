#!/c/Python27/python
# coding: utf-8

#インポートおよびグローバルな変数の設定を行う。
import pandas as pd
import numpy as np
import winsound
#プロット
from matplotlib import pyplot as plt

#機械学習ライブラリACCESS_AND_PREPROCESS_DATA
from sklearn import *
from sklearn import tree
from sklearn.metrics import *
from sklearn import ensemble
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import *
from sklearn.learning_curve import learning_curve

#リファレンスを明記する。	
#print 'REFERENCE:http://tanajun99.hatenablog.com/entry/2015/06/24/020007'

#MAIN関数
def main():
    

	#CSVの読み込み
	print '\n-----CSVの読み込み-----\n';
	dftrain = pd.read_csv("C:/Users/perfectsugar/Documents/TitanicMachineLearningfromDisaster/train.csv")
	dftest = pd.read_csv("C:/Users/perfectsugar/Documents/TitanicMachineLearningfromDisaster/test.csv")
	
	# 生データにある欠損値の処理
	dftrain_fillna = MISSING_VALUE_PREPROCESSING(dftrain);
	dftest_fillna = MISSING_VALUE_PREPROCESSING_TEST(dftest);	
	# 生データ時点での可視化
	#RAW_DATA_VISUALIZATION(dftrain_fillna);

	#PREPROCESSING  TRAIN
	#train_fit = ACCESS_AND_PREPROCESS_DATA(dftrain_fillna)
	#print train_fit.info()

	#PREPROCESSING  TEST
	#test_data = ACCESS_AND_PREPROCESS_DATA(dftest)
	#print test_data.info()

	#DATA ANALYSIS  
	DATA_ANALYSIS_ML(dftrain_fillna,dftest_fillna)
	print 'fin';
	pass

def MISSING_VALUE_PREPROCESSING(df):
	
	print '\n生データの可視化\n';
	print '①インポートしたpandasのバージョン:\t{0}\n'.format(pd.version.version);

	print '②各列の型(by pandas)の確認:\n{0}\n'.format(df.dtypes);

	print '③データ内容の確認:\n列名:{0}\n\n'.format(df.columns);
	print '\n長さ:{0}\n\n次元数:{1}\n\n.info():{2}\n\n要約(SUMMARY):{3}\n\n'.format(len(df),df.shape,df.info(),df.describe());

	print '\n（解説）\n# ixを使った列選択\n# 列名と列番号両方が使える。基本これを使っておけば良い感\ndf_sample.ix[:,"day_no"] # なお、単列選択の場合には結果はPandas.Series Object\ndf_sample.ix[:,["day_no","score1"]] # 複数列選択の場合には結果はPandas.Dataframeになる\ndf_sample.ix[0:4,"score1"] # 行は番号で、列は列名で選択することもできる\n\n';
	print '\nPassengerIDとクラスを抽出:\n{0}'.format(df.ix[:,["PassengerId","Pclass"]]);

	print '\n（解説）\n#列名の部分一致による選択:\n#R DplyrにはSelect(Contains()）という、列名部分一致選択のための便利スキームがある\n#Pandasにはそれに該当する機能はないため、少し工程を踏む必要がある\nscore_select = pd.Series(df_sample.columns).str.contains("score") # "score"を列名に含むかどうかの論理判定\ndf_sample.ix[:,np.array(score_select)]   # 論理配列を使って列選択';
	# "Survive"を列名に含むかどうかの論理判定
	Survive_select = pd.Series(df.columns).str.contains("Survive");
	print '\n\n列名部分にSurviveを含むものを抽出:\n{0}'.format(df.ix[:,np.array(Survive_select)]  );

	#plt.subplot(1,2,1);
	#plt.hist(df["Survived"],histtype="barstacked",bins=2);
	
	#---欠損値を調整しなければ，プロットはできない。-------------------------------------------------------------------------------------------------------------
	
	#論理値で欠損値を確認する
	#ages = df["Age"].isnull();	print ages;
	
	#欠損値を置き換えるfillna関数(こっちでも可能）
	ages_fillna = df["Age"].copy();	ages_fillna=ages_fillna.fillna(20); #置き換えたい数値を代入する
	print ages_fillna;

	#欠損値を-1で置き換える(こっちでも可能）
	ages_minus = df["Age"].copy();	print ages_minus;
	ages_minus[np.isnan(ages_minus)] = -1.0;

	# 欠損値を平均値で置き換える
	ages_mean = df["Age"].copy();	print ages_mean;
	ages_mean[np.isnan(ages_mean)] = np.nanmean(ages_mean);

	#元データに反映する
	#df["Age"]=ages;	print df["Age"];

	#plt.subplot(1,3,1);
	#plt.title("JUST DELETE NA") ;	plt.xlabel("Age");plt.ylim(0,300);
	#plt.hist(df["Age"].dropna(),histtype="barstacked",bins=16);

	#plt.subplot(1,3,2);
	#plt.title("NA = -1") ;	plt.xlabel("Age");plt.ylim(0,300);
	#plt.hist(ages_minus,histtype="barstacked",bins=16);

	#plt.subplot(1,3,3);
	#plt.title("NA = MEAN") ;	plt.xlabel("Age");plt.ylim(0,300);
	#plt.hist(ages_mean,histtype="barstacked",bins=16);
	#plt.show();

	#-----データを可視化していく-----------------------------------------------------------------------------------------------------------------------------
	split_data=[];
	for did_survive in [0,1]:
		split_data.append(df[df["Survived"]==did_survive]);

	print '\n---------Survived or deadでデータセットを分ける----------';	
	print '\n\tsplit_data=[];\n\tfor did_survive in [0,1]:\n\tsplit_data.append(df[df["Survived"]==did_survive]);\n\n---------↑Survived or deadでデータセットを分ける↑----------\n\n';
	print split_data
	temp_dropna = [i["Age"].dropna() for i in split_data];								#該当インデックス削除
	temp_fillna_minus = [i["Age"].fillna(-1) for i in split_data];						#-1に置き換え
	temp_mean = [i["Age"].fillna(df["Age"].mean()) for i in split_data];			#平均値で置き換え

	#plt.subplot(1,3,1);		plt.xlabel("Age");plt.ylim(0,300);		plt.hist(temp_dropna,histtype="barstacked",bins=16, color=['blue','red']);			#該当インデックス削除
	#plt.subplot(1,3,2);		plt.xlabel("Age");plt.ylim(0,300);		plt.hist(temp_fillna_minus,histtype="barstacked",bins=16, color=['blue','red']);	#-1に置き換え
	#plt.subplot(1,3,3);		plt.xlabel("Age");plt.ylim(0,300);		plt.hist(temp_mean,histtype="barstacked",bins=16, color=['blue','red']);			#平均値で置き換え
	#plt.show();

	temp_dropna=pd.DataFrame(temp_dropna);	#いったんデータフレーム型に変換
	temp_mean=pd.DataFrame(temp_mean);		#いったんデータフレーム型に変換
	print '\n順に, ALIVE / DEAD の Age \n\ntemp_dropna:\n{0}\n\ntemp_mean:\n{1}\n\n'.format(temp_dropna.T.describe(),temp_mean.T.describe())#次回はFIMLなどの方法を見ていく。
	
	#-----ここから実際に修正した欠損値を元データに反映していく-----#

	#年齢を補完する
	df["Age"]=df["Age"].fillna(df["Age"].mean());
	
	#カテゴリ変数をダミー変数に置換    #男→1、女→0に変換
	df["Sex"] = df["Sex"].map( {'female': 0, "male": 1} ).astype(int);
	
	Pclass_dum  = pd.get_dummies(df['Pclass']);
	Pclass_dum.columns = ['Class1','Class2','Class3'];print Pclass_dum;
	Pclass_dum=pd.DataFrame(Pclass_dum);
	df=pd.concat([df,Pclass_dum],axis=1);print df;
	
	#####特徴量を選択する##############
	print df.columns;
	df=df.drop(["Name","Pclass","Ticket","Cabin","Embarked","Fare","Parch","SibSp"],axis=1);
	
	#欠損値を補完したデータセットを返す
	print df;
	return df;

	pass

def MISSING_VALUE_PREPROCESSING_TEST(df):
	#-----ここから実際に修正した欠損値を元データに反映していく-----#

	#年齢を補完する
	df["Age"]=df["Age"].fillna(df["Age"].mean());
	
	#カテゴリ変数をダミー変数に置換    #男→1、女→0に変換
	df["Sex"] = df["Sex"].map( {'female': 0, "male": 1} ).astype(int);
	
	Pclass_dum  = pd.get_dummies(df['Pclass']);
	Pclass_dum.columns = ['Class1','Class2','Class3'];print Pclass_dum;
	Pclass_dum=pd.DataFrame(Pclass_dum);
	df=pd.concat([df,Pclass_dum],axis=1);print df;
	
	#####特徴量を選択する##############
	print df.columns;
	df=df.drop(["Name","Pclass","Ticket","Cabin","Embarked","Fare","Parch","SibSp"],axis=1);
	
	#欠損値を補完したデータセットを返す
	print df;
	return df;

	pass

def ACCESS_AND_PREPROCESS_DATA(df):

	print '\n------------CALL ACCESS_AND_PREPROCESS_DATA------------'

    # 各カラムの型がpandasにはどのように解釈されているかを表示
    # pandasは小数点がある場合は自動的に浮動小数点として解釈する
	print '\n-----データの型-----\n';df.dtypes
	## 先頭から2行を選択
	print '\n-----先頭から2行を選択-----\n';df.head(2);	print udf.head(2)

	# 末尾から5行を選択
	print '\n-----末尾から5行を選択-----\n';	df.tail();	print udf.tail()

	# name列に絞り込み、先頭から3行を選択
	print '\n-----name列に絞り込み、先頭から3行を選択-----\n';	df['Name'].head(3);	print udf['Name'].head(3)

	# name, age列に絞り込み
	print '\n-----name, age列に絞り込み-----\n';	df[['Name', 'Age']];print udf[['Name', 'Age']]

	print '\n\n---------------------実際に前処理を行っていきます---------------------------------------------\nhttp://corpocrat.com/2014/08/29/tutorial-titanic-dataset-machine-learning-for-kaggle/\n'
	
	print '\n-----Lets take a look at the data format below-----\n'
	df.info()

	print '\n-----Lets try to drop some of the columns which many not contribute much to our machine learning model such as Name, Ticket, Cabin etc.w-----\n'
	cols = ['Name','Ticket','Cabin']
	drop_df = df.drop(cols,axis=1)
	print udrop_df.info()

	#欠損の無いデータセットを使いたいので飛ばす::#print '\n-----Next if we want we can drop all rows in the data that has missing values (NaN).  You can do it like-----\n'	#dropna_dropcol_df = drop_df.dropna()	#print udropna_dropcol_df.info()
	dropna_dropcol_df=drop_df

	print '\n-----Now we convert the Pclass, Sex, Embarked to columns in pandas and drop them after conversion.-----\n'
	dummies = [];	cols = ['Pclass','Sex','Embarked']

	print '\nAbout get_dummies:http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html\n'
	for col in cols:	dummies.append(pd.get_dummies(dropna_dropcol_df[col]))

	print '\nAbout concat:http://sinhrks.hatenablog.com/entry/2015/01/28/073327\n'
	titanic_dummies = pd.concat(dummies, axis=1)

	print  titanic_dummies.info();	print titanic_dummies.dtypes;	print	titanic_dummies.describe()

	print '\n----finally we concatenate to the original dataframe columnwise------\n'
	dummies_dropna_dropcol_df = pd.concat((dropna_dropcol_df,titanic_dummies),axis=1);	print udummies_dropna_dropcol_df.info()


	print '\n----Now that we converted Pclass, Sex, Embarked values into columns, we drop the redundant same columns from the dataframe------\n'
	redundantremoved_dummies_dropna_dropcol_df = dummies_dropna_dropcol_df.drop(['Pclass','Sex','Embarked'],axis=1);	print uredundantremoved_dummies_dropna_dropcol_df.info()


	print '\n----All is good, except age which has lots of missing values. Lets compute a median or interpolate() all the ages\nand fill those missing age values. Pandas has a nice interpolate() function that will replace all the missing NaNs to interpolated values.------\n'
	redundantremoved_dummies_dropna_dropcol_df = redundantremoved_dummies_dropna_dropcol_df.interpolate();	print uredundantremoved_dummies_dropna_dropcol_df.info()

	print 'http://www.turbare.net/transl/scipy-lecture-notes/intro/matplotlib/matplotlib.html \nNOW IT IS PLOT TIME\n'
	#plt.subplot(2,2,1);	plt.hist(redundantremoved_dummies_dropna_dropcol_df['Age']);	#plt.subplot(2,2,2);	plt.bar(redundantremoved_dummies_dropna_dropcol_df['male'],redundantremoved_dummies_dropna_dropcol_df['female']);	#plt.subplot(2,2,3);	redundantremoved_dummies_dropna_dropcol_df['Survived'].plot(alpha=0.5,kind='hist',bins=2);plt.xlabel('Survived');plt.ylabel('N')#plt.subplot(2,2,4);	plt.hist(redundantremoved_dummies_dropna_dropcol_df['Pclass']);	#plt.show();

	print '\n\n---------------------【↑終わり↑】前処理【↑終わり↑】-----------------------------\nhttp://corpocrat.com/2014/08/29/tutorial-titanic-dataset-machine-learning-for-kaggle/\n'
	return redundantremoved_dummies_dropna_dropcol_dfpass

def USING_PANDAs_DEFALT_FUNCTION():
	print '\n------------CALL USING_PANDAs_DEFALT_FUNCTION------------\nhttp://codezine.jp/article/detail/8636?p=2\n'
	
	print '先頭行1を表示'
	print udf.head(1)

	print '\nデータが読み込めたところで、データの集計を行います。describe() 関数を利用して、データフレームのデータの概要を把握することができます。\n'
	df.describe()
	print udf.describe()

	print '\nPandasには他にも、さまざまな集計、統計用関数が用意されています。\nhttp://codezine.jp/article/detail/8636?p=2\n'

	# Age列の最大値
	a = df['Age'].max()

	# Age列の最小値
	b = df['Age'].min()

	# Age列の平均値
	c = df['Age'].mean()

	# Age列の分散
	d = df['Age'].var(ddof=False)#n-1 or nのこと

	#書式付出力について
	print 'Age列の最大値などを書式付で出力\nMaxAge:\t{0}\nMinAge:\t{1}\nMeamAge:{2}\nVarAge:\t{3}\n'.format(a,b,c,d)
	pass

def PREPROCESSING_RAW_DATA():
	print '\n------------PREPROCESSING_RAW_DATA------------\nhttp://codezine.jp/article/detail/8636?p=2\n'
	print 'データの傾向が掴めたら、必要に応じて前処理を行います。\n前処理にはいくつかの観点があります。分析の精度に悪影響を与える\nノイズとなるデータを除去する目的の前処理は\nよく行われます。サイズの大きなデータを扱う場合に、不要なデータを削除することで\n読み込みや集計速度の向上を目的とする前処理を行う\nこともあります。分析ツールが数値データしか\n受け付けない仕様に対応するために、「男／女」の文字列データ\nを「0／1」の数値データに置換する、などの制約に\n対応する目的でも行います。\n'
	print '\n不要列の削除\ndrop()関数を利用すると、データフレームから指定列の除去が行えます。\n以下のコード例では、Ticket列を削除します。\ndf.drop(''Ticket'', axis=1)\n'
	print '\nfillna()関数を利用すると、欠損値を埋めることができます。\n# 欠損データを 50 で埋める\ndf.fillna(50)\n\n# 欠損データをデータの平均値で埋める\ndf.fillna(df.mean())\nデータを補間するinterpolate()関数も用意されています。\ndf[[Name, Age]].interpolate()\ninterpolate()関数にはいつくかの補間のためのアルゴリズムが用意されています。デフォルトでは線型補間が実行されます。\nfillna()やinterpolate()関数で簡単に欠損値を埋めることができましたが、この工程には注意が必要です。\n今回のデータで年齢に対して線形補間を行っても正しい（本来の）年齢、またはそれに近い年齢が入力できるとは考えにくいです。\nさらに、推定したデータを元にして推定をすることになるので、結果に対して与える影響を慎重に検討する必要があります。\n'

	print '\n\nNaN値の補間について:http://smrmkt.hatenablog.jp/entry/2013/01/06/164758 \n\n一覧:http://kagglechallenge.hatenablog.com/entry/2015/02/13/193155 \n'
	print 'Rで前処理，Pythonで機械学習コースでよくね...???'
	pass

def DATA_ANALYSIS_ML(train_fit,test_data):

	print '\n------------CALL DATA_ANALYSIS_ML------------\nhttp://corpocrat.com/2014/08/29/tutorial-titanic-dataset-machine-learning-for-kaggle/\n'
	print train_fit.info();
	print test_data.info();

	print '\n---説明変数と応答変数を作成する---\n'
	print train_fit.columns;print test_data.columns;
	
	y_X = train_fit.values
	y = train_fit['Survived'].values
	
	X = test_data.values

	print '\n-----訓練・テストデータの特徴量を揃える-----\n'
	#訓練データにおいて，PassengerId,Survivedは省く
	#テストデータにおいて，PassengerIdは省く
	y_X = y_X[:,2:];

	"""-------------------------------------------------------------------------------------------------"""
	print '\n---モデルプリディクティング---\n'

	"""
		4.1 Cross Validation
	ラベル付きデータが少ないときに有効な評価法であるK-fold cross-validationについての説明。
	訓練データをK個のサブセットに分割し、そのうち1つのサブセットをテストデータに残りK-1個
	のサブセットを訓練データにして評価する。これをテストデータを入れ替えながらK回評価し、
	その平均を求める。
	
	"""
	######RandomForestClassifier(n_estimators=50)でのテスト実行######################
	
	#print '-----RandomForestClassifier(n_estimators=50)でのテスト実行-----\n';
	#clf = ensemble.RandomForestClassifier(n_estimators=50);
	#print clf;
	
	##訓練データとテストデータに分割する
	#X_train, X_test, y_train, y_test = train_test_split(y_X, y, random_state=0);
	#print [d.shape for d in [X_train, X_test, y_train, y_test]];
	
	##訓練データを使ってモデルを学習する
	#clf.fit(X_train,y_train);

	##テストデータで精度を評価する
	#print "再代入誤り率：", 1 - clf.score(X_train, y_train) ;
	#print "ホールドアウト誤り率：", 1 - clf.score(X_test, y_test) ;
	#print clf.score(X_test,y_test);
	#print clf.feature_importances_;

	######RandomForestClassifier(n_estimators=50)でのテスト実行######################END
	
	"""-------------------------------------------------------------------------------------------------"""
	
	######CV RandomForestClassifier(n_estimators=50)でのテスト実行######################
	
	#print '\n-----CV RandomForestClassifier(n_estimators=50)でのテスト実行-----\n';
	#clf = ensemble.RandomForestClassifier(n_estimators=50);
	#print clf;
	
	##訓練データとテストデータに分割する
	#X_train, X_test, y_train, y_test = train_test_split(y_X, y, random_state=0);
	#print [d.shape for d in [X_train, X_test, y_train, y_test]];
	
	#"""
	##訓練データを使ってCross Validationでモデルの最適なハイパーパラメータを見つける
	##訓練データを多くすると学習の精度は上がるがモデルの評価の精度が下がる。
	##テストデータを多くするとモデル評価の精度は上がるが学習の精度が下がる、
	#"""
	
	##クロスバリデーション(分割したデータで学習していく)
	
	##マニュアル
	#cv = KFold(n=len(X_train), n_folds=10, shuffle=True);
	#"""
	##選択した訓練データを表示
	#for tr, ts in cv:
	#    print("%s %s" % (tr, ts));
	#"""
	#print("\n----Subsets of the data K-fold :----\n")
	#scores_manual = []
	#for train, test in cv:
	#	X_train, y_train, X_test, y_test = y_X[train], y[train], y_X[test], y[test]
	#	clf.fit(X_train, y_train);
	#	scores_manual.append(clf.score(X_test, y_test))
	#print '\n(MANUAL)K-foldの結果\nmean:%f:std:%f' % (np.mean(scores_manual),np.std(scores_manual));

	##オートマティック
	#scores =cross_val_score(clf, y_X, y, cv=cv);
	#print '\n(AUTOMATIC)K-foldの結果\nmean:%f:std:%f' % (np.mean(scores),np.std(scores));

	##テストデータで精度を評価する
	#print "\n再代入誤り率：\t", 1 - clf.score(X_train, y_train) ;
	#print "ホールドアウト誤り率：\t", 1 - clf.score(X_test, y_test) ;
	#print 'テストデータスコア:\t',clf.score(X_test,y_test);
	#print '特徴量の効果:\t',clf.feature_importances_;

	######CV RandomForestClassifier(n_estimators=50)でのテスト実行######################END

	"""-------------------------------------------------------------------------------------------------"""

	#####GRID SEARCH RandomForestClassifierでのテスト実行######################
	
	print '\n-----GRID SEARCH RandomForestClassifierでのテスト実行-----\n';

	#訓練データとテストデータに分割する
	##訓練データを多くすると学習の精度は上がるがモデルの評価の精度が下がる。
	##テストデータを多くするとモデル評価の精度は上がるが学習の精度が下がる、
	X_train, X_test, y_train, y_test = train_test_split(y_X, y, random_state=0,test_size=0.2);
	print [d.shape for d in [X_train, X_test, y_train, y_test]];
	
	"""
			より良いハイパーパラメーターの探し方
			n_estimatorsによっても、モデルの精度は異なる（ハイパーパラメータ）
			このようなハイパーパラメータの探索は、グリッドサーチと呼ばれる。
			scikit-learnではグリッドサーチを簡単に行うGridSearchCVが提供されている。
			s
	"""

	#グリッドサーチ
	param_grid = {'n_estimators': [1,3,5,10,30,50,100],'max_features': ['auto', 'sqrt', 'log2', None]}
	clf = ensemble.RandomForestClassifier();
	cv = KFold(n=len(X_train), n_folds=10, shuffle=True)
	grid = GridSearchCV(clf, param_grid=param_grid, cv=cv,scoring='accuracy',n_jobs=-1)
	res=grid.fit(X_train, y_train);

	print("\nベストパラメタを表示\n")
	print(grid.best_estimator_)

	print("\nトレーニングデータでCVした時の平均スコア\n")
	for params, mean_score, all_scores in grid.grid_scores_:
		print("{:.3f} (+/- {:.3f}) for {}\n".format(mean_score, all_scores.std() / 2, params))

	print "\n+ テストデータでの識別結果（評価方法１）:\n"
	y_true, y_pred_test = y_test, grid.predict(X_test)
	print classification_report(y_true, y_pred_test);
	print "\nAccuracy（評価方法２）: %f\n" % grid.score(X_test, y_test);

	#グリッドサーチによって最適化されたパイパーパラメータによる"Kaggle:test.csv"予測出力
	y_pred = grid.predict(X[:,1:]);

	#最適なパラメータ時の重要度を確認
	print grid.best_params_ ;
	#####GRID SEARCH RandomForestClassifierでのテスト実行######################END
	"""-------------------------------------------------------------------------------------------------"""


	##############################学習曲線##########################
	#"""
	#このような最適モデルを探す一般的な方法はなく、
	#モデルのハイパーパラメータの組み合わせを力づくで評価して見つけるしかない。
	#この最適なハイパーパラメータの探索にもCross Validationが使える。

	#"""

	#for n_estimators in [10,30,50,100,150,200,250]:
	#	cv = ShuffleSplit(len(y_X), n_iter=3, test_size=.2)
	#	#訓練データと学習データに自動的に分けつつ，一番よいスコアを返す
	#	scores=cross_val_score(ensemble.RandomForestClassifier(n_estimators=n_estimators), y_X, y, cv=cv);
	#	print "n_estimators: %d, average score: %f" % (n_estimators, np.mean(scores));

	##最適なn_estimators
	#n_estimators =   [1,3,5,10,30,50,100,150,200]
	#train_scores, test_scores = validation_curve(RandomForestClassifier(), y_X, y, param_name="n_estimators",param_range=n_estimators, cv=cv,n_jobs=-1)
	#train_scores_mean = np.mean(train_scores, axis=1)
	#train_scores_std = np.std(train_scores, axis=1)
	#test_scores_mean = np.mean(test_scores, axis=1)
	#test_scores_std = np.std(test_scores, axis=1)

	#plt.title("Validation Curve with RF (n_estimators)")
	#plt.semilogx(n_estimators, train_scores_mean, label="Training score", color="r")
	#plt.fill_between(n_estimators, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.2, color="r")
	#plt.semilogx(n_estimators, test_scores_mean, label="Cross-validation score",color="g")
	#plt.fill_between(n_estimators, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.2, color="g")
	#plt.legend(loc="best");plt.show()

	#使用しているモデルのROC曲線（AUC）を描く
	print '\n使用しているモデルのROC曲線（AUC）を描く（評価方法３）\n';
	training_sizes, train_scores, test_scores = learning_curve(clf,
                                                y_X, y, cv=cv,
                                                scoring="mean_squared_error",
                                                train_sizes=[0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9, 1.0])
	plt.plot(training_sizes, train_scores.mean(axis=1), label="training scores")
	plt.plot(training_sizes, test_scores.mean(axis=1), label="test scores")
	plt.title("RandomForestClassifier (cv=cv)");
	plt.legend(loc="best");plt.show();

	#混合行列：各クラスのサンプルがどんな間違われ方をしたのか直感的に把握：多クラスの分類問題で有効な評価法
	print '\n混同行列　Confusion Matrix（評価方法３）\n';
	plt.matshow(confusion_matrix(y_true, y_pred_test))
	plt.colorbar()
	plt.xlabel("Predicted label")
	plt.ylabel("True label");plt.show();

	##最適なパラメータ探索によってOverfittingかUnderfittingかを調べるコード END####
	#KAGGLE
	output = np.column_stack((X[:,0],y_pred))
	df_results = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])
	df_results.to_csv('titanic_results.csv',index=False)

	#VIA SOUND
	for hz in [500,1000,1500,2000]:
		winsound.Beep(hz,400);
		pass
	print '\n\n>>end';
	pass

def plot_learning_curve(model_func, dataset):
	from sklearn.learning_curve import learning_curve
	from sklearn.grid_search import GridSearchCV
	from sklearn.ensemble import RandomForestClassifier
	import matplotlib.pyplot as plt
	import numpy as np
	sizes = [i / 10 for i in range(1, 11)]
	train_sizes, train_scores, valid_scores = learning_curve(model_func(), dataset.data, dataset.target, train_sizes=sizes, cv=5)
	take_means = lambda s: np.mean(s, axis=1)
	plt.plot(sizes, take_means(train_scores), label="training")
	plt.plot(sizes, take_means(valid_scores), label="test")
	plt.ylim(0, 1.1)
	plt.title("learning curve")
	plt.legend(loc="lower right")
	plt.show();

def RAW_DATA_VISUALIZATION(df):
	print'';
pass

if __name__ == '__main__':
	main()
