#!/c/Python27/python
# coding: utf-8

#�C���|�[�g����уO���[�o���ȕϐ��̐ݒ���s���B
import pandas as pd
import numpy as np
import winsound
#�v���b�g
from matplotlib import pyplot as plt

#�@�B�w�K���C�u����ACCESS_AND_PREPROCESS_DATA
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

#���t�@�����X�𖾋L����B	
#print 'REFERENCE:http://tanajun99.hatenablog.com/entry/2015/06/24/020007'

#MAIN�֐�
def main():
    

	#CSV�̓ǂݍ���
	print '\n-----CSV�̓ǂݍ���-----\n';
	dftrain = pd.read_csv("C:/Users/perfectsugar/Documents/TitanicMachineLearningfromDisaster/train.csv")
	dftest = pd.read_csv("C:/Users/perfectsugar/Documents/TitanicMachineLearningfromDisaster/test.csv")
	
	# ���f�[�^�ɂ��錇���l�̏���
	dftrain_fillna = MISSING_VALUE_PREPROCESSING(dftrain);
	dftest_fillna = MISSING_VALUE_PREPROCESSING_TEST(dftest);	
	# ���f�[�^���_�ł̉���
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
	
	print '\n���f�[�^�̉���\n';
	print '�@�C���|�[�g����pandas�̃o�[�W����:\t{0}\n'.format(pd.version.version);

	print '�A�e��̌^(by pandas)�̊m�F:\n{0}\n'.format(df.dtypes);

	print '�B�f�[�^���e�̊m�F:\n��:{0}\n\n'.format(df.columns);
	print '\n����:{0}\n\n������:{1}\n\n.info():{2}\n\n�v��(SUMMARY):{3}\n\n'.format(len(df),df.shape,df.info(),df.describe());

	print '\n�i����j\n# ix���g������I��\n# �񖼂Ɨ�ԍ��������g����B��{������g���Ă����Ηǂ���\ndf_sample.ix[:,"day_no"] # �Ȃ��A�P��I���̏ꍇ�ɂ͌��ʂ�Pandas.Series Object\ndf_sample.ix[:,["day_no","score1"]] # ������I���̏ꍇ�ɂ͌��ʂ�Pandas.Dataframe�ɂȂ�\ndf_sample.ix[0:4,"score1"] # �s�͔ԍ��ŁA��͗񖼂őI�����邱�Ƃ��ł���\n\n';
	print '\nPassengerID�ƃN���X�𒊏o:\n{0}'.format(df.ix[:,["PassengerId","Pclass"]]);

	print '\n�i����j\n#�񖼂̕�����v�ɂ��I��:\n#R Dplyr�ɂ�Select(Contains()�j�Ƃ����A�񖼕�����v�I���̂��߂֗̕��X�L�[��������\n#Pandas�ɂ͂���ɊY������@�\�͂Ȃ����߁A�����H���𓥂ޕK�v������\nscore_select = pd.Series(df_sample.columns).str.contains("score") # "score"��񖼂Ɋ܂ނ��ǂ����̘_������\ndf_sample.ix[:,np.array(score_select)]   # �_���z����g���ė�I��';
	# "Survive"��񖼂Ɋ܂ނ��ǂ����̘_������
	Survive_select = pd.Series(df.columns).str.contains("Survive");
	print '\n\n�񖼕�����Survive���܂ނ��̂𒊏o:\n{0}'.format(df.ix[:,np.array(Survive_select)]  );

	#plt.subplot(1,2,1);
	#plt.hist(df["Survived"],histtype="barstacked",bins=2);
	
	#---�����l�𒲐����Ȃ���΁C�v���b�g�͂ł��Ȃ��B-------------------------------------------------------------------------------------------------------------
	
	#�_���l�Ō����l���m�F����
	#ages = df["Age"].isnull();	print ages;
	
	#�����l��u��������fillna�֐�(�������ł��\�j
	ages_fillna = df["Age"].copy();	ages_fillna=ages_fillna.fillna(20); #�u�������������l��������
	print ages_fillna;

	#�����l��-1�Œu��������(�������ł��\�j
	ages_minus = df["Age"].copy();	print ages_minus;
	ages_minus[np.isnan(ages_minus)] = -1.0;

	# �����l�𕽋ϒl�Œu��������
	ages_mean = df["Age"].copy();	print ages_mean;
	ages_mean[np.isnan(ages_mean)] = np.nanmean(ages_mean);

	#���f�[�^�ɔ��f����
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

	#-----�f�[�^���������Ă���-----------------------------------------------------------------------------------------------------------------------------
	split_data=[];
	for did_survive in [0,1]:
		split_data.append(df[df["Survived"]==did_survive]);

	print '\n---------Survived or dead�Ńf�[�^�Z�b�g�𕪂���----------';	
	print '\n\tsplit_data=[];\n\tfor did_survive in [0,1]:\n\tsplit_data.append(df[df["Survived"]==did_survive]);\n\n---------��Survived or dead�Ńf�[�^�Z�b�g�𕪂��遪----------\n\n';
	print split_data
	temp_dropna = [i["Age"].dropna() for i in split_data];								#�Y���C���f�b�N�X�폜
	temp_fillna_minus = [i["Age"].fillna(-1) for i in split_data];						#-1�ɒu������
	temp_mean = [i["Age"].fillna(df["Age"].mean()) for i in split_data];			#���ϒl�Œu������

	#plt.subplot(1,3,1);		plt.xlabel("Age");plt.ylim(0,300);		plt.hist(temp_dropna,histtype="barstacked",bins=16, color=['blue','red']);			#�Y���C���f�b�N�X�폜
	#plt.subplot(1,3,2);		plt.xlabel("Age");plt.ylim(0,300);		plt.hist(temp_fillna_minus,histtype="barstacked",bins=16, color=['blue','red']);	#-1�ɒu������
	#plt.subplot(1,3,3);		plt.xlabel("Age");plt.ylim(0,300);		plt.hist(temp_mean,histtype="barstacked",bins=16, color=['blue','red']);			#���ϒl�Œu������
	#plt.show();

	temp_dropna=pd.DataFrame(temp_dropna);	#��������f�[�^�t���[���^�ɕϊ�
	temp_mean=pd.DataFrame(temp_mean);		#��������f�[�^�t���[���^�ɕϊ�
	print '\n����, ALIVE / DEAD �� Age \n\ntemp_dropna:\n{0}\n\ntemp_mean:\n{1}\n\n'.format(temp_dropna.T.describe(),temp_mean.T.describe())#�����FIML�Ȃǂ̕��@�����Ă����B
	
	#-----����������ۂɏC�����������l�����f�[�^�ɔ��f���Ă���-----#

	#�N���⊮����
	df["Age"]=df["Age"].fillna(df["Age"].mean());
	
	#�J�e�S���ϐ����_�~�[�ϐ��ɒu��    #�j��1�A����0�ɕϊ�
	df["Sex"] = df["Sex"].map( {'female': 0, "male": 1} ).astype(int);
	
	Pclass_dum  = pd.get_dummies(df['Pclass']);
	Pclass_dum.columns = ['Class1','Class2','Class3'];print Pclass_dum;
	Pclass_dum=pd.DataFrame(Pclass_dum);
	df=pd.concat([df,Pclass_dum],axis=1);print df;
	
	#####�����ʂ�I������##############
	print df.columns;
	df=df.drop(["Name","Pclass","Ticket","Cabin","Embarked","Fare","Parch","SibSp"],axis=1);
	
	#�����l��⊮�����f�[�^�Z�b�g��Ԃ�
	print df;
	return df;

	pass

def MISSING_VALUE_PREPROCESSING_TEST(df):
	#-----����������ۂɏC�����������l�����f�[�^�ɔ��f���Ă���-----#

	#�N���⊮����
	df["Age"]=df["Age"].fillna(df["Age"].mean());
	
	#�J�e�S���ϐ����_�~�[�ϐ��ɒu��    #�j��1�A����0�ɕϊ�
	df["Sex"] = df["Sex"].map( {'female': 0, "male": 1} ).astype(int);
	
	Pclass_dum  = pd.get_dummies(df['Pclass']);
	Pclass_dum.columns = ['Class1','Class2','Class3'];print Pclass_dum;
	Pclass_dum=pd.DataFrame(Pclass_dum);
	df=pd.concat([df,Pclass_dum],axis=1);print df;
	
	#####�����ʂ�I������##############
	print df.columns;
	df=df.drop(["Name","Pclass","Ticket","Cabin","Embarked","Fare","Parch","SibSp"],axis=1);
	
	#�����l��⊮�����f�[�^�Z�b�g��Ԃ�
	print df;
	return df;

	pass

def ACCESS_AND_PREPROCESS_DATA(df):

	print '\n------------CALL ACCESS_AND_PREPROCESS_DATA------------'

    # �e�J�����̌^��pandas�ɂ͂ǂ̂悤�ɉ��߂���Ă��邩��\��
    # pandas�͏����_������ꍇ�͎����I�ɕ��������_�Ƃ��ĉ��߂���
	print '\n-----�f�[�^�̌^-----\n';df.dtypes
	## �擪����2�s��I��
	print '\n-----�擪����2�s��I��-----\n';df.head(2);	print udf.head(2)

	# ��������5�s��I��
	print '\n-----��������5�s��I��-----\n';	df.tail();	print udf.tail()

	# name��ɍi�荞�݁A�擪����3�s��I��
	print '\n-----name��ɍi�荞�݁A�擪����3�s��I��-----\n';	df['Name'].head(3);	print udf['Name'].head(3)

	# name, age��ɍi�荞��
	print '\n-----name, age��ɍi�荞��-----\n';	df[['Name', 'Age']];print udf[['Name', 'Age']]

	print '\n\n---------------------���ۂɑO�������s���Ă����܂�---------------------------------------------\nhttp://corpocrat.com/2014/08/29/tutorial-titanic-dataset-machine-learning-for-kaggle/\n'
	
	print '\n-----Lets take a look at the data format below-----\n'
	df.info()

	print '\n-----Lets try to drop some of the columns which many not contribute much to our machine learning model such as Name, Ticket, Cabin etc.w-----\n'
	cols = ['Name','Ticket','Cabin']
	drop_df = df.drop(cols,axis=1)
	print udrop_df.info()

	#�����̖����f�[�^�Z�b�g���g�������̂Ŕ�΂�::#print '\n-----Next if we want we can drop all rows in the data that has missing values (NaN).  You can do it like-----\n'	#dropna_dropcol_df = drop_df.dropna()	#print udropna_dropcol_df.info()
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

	print '\n\n---------------------�y���I��聪�z�O�����y���I��聪�z-----------------------------\nhttp://corpocrat.com/2014/08/29/tutorial-titanic-dataset-machine-learning-for-kaggle/\n'
	return redundantremoved_dummies_dropna_dropcol_dfpass

def USING_PANDAs_DEFALT_FUNCTION():
	print '\n------------CALL USING_PANDAs_DEFALT_FUNCTION------------\nhttp://codezine.jp/article/detail/8636?p=2\n'
	
	print '�擪�s1��\��'
	print udf.head(1)

	print '\n�f�[�^���ǂݍ��߂��Ƃ���ŁA�f�[�^�̏W�v���s���܂��Bdescribe() �֐��𗘗p���āA�f�[�^�t���[���̃f�[�^�̊T�v��c�����邱�Ƃ��ł��܂��B\n'
	df.describe()
	print udf.describe()

	print '\nPandas�ɂ͑��ɂ��A���܂��܂ȏW�v�A���v�p�֐����p�ӂ���Ă��܂��B\nhttp://codezine.jp/article/detail/8636?p=2\n'

	# Age��̍ő�l
	a = df['Age'].max()

	# Age��̍ŏ��l
	b = df['Age'].min()

	# Age��̕��ϒl
	c = df['Age'].mean()

	# Age��̕��U
	d = df['Age'].var(ddof=False)#n-1 or n�̂���

	#�����t�o�͂ɂ���
	print 'Age��̍ő�l�Ȃǂ������t�ŏo��\nMaxAge:\t{0}\nMinAge:\t{1}\nMeamAge:{2}\nVarAge:\t{3}\n'.format(a,b,c,d)
	pass

def PREPROCESSING_RAW_DATA():
	print '\n------------PREPROCESSING_RAW_DATA------------\nhttp://codezine.jp/article/detail/8636?p=2\n'
	print '�f�[�^�̌X�����͂߂���A�K�v�ɉ����đO�������s���܂��B\n�O�����ɂ͂������̊ϓ_������܂��B���͂̐��x�Ɉ��e����^����\n�m�C�Y�ƂȂ�f�[�^����������ړI�̑O������\n�悭�s���܂��B�T�C�Y�̑傫�ȃf�[�^�������ꍇ�ɁA�s�v�ȃf�[�^���폜���邱�Ƃ�\n�ǂݍ��݂�W�v���x�̌����ړI�Ƃ���O�������s��\n���Ƃ�����܂��B���̓c�[�������l�f�[�^����\n�󂯕t���Ȃ��d�l�ɑΉ����邽�߂ɁA�u�j�^���v�̕�����f�[�^\n���u0�^1�v�̐��l�f�[�^�ɒu������A�Ȃǂ̐����\n�Ή�����ړI�ł��s���܂��B\n'
	print '\n�s�v��̍폜\ndrop()�֐��𗘗p����ƁA�f�[�^�t���[������w���̏������s���܂��B\n�ȉ��̃R�[�h��ł́ATicket����폜���܂��B\ndf.drop(''Ticket'', axis=1)\n'
	print '\nfillna()�֐��𗘗p����ƁA�����l�𖄂߂邱�Ƃ��ł��܂��B\n# �����f�[�^�� 50 �Ŗ��߂�\ndf.fillna(50)\n\n# �����f�[�^���f�[�^�̕��ϒl�Ŗ��߂�\ndf.fillna(df.mean())\n�f�[�^���Ԃ���interpolate()�֐����p�ӂ���Ă��܂��B\ndf[[Name, Age]].interpolate()\ninterpolate()�֐��ɂ͂������̕�Ԃ̂��߂̃A���S���Y�����p�ӂ���Ă��܂��B�f�t�H���g�ł͐��^��Ԃ����s����܂��B\nfillna()��interpolate()�֐��ŊȒP�Ɍ����l�𖄂߂邱�Ƃ��ł��܂������A���̍H���ɂ͒��ӂ��K�v�ł��B\n����̃f�[�^�ŔN��ɑ΂��Đ��`��Ԃ��s���Ă��������i�{���́j�N��A�܂��͂���ɋ߂��N����͂ł���Ƃ͍l���ɂ����ł��B\n����ɁA���肵���f�[�^�����ɂ��Đ�������邱�ƂɂȂ�̂ŁA���ʂɑ΂��ė^����e����T�d�Ɍ�������K�v������܂��B\n'

	print '\n\nNaN�l�̕�Ԃɂ���:http://smrmkt.hatenablog.jp/entry/2013/01/06/164758 \n\n�ꗗ:http://kagglechallenge.hatenablog.com/entry/2015/02/13/193155 \n'
	print 'R�őO�����CPython�ŋ@�B�w�K�R�[�X�ł悭��...???'
	pass

def DATA_ANALYSIS_ML(train_fit,test_data):

	print '\n------------CALL DATA_ANALYSIS_ML------------\nhttp://corpocrat.com/2014/08/29/tutorial-titanic-dataset-machine-learning-for-kaggle/\n'
	print train_fit.info();
	print test_data.info();

	print '\n---�����ϐ��Ɖ����ϐ����쐬����---\n'
	print train_fit.columns;print test_data.columns;
	
	y_X = train_fit.values
	y = train_fit['Survived'].values
	
	X = test_data.values

	print '\n-----�P���E�e�X�g�f�[�^�̓����ʂ𑵂���-----\n'
	#�P���f�[�^�ɂ����āCPassengerId,Survived�͏Ȃ�
	#�e�X�g�f�[�^�ɂ����āCPassengerId�͏Ȃ�
	y_X = y_X[:,2:];

	"""-------------------------------------------------------------------------------------------------"""
	print '\n---���f���v���f�B�N�e�B���O---\n'

	"""
		4.1 Cross Validation
	���x���t���f�[�^�����Ȃ��Ƃ��ɗL���ȕ]���@�ł���K-fold cross-validation�ɂ��Ă̐����B
	�P���f�[�^��K�̃T�u�Z�b�g�ɕ������A���̂���1�̃T�u�Z�b�g���e�X�g�f�[�^�Ɏc��K-1��
	�̃T�u�Z�b�g���P���f�[�^�ɂ��ĕ]������B������e�X�g�f�[�^�����ւ��Ȃ���K��]�����A
	���̕��ς����߂�B
	
	"""
	######RandomForestClassifier(n_estimators=50)�ł̃e�X�g���s######################
	
	#print '-----RandomForestClassifier(n_estimators=50)�ł̃e�X�g���s-----\n';
	#clf = ensemble.RandomForestClassifier(n_estimators=50);
	#print clf;
	
	##�P���f�[�^�ƃe�X�g�f�[�^�ɕ�������
	#X_train, X_test, y_train, y_test = train_test_split(y_X, y, random_state=0);
	#print [d.shape for d in [X_train, X_test, y_train, y_test]];
	
	##�P���f�[�^���g���ă��f�����w�K����
	#clf.fit(X_train,y_train);

	##�e�X�g�f�[�^�Ő��x��]������
	#print "�đ����藦�F", 1 - clf.score(X_train, y_train) ;
	#print "�z�[���h�A�E�g��藦�F", 1 - clf.score(X_test, y_test) ;
	#print clf.score(X_test,y_test);
	#print clf.feature_importances_;

	######RandomForestClassifier(n_estimators=50)�ł̃e�X�g���s######################END
	
	"""-------------------------------------------------------------------------------------------------"""
	
	######CV RandomForestClassifier(n_estimators=50)�ł̃e�X�g���s######################
	
	#print '\n-----CV RandomForestClassifier(n_estimators=50)�ł̃e�X�g���s-----\n';
	#clf = ensemble.RandomForestClassifier(n_estimators=50);
	#print clf;
	
	##�P���f�[�^�ƃe�X�g�f�[�^�ɕ�������
	#X_train, X_test, y_train, y_test = train_test_split(y_X, y, random_state=0);
	#print [d.shape for d in [X_train, X_test, y_train, y_test]];
	
	#"""
	##�P���f�[�^���g����Cross Validation�Ń��f���̍œK�ȃn�C�p�[�p�����[�^��������
	##�P���f�[�^�𑽂�����Ɗw�K�̐��x�͏オ�邪���f���̕]���̐��x��������B
	##�e�X�g�f�[�^�𑽂�����ƃ��f���]���̐��x�͏オ�邪�w�K�̐��x��������A
	#"""
	
	##�N���X�o���f�[�V����(���������f�[�^�Ŋw�K���Ă���)
	
	##�}�j���A��
	#cv = KFold(n=len(X_train), n_folds=10, shuffle=True);
	#"""
	##�I�������P���f�[�^��\��
	#for tr, ts in cv:
	#    print("%s %s" % (tr, ts));
	#"""
	#print("\n----Subsets of the data K-fold :----\n")
	#scores_manual = []
	#for train, test in cv:
	#	X_train, y_train, X_test, y_test = y_X[train], y[train], y_X[test], y[test]
	#	clf.fit(X_train, y_train);
	#	scores_manual.append(clf.score(X_test, y_test))
	#print '\n(MANUAL)K-fold�̌���\nmean:%f:std:%f' % (np.mean(scores_manual),np.std(scores_manual));

	##�I�[�g�}�e�B�b�N
	#scores =cross_val_score(clf, y_X, y, cv=cv);
	#print '\n(AUTOMATIC)K-fold�̌���\nmean:%f:std:%f' % (np.mean(scores),np.std(scores));

	##�e�X�g�f�[�^�Ő��x��]������
	#print "\n�đ����藦�F\t", 1 - clf.score(X_train, y_train) ;
	#print "�z�[���h�A�E�g��藦�F\t", 1 - clf.score(X_test, y_test) ;
	#print '�e�X�g�f�[�^�X�R�A:\t',clf.score(X_test,y_test);
	#print '�����ʂ̌���:\t',clf.feature_importances_;

	######CV RandomForestClassifier(n_estimators=50)�ł̃e�X�g���s######################END

	"""-------------------------------------------------------------------------------------------------"""

	#####GRID SEARCH RandomForestClassifier�ł̃e�X�g���s######################
	
	print '\n-----GRID SEARCH RandomForestClassifier�ł̃e�X�g���s-----\n';

	#�P���f�[�^�ƃe�X�g�f�[�^�ɕ�������
	##�P���f�[�^�𑽂�����Ɗw�K�̐��x�͏オ�邪���f���̕]���̐��x��������B
	##�e�X�g�f�[�^�𑽂�����ƃ��f���]���̐��x�͏オ�邪�w�K�̐��x��������A
	X_train, X_test, y_train, y_test = train_test_split(y_X, y, random_state=0,test_size=0.2);
	print [d.shape for d in [X_train, X_test, y_train, y_test]];
	
	"""
			���ǂ��n�C�p�[�p�����[�^�[�̒T����
			n_estimators�ɂ���Ă��A���f���̐��x�͈قȂ�i�n�C�p�[�p�����[�^�j
			���̂悤�ȃn�C�p�[�p�����[�^�̒T���́A�O���b�h�T�[�`�ƌĂ΂��B
			scikit-learn�ł̓O���b�h�T�[�`���ȒP�ɍs��GridSearchCV���񋟂���Ă���B
			s
	"""

	#�O���b�h�T�[�`
	param_grid = {'n_estimators': [1,3,5,10,30,50,100],'max_features': ['auto', 'sqrt', 'log2', None]}
	clf = ensemble.RandomForestClassifier();
	cv = KFold(n=len(X_train), n_folds=10, shuffle=True)
	grid = GridSearchCV(clf, param_grid=param_grid, cv=cv,scoring='accuracy',n_jobs=-1)
	res=grid.fit(X_train, y_train);

	print("\n�x�X�g�p�����^��\��\n")
	print(grid.best_estimator_)

	print("\n�g���[�j���O�f�[�^��CV�������̕��σX�R�A\n")
	for params, mean_score, all_scores in grid.grid_scores_:
		print("{:.3f} (+/- {:.3f}) for {}\n".format(mean_score, all_scores.std() / 2, params))

	print "\n+ �e�X�g�f�[�^�ł̎��ʌ��ʁi�]�����@�P�j:\n"
	y_true, y_pred_test = y_test, grid.predict(X_test)
	print classification_report(y_true, y_pred_test);
	print "\nAccuracy�i�]�����@�Q�j: %f\n" % grid.score(X_test, y_test);

	#�O���b�h�T�[�`�ɂ���čœK�����ꂽ�p�C�p�[�p�����[�^�ɂ��"Kaggle:test.csv"�\���o��
	y_pred = grid.predict(X[:,1:]);

	#�œK�ȃp�����[�^���̏d�v�x���m�F
	print grid.best_params_ ;
	#####GRID SEARCH RandomForestClassifier�ł̃e�X�g���s######################END
	"""-------------------------------------------------------------------------------------------------"""


	##############################�w�K�Ȑ�##########################
	#"""
	#���̂悤�ȍœK���f����T����ʓI�ȕ��@�͂Ȃ��A
	#���f���̃n�C�p�[�p�����[�^�̑g�ݍ��킹��͂Â��ŕ]�����Č����邵���Ȃ��B
	#���̍œK�ȃn�C�p�[�p�����[�^�̒T���ɂ�Cross Validation���g����B

	#"""

	#for n_estimators in [10,30,50,100,150,200,250]:
	#	cv = ShuffleSplit(len(y_X), n_iter=3, test_size=.2)
	#	#�P���f�[�^�Ɗw�K�f�[�^�Ɏ����I�ɕ����C��Ԃ悢�X�R�A��Ԃ�
	#	scores=cross_val_score(ensemble.RandomForestClassifier(n_estimators=n_estimators), y_X, y, cv=cv);
	#	print "n_estimators: %d, average score: %f" % (n_estimators, np.mean(scores));

	##�œK��n_estimators
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

	#�g�p���Ă��郂�f����ROC�Ȑ��iAUC�j��`��
	print '\n�g�p���Ă��郂�f����ROC�Ȑ��iAUC�j��`���i�]�����@�R�j\n';
	training_sizes, train_scores, test_scores = learning_curve(clf,
                                                y_X, y, cv=cv,
                                                scoring="mean_squared_error",
                                                train_sizes=[0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8, 0.9, 1.0])
	plt.plot(training_sizes, train_scores.mean(axis=1), label="training scores")
	plt.plot(training_sizes, test_scores.mean(axis=1), label="test scores")
	plt.title("RandomForestClassifier (cv=cv)");
	plt.legend(loc="best");plt.show();

	#�����s��F�e�N���X�̃T���v�����ǂ�ȊԈ�����������̂������I�ɔc���F���N���X�̕��ޖ��ŗL���ȕ]���@
	print '\n�����s��@Confusion Matrix�i�]�����@�R�j\n';
	plt.matshow(confusion_matrix(y_true, y_pred_test))
	plt.colorbar()
	plt.xlabel("Predicted label")
	plt.ylabel("True label");plt.show();

	##�œK�ȃp�����[�^�T���ɂ����Overfitting��Underfitting���𒲂ׂ�R�[�h END####
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
