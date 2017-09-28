（1）数据读取：改写了数据读取脚本，自动读取和编码。
（2）处理过程分：数据集、测试集和验证集。
（3）baseline实验：
		freesense：
			data->butterworth II ->PCA->D4 DWT PRESS->K-NN based on dtw
		问题：
			a.数据运算量非常大，实验中只给出人数低于10的结果，在尝试15个人的时候，计算knn时间很长。
			b.读取视距路径时参数不详细。
		svm:
			data->butterworth II ->PCA->svm
		Smart User Authentication：
			未做
