
############
#    ��������    #
############

����> �Ƽ�ʹ�ã�
Deep Belief Network (DBN) 
Stacked Autoencoder (sAE) 
Stacked Sparse Autoencoder (sSAE) 
Stacked Denoising Autoencoders (sDAE) 
����> ���Ը��õ�ģ�ͣ�
Convolutional Neural Network (CNN) 
Recurrent Neural Network (RNN) 
Long Short Term Memory (LSTM) 

############
#    ��������    #
############

pip install tensorflow
pip install keras
pip install librosa �������������࣬ѡװ��
pip install --upgrade numpy pandas��������һ����������İ���Ҫ�����ˣ�

############
#    �汾��Ϣ    #
############

Note �û�����ͨ��model.py�ļ�����һЩ���ܵĿ��أ� 
���� self.show_pic =True       # show curve in 'Console'?
���� self.tbd = False              # open/close tensorboard
���� self.save_model = False  # save/ not save model
���� self.plot_para=False       # plot W image or not
���� self.save_weight = False # save W matrix or not
���� self.do_tSNE = False       # do t-SNE or not

Version 2018.11.7
New �������������ݼ���һ�����ڷ��࣬һ������Ԥ��
New ����t-SNE��ά���ӻ�
Chg �������� use_for = 'prediction' ʱ��Bug

Version 2018.6.1 
New �����˻���ѵ������ͼ��Ԥ���ǩ�ֲ�ͼ��Ȩֵͼ�Ĺ��� 
Chg ��д��SAE�����ڿ��Է���ʹ���� 
Chg ������������к���run_sess�ŵ���base_func.py 
Chg �ع��ǿ���ʵ�ֵģ���Ҫ���� use_for = 'prediction' 

############
#    ���Խ��   #
############

����minst���ݼ����࣬���еõ���ȷ�ʿɴ�98.78%��
����Urban Sound Classification�������࣬��ȷ�ʴ�73.37%��
����Big Mart Sales IIIԤ�⣬RMSEΪ1152.04

�ܵĽ��������̫�ߣ��и��õķ�����ͽ̡�
��������δ�������׷�����ӭ�����ĺ��ҽ�����

############
#    �ο�����  #
############

TF����������http://www.cnblogs.com/wuzhitj/p/6431381.html 
RBMԭ��https://blog.csdn.net/itplus/article/details/19168937 
HintonԴ�룺http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html 
sDAEԭ���ģ�http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf 
sSAE����TE��https://www.sciencedirect.com/science/article/pii/S0169743917302496 
RNNԭ��https://zhuanlan.zhihu.com/p/28054589 
LSTM��https://www.jianshu.com/p/9dc9f41f0b29 
Tensorboard��https://blog.csdn.net/sinat_33761963/article/details/62433234 

############
#    My blog  #
############

֪����https://www.zhihu.com/people/fu-zi-36-41/posts 
CSDN��https://blog.csdn.net/fuzimango/article/list/ 
QQȺ��640571839 