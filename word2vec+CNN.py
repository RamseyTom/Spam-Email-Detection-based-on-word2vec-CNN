import jieba
import re
import numpy as np
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense,Dropout
from keras.utils import to_categorical
import seaborn as sb

class ChineseEmailClassification():
    #初始化
    def __init__(self):
        self.max_sequence_length = 300  #最大序列长度
        self.embedding_dim = 64         #词向量维度
        self.epochs = 10                #训练次数
        self.batch_size = 2048          #每批样本数
        self.num_filters = 64           #卷积核数目
        self.kernel_size = 5            #卷积核尺寸


    """                               数据预处理【得到完整的路径，将邮件信息转为0∪1】                                   """

    #获取jieba分词后的文本，如" 我 爱 中文 ..."
    def get_mail_text(self, mailPath):
        mail = open(mailPath, "r", encoding="gb2312", errors="ignore")
        mailTestList = [text for text in mail]
        #去除文件开头部分
        XindexList = [mailTestList.index(i) for i in mailTestList if re.match("[a-zA-Z0-9]", i)]
        textBegin = int(XindexList[-2]) + 1
        text = "".join(mailTestList[textBegin:])
        # 匹配汉字
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
        text = ''.join(chinese_chars)
        #jieba分词
        seg_list = jieba.cut(text, cut_all=False)
        text = " ".join(seg_list)
        return text

    # 通过index文件获取所有文件路径与标签值
    def get_paths_labels(self):
        targets = open("../trec06c/full/index", "r", encoding="gb2312", errors="ignore")
        targetList = [t for t in targets]
        newTargetList = [target.split() for target in targetList if len(target.split()) == 2]
        pathList = [path[1].replace("..", "../trec06c") for path in newTargetList]  #完整路径
        labelList = [label[0] for label in newTargetList]
        return pathList, labelList

    #对准备识别的文本预处理
    def preprocess_email_text(self,text):
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]', text)
        text = ''.join(chinese_chars)
        seg_list = jieba.cut(text, cut_all=False)
        preprocessed_text = " ".join(seg_list)
        return preprocessed_text

    #标签转化，spam，ham 分别对应 1，0
    def transform_label(self, labelList):
        return [1 if label == "spam" else 0 for label in labelList]

    """                            数据向量化与搭建模型                        """
    #文本填充序列,标签转为独热编码,得到词汇表
    def prepare_data_for_cnn(self, content_list, labels):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(content_list)
        sequenses = tokenizer.texts_to_sequences(content_list)#转化整数编码[[3,23,2],[3,54,...],...]
        X_cnn = pad_sequences(sequenses,maxlen=self.max_sequence_length)
        y_cnn = to_categorical(labels)
        """独热编码：[0,1,1,1,0]转换为
                [[1. 0.]
                 [0. 1.]
                 [0. 1.]
                 [0. 1.]
                 [1. 0.]]  
        """
        return X_cnn, y_cnn, tokenizer.word_index

    #搭建CNN模型
    def build_cnn_model(self, vocab_size, embedding_matrix):
        model_cnn = Sequential()
        model_cnn.add(
            Embedding(input_dim=vocab_size + 1, output_dim=self.embedding_dim, input_length=self.max_sequence_length,
                      weights=[embedding_matrix], trainable=False))#trainable不更新权重
        model_cnn.add(Conv1D(self.num_filters, self.kernel_size, activation='relu'))
        model_cnn.add(GlobalMaxPooling1D())
        model_cnn.add(Dense(32, activation="relu"))
        model_cnn.add(Dropout(0.5))
        model_cnn.add(Dense(2, activation='softmax'))
        model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])#交叉熵损失
        return model_cnn

    #训练模型
    def train_model(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_test, y_test))

    #预测函数
    def predict(self, email_texts):
        preprocessed_texts = [self.preprocess_email_text(text) for text in email_texts]
        sequences = self.tokenizer.texts_to_sequences(preprocessed_texts)
        X_pred = pad_sequences(sequences, maxlen=self.max_sequence_length)
        predictions = self.model.predict(X_pred)
        print(predictions)#[[0.232,0.768],[...]]
        class_labels = np.argmax(predictions, axis=1)
        return list(class_labels)

    #评估函数
    def evaluate_model(self, X_test, y_test):
        #预测值
        y_pred = self.model.predict(X_test)
        #预测值转换
        y_pred_labels = np.argmax(y_pred, axis=1)
        #真实值转换
        y_true_labels = np.argmax(y_test, axis=1)

        #混淆矩阵
        conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
        #(精度，召回率，F1分数，样本数)
        class_report = classification_report(y_true_labels, y_pred_labels)
        #混淆矩阵热力图
        self.plot_confusion_matrix(conf_matrix)

        return conf_matrix, class_report

    #绘制混淆矩阵
    def plot_confusion_matrix(self, conf_matrix):
        plt.figure(figsize=(8, 6))
        sb.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Spam', 'Spam'], yticklabels=['Non-Spam', 'Spam'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()


    #主函数
    def main(self):
        #数据加载
        path_list, label_list = self.get_paths_labels()

        #获得分词文本
        content_list = [self.get_mail_text(file_path) for file_path in path_list]

        #转换标签
        label_list = self.transform_label(label_list)

        #训练word2vec模型
        sentences = [jieba.lcut(text) for text in content_list]#lcut是列表类型，cut为生成器类型
        self.word2vec_model = Word2Vec(sentences, vector_size=self.embedding_dim, window=5, min_count=1, workers=4)

        #准备数据
        X_cnn, y_cnn, word_index = self.prepare_data_for_cnn(content_list, label_list)

        #准备嵌入矩阵
        embedding_matrix = np.zeros((len(word_index) + 1, self.embedding_dim))#全零矩阵行和列
        for word, i in word_index.items():
            if word in self.word2vec_model.wv:#word2vec_model.wv结果KeyedVectors<vector_size=100, 99 keys>
                embedding_matrix[i] = self.word2vec_model.wv[word]

        #搭建CNN模型
        self.model = self.build_cnn_model(len(word_index), embedding_matrix)

        #获取词汇表
        self.tokenizer = Tokenizer()
        self.tokenizer.word_index = word_index

        # 划分训练集与验证集
        X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_cnn, test_size=0.3, random_state=42)

        # 训练模型
        self.train_model(X_train, y_train, X_test, y_test)
        # 混淆矩阵，(精度，召回率，F1分数，样本数)
        conf_matrix, class_report = self.evaluate_model(X_test, y_test)
        print("Confusion Matrix:\n", conf_matrix)
        print("\nClassification Report:\n", class_report)


a = ChineseEmailClassification()
a.main()
test_emails = ["""尊敬的贵公司(财务/经理)负责人您好！  
        我是深圳金海实业有限公司（广州。东莞）等省市有分公司。  
    我司有良好的社会关系和实力，因每月进项多出项少现有一部分  
    发票可优惠对外代开税率较低，增值税发票为5%其它国税.地税.     
    运输.广告等普通发票为1.5%的税点，还可以根据数目大小来衡  
    量优惠的多少，希望贵公司.商家等来电商谈欢迎合作。""","港澳台代表团参加亚运会","""希望这封邮件能找到你一切安好。感谢你一直的支持和友谊。
    最近天气转暖，我计划组织一次小型聚会，希望你能参加。我们可以分享一些美食，畅谈近况，共度愉快时光。请告诉我你的方便时间，期待和你见面。""",
               """在实际应用中，你可能需要检查模型的训练数据，尝试调整模型的架构或超参数，以及考虑更多的特征工程，
               以提高模型的性能和分类准确度。这可能包括调整词向量维度、卷积核大小等"""]
xs = a.predict(test_emails)
print(xs)
