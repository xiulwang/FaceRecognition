import numpy as np
import scipy.io as sio
import sklearn.metrics as metrics
from skimage.feature import hog
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


def get_data():
    data = []
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for i in ['05','07','09','27','29']:
        data.append(sio.loadmat('PIE/Pose'+i+'_64x64.mat'))

    for i in range(len(data)):
        for j in range(data[i]['fea'].shape[0]):
            if data[i]['isTest'][j][0] < 1.0:
                train_data.append(data[i]['fea'][j].reshape(64,64))
                train_label.append(int(data[i]['gnd'][j]))
            else:
                test_data.append(data[i]['fea'][j].reshape(64,64))
                test_label.append(int(data[i]['gnd'][j]))
    return train_data, train_label, test_data, test_label


def hog_show(img, orient, pix_per_cell, cell_per_block):
    fea, img_show = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                               cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=True,
                               feature_vector=False)
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(img, 'gray')
    plt.title('Original')
    plt.subplot(122)
    plt.imshow(img_show, 'gray')
    plt.title('HOG')
    plt.show()

	
def get_hog_feature(x_train,x_test, orient, pix_per_cell, cell_per_block):
    train_hog_feature = []
    test_hog_feature = []
    for i in range(len(x_train)):
        img = x_train[i]
        train_hog_feature.append(hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                               cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=False,
                               feature_vector=False))
    for j in range(len(x_test)):
        img = x_test[j]
        test_hog_feature.append(hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                               cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=False,
                               feature_vector=False))
    return train_hog_feature, test_hog_feature


train_data, train_label, test_data, test_label = get_data()
orient = 9
pix_per_cell = 8
cell_per_block = 2

train_hog_feature, test_hog_feature = get_hog_feature(train_data, test_data, orient, pix_per_cell, cell_per_block)
train_hog_feature = np.array(train_hog_feature)
test_hog_feature = np.array(test_hog_feature)
train_hog_feature = train_hog_feature.reshape(len(train_label), -1)
test_hog_feature = test_hog_feature.reshape(len(test_hog_feature), -1)

#classifier
svc = LinearSVC()
svc.fit(train_hog_feature, train_label)
print 'Accuracy = ', round(svc.score(test_hog_feature, test_label), 4)
predict_y = svc.predict(test_hog_feature)
print(metrics.classification_report(test_label,predict_y))

# save the misclassification information
fea = []
hog_fea = []
ans = []
pre_ans = []
predict_y = svc.predict(test_hog_feature)
for i in range(len(predict_y)):
    if predict_y[i] != test_label[i]:
        fea.append(test_data[i])
        hog_fea.append(test_hog_feature[i])
        ans.append(test_label[i])
        pre_ans.append(predict_y[i])

sio.savemat('wrongClassification', {
    'features': fea,
    'hogFeatures': hog_fea,
    'label': ans,
    'predictLabel': pre_ans
})

# show misclassification picture
for i in range(len(fea)):
    plt.subplot(3,4,i+1)
    plt.imshow(fea[i],'gray')
    plt.xticks([])
    plt.yticks([])
    plt.title(i+1)
plt.show()

# show origi picture and hog picture
hog_show(train_data[0], orient, pix_per_cell, cell_per_block)













