import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image


# 1. 加载并预处理数据集
def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 归一化像素值 (0-255) -> (0.0-1.0)
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # 添加通道维度 (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # 将标签转为one-hot编码
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


# 2. 构建CNN模型
def build_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# 3. 训练模型
def train_model(model, x_train, y_train, x_test, y_test):
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=5,
        validation_split=0.1,
        verbose=1
    )

    # 在测试集上评估
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n测试集损失: {score[0]:.4f}")
    print(f"测试集准确率: {score[1] * 100:.2f}%")



    return model, history


# 4. 预测单张图像
def predict_image(model, img):
    if img.ndim == 3:
        img = np.expand_dims(img, 0)  # 添加批次维度

    predictions = model.predict(img, verbose=0)
    predicted_num = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    return predicted_num, confidence


# 5. 可视化预测结果
def display_prediction(img, prediction, confidence):
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"Predict: {prediction} | Confidence: {confidence * 100:.1f}%")
    plt.axis('off')
    plt.show()


def load_custom_image(path):
    img = Image.open(path).convert('L')  # 转为灰度
    img = img.resize((28, 28))  # 调整大小
    img_array = np.array(img)

    # 反转颜色：黑底白字 → 白底黑字
    img_array = 255 - img_array

    # 归一化并添加维度
    img_array = img_array.astype("float32") / 255
    return np.expand_dims(img_array, -1)  # 添加通道维度


# 主程序
if __name__ == "__main__":
    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_data()

    # 构建模型
    model = build_model()
    model.summary()

    # 训练模型
    print("\ntraining...")
    model, history = train_model(model, x_train, y_train, x_test, y_test)

    # 随机选择测试图像进行预测
    sample_idx = np.random.randint(0, x_test.shape[0])
    sample_img = x_test[sample_idx]
    true_label = np.argmax(y_test[sample_idx])

    # 预测
    predicted_num, confidence = predict_image(model, sample_img)

    # 显示结果

    display_prediction(sample_img, predicted_num, confidence)





    # 使用示例：
    custom_img = load_custom_image("/Users/ywsun/Desktop/pythonProject/ITSC/samples/IMG_2618 小.png")
    predicted_num, confidence = predict_image(model, custom_img)
    print(f"Prediction: {predicted_num} (Confidence: {confidence * 100:.1f}%)")
    display_prediction(custom_img, predicted_num, confidence)