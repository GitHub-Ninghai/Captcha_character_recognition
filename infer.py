import os
import numpy as np
import paddle
from utils.model import Model
from utils.data import process
from utils.decoder import ctc_greedy_decoder
import streamlit as st
from PIL import Image
from io import BytesIO

def infer(path):
    with open('dataset/vocabulary.txt', 'r', encoding='utf-8') as f:
        vocabulary = f.readlines()

    vocabulary = [v.replace('\n', '') for v in vocabulary]

    save_model = 'models/'
    model = Model(vocabulary, image_height=32)
    model.set_state_dict(paddle.load(os.path.join(save_model, 'model.pdparams')))
    model.eval()
    # 定义保存上传图片的文件夹

    data = process(path, img_height=32)
    data = data[np.newaxis, :]
    data = paddle.to_tensor(data, dtype='float32')
    # 执行识别
    out = model(data)
    out = paddle.transpose(out, perm=[1, 0, 2])
    out = paddle.nn.functional.softmax(out)[0]
    # 解码获取识别结果
    out_string = ctc_greedy_decoder(out, vocabulary)
    return out_string
    # print('预测结果：%s' % out_string)

# 创建一个简单的Streamlit应用
def main():
    st.title('图片识别应用')
    st.write('请上传图片进行识别：')
    UPLOAD_FOLDER = 'dataset/'
    # 上传图片
    uploaded_file = st.file_uploader("选择图片", type=["jpg", "png"])

    if uploaded_file is not None:
        # 读取上传的文件内容
        image_bytes = uploaded_file.read()
        image = Image.open(BytesIO(image_bytes))
        st.image(image, caption='上传的图片')

        # 构造保存图片的路径
        image_name = uploaded_file.name
        image_path = os.path.join(UPLOAD_FOLDER, image_name)

        # 保存图片到本地
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        st.success(f'图片已保存至 {image_path}')

        # 执行识别并显示结果
        if st.button('运行识别'):
            prediction = infer(image_path)
            st.write('识别结果：', prediction)
    else:
        st.write('请先上传图片。')


if __name__ == '__main__':
    main()