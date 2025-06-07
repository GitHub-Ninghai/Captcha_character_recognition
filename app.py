import numpy as np
import paddle
from utils.model import Model
from utils.data import process
from utils.decoder import ctc_greedy_decoder
import os
import random
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




# 假设这是验证成功的函数，您需要根据实际情况来实现它
def extract_verification_code_from_filename(filename):
    # 假设文件名格式是 "数字_验证码.jpg"
    # 使用split方法按'_'和'.jpg'分割文件名，并提取验证码部分
    parts = filename.split('_')
    if len(parts) > 1:
        code_with_extension = parts[-1]
        # 去除'.jpg'后缀
        if code_with_extension.endswith('.jpg'):
            return code_with_extension[:-4]  # 返回验证码字符串
    return None  # 如果格式不正确，返回None


def is_verification_successful(prediction, filename):
    # 从文件名中提取验证码
    expected_code = extract_verification_code_from_filename(filename)
    if expected_code is not None:
        # 比较预测结果和期望的验证码是否一致
        return prediction == expected_code
    else:
        # 如果无法从文件名中提取验证码，则认为验证失败
        return False

# 其他函数保持不变...

def main():
    # 设置上方导航栏
    st.sidebar.title('功能导航')
    page = st.sidebar.radio("请选择页面", ("熊光豪毕设简介","验证码上传识别", "辅助输入验证码"))
    if page == "熊光豪毕设简介":
        with open('README.md', 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        image1 = Image.open('./asset/1.png')
        st.image(image1, caption='图1：CRNN的结构1')
        image2 = Image.open('./asset/2.png')
        st.image(image2, caption='图2：CRNN的结构2')
        image3 = Image.open('./asset/3.png')
        st.image(image3, caption='图3：Learning rate')
        image4 = Image.open('./asset/3.jpg')
        st.image(image4, caption='图4：Train Loss')


        st.markdown(markdown_content)
    elif page == "验证码上传识别":

        st.title('验证码上传识别应用')
        # 验证码上传识别的代码保持不变...
        # st.title('验证码识别应用')
        st.write('请上传图片进行识别：')
        UPLOAD_FOLDER = 'dataset/upload'
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

    elif page == "辅助输入验证码":
        st.title('辅助输入验证码功能')

        image_folder = './dataset/images'
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        if image_files:
            random_image_file = random.choice(image_files)
            random_image_path = os.path.join(image_folder, random_image_file)
            random_image = Image.open(random_image_path)
            random_image_bytes = BytesIO()
            random_image.save(random_image_bytes, format='JPEG')
            random_image_bytes.seek(0)

            # 显示图片
            # st.title('验证码如下，请在输入框内输入正确的验证码')
            st.image(random_image, caption='随机验证码图片')


            # 初始化验证码输入框的默认值
            verification_code_input = ''

            # 创建 Streamlit 应用
            if st.button('帮我输入'):
                # 假设 infer 函数返回验证码的预测结果

                prediction = infer(random_image_path)
                # st.write('识别结果：', prediction)

                # 更新验证码输入框的默认值
                verification_code_input = prediction

                # 检查验证是否成功
                if is_verification_successful(prediction, random_image_file):
                    st.info('辅助输入成功！')
                else:
                    st.error('验证码错误，登录失败。')
                    # 输入框和按钮
            verification_code = st.text_input('请输入识别出的验证码', value=verification_code_input)
    else:
            st.error('图片文件夹中没有找到图片文件。')


if __name__ == '__main__':
    # 创建 Streamlit 应用
    st.set_page_config(layout="wide")  # 设置为宽布局
    main()