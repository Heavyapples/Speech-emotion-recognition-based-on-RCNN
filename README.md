# 基于随机森林的语音情感识别

### 原数据集enterface database下载
链接：[https://pan.baidu.com/s/1AXb31ov3kJhg5_Bo4C-ElA?pwd=5kxk](https://pan.baidu.com/s/1AXb31ov3kJhg5_Bo4C-ElA?pwd=5kxk)  
提取码：5kxk

## 系统要求
建议使用 python3 或以上版本。

## 数据集格式
请将数据集调整为以下格式：
- 数据集主文件夹包含若干子文件夹
- 每个子文件夹中有6中情绪的子文件夹，每个子文件夹名对应情绪标签
- 每个情绪子文件夹中包含若干个子文件夹，每个子文件夹中包含一个语音文件

调整每个.py代码中的路径参数为相应的路径。

## 项目运行流程
1. 运行 `trans_to_wav.py`：调整语音文件格式（此步可省略如果语音文件已经为wav格式）
2. 运行 `get_feature.py`：提取特征
3. 运行 `train_rcnn.py`：训练模型
4. 运行 `test.py`：测试模型

## 可视化界面
项目包含一个可视化界面文件 `detect_gui.py`，其中集成了完整功能，方便用户使用。

## 使用说明
1. 下载并解压项目文件。
2. 打开 MATLAB，并将当前文件夹设置为项目根目录。
3. 按照上述运行流程执行对应的 MATLAB 文件。
4. 运行 `detect_gui.py` 以使用可视化界面进行导入模型、导入待测音频、预处理音频、提取特征、情感识别。

## 联系我们
如果您在使用过程中有任何问题，请通过以下方式联系我：
- 邮箱：w1372988970@gmail.com

![star-history-2025627 (2)](https://github.com/user-attachments/assets/abf72bca-c068-4839-a220-c3c9cea6e789)
