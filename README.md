# map_matching 使用说明



## 配置环境

1.   如在本机环境中已经配置好anaconda（建议）
     +   在项目文件夹中打开终端
     +   windows系统运行指令：`bash run_win.sh`
     +   macOS系统运行指令：`bash run_mac.sh`
2.   如未配置anaconda，则直接导入python3所需库
     +   在项目文件夹中打开终端
     +   运行指令：`pip3 install -r requirements.txt`



## 运行程序

>   注：若正确执行 bash run.sh 指令则会直接激活环境并运行程序

1.   激活anaconda环境

     `conda activate map_match`

2.   在项目文件夹打开终端，执行：

     `python -u 'main.py'`



## 初次使用参数设置

首次运行时会弹出提示：`  -->  Choose the task ID [1,2,3,4,5]:`

只需根据下表的任务分配*【暂定】* 输入对应数字即可。

|          用户          | Task ID |
| :--------------------: | :-----: |
|         杨宇豪         |    1    |
|      田老师电脑1       |    2    |
|      田老师电脑2       |    3    |
| 付学姐（如果方便的话） |    4    |
| 吴学长（如果方便的话） |    5    |



## 结果文件

匹配结果记录在`result/result.json`文件中



## 特别说明

程序运行过程中可以随时通过`ctrl+c`中断，程序会自动记录当前进度。

如需再次运行仅需按**运行步骤**部分操作即可，程序会接续中断前的位置继续运行。