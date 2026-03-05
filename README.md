# DCLPV2_from0305

本仓库当前包含两个尚未合并的实验版本，用于机器人导航训练（基于 Stage/StageROS）：

* `0205_TDE_AddPolicy`：主要对**输入**做了调整
* `SingleStreamCNN`：主要对**网络结构**做了调整

> 说明：这两个版本目前是独立分支思路，**暂时还没有合并**。

---

## 项目结构（示意）

```text
DCLPV2_from0305/
├── 0205_TDE_AddPolicy/
├── SingleStreamCNN/
└── stage_ros/   (StageROS 相关)
```

---

## 环境安装

建议先使用 `environment.yml` 创建虚拟环境。

### 使用 conda 安装虚拟环境

```bash
conda env create -f environment.yml
conda activate <你的环境名>
```

> 环境名以 `environment.yml` 中定义的 `name:` 为准。

---

## StageROS 部署流程（重要）

当前推荐的部署顺序如下：

### 第一步：先只部署 `stage_ros`

先将 `stage_ros` 放到 `catkin_ws/src` 目录下，然后编译：

```bash
cd ~/catkin_ws/src
# 将 stage_ros 放到这里（拷贝或软链接均可）

cd ~/catkin_ws
catkin_make
```

### 第二步：再放入算法代码

`catkin_make` 成功后，再将以下两个目录放到 `catkin_ws/src` 中：

* `0205_TDE_AddPolicy`
* `SingleStreamCNN`

这两个目录对应两个不同版本（一个改输入，一个改网络），目前未合并。

---

## 部署完成后的检查（建议）

在完成 `stage_ros` 部署后，建议先分别运行测试，确认 **动态障碍物能够正常移动**。
如果动态障碍物运动正常，再进行后续训练。

---

## 训练运行方式

运行训练脚本（直接执行即可）：

```bash
./run_mpi_training_fixed_size.sh
```

> 请先确认：
>
> * 已进入正确的代码目录
> * 虚拟环境已激活
> * ROS / catkin 工作空间环境已 source（如 `devel/setup.bash`）

---

## 版本说明（当前状态）

### 1) `0205_TDE_AddPolicy`

* 主要修改：**输入处理方式调整**
* 用途：用于验证输入侧改动对训练效果的影响

### 2) `SingleStreamCNN`

* 主要修改：**网络结构调整**
* 用途：用于验证网络结构改动对性能的影响

> 两个版本目前用于并行对比实验，后续可根据结果考虑合并。

---

## 常见注意事项

* 请确保 `stage_ros` 先编译成功，再加入算法目录。
* 若运行异常，优先检查：

  * ROS 环境是否正确 source
  * `catkin_make` 是否成功
  * 动态障碍物是否正常运动
  * 脚本执行权限是否开启（必要时 `chmod +x run_mpi_training_fixed_size.sh`）


```
```
