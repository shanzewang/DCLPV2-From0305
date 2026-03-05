#!/bin/bash
# MPI多进程机器人导航训练启动脚本
# 参考RL collision项目的MPI架构设计

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 设置ROS环境
source /opt/ros/noetic/setup.bash
source ../../../devel/setup.bash    #! 需要改到自己的stage文件夹下
# source /home/eit-robot/sz_ipaprec_ws/devel/setup.bash

# 正确初始化conda
eval "$(conda shell.bash hook)"
conda activate dclpv2   # ! 改为自己的虚拟环境名称

# 配置参数
NUM_PROCESSES=10  # 1个主控机器人 + 8个动态障碍物 + 1个目标点发布机器人
WORLD_FILE="map/dynamic_2ped_static_singlestream.world"
ROS_PORT=$((RANDOM % 5001 + 10000))


# 定义清理函数
cleanup() {
    echo "🧹 正在清理相关进程..."
    
    # 杀死MPI训练进程
    if [ ! -z "$MPI_PID" ]; then
        echo "🔄 关闭 MPI训练进程 (PID: $MPI_PID)"
        kill -TERM $MPI_PID 2>/dev/null
        sleep 2
        kill -9 $MPI_PID 2>/dev/null
    fi
    
    # 杀死Stage仿真器
    if [ ! -z "$STAGE_PID" ]; then
        echo "🎮 关闭 Stage仿真器 (PID: $STAGE_PID)"
        kill -TERM $STAGE_PID 2>/dev/null
        sleep 1
        kill -9 $STAGE_PID 2>/dev/null
    fi
    
    # 杀死roscore
    if [ ! -z "$ROSCORE_PID" ]; then
        echo "🌐 关闭 ROS Master (PID: $ROSCORE_PID)"
        kill -TERM $ROSCORE_PID 2>/dev/null
        sleep 1
        kill -9 $ROSCORE_PID 2>/dev/null
    fi
    
    # 额外清理可能遗留的进程
    echo "🔍 查找并清理遗留进程..."
    pkill -f "stageros.*$WORLD_FILE" 2>/dev/null
    pkill -f "roscore.*$ROS_PORT" 2>/dev/null
    pkill -f "mpiexec.*torchdclp_simple_singlestream.py" 2>/dev/null
    
    echo "✅ 清理完成"
}

# 捕获退出信号
trap "cleanup; exit" SIGINT SIGTERM

echo "🚀 启动MPI多进程机器人导航训练系统"
echo "📊 配置: $NUM_PROCESSES 个进程 (1主控 + 2动态障碍物 + 1目标点发布机器人)"

# ============= 步骤1: 启动ROS Master =============
echo "🌐 步骤1: 启动ROS Master (端口: $ROS_PORT)"
export ROS_MASTER_URI="http://localhost:$ROS_PORT"
roscore -p $ROS_PORT &
ROSCORE_PID=$!
echo "✅ ROS Master启动完成 (PID: $ROSCORE_PID)"
sleep 3

# ============= 步骤2: 启动Stage仿真器 =============
echo "🎮 步骤2: 启动Stage仿真器"
echo "📁 World文件: $WORLD_FILE"

# 检查world文件是否存在
if [ ! -f "$WORLD_FILE" ]; then
    echo "❌ 错误: World文件 $WORLD_FILE 不存在!"
    cleanup
    exit 1
fi


rosrun stage_ros1 stageros $WORLD_FILE &
STAGE_PID=$!
echo "✅ Stage仿真器启动完成 (PID: $STAGE_PID) - 🎮 GUI可见"
sleep 5

# ============= 步骤3: 等待Stage完全启动 =============
echo "🔍 步骤3: 等待Stage仿真器完全启动"
sleep 5
echo "✅ Stage启动等待完成"

# ============= 步骤4: 启动MPI多进程训练 =============
echo "🔥 步骤4: 启动MPI多进程训练"
echo "🤖 进程分配:"
echo "   - 进程 0: 主控机器人 (robot_0)"
echo "   - 进程 1-2: 动态障碍物 "
echo "   - 进程 3: 目标点发布 (robot_45)"

# 使用mpiexec启动多进程训练
# 参考 RL collision 项目: mpiexec -np 24 python ppo_stage1.py
mpiexec -np $NUM_PROCESSES python torchdclp_simple_singlestream.py & 
MPI_PID=$!

echo "✅ MPI训练启动完成 (PID: $MPI_PID)"
echo "📊 监控训练进度，按 Ctrl+C 停止训练"

# ============= 步骤5: 监控训练进程 =============
# 等待MPI训练进程结束
wait $MPI_PID

echo "🎯 训练进程结束"

# 清理资源
cleanup
exit 0 