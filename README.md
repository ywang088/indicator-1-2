# 指标 1.2测试 运行方法

本项目提供两个独立脚本：

| 任务 | 脚本 | 输入 | 输出 | 计算设备 |
|---|---|---|---|---|
| 18 条轨迹建图 | `scripts/01_build_18.sh` | RGB、轨迹清单、同步几何、SAM3/OCR 权重 | 每条轨迹的图、检测框图片、更新与查询日志、本轮合并图 | NVIDIA GPU |
| 全场景图存储比较 | `scripts/02_compare_full_map.sh` | 已保存的全场景图和对应点云 | 存储对比表、范围对照图、检查结果 | CPU |

两个脚本可以单独执行。建图结果保存在 `results/`，存储比较使用 `data/full_map/` 中预存的封存图。

## 1. 准备运行环境

推荐在 Linux x86_64 服务器上使用 Docker。以下安装命令以 Ubuntu 为例，所有项目命令均在包含 `compose.yaml` 的项目根目录执行。

| 项目 | 配置 |
|---|---|
| 建图显卡 | 建议 RTX 4090 24 GB 或 H100；参考运行使用 H100 80 GB |
| 显卡驱动 | 支持 CUDA 13.0 的 NVIDIA 驱动 |
| CPU | 建议 4 核以上 |
| 主机内存 | 建议 32 GB 以上 |
| 磁盘 | 建议预留 50 GB 以上，保存数据、权重、镜像和结果 |
| 容器工具 | Docker Engine、Docker Compose 插件；运行建图另需 NVIDIA Container Toolkit |
| 网络 | 首次下载数据和镜像时联网；加载镜像后可离线运行 |

发布镜像已包含 Python 3.11、PyTorch 2.12.0 + CUDA 13.0、torchvision 0.27.0、SAM3 0.1.4、EasyOCR 1.7.2，以及 SAM3、BPE、OCR 模型文件。使用 Docker 时，主机准备好驱动和容器工具即可。

只运行第二个脚本时，使用 CPU 环境并跳过 NVIDIA 工具包和 GPU 检查步骤。

### 1.1 安装 Docker

已有 Docker 的机器先检查：

```bash
docker version
docker compose version
```

Ubuntu 首次安装命令如下，来源见 [Docker 安装说明](https://docs.docker.com/engine/install/ubuntu/)：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git python3
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF_DOCKER
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF_DOCKER
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run --rm hello-world
```

后续示例使用 `docker`。如果当前账号没有 Docker 权限，在 Docker 命令前加 `sudo`。

### 1.2 配置 GPU 容器支持（运行建图时需要）

先检查主机显卡：

```bash
nvidia-smi
```

确认显卡正常显示、驱动支持 CUDA 13.0。535 系列驱动需先升级到兼容版本。

按 [NVIDIA Container Toolkit 安装说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)安装：

```bash
sudo apt-get install -y curl gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 2. 获取代码和数据

有完整交付文件夹时，直接进入该文件夹，对照第 3 节检查文件，再执行第 4 节。通过 GitHub 获取时，按下面步骤下载。

### 2.1 下载代码

仓库公开可访问，使用 Git 直接下载：

```bash
git clone https://github.com/ywang088/indicator-1-2.git
cd indicator-1-2
mkdir -p release_assets results
```

已有 Docker 的 Ubuntu 主机，可执行 `sudo apt-get install -y git python3` 准备下载工具。后续下载脚本使用 Python 标准库，支持匿名下载与 SHA256 校验。

### 2.2 下载固定数据（两个脚本都需要）

固定数据包含 6,723 张 RGB、18 条轨迹清单、370 个关键帧对应的几何数据、点云和预存图。下载 108 个数据分卷并解压：

```bash
python3 code/download_assets.py data
(cd release_assets && sha256sum -c data-SHA256SUMS --ignore-missing)
cat release_assets/data.tar.gz.part* | tar -xzf -
tar -xzf release_assets/frozen-map-metadata.tar.gz
```

下载脚本同时获取封存图的配置和校验文件，已完成且校验通过的分卷会自动跳过。

### 2.3 下载内含模型的 Docker 镜像

接手运行者下载完整镜像即可，SAM3 和 OCR 权重已包含在镜像内。

```bash
python3 code/download_assets.py image
(cd image_assets && sha256sum -c SHA256SUMS)
cat image_assets/gpu-with-models.tar.gz.part* | gzip -dc | docker load
```

成功后显示 `Loaded image: indicator-1-2:bundled-gpu`。模型位于镜像的 `/opt/indicator/models/`，数据挂载在 `/workspace/data/`，运行结果写入 `/workspace/results/`。

所有数据压缩包在项目根目录解压，解压后应直接出现 `data/dataset/` 和 `data/full_map/`。镜像分卷通过 `docker load` 加载。

## 3. 检查文件夹结构

```text
indicator-1-2/
├── README.md
├── Dockerfile
├── compose.yaml
├── scripts/
│   ├── 01_build_18.sh              # 18条轨迹建图
│   └── 02_compare_full_map.sh      # 预存全场景图比较
├── code/                          # 实现代码、依赖清单、SAM3安装包
├── data/
│   ├── dataset/
│   │   ├── rgb/                   # 原始RGB图片
│   │   ├── config.json            # 轨迹、关键帧、位姿等配置
│   │   └── trajectories/          # T01至T18清单
│   ├── models/                    # 仅自行构建镜像时准备；发布镜像已内含
│   │   ├── sam3.pt
│   │   ├── bpe_simple_vocab_16e6.txt.gz
│   │   └── ocr/
│   │       ├── craft_mlt_25k.pth
│   │       └── zh_sim_g2.pth
│   └── full_map/                  # 预存封存图及对应点云
│       ├── graph.sqlite
│       ├── graph.json
│       ├── points.npy
│       ├── graph_scope.json
│       ├── pointcloud_scope.json
│       ├── frozen_manifest.json
│       ├── audit/                 # 封存配置及检测记录
│       └── scope_evidence/        # 逐帧几何和扫描记录
└── results/                       # 运行后生成的结果
```

## 4. 检查镜像和运行环境

两个脚本使用同一个已加载镜像：

```bash
docker image inspect indicator-1-2:bundled-gpu --format '{{.Os}}/{{.Architecture}}'
```

预期输出 `linux/amd64`。按第 2.3 节加载发布镜像后，可直接执行下面的命令。

运行建图前检查容器内 GPU：

```bash
docker compose run --rm mapping python3 -c "import torch; assert torch.cuda.is_available(); print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name())"
```

成功时会打印 PyTorch 版本、CUDA 版本和显卡名称。

首次运行建图，建议先做两帧检查：

```bash
docker compose run --rm mapping bash scripts/01_build_18.sh --smoke-frames 2
```

终端出现 `GPU smoke run complete` 表示两帧推理完成，输出位于 `results/runs/运行编号/`。

## 5. 运行脚本1：18条轨迹建图

### 5.1 启动

```bash
docker compose run --rm mapping bash scripts/01_build_18.sh
```

脚本校验原始数据，对 370 个关键帧运行 SAM3 和 OCR，完成物体三维定位、柱子颜色与区域字母绑定，再按轨迹时间顺序增量更新四层图。每条轨迹从空图开始，四层分别为帧、物体、门户、区域。三维定位使用数据包提供的同步几何、标定和位姿。

H100 80 GB 的参考运行约 26 分钟。运行过程中，终端显示当前帧或处理阶段。

### 5.2 查看输出

```text
results/
├── trajectories/T01...T18/
│   ├── summary.json              # 本条轨迹统计
│   ├── graph.json                # 图字典：nodes、edges、metadata
│   ├── graph.sqlite              # 可查询的图数据库
│   ├── frames/
│   │   ├── 帧号.jpg              # RGB检测框、类别和物体ID
│   │   └── 帧号.json             # 检测框与图节点ID对应表
│   ├── all_steps.jsonl           # 按时间顺序记录全部输入
│   ├── updates.jsonl             # 图更新日志
│   └── query_log.jsonl           # 查询记录
├── online_full_map/              # 本轮18条轨迹的合并图和summary
├── runs/运行编号/                 # 各阶段检测、掩膜、中间数据和日志
├── execution.json               # 本轮执行信息和完成标记
├── online_automated_check.json   # 建图输出检查
└── report.json                  # 18条轨迹汇总
```

直接打开 JPG 即可查看带框图像。每条 `summary.json` 的格式示例：

```json
{
  "summary": {
    "frames": 21,
    "objects": 855,
    "regions": 2,
    "portals": 1,
    "unlocalized_objects": 523
  }
}
```

以上数字来自参考运行；`unlocalized_objects` 表示已识别但没有三维位置的物体数量。全部轨迹共输出 370 张检测图和 370 份对应 JSON。

### 5.3 判断是否成功

终端最后出现 `Fresh SAM3 + OCR + geometry + 18-track mapping completed`，同时：

- `results/execution.json` 中 `completed` 为 `true`，`processed_frames` 为 `370`。
- `results/online_automated_check.json` 中 `automated_checks_passed` 为 `true`。
- `results/trajectories/` 下存在 T01 至 T18。

重新运行会创建新的 `runs/运行编号/`，并更新轨迹导出和 `online_full_map/`。需要保留多轮导出时，先将当前 `results/` 归档。

## 6. 运行脚本2：读取预存图进行比较

### 6.1 启动

```bash
docker compose run --rm compare
```

该命令在 CPU 上运行，直接读取 `data/full_map/` 中的封存图和点云，`results/` 为空时也可以启动。

### 6.2 查看输出

| 文件 | 内容 |
|---|---|
| `results/storage_comparison.json` | 基线字节数、认知图字节数、降低率 |
| `results/storage_comparison.csv` | 纳入统计的文件、字节数和哈希 |
| `results/storage_report.json` | 存储格式和统计说明 |
| `results/scope_consistency.json` | 帧集合、坐标范围和几何一致性检查 |
| `results/scope_consistency.png` | 预存图、点云和18条轨迹的范围对照图 |
| `results/frozen_map_automated_check.json` | 封存输入与比较结果检查 |

配套封存数据的预期存储结果：

```json
{
  "baseline_bytes": 14653609,
  "cognitive_bytes": 11044558,
  "reduction_percent": 24.62909307870846
}
```

统计对象为 NPY 点云及其范围文件，与 SQLite 图及其范围文件。范围对应固定 18 条轨迹的观测地图。

### 6.3 判断是否成功

终端出现 `Frozen-map comparison checks passed`，随后打印上述存储结果。同时：

- `scope_consistency.json` 中 `scope_consistent` 为 `true`。
- `frozen_map_automated_check.json` 中 `automated_checks_passed` 为 `true`。

## 7. 常见问题

| 现象 | 处理方法 |
|---|---|
| 下载失败或连接中断 | 检查 GitHub 网络连接后重新执行下载命令；校验通过的完整文件会跳过 |
| 提示 Verified existing | 对应文件已通过哈希校验，可继续下一步 |
| 提示找不到bundled-gpu镜像 | 完成第2.3节镜像下载与加载 |
| 提示缺少原图、配置或封存校验文件 | 检查第2节两个固定数据下载步骤和解压目录 |
| `Model file missing` 或缺少OCR权重 | 加载第2.3节的bundled-gpu镜像；自行构建时检查data/models目录 |
| CUDA不可用或驱动版本不足 | 检查 `nvidia-smi`、CUDA 13.0驱动兼容性和NVIDIA Container Toolkit，再执行第4节GPU检查 |
| Docker socket访问被拒绝 | 在Docker命令前加 `sudo`，或配置当前账号的Docker使用权限 |
| 只得到两帧结果 | 去掉 `--smoke-frames 2`，使用第5节正式命令 |
| 原图或封存图哈希不匹配 | 重新下载、校验并解压对应固定数据包 |
| 显存、内存或磁盘不足 | 释放所需资源，或换用符合第1节配置的机器 |
| 进程报错退出 | 查看终端最后的错误，以及 `results/runs/运行编号/*.py.log` |

## 8. 其他运行方式

### 直接使用 Docker 命令

已加载镜像时，可以执行：

```bash
# 脚本1：GPU建图
docker run --rm --gpus all --shm-size=8g -v "$PWD/data:/workspace/data" -v "$PWD/results:/workspace/results" indicator-1-2:bundled-gpu bash scripts/01_build_18.sh

# 脚本2：CPU比较
docker run --rm -v "$PWD/data:/workspace/data:ro" -v "$PWD/results:/workspace/results" indicator-1-2:bundled-gpu bash scripts/02_compare_full_map.sh
```

### 自行构建镜像（开发或修改代码时）

先按第 3 节目录准备本地 `data/models/`。可从原模型附件获取：

```bash
python3 code/download_assets.py models
(cd release_assets && sha256sum -c SHA256SUMS --ignore-missing)
cat release_assets/models.tar.gz.part* | tar -xzf -
docker compose build mapping
```

Dockerfile 会将模型复制到镜像内。首次自行构建需要联网安装依赖。

如需把自行构建的镜像交给其他机器：

```bash
docker save indicator-1-2:bundled-gpu | gzip > gpu-with-models.tar.gz
# 在接收机器上加载
gzip -dc gpu-with-models.tar.gz | docker load
```

### HPC已有Python环境

使用包含上述版本依赖的 Python 环境，设置解释器路径后执行相同入口：

```bash
export INDICATOR_PYTHON=/实际环境路径/bin/python
bash scripts/01_build_18.sh
# 或独立运行比较
bash scripts/02_compare_full_map.sh
```

HPC 原生环境使用本地 data/models/ 中的权重，可按上述模型下载命令准备。HPC 建图需在 GPU 作业中运行；Slurm 模板位于 `code/fresh_hpc.sbatch`，按所在集群填写项目路径、环境路径、分区和账号。文件夹布局与 Docker 方式一致。
