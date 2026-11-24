# 停车场指标1.2 Docker交付包

本项目以固定的18条IPS-3D停车场轨迹为输入，构建帧—物体—门户—区域四层认知图，并与同范围点云比较实际存储字节数。运行两个入口即可生成图字典、日志、带物体ID的检测图、存储比较表、范围对照图及复核结果。

## 1. 在什么机器和显卡上运行

**本交付包的两个入口不需要显卡，也不需要CUDA。** 它们使用已保存的真实SAM3检测、OCR及几何定位结果，重新执行18条独立建图回放，不会重新运行SAM3神经网络。不会把未检测原始帧伪装成检测帧。

- 可在Linux x86_64服务器（包括装有RTX 4090的服务器）、Intel/Apple Silicon Mac的Docker环境运行。
- 建议至少4个CPU核、8 GB可用内存、30 GB可用磁盘；构建镜像或保存多份压缩包时应预留更多空间。该配置是部署建议，不是性能保证。
- 镜像使用Python 3.11.11及锁定版本的NumPy、SciPy、Pillow、Matplotlib，基础环境见Dockerfile。
- 当前交付回放此前在Mac主机执行；原始LiDAR来源包括4090服务器，补充SAM3检测在HPC上执行。不能将本次回放描述为“4090实测GPU推理”。
- 若以后要从新图像重新执行SAM3，需要另外准备模型权重、对应PyTorch/CUDA环境和GPU；本镜像不包含权重，也不承诺4090上的新图像推理吞吐或显存需求。`code/requirements.txt`保留上游感知依赖参考，**不要用它代替本包的requirements-replay.txt**。

## 2. 文件夹结构

```text
parking_docker/
├── Dockerfile                    # 缓存回放的完整运行环境
├── compose.yaml                  # data、results挂载规则
├── README.md                     # 本说明
├── scripts/
│   ├── 01_build_18.sh            # 第一个入口：18条轨迹建图
│   └── 02_compare_full_map.sh    # 第二个入口：存储比较及完整检查
├── code/
│   ├── delivery_runner.py       # 两个入口的执行器
│   ├── requirements-replay.txt  # 镜像锁定依赖
│   ├── map_pipeline.py          # 图构建和按ID查询
│   ├── color_topology.py        # 柱色与字母绑定、区域连接
│   ├── render_detection_frames.py
│   ├── check_scope_consistency.py
│   ├── final_delivery_check.py
│   ├── export_scene_review.py
│   ├── result_notes/            # 随运行复制到results的说明
│   └── ...                     # 感知、几何、可视化及原有辅助代码
├── data/
│   ├── dataset/
│   │   ├── rgb/                # 6,723张原始RGB
│   │   ├── trajectories/       # T01至T18固定清单
│   │   ├── detections.jsonl    # 370个检测关键帧
│   │   ├── masks/              # SAM3掩膜
│   │   ├── images/              # 原感知缓存预览
│   │   ├── config.json
│   │   ├── provenance.json
│   │   └── selection_manifest.json
│   └── full_map/
│       ├── points.npy          # 同范围合并点云，运行所需输入
│       ├── graph.sqlite        # 合并图运行数据库，入口1重新生成
│       ├── graph.json          # 合并图字典，入口1重新生成
│       ├── graph_scope.json
│       ├── pointcloud_scope.json
│       └── scope_evidence/     # 逐帧几何及来源审核记录
└── results/
    ├── trajectories/T01...T18/
    │   ├── summary.json
    │   ├── graph.json
    │   ├── graph.sqlite
    │   ├── all_steps.jsonl
    │   ├── updates.jsonl
    │   ├── query_log.jsonl
    │   └── frames/             # 帧号.jpg + 帧号.json ID对应表
    ├── report.json
    ├── run_config.json
    ├── data_integrity.json
    ├── storage_comparison.json
    ├── storage_comparison.csv
    ├── storage_report.json
    ├── scope_consistency.json
    ├── scope_consistency.png
    ├── final_automated_check.json
    ├── fresh_process_queries.json
    ├── scene_review/           # 固定出口标识/路口/疑似电梯原图
    ├── 范围对照图说明.txt
    └── 指标1.2文档核对.txt
```

Git仓库存放代码、Docker配置及说明；约7.7 GB的数据和结果通过配套附件提供，不直接提交进Git历史。**只克隆代码不能运行**，必须同时取得数据附件。完整本地文件夹已经包含data及results。

## 3. 安装Docker

### Ubuntu 22.04/24.04服务器

如果已有可用Docker，直接执行`sudo docker run --rm hello-world`验证，无需重复安装。没有Docker时，参考[Docker官方Ubuntu安装文档](https://docs.docker.com/engine/install/ubuntu/)，使用官方软件源：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
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
sudo docker compose version
```

如系统已安装其他Docker发行包，按官方文档先处理冲突。以下示例默认当前用户有Docker访问权限；无权限时在docker命令前加sudo。当前缓存回放不需要安装NVIDIA Container Toolkit，也不需要`--gpus all`。

### Mac

从[Docker Desktop官方页面](https://docs.docker.com/desktop/setup/install/mac-install/)安装匹配Apple Silicon或Intel的版本。启动Docker Desktop，等待服务就绪，在设置中分配至少8 GB内存，再执行：

```bash
docker version
docker compose version
docker run --rm hello-world
```

已有Colima环境也可以使用，确保`colima start --cpu 4 --memory 8`成功后再构建镜像。Docker仅安装命令行而服务未启动，会报“Cannot connect to Docker daemon”。

## 4. 安装本项目和构建镜像

进入完整交付文件夹。确认`data/dataset/rgb/000119.png`与`data/full_map/points.npy`存在。

```bash
cd parking_docker
docker compose build
```

首次构建需要联网下载基础镜像及Python依赖。数据、结果和用户凭证不会打进镜像；镜像中包含运行代码，数据通过挂载读取。构建完成后两个入口可在断网条件下运行。

仅从私有仓库下载时，先登录有访问权限的GitHub账户，再下载配套数据附件，按附件的SHA256清单校验，解压到本目录。不要把数据误放进`parking_docker/parking_docker/data`。

## 5. 按顺序运行两个脚本

```bash
# 入口1：从空图分别回放18条轨迹，再输出合并图
docker compose run --rm parking bash scripts/01_build_18.sh

# 入口2：测量存储、检查范围、独立加载查询、导出说明
docker compose run --rm parking bash scripts/02_compare_full_map.sh
```

两个命令均正常退出（退出码0）才表示本次运行完成。每次重跑会更新当前`results`及`data/full_map`中的生成地图，需留旧结果时先自行复制。不要同时运行两个入口；入口2必须在入口1成功后执行。代码更新后需重新`docker compose build`。

Linux多人服务器如不希望生成root所有的结果，可使用：

```bash
docker compose run --rm --user "$(id -u):$(id -g)" parking bash scripts/01_build_18.sh
docker compose run --rm --user "$(id -u):$(id -g)" parking bash scripts/02_compare_full_map.sh
```

## 6. 预期输出和如何查看

入口1生成18份独立summary与graph，共370张带检测框的JPG，以及370份逐框ID对应JSON。共有6,723张原始输入，但只有370个固定关键帧执行检测和建图更新，其余帧在all_steps中记录为固定策略跳帧。

T01参考summary：

```json
{"summary":{"frames":21,"objects":845,"regions":2,"portals":1,"unlocalized_objects":517}}
```

合并图参考规模：370帧、10,876个物体、3个区域、8个门户候选。图字典格式：

```json
{"nodes":{"object:004165:014":{"kind":"object","attributes":{}}},"edges":[{"source":"paintyellow","target":"object:004165:014","kind":"candidate_contains","attributes":{}}],"metadata":{}}
```

上例仅说明结构。真实属性见生成文件。区域ID为paintred/paintyellow/paintblue，连接属性不输出method。框图右侧有完整物体ID；图上省略共同的object:前缀。每帧JSON的bbox使用原始RGB坐标，图像标题栏不计入该坐标。

入口2生成三个主要存储数值，前次主机参考结果：

```json
{"baseline_bytes":14653675,"cognitive_bytes":11044558,"reduction_percent":24.629432548490396}
```

**以本次生成值为准**。运行路径、范围元数据和SQLite版本可能使文件字节数略有变化，不应硬编码以上数值作为验收结果。达到20%的条件由自动检查读取实际字节数判定。

直接打开以下文件，无需浏览器服务：

- `results/scope_consistency.png`：点云、已定位物体、18条轨迹及共同空间边界。
- `results/trajectories/T01/frames/004305.jpg`：检测框及物体ID示例。
- `results/scene_review/000299.png`：出口导向牌。
- `results/scene_review/001753.png`：路口抽查候选。
- `results/scene_review/008976.png`：疑似电梯候选，当前不能确认电梯。
- `results/范围对照图说明.txt`及`results/指标1.2文档核对.txt`：图例、计量口径和人工核验事项。

## 7. 如何测试及按ID查询

入口2自动执行以下检查，任何失败都会以非零退出码停止：

1. 点云与graph的场景、18条轨迹、370帧、配置与预测哈希一致。
2. 真实RGB哈希、图中相机位姿一致；所有6,723张原图重新校验。
3. 点云为有限XYZ、全部位于共同边界；已定位物体在边界内。
4. 逐帧几何属于对应RGB视锥；由逐帧几何重新合成点云并逐点比较。
5. SQLite完整性、图边引用、逐框物体ID匹配。
6. 在新进程分别加载合并图和18份独立图，执行物体ID查询；另一个新进程加载点云。
7. 按实际文件大小计算降低率，并检查是否≥20%。

结果在`scope_consistency.json`与`final_automated_check.json`，预期布尔检查全部为true。自动检查不替代原图人工核验，也不把电梯候选视为已确认设施。

```bash
# 查看T01中的指定物体及其观测与邻接关系
docker compose run --rm parking python3 code/map_pipeline.py query \
  --db results/trajectories/T01/graph.sqlite --object-id object:004165:014

# 单独重新生成370张检测图
docker compose run --rm parking python3 code/render_detection_frames.py

# 单独进行最终文件及新进程加载复核
docker compose run --rm parking python3 code/final_delivery_check.py
```

图仅按其真实观测表达。当前“全量地图”指固定18条轨迹合并结果，物体位置不是密集表面；未定位物体没有三维坐标。比较口径为点云NPY+范围文件，对比SQLite图+范围文件。SQLite内已含物体记录及查询所需关系；RGB、模型、离线中间缓存、可视化和JSON重复导出不计入该运行地图比较。数据与结果压缩归档只是交付方式，不以压缩包大小计算降低率。

## 8. 常见问题

- `Missing or changed original RGB`：数据不完整或被修改。重新核验附件，不要跳过哈希检查。
- 找不到points.npy：缺少full_map数据，只有代码仓库不能运行。
- 运行找不到本机`/Users/...`路径：两个正式入口会重绑定输入路径；不要绕过入口直接使用旧config调用底层build。
- scope检查不通过：查看JSON中为false的项；不要手改成true。确认未混用其他批次点云、检测或逐帧几何。
- PNG中文字体显示：范围图采用英文轴标，中文解释在同目录TXT；框图使用英文类别及数字ID。
- 容器退出137：通常是内存不足，增加Docker可用内存再重试。
- HPC不允许Docker：在有Docker权限的普通服务器执行；不能假定HPC登录节点允许启动Docker服务。

## 9. 封存及环境记录

用户已确认上传数据是固定测试数据。本包按现有18条清单回放，不自动重新筛选轨迹。文件来源、配置、原始RGB哈希及几何记录均保留。最初的感知/几何开发辅助程序随code归档，但并非所有辅助脚本都属于这两个Docker入口的运行契约；完整重新训练或重新推理不在本缓存回放镜像内。

实际容器构建与执行验证记录见`results/docker_verification.json`（若存在）。它与每次实际运行生成的`final_automated_check.json`共同记录环境，不能把本机验证冒充指定HPC环境验证。

## 10. 私有附件的恢复及归档

数据附件按最多64 MiB分卷，文件名按顺序为`parking-data.tar.gz.part0000`等；结果附件为`parking-results.tar.gz`。它们需要与同一版本代码一起使用。下载到仓库根目录的`release_assets/`后：

```bash
# Linux校验；Mac可用 shasum -a 256 -c SHA256SUMS
cd release_assets
sha256sum -c SHA256SUMS
cd ..
cat release_assets/parking-data.tar.gz.part* | tar -xzf -
# 可选：查看交付时已有结果；不解压该附件也能由两个入口重新生成
tar -xzf release_assets/parking-results.tar.gz
```

制作者归档命令为`python3 code/package_release.py`。附件分卷只用于传输，测量存储时使用解压后的原生地图文件。

如当前Docker只有Engine、没有Compose，也可以直接使用：

```bash
docker build -t parking-indicator-1-2:replay .
docker run --rm -v "$PWD/data:/workspace/data" -v "$PWD/results:/workspace/results" \
  parking-indicator-1-2:replay bash scripts/01_build_18.sh
docker run --rm -v "$PWD/data:/workspace/data" -v "$PWD/results:/workspace/results" \
  parking-indicator-1-2:replay bash scripts/02_compare_full_map.sh
```

Dockerfile会在目标主机构建对应CPU架构的镜像。不要把Apple Silicon生成的arm64镜像当作4090服务器的amd64镜像使用；在4090服务器上按上述命令构建即可选择本机架构。

本次验证（2026-11-24）：已在Linux arm64 Docker容器中从空results目录顺序执行两个入口，退出码均为0，370张检测图及对应JSON、18条轨迹图、范围图、说明和新进程查询报告均生成成功。运行镜像未使用GPU。存储结果为baseline 14,653,609字节、cognitive 11,044,558字节、降低24.62909307870846%。Linux amd64/4090主机尚未进行本次容器实跑验证。
