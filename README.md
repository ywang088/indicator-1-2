# 停车场指标1.2 完整推理与建图

第一个入口从原始 RGB 运行 SAM3 和 OCR，构建固定 18 条轨迹的认知图。第二个入口独立读取 data/full_map 中预存的封存全场景图与同范围点云，执行存储比较和范围检查。

## 1. 实际计算流程

1. 校验固定6,723张RGB的SHA256，按封存清单选择370个关键帧。
2. 对370帧逐帧调用SAM3，按固定提示词检测、分割并按类别去重。
3. 本次运行EasyOCR：整帧识别和柱子裁剪放大识别，重新读取字母。
4. 使用同帧LiDAR几何、固定标定及相机位姿，重新计算物体三维位置，清理车辆门等候选误检。
5. 根据本次SAM3柱子掩膜提取颜色，再用当前及历史字母观测绑定区域；按通道两侧柱子建立候选连接。
6. 每条轨迹从空图开始按时间顺序增量建图，输出帧、物体、门户、区域四层关系；本轮合并图另存 results/online_full_map。
7. 保存本次检测框图、ID对应表、输入/更新/查询日志和执行证据。

在线建图使用封存 RGB 序列：完成 370 个关键帧的识别后，按时间顺序增量更新图。其余原始帧按固定采样策略记录跳帧。三维定位使用固定的 LiDAR XYZ、相机标定和位姿。

## 2. 硬件及环境

- Linux x86_64 + NVIDIA CUDA显卡。默认Docker使用Python 3.11、PyTorch 2.12.0 + CUDA 13.0、torchvision 0.27.0、SAM3 0.1.4。
- 主机驱动支持 CUDA 13.0，安装 NVIDIA Container Toolkit。
- 实跑配置：H100 80GB；全量作业 877727 成功完成，用时 26 分 21 秒。
- RTX 4090 部署需使用兼容 CUDA 13.0 的驱动；当前服务器的 535 驱动需要升级。
- 建议 CPU 4 核以上、内存 32 GB 以上、磁盘可用空间 50 GB 以上。
- Mac 可用于查看结果、代码开发和第二个入口的 CPU 比较。

主要依赖见 `code/requirements-gpu.txt`，SAM3 安装包位于 `code/vendor/`。

## 3. 文件夹结构

```text
parking_docker/
├── Dockerfile                   # 完整推理的GPU环境
├── compose.yaml                 # parking使用GPU；compare只需CPU
├── README.md
├── scripts/
│   ├── 01_build_18.sh           # 重新检测、OCR、定位和建图
│   └── 02_compare_full_map.sh   # 读取预存全场景图，独立比较
├── code/
│   ├── fresh_build.py          # 入口1流程编排
│   ├── perception.py           # SAM3、整帧和柱子OCR
│   ├── refine.py               # 同帧几何定位及候选清理
│   ├── prepare_color_inputs.py # 本次柱色测量
│   ├── color_topology.py
│   ├── map_pipeline.py
│   ├── delivery_runner.py
│   ├── check_scope_consistency.py
│   ├── final_delivery_check.py
│   ├── requirements-gpu.txt
│   └── vendor/                 # 固定SAM3安装包
├── data/
│   ├── dataset/
│   │   ├── rgb/               # 6,723张固定原始RGB
│   │   ├── config.json
│   │   ├── trajectories/      # T01...T18清单
│   │   └── ...                # 历史检测归档
│   ├── models/
│   │   ├── sam3.pt
│   │   ├── bpe_simple_vocab_16e6.txt.gz
│   │   ├── manifest.json
│   │   └── ocr/               # craft_mlt_25k.pth、zh_sim_g2.pth
│   └── full_map/
│       ├── points.npy         # 同范围LiDAR合并基线
│       ├── scope_evidence/    # 逐帧几何和扫描来源记录
│       ├── graph.sqlite       # 预先保存的封存图
│       └── graph.json         # 封存图字典；另有audit/和frozen_manifest.json
└── results/
    ├── online_full_map/         # 入口1本轮合并图和summary
    ├── runs/运行编号/
    │   ├── input_config.json
    │   ├── execution.json
    │   ├── perception/        # 本次原始检测、OCR、掩膜、耗时
    │   ├── refined/           # 本次几何定位和清理
    │   ├── colored/           # 本次柱色识别和配置
    │   ├── map/               # 本次原始图输出及审核记录
    │   └── *.py.log           # 各阶段运行日志
    ├── execution.json         # 最近一次完整运行尝试的状态与证据
    ├── detections.jsonl       # 本次用于建图的检测
    ├── run_config.json
    ├── data_integrity.json
    ├── report.json
    ├── trajectories/T01...T18/
    │   ├── summary.json
    │   ├── graph.json
    │   ├── graph.sqlite
    │   ├── all_steps.jsonl
    │   ├── updates.jsonl
    │   ├── query_log.jsonl
    │   └── frames/            # 帧号.jpg + 帧号.json框与ID对应表
    ├── storage_comparison.json
    ├── storage_comparison.csv
    ├── storage_report.json
    ├── scope_consistency.json
    ├── scope_consistency.png
    ├── online_automated_check.json      # 入口1独立检查
    ├── frozen_map_automated_check.json  # 入口2独立检查
    ├── final_automated_check.json       # 交付包综合复核
    ├── fresh_process_queries.json
    ├── scene_review/
    ├── 范围对照图说明.txt
    └── 指标1.2文档核对.txt
```

## 4. 安装Docker及NVIDIA运行环境

已有Docker时无需重复安装。Ubuntu服务器按[Docker官方安装说明](https://docs.docker.com/engine/install/ubuntu/)配置官方软件源并安装Engine、Buildx与Compose：

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
```

再按[NVIDIA官方Container Toolkit说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)安装GPU容器支持：

```bash
sudo apt-get install -y curl gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
nvidia-smi
```

主机显卡驱动和容器工具包分别安装。

后续命令默认当前账号有 Docker 使用权限；如提示无权访问 Docker socket，请在 Docker 命令前加 `sudo`，或由管理员配置账号权限。

## 5. 准备数据、模型并构建镜像

代码仓库是私有仓库，先使用有权限的GitHub账号登录。固定数据沿用`sealed-18-v1`中的108个64 MiB分卷；`fresh-18-v2`提供模型、完整推理代码和新结果。下载后先按各附件随附的SHA256清单核验，再解压到仓库根目录，避免多套一层文件夹。

```bash
gh repo clone ywang088/parking-indicator-1-2
cd parking-indicator-1-2
mkdir -p release_assets
gh release download sealed-18-v1 --repo ywang088/parking-indicator-1-2 --pattern 'parking-data.tar.gz.part*' --dir release_assets
gh release download sealed-18-v1 --repo ywang088/parking-indicator-1-2 --pattern SHA256SUMS --output release_assets/data-SHA256SUMS
(cd release_assets && sha256sum -c data-SHA256SUMS --ignore-missing)
cat release_assets/parking-data.tar.gz.part* | tar -xzf -
gh release download fresh-18-v2 --repo ywang088/parking-indicator-1-2 --pattern 'parking-models.tar.gz.part*' --pattern SHA256SUMS --pattern release_manifest.json --pattern parking-frozen-map-metadata.tar.gz --dir release_assets
(cd release_assets && sha256sum -c SHA256SUMS --ignore-missing)
cat release_assets/parking-models.tar.gz.part* | tar -xzf -
tar -xzf release_assets/parking-frozen-map-metadata.tar.gz
docker compose build
```

也可直接把已有模型放到`data/models/`。模型必须与本项目相匹配；本地原SAM3权重SHA256为`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`。完整本地交付文件夹已经放入原SAM3、BPE及两份OCR权重。

首次构建需要联网下载Python/CUDA依赖，数据和权重不打进镜像，作为data卷挂载。镜像和权重齐全后可断网执行。

如只想先查看本次已完成的实跑结果，可在上述固定数据解压后下载新结果包（只更新 results，不覆盖 data 中的封存全场景图）：

```bash
gh release download fresh-18-v2 --repo ywang088/parking-indicator-1-2 --pattern parking-fresh-results.tar.gz --dir release_assets
(cd release_assets && sha256sum -c SHA256SUMS --ignore-missing)
tar -xzf release_assets/parking-fresh-results.tar.gz
```

## 6. 运行两个独立入口

```bash
# 检查容器中的GPU及驱动兼容性
docker compose run --rm parking python3 -c "import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name())"

# 可选：两帧推理检查，输出到独立runs子目录
docker compose run --rm parking bash scripts/01_build_18.sh --smoke-frames 2

# 正式入口1：所有370个固定关键帧重新推理并建图
docker compose run --rm parking bash scripts/01_build_18.sh

# 正式入口2：独立读取预存全场景图，比较存储并检查范围
docker compose run --rm compare
```

两个入口分别运行。入口1使用 GPU，将在线图、轨迹和中间结果写入 results。入口2使用 CPU，读取 data/full_map 并输出存储比较。每次建图分配独立运行编号，历史 runs 可归档保存。

如无Compose，可直接运行：

```bash
docker build --platform linux/amd64 -t parking-indicator-1-2:fresh-gpu .
docker run --rm --gpus all --shm-size=8g -v "$PWD/data:/workspace/data" -v "$PWD/results:/workspace/results" parking-indicator-1-2:fresh-gpu bash scripts/01_build_18.sh
docker run --rm -v "$PWD/data:/workspace/data:ro" -v "$PWD/results:/workspace/results" parking-indicator-1-2:fresh-gpu bash scripts/02_compare_full_map.sh
```

HPC禁止Docker时，在已安装同版本SAM3/CUDA依赖的环境中通过Slurm执行相同scripts入口，设置`PARKING_PYTHON`为该环境Python。HPC 和 Docker 的测试记录分别保存在 results/verification。

## 7. 运行记录

`results/execution.json`必须满足：

```json
{"pipeline_mode":"fresh_sam3","completed":true,"processed_frames":370,"reuses_detection_cache":false,"reuses_ocr_cache":false}
```

同时有新运行编号、GPU型号、模型SHA256及本次预测SHA256。`results/runs/运行编号/perception/detections.jsonl`记录每帧本次检测、OCR和耗时；同目录provenance记录环境和模型信息。入口1自己的 online_automated_check.json 会核对这份执行证据和预测哈希。入口2只验证独立封存图的 frozen_manifest.json，不使用在线建图记录。

运行失败时，入口返回非零退出码；具体原因见对应阶段日志。

## 8. 预期输出及测试判定

结构固定为18条轨迹结果、370张关键帧检测框图及370份框-ID对应JSON。每条summary包括frames、objects、regions、portals和unlocalized_objects；graph为nodes字典、edges列表和metadata。框上显示本轨迹graph物体ID，旁边JSON保留原始RGB坐标的bbox和检测ID。

入口1记录本轮检测的物体、区域和门户。入口2记录预存图与点云的固定文件大小。 data/full_map 已从 sealed-18-v1 原始数据附件逐字节恢复，其来源和所有输入哈希记录在 frozen_manifest.json。

第二个脚本输出baseline_bytes、cognitive_bytes、reduction_percent和逐文件CSV，同时检查封存文件哈希、共同帧集合、对应关键帧原图哈希、位姿、空间边界、逐帧点云重建一致性和独立进程加载查询。入口1另行检查18条轨迹的图边端点、输入日志顺序、框ID对应关系和全部原图哈希。封存图的存储检查要求实际降低率≥20%，检查结果写入 frozen_map_automated_check.json。

直接打开`scope_consistency.png`和`trajectories/T01/frames/*.jpg`即可看结果，无需HTML。`scene_review`保留固定出口标识、路口、疑似电梯的原图，相关检测记录来自本次在线图。scope_consistency.png 对应预存全场景图，与入口1输出目录分开。

```bash
# 查询当前结果里的物体；把示例ID替换为本次graph中的实际ID
docker compose run --rm parking python3 code/map_pipeline.py query --db results/trajectories/T01/graph.sqlite --object-id object:004165:014
```

## 9. 常见问题

- CUDA不可用：确认在NVIDIA Linux主机、驱动兼容CUDA 13.0、Container Toolkit正确安装且gpus配置生效。
- 找不到权重：检查data/models及OCR两份pth文件。
- 原图SHA256失败：数据缺失或被改动，重新校验固定附件。
- 只有两帧输出：执行的是`--smoke-frames 2`；正式运行请去掉该参数。
- 在线结果与历史不同：新检测和OCR可能改变在线图；第二个脚本仍读取固定的封存图。
- 封存图哈希失败：检查是否修改了 data/full_map。从固定数据附件恢复对应文件。
- 峰值显存或CPU内存不足：使用空闲且显存更大的GPU，增加主机内存，保留失败日志。
- HPC登录失败且解析成198.18.x.x：可能是代理Fake-IP接管；检查校园网DNS和路由后再直连。

## 10. 本版验证记录（2026-11-24）

- HPC H100 80GB，作业 877727，COMPLETED，退出码 0:0，用时 26 分 21 秒；370 个关键帧重新运行 SAM3 和 OCR，未使用旧检测和 OCR 缓存。
- 新在线合并图：10,943 个物体、3 个区域、8 个候选门户；T01：21 帧、855 个物体、2 个区域、1 个候选门户。18 条轨迹均形成区域，370 张检测图附带 ID 和同名 JSON。
- 上述 GPU 运行完成后，按独立入口要求拆分输出位置：在线图移入 results/online_full_map；data/full_map 从原始封存附件恢复。重新导出和检查不改变本轮推理结果。
- 第二个入口读取预存图：基线 14,653,609 字节，认知图运行文件 11,044,558 字节，降低 24.629093%。
- 测试范围：HPC 原生环境完成 GPU 推理；Linux amd64 Docker 完成镜像构建、模型导入、35 项自动测试和入口2运行。
- 额外验证：在没有在线输出的空 results 目录中，以只读方式挂载 data，单独运行入口2；验证记录随包提供。
- 本轮 GPU 峰值已分配显存约 4.71 GiB。

详见 results/fresh_run_verification.json、online_automated_check.json、frozen_map_automated_check.json、final_automated_check.json 和 verification/。
