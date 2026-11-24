# 停车场指标1.2 完整推理与建图

第一个入口现在从原始RGB重新运行SAM3和OCR，再构建固定18条轨迹的认知图。它不读取旧detections.jsonl，不复用旧OCR或人工抄录结果，也不会在GPU不可用时切换到缓存回放。第二个入口独立读取 data/full_map 中预先保存的封存全场景图与同范围点云，执行存储比较和范围检查。它不读取本次在线图，也不要求先执行第一个入口。

旧版`sealed-18-v1`是缓存回放交付。完整推理的代码、权重补充包和新结果以本版运行记录为准，不能把旧版回放记录当作本版GPU实跑证明。

## 1. 实际计算流程

1. 校验固定6,723张RGB的SHA256，按封存清单选择370个关键帧。
2. 对370帧逐帧调用SAM3，按固定提示词检测、分割并按类别去重。
3. 本次运行EasyOCR：整帧识别和柱子裁剪放大识别，重新读取字母。
4. 使用同帧LiDAR几何、固定标定及相机位姿，重新计算物体三维位置，清理车辆门等候选误检。
5. 根据本次SAM3柱子掩膜提取颜色，再用当前及历史字母观测绑定区域；按通道两侧柱子建立候选连接。
6. 每条轨迹从空图开始按时间顺序增量建图，输出帧、物体、门户、区域四层关系；本轮合并图另存 results/online_full_map。
7. 保存本次检测框图、ID对应表、输入/更新/查询日志和执行证据。

这里的在线建图指固定离线轨迹上的逐帧增量更新，输入是封存 RGB 序列，不是实时摄像头采集；识别阶段完成后按时间顺序更新图。370个关键帧的检测全部重新计算；其余原始帧属于固定采样策略的跳帧输入。固定的LiDAR XYZ、相机标定和位姿是测量输入，本入口不重新运行LiDAR里程计或标定求解。

## 2. 硬件及环境

- Linux x86_64 + NVIDIA CUDA显卡。默认Docker使用Python 3.11、PyTorch 2.12.0 + CUDA 13.0、torchvision 0.27.0、SAM3 0.1.4。
- 主机驱动必须支持CUDA 13.0。容器包含CUDA运行库，但不包含或升级主机显卡驱动。先完成下文的GPU检查，再执行全量任务。
- H100是本项目已有SAM3环境使用的GPU；本次新入口的具体GPU、任务号和完成情况见执行记录。
- RTX 4090拥有24 GB显存，可作为需要进一步实测的部署目标；不能仅凭显存容量声明本环境已在4090跑通。当前访问到的4090主机驱动535不兼容本镜像的CUDA 13.0，本次没有修改该主机驱动。两帧真实推理已在H100 80GB上通过（HPC作业877667，退出码0）；全量作业877727已成功完成，退出码0，用时26分21秒。
- 建议CPU 4核以上、内存32 GB以上，完整数据、模型、镜像和中间输出预留50 GB以上磁盘。
- Mac可以查看结果或开发代码，不能用本机Apple GPU代替此入口要求的NVIDIA CUDA GPU。

SAM3 安装包漏声明的 psutil 依赖已在镜像中显式补齐。

锁定的主要依赖见`code/requirements-gpu.txt`，SAM3 wheel随`code/vendor/`保存。正式入口1不支持无GPU缓存回退。

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
│   ├── fresh_build.py          # 入口1编排，无检测缓存输入选项
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
│   │   └── ...                # 历史检测缓存仅留档，正式入口不读取
│   ├── models/
│   │   ├── sam3.pt
│   │   ├── bpe_simple_vocab_16e6.txt.gz
│   │   ├── manifest.json
│   │   └── ocr/               # craft_mlt_25k.pth、zh_sim_g2.pth
│   └── full_map/
│       ├── points.npy         # 同范围LiDAR合并基线
│       ├── scope_evidence/    # 逐帧几何和扫描来源记录
│       ├── graph.sqlite       # 预先保存的封存图；入口1不会覆盖
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

驱动安装/升级属于主机维护操作，尤其共享服务器上应由管理员安排，不要在其他GPU任务运行时升级驱动。以上工具包安装并不自动升级驱动。

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

也可直接把已有模型放到`data/models/`。模型必须与本项目相匹配；本地原SAM3权重SHA256为`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`。完整本地交付文件夹已经放入原SAM3、BPE及两份OCR权重。不要把GitHub或Hugging Face凭证写进镜像。

首次构建需要联网下载Python/CUDA依赖，数据和权重不打进镜像，作为data卷挂载。镜像和权重齐全后可断网执行。

如只想先查看本次已完成的实跑结果，可在上述固定数据解压后下载新结果包（只更新 results，不覆盖 data 中的封存全场景图）：

```bash
gh release download fresh-18-v2 --repo ywang088/parking-indicator-1-2 --pattern parking-fresh-results.tar.gz --dir release_assets
(cd release_assets && sha256sum -c SHA256SUMS --ignore-missing)
tar -xzf release_assets/parking-fresh-results.tar.gz
```

## 6. 先检查GPU，再顺序运行两个入口

```bash
# 检查容器中的GPU及驱动兼容性
docker compose run --rm parking python3 -c "import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name())"

# 可选：两帧真实推理检查，只写独立runs子目录，不覆盖18条最终结果
docker compose run --rm parking bash scripts/01_build_18.sh --smoke-frames 2

# 正式入口1：所有370个固定关键帧重新推理并建图
docker compose run --rm parking bash scripts/01_build_18.sh

# 正式入口2：独立读取预存全场景图，比较存储并检查范围
docker compose run --rm compare
```

两个入口可以分别运行。入口2即使在 results 为空、从未运行入口1时也能执行，不需要 GPU 或 SAM3 权重。入口1只更新 results 中的在线图和轨迹输出；data/full_map 的封存图、点云和校验文件保持不变。每次入口1创建独立运行编号保留中间结果。磁盘空间不足时应先归档过往runs，不能删掉当前运行依赖的中间结果。

如无Compose，可直接运行：

```bash
docker build --platform linux/amd64 -t parking-indicator-1-2:fresh-gpu .
docker run --rm --gpus all --shm-size=8g -v "$PWD/data:/workspace/data" -v "$PWD/results:/workspace/results" parking-indicator-1-2:fresh-gpu bash scripts/01_build_18.sh
docker run --rm -v "$PWD/data:/workspace/data:ro" -v "$PWD/results:/workspace/results" parking-indicator-1-2:fresh-gpu bash scripts/02_compare_full_map.sh
```

HPC禁止Docker时，在已安装同版本SAM3/CUDA依赖的环境中通过Slurm执行相同scripts入口，设置`PARKING_PYTHON`为该环境Python。使用原环境执行成功不能冒充Docker GPU实跑，运行证据分别记录。

## 7. 如何确认这次真的重新跑了

`results/execution.json`必须满足：

```json
{"pipeline_mode":"fresh_sam3","completed":true,"processed_frames":370,"reuses_detection_cache":false,"reuses_ocr_cache":false}
```

同时有新运行编号、GPU型号、模型SHA256及本次预测SHA256。`results/runs/运行编号/perception/detections.jsonl`记录每帧本次检测、OCR和耗时；同目录provenance记录环境和模型信息。入口1自己的 online_automated_check.json 会核对这份执行证据和预测哈希。入口2只验证独立封存图的 frozen_manifest.json，不使用在线建图记录。

若权重不存在、CUDA不可用或SAM3/OCR失败，入口1以非零码终止，没有自动缓存回退。

## 8. 预期输出及测试判定

结构固定为18条轨迹结果、370张关键帧检测框图及370份框-ID对应JSON。每条summary包括frames、objects、regions、portals和unlocalized_objects；graph为nodes字典、edges列表和metadata。框上显示本轨迹graph物体ID，旁边JSON保留原始RGB坐标的bbox和检测ID。

**入口1的物体数量、字母绑定和门户数量来自本次真实推理，可能变化。入口2固定比较预存图和点云，因此其存储结果不随入口1变化。** data/full_map 已从 sealed-18-v1 原始数据附件逐字节恢复，其来源和所有输入哈希记录在 frozen_manifest.json。

第二个脚本输出baseline_bytes、cognitive_bytes、reduction_percent和逐文件CSV，同时检查封存文件哈希、共同帧集合、对应关键帧原图哈希、位姿、空间边界、逐帧点云重建一致性和独立进程加载查询。入口1另行检查18条轨迹的图边端点、输入日志顺序、框ID对应关系和全部原图哈希。封存图的存储检查要求实际降低率≥20%；若不达标会报错并保留本次数字，不能修改数字或删掉必要文件来达标。

直接打开`scope_consistency.png`和`trajectories/T01/frames/*.jpg`即可看结果，无需HTML。`scene_review`保留固定出口标识、路口、疑似电梯的原图，相关检测记录来自本次在线图。scope_consistency.png 对应预存全场景图，与入口1输出目录分开。人工核验仍需按原图进行，候选节点不等于人工通过。

```bash
# 查询当前结果里的物体；把示例ID替换为本次graph中的实际ID
docker compose run --rm parking python3 code/map_pipeline.py query --db results/trajectories/T01/graph.sqlite --object-id object:004165:014
```

## 9. 常见问题

- CUDA不可用：确认在NVIDIA Linux主机、驱动兼容CUDA 13.0、Container Toolkit正确安装且gpus配置生效。
- 找不到权重：检查data/models及OCR两份pth文件。不会替你自动改用历史检测。
- 原图SHA256失败：数据缺失或被改动，重新校验固定附件；不能关闭哈希检查。
- 只有两帧输出：执行的是`--smoke-frames 2`；正式运行请去掉该参数。
- 在线结果与历史不同：新检测和OCR可能改变在线图；第二个脚本仍读取固定的封存图。
- 封存图哈希失败：检查是否修改了 data/full_map。应恢复封存原文件，不要用新在线图替换后继续声称同一份固定输入。
- 峰值显存或CPU内存不足：使用空闲且显存更大的GPU，增加主机内存，保留失败日志。不要与他人任务争抢同一张GPU。
- HPC登录失败且解析成198.18.x.x：可能是代理Fake-IP接管；检查校园网DNS和路由后再直连。具体个人网络地址不写入通用Docker配置。

## 10. 本版验证记录（2026-11-24）

- HPC H100 80GB，作业 877727，COMPLETED，退出码 0:0，用时 26 分 21 秒；370 个关键帧重新运行 SAM3 和 OCR，未使用旧检测和 OCR 缓存。
- 新在线合并图：10,943 个物体、3 个区域、8 个候选门户；T01：21 帧、855 个物体、2 个区域、1 个候选门户。18 条轨迹均形成区域，370 张检测图附带 ID 和同名 JSON。
- 上述 GPU 运行完成后，按独立入口要求拆分输出位置：在线图移入 results/online_full_map；data/full_map 从原始封存附件恢复。重新导出和检查不改变本轮推理结果。
- 第二个入口读取预存图：基线 14,653,609 字节，认知图运行文件 11,044,558 字节，降低 24.629093%。这是固定封存图的比较值，不是新在线图的指标。
- Linux amd64 Docker 镜像构建、模型依赖导入和 35 项自动测试通过。容器内 GPU 推理尚未实测；HPC 原生 GPU 运行证据独立保存。
- 额外验证：在没有在线输出的空 results 目录中，以只读方式挂载 data，单独运行入口2；验证记录随包提供。
- 本轮 GPU 峰值已分配显存约 4.71 GiB，不代表全部部署开销。区域边界、标定和门户人工审核仍需按原始证据确认。

详见 results/fresh_run_verification.json、online_automated_check.json、frozen_map_automated_check.json、final_automated_check.json 和 verification/。
