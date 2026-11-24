固定数据来自 sealed-18-v1 附件。full_map 是之前预存的封存全场景图和同范围点云，frozen_manifest.json 绑定原始哈希，audit 保存范围检查所需的封存配置与检测。入口1不修改这里；其新图写入 results/online_full_map。入口2直接只读这里，不依赖在线输出或SAM3权重。
