# VectorMerge Global Configuration Guide

VectorMerge 提供了一个强大的全局配置系统，让您可以设置个人偏好，避免每次都重复指定相同的参数。

## 🎯 配置优先级

配置值按以下优先级顺序应用：

1. **CLI 参数** (最高优先级) - 命令行直接指定的参数
2. **项目配置文件** - 通过 `--config` 指定的项目配置文件
3. **全局配置文件** - `~/.vectormerge/config.yaml` 中的用户全局配置
4. **内置默认值** (最低优先级) - 代码中的默认值

## 📁 全局配置文件位置

全局配置文件位于：`~/.vectormerge/config.yaml`

## 🚀 快速开始

### 1. 初始化全局配置

```bash
# 创建默认的全局配置文件
vectormerge config init
```

### 2. 查看当前配置

```bash
# 显示当前的全局配置
vectormerge config show
```

### 3. 编辑配置文件

```bash
# 在编辑器中打开配置文件
vectormerge config edit
```

### 4. 设置单个配置值

```bash
# 设置默认数据路径
vectormerge config set default_data_path "/path/to/your/data"

# 设置默认模型
vectormerge config set default_model_1 "sentence-transformers/all-MiniLM-L6-v2"
vectormerge config set default_model_2 "sentence-transformers/all-mpnet-base-v2"

# 设置API密钥
vectormerge config set openai_api_key "your-api-key-here"

# 设置批处理大小
vectormerge config set default_batch_size 64
```

### 5. 获取配置值

```bash
# 获取特定配置值
vectormerge config get default_data_path
vectormerge config get default_batch_size
```

## ⚙️ 配置选项详解

### 🗂️ 默认路径设置

```yaml
default_data_path: "./data/raw/beir/"              # 数据集存储路径
default_embedding_path: "./data/processed/embeddings/"  # 嵌入文件存储路径
default_reference_path: "./data/processed/references/"  # 参考点文件存储路径
default_mapping_path: "./data/processed/mappings/"      # 映射文件存储路径
default_output_path: "./results/"                       # 结果输出路径
```

### 🤖 模型设置

```yaml
default_model_1: "bert-base-uncased"    # 默认第一个模型
default_model_2: "roberta-base"         # 默认第二个模型
default_dataset: "scifact"              # 默认数据集
default_batch_size: 32                  # 默认批处理大小
default_device: "auto"                  # 默认设备 ("auto", "cpu", "cuda")
```

### 🧮 算法设置

```yaml
default_mapping_method: "procrustes"     # 默认映射方法
default_reference_strategy: "ours"       # 默认参考点策略
default_num_clusters: 50                 # 默认聚类数量
default_d0_ratio: 0.33                   # 默认D0比例
```

### 📊 评估设置

```yaml
default_k_list: [10, 100, 1000]         # 默认k值列表
default_metrics: ["recall", "ndcg", "map"]  # 默认评估指标
```

### 🔑 API密钥

```yaml
openai_api_key: null      # OpenAI API密钥
mistral_api_key: null     # Mistral API密钥
```

### 🔧 高级设置

```yaml
force_download: false     # 是否强制重新下载数据集
verbose: false           # 是否默认启用详细日志
use_cache: true          # 是否使用缓存
```

## 💡 实际使用示例

### 场景1: 设置个人工作目录

```bash
# 设置您的工作目录
vectormerge config set default_data_path "/home/user/research/data"
vectormerge config set default_output_path "/home/user/research/results"

# 现在所有命令都会使用您的个人目录
vectormerge download-dataset -d scifact  # 下载到您的个人目录
```

### 场景2: 配置您的GPU设备

```bash
# 如果您有多GPU环境，固定使用GPU
vectormerge config set default_device "cuda"
vectormerge config set default_batch_size 128  # 增大批处理大小
```

### 场景3: 设置常用模型组合

```bash
# 设置您常用的模型组合
vectormerge config set default_model_1 "sentence-transformers/all-MiniLM-L6-v2"
vectormerge config set default_model_2 "sentence-transformers/all-mpnet-base-v2"

# 现在可以省略模型参数
vectormerge generate-embedding -d scifact  # 使用默认模型
```

### 场景4: 项目特定配置

```bash
# 为特定项目创建配置文件
cat > project_config.yaml << EOF
default_model_1: "bert-large-uncased"
default_model_2: "roberta-large"
default_batch_size: 16
default_output_path: "./project_results/"
EOF

# 使用项目配置
vectormerge --config project_config.yaml generate-embedding -d scifact
```

## 🔍 配置验证

您可以使用 `doctor` 命令来检查配置状态：

```bash
vectormerge doctor
```

这将显示：
- 全局配置文件是否存在
- 当前配置的主要设置
- 系统依赖状态
- 目录结构状态

## 🛠️ 故障排除

### 配置文件损坏

如果配置文件损坏，您可以重新初始化：

```bash
# 删除现有配置文件
rm ~/.vectormerge/config.yaml

# 重新初始化
vectormerge config init
```

### 权限问题

如果遇到权限问题：

```bash
# 确保配置目录权限正确
chmod 755 ~/.vectormerge/
chmod 644 ~/.vectormerge/config.yaml
```

### 配置值类型错误

使用 `set` 命令时注意类型转换：

```bash
# 布尔值
vectormerge config set verbose true
vectormerge config set use_cache false

# 数字
vectormerge config set default_batch_size 64

# 列表 (用逗号分隔)
vectormerge config set default_k_list "10,100,1000"
vectormerge config set default_metrics "recall,ndcg,map"
```

## 📋 完整配置示例

查看 `docs/global_config_example.yaml` 获取完整的配置示例文件。

## 🔒 安全注意事项

1. **API密钥安全**: 不要将包含API密钥的配置文件提交到版本控制系统
2. **文件权限**: 确保配置文件权限设置正确，防止其他用户访问
3. **敏感信息**: 考虑使用环境变量而不是配置文件存储敏感信息

## 🎉 结论

全局配置系统让您可以：

- ✅ 避免重复输入相同的参数
- ✅ 为不同项目设置不同的默认值
- ✅ 安全地存储API密钥
- ✅ 在团队中共享配置模板
- ✅ 根据硬件环境调整性能参数

通过合理使用配置系统，您可以显著提高使用 VectorMerge 的效率！ 