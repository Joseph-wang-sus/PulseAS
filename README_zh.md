# PulseAS

基于短时PPG、波形标志物和常规临床信息的主动脉瓣狭窄（AS）评估框架。

[English](README.md) · [快速运行](#快速运行) · [输入格式](docs/data_format.md) · [分析说明](docs/reproducibility.md)

## 研究概览

[![Fig. 1：PulseAS研究概览](assets/fig1workflow.png)](assets/fig1workflow.svg)

**Fig. 1：PulseAS研究概览。** 研究以15秒PPG和临床信息开展AS评估，并分析疾病相关脉搏形态、瓣膜干预前后变化，以及MIMIC-III至EDU的外部验证和本地适配。[查看完整图片](assets/fig1workflow.png)。

## 数据说明

仓库包含核心代码、配置和合成示例。临床数据、患者级预测及模型权重不随仓库发布。EDU和MIMIC-III数据分别遵循机构及PhysioNet的访问要求。

使用者需提供符合[输入格式](docs/data_format.md)的分析数据。患者匹配、人工筛选和临床标注在进入本流程前完成，仓库不包含这些操作脚本。

## 快速运行

建议使用Python 3.11；CPU示例也支持Python 3.12。

```bash
git clone https://github.com/Joseph-wang-sus/PulseAS.git
cd PulseAS
python -m venv .venv
# Linux/macOS：source .venv/bin/activate
# Windows：.venv/Scripts/activate
python -m pip install -e '.[dev]'
python -m pulseas demo --output outputs/demo
```

示例生成虚构患者、模拟特征和纵向记录，执行验证、迁移适配、性能评价、评分分层及形态统计。无需临床数据或预训练权重。模拟表征不是AnyPPG输出，示例结果不代表论文性能。结果保存在`outputs/demo`中，生成的数据、模型和输出均不提交至Git。

## 核心流程

```text
15秒PPG → 信号预处理 → 波形标志物与冻结AnyPPG表征
        → 融合临床信息 → 队列内验证／跨队列迁移
        → 性能评价、AS/pre-TAVI评分分层及形态分析
```

EDU与MIMIC-III分别开发和内部验证模型。跨队列分析使用MIMIC-III来源模型，在EDU中进行外部验证及本地适配。围术期记录用于独立的形态分析。

| 入口 | 功能 |
| --- | --- |
| `prepare`、`extract` | 信号处理、形态特征和冻结表征提取 |
| `validate`、`transfer` | 队列内验证、外部验证和本地适配 |
| `evaluate`、`stratify` | 性能、临床增量和AS/pre-TAVI病例富集 |
| `morphology` | 疾病相关形态及术前、术后、随访分析 |
| `demo` | 合成数据流程演示 |

统一入口为`python -m pulseas`。命令示例见[英文README](README.md)，参数可通过`python -m pulseas <command> --help`查看。全局`--config`放在子命令前。正式设置位于`configs/study.yaml`，轻量示例设置位于`configs/demo.yaml`。

## 项目结构

`assets`存放论文Fig. 1；`src/pulseas`包含信号、特征、模型、验证、迁移、评价、分层与形态模块；`configs`存放配置及特征定义；`examples`提供合成数据生成器；`docs`说明输入及分析流程；`tests`核查核心行为。

## 第三方模型与依赖

完整信号及神经模型依赖可通过`python -m pip install -e '.[neural,signals,dev]'`安装。正式AnyPPG、XGBoost和神经训练路径需要兼容的CUDA/PyTorch环境；研究环境版本记录于`requirements-lock.txt`。

从[AnyPPG](https://github.com/Ngk03/AnyPPG)按上游要求获取编码器及权重，使用`extract`传入路径。Net1D架构注明[Shenda Hong的resnet1d实现](https://github.com/hsd1503/resnet1d)。信号处理使用pyPPG和NeuroKit2，原始MIMIC适配器所需的Ext-PPG工具另行提供。

运行检查：`python -m pytest -q -m 'not gpu'`。测试及GitHub自动检查使用合成输入。精确复现论文数值还需对应的研究数据、划分和第三方权重。

自有代码采用[MIT许可](LICENSE)，第三方内容维持原有许可。详见[分析说明](docs/reproducibility.md)及[第三方说明](THIRD_PARTY_NOTICES.md)。
