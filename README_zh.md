# PulseAS

本项目整理PPG主动脉瓣狭窄（AS）研究的核心处理和分析代码，包括信号质量处理、形态特征与冻结AnyPPG表征、八种内部模型策略、MIMIC→EDU外部迁移和本地适配、临床增量比较、AS/pre-TAVI评分分层及纵向形态统计。

## 架构

```text
15秒PPG
 ├─ EDU质量检查／MIMIC Ext-PPG筛选
 ├─ 918项pyPPG汇总特征 ─ 训练折内选择Top-10 ─┐
 └─ 中央10秒输入 ─ 冻结AnyPPG的512维表征 ───┼─ 融合与临床联合模型
                                          ├─ 两队列分别开展患者级嵌套交叉验证
                                          ├─ MIMIC→EDU外部验证及20%本地适配
                                          └─ OOF评分的AS/pre-TAVI病例富集
510项特征检验族 ─ 疾病相关形态、术前术后混合模型及随访配对检验
```

`src/pulseas`中的各模块分别负责输入、信号、特征、模型、验证、迁移、评价、分层和形态分析。代码不依赖原来的私人磁盘目录，也不需要导入历史AR实验模块。

## 安装与示例

正式运行环境建议Python 3.11；CPU特征级示例也支持Python 3.12。在仓库根目录执行：

```bash
python -m venv .venv
# Windows激活：.venv\Scripts\activate
# Linux/macOS激活：source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q -m 'not gpu'
python -m pulseas demo --output outputs/demo
```

示例自动生成虚构患者、模拟的918维形态特征、512维表征、原始波形示例和纵向记录，并执行内部验证、迁移适配、性能评价、病例富集及形态检验。**示例中的表征为模拟数据，不是AnyPPG输出；示例结果不代表论文性能或临床验证。**

完整信号与GPU模型依赖可通过`python -m pip install -e '.[neural,signals,dev]'`安装。正式AnyPPG、XGBoost及神经模型路径要求CUDA，不会自动换成另一种模型或悄悄切换到CPU。原始研究版本见`requirements-lock.txt`，其中PyTorch原始CUDA构建为2.0.1+cu117。

## 运行接口

统一入口为`python -m pulseas`，提供：

| 子命令 | 功能 |
|---|---|
| `prepare` | 原始信号处理，或已生成特征的输入核查 |
| `extract` | pyPPG及真实冻结AnyPPG表征提取 |
| `validate` | 各队列分别进行患者级嵌套验证 |
| `transfer` | MIMIC来源模型、EDU零样本评价与本地适配 |
| `evaluate` | 性能、bootstrap、亚组及匹配临床比较 |
| `stratify` | AS与pre-TAVI病例捕获和富集 |
| `morphology` | 510项形态检验及完整脉搏模板 |
| `demo` | 不需要临床数据和权重的合成示例 |

完整参数示例见[英文README](README.md)。全局`--config`参数放在子命令前。`configs/study.yaml`保留正式预算；`configs/demo.yaml`仅用于轻量演示。PulseAS–XGB、PulseAS–L2和PulseAS–MLP分别对应`clinical_fusion`、`clinical_fusion_logistic`及`clinical_neural_fusion`。

## 数据与方法边界

- EDU与MIMIC分别开发和内部验证模型，不合并为一个训练池。
- 特征选择、插补、缩放、调参、校准和阈值均限定在对应训练或开发数据内。
- MIMIC患者等权，先聚合片段评分再校准；同一患者不跨验证折。
- 正式EDU内部分析继承原有折分，需要通过`--folds`传入相同患者顺序和折分；公开代码不会把重新生成的AS单终点折分冒充原有折分。
- 外部迁移使用20%的合格EDU患者作为适配与验证池；适配前后在同一80%留出测试集评价。
- AnyPPG保持冻结；XGBoost添加树；逻辑回归向源系数约束更新；神经融合更新预测头；ExtraTrees仅再校准；来源Net1D接受全网络适配。
- 918项提取特征与510项统计检验族分开；不能只对展示的代表特征进行BH校正。
- pre-TAVI病例来自独立整理的术前状态和手术证据，不由AS分级推断，不进入AS模型输入。未知手术状态不等于没有手术。
- 纵向数据不进入AS训练。术前术后模型估计的阶段差异不等于两个阶段中位数之差。

## 获取第三方代码与权重

从[AnyPPG原项目](https://github.com/Ngk03/AnyPPG)按上游许可获取`load_anyppg/resnet1d.py`及对应权重。代码保存研究使用的实现与权重哈希，用于识别版本，不分发权重。Net1D架构同时注明[Shenda Hong的resnet1d实现](https://github.com/hsd1503/resnet1d)。MIMIC筛选依赖的Ext-PPG工具另行提供，研究记录的上游提交为`667dc0f307de93d938c22f83feaeccc0b8e08cdf`。

本仓库不包含EDU或MIMIC临床数据、患者预测结果、身份对照表、模型权重或访问凭据。获取真实数据仍须遵循相关机构和PhysioNet的访问要求。

详见[输入格式](docs/data_format.md)、[复现说明](docs/reproducibility.md)与[第三方说明](THIRD_PARTY_NOTICES.md)。本项目自有代码采用MIT许可，第三方内容维持原有许可。
