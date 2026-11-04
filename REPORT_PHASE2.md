# CAR1 自动初值修复：根因、最小变更与科学验收

版本：carma_pack_modern 0.1.1。日期：2026-11-02（北京时间）。
工作全部在 Work 云端 Linux 完成，未连接、操作、安装或修改用户 Mac。

**结果：新 wheel 在全新云端 Python 环境安装后，121 passed、2 warnings；pip check 无依赖冲突。** 上一阶段 0.1.0 交付包原样保留。

## 1. 开始前核验与失败用例

核对了现有项目的 Phase 1 提交 `9508ecfdaec59b0b9a64197b12dbb379a9ccac61`、真实 src 源码、pytest 日志、已安装 0.1.0 wheel，以及上一版完整压缩包。压缩包内所有成员与其 MANIFEST 中的 SHA256 相符；旧包整体 SHA256 及核验记录在 `audit/phase2/phase1_artifact_verified.json`。

本次先新增 `modern_tests/test_car1_initialization.py`，在**产品源码尚未修改、仍安装 0.1.0** 的 cloud-test-env 上实际运行：**15 failed、23 passed**。失败包括 7 个固定种子的冻结链，以及 8 个无效显式初值未报错而被静默替换。原始 stdout 与 JUnit 在 `audit/phase2/before_fix.log` / `before_fix.xml`，不是修复后伪造的预期失败。

固定复现输入在 `audit/phase2/reproducer_input.txt`。时间为 [0,0.4,1.7,2.1,4.8,7,10]；y 为 [0.2,-0.1,0.5,0.3,-0.4,0.1,0.2]；各点异方差误差见文件。Boost/Armadillo 均设置 seed=314159，原 0.1.0 的 12 samples / 20 burnin / thin=2 返回全 -inf 的记录值和完全不动的链。

另外从 pinned 上游 Git 对象提取、逐字节校验了 7 个核心实现及全部原头文件，保存在 `modern_tests/upstream_core`；独立编译，不与修改后的 carma_core 混合链接。PROVENANCE.json 记录来源和哈希。源包自带原始快照，复验不依赖外部网络。

## 2. 有证据的根因

### 参数化与先验

实际四参数是：`[stationary_stdev, measurement_error_variance_scale, mean, log(omega)]`。原文件若干注释的顺序不正确，应以 `ExtractAR`、`ExtractSigsqr`、`LogDensity` 的实现为准。

原模型驱动噪声方差为 `2 * stationary_stdev² * omega`，OU 平稳方差是 `stationary_stdev²`。该数学参数化没有改变。

原 prior 支持：omega 在 `[1/time_span, 1/min(dt)]`，stdev 在 `[0,max_stdev]`，误差方差尺度在 `[0.5,2]`。采样入口的 max_stdev 继续采用原 `10*sqrt(population_variance)`。

原 `StartingValue` 先抽 `tau=median(dt)*Uniform(1,50)`，再求 `log_omega=-log(tau)`，却用 `min(log_omega,max_freq_)` 将对数频率与线性频率比较，且从未限制 omega 的下界。对于上述种子，得到 log_omega=-3.6308725422420389，即 omega≈0.0265，小于 prior 下界 0.1。上界单位错误也是真实代码缺陷，但这个具体失败是由缺失下界检查触发的。

### 初始密度缓存与接受步骤

`Sampler::Run` 通过 `Parameter<arma::vec>*` 执行 `Save(StartingValue())`；实际虚调用是 `Parameter::Save(arma::vec)`，它重新调用 LogDensity 并缓存。`CARMA_Base::Save(arma::vec&)` 是签名不同的隐藏重载，不会替代这条虚调用。这一点由 native probe 按真实路径复现，不能凭名称推断其缓存行为。

真实 probe 输出（原样保存在 upstream_root_cause.txt）：

```text
initial 0.67361584091944859 1.2969799521460408 -0.0088652037425425712 -3.6308725422420389
bounds 0.10000000000000001 2.5 2.6954231805876017
legal 0
density -inf -inf
finite_candidate -20.232726570194188 0 0
after 0 -inf
```

其中 finite_candidate 最后两项是原 Accept 返回 false 和 alpha 被重置为 0。即使新候选点密度有限，`finite - (-inf)` 得到 +inf，原 `AdaptiveMetro::Accept` 对所有非有限差值一律拒绝；两个 -inf 的差是 NaN，也被拒绝。因此起点既非法又无法脱困。连续 20 次更新的状态变化范数为 0，记录密度一直 -inf。

proposal 初始矩阵来自原数据方差且为正定，构造 Cholesky 未失败；拒绝率为 100% 后 RAM 按原规则收缩 proposal。不是“初始 proposal 矩阵本来奇异”造成这次故障。完整前后 proposal 对角线在探针输出中。

原显式初值错误路径还会打印警告后转入同一个自动初值流程，既改变用户指定条件，也可能再次生成非法起点。

## 3. 科学依据与最小修复范围

只修改了两个原核心实现文件：

| 文件 | 具体差异与理由 |
|---|---|
| src/carpack.cpp | CAR1::StartingValue 将原始 tau 初值分布条件化在合法范围：a~Uniform(1,min(50,time_span/median(dt)))，tau=a*median(dt)，omega=1/tau；每个候选必须有限、满足原 prior 且具有有限原 LogDensity，最多尝试 128 次；CAR1::SetStartingValue 对无效显式参数报错，不再静默随机替换；更正局部错误参数注释。 |
| src/carmcmc.cpp | RunCar1Sampler 在运行前检查数据、长度、时间/误差、计数溢出、原经验方差和 proposal 数值可表示性；显式参数在添加采样步骤前校验，避免落入 Sampler 按错误长度自动选随机初值的路径。合法输入的原 proposal 方差公式保持不变。 |

精确上游哈希、新哈希及逐文件理由见 `audit/phase2/core_changes.json`；只含核心修复的补丁和完整源码补丁均随交付包提供。**不能再声称所有核心哈希一致。** 其余 14 个先前记录的原 C++/头文件哈希仍一致，测试强制校验；包括 kfilter.cpp、proposals.cpp、steps.cpp、samplers.cpp、random.cpp 和所有原头文件。

没有修改：原 prior 或 posterior 定义、CARMA/OU 数学、Kalman 递推、Student proposal、RAM 更新、温度、转移核或接受准则。接受步骤仍保留原非有限差值处理；修复是在进入转移前保证合法、有限的初始状态。初始化不是 posterior transition，条件化初值分布不改变目标分布。自动初值/后续随机序列允许与旧版不同，不作旧自动链逐位不变的承诺。

正常显式初值走原确定性滤波与相同 RNG 路径；新增提前校验不消费随机数。原 proposal 方差的消减消去问题没有通过改变公式来掩盖：若原表达式在极端数据尺度下变成非正、无穷或下溢，明确报错，提示该数值尺度不可用。

## 4. 修复后同一失败用例

独立当前核心 probe 的原始输出在 fixed_root_cause.txt：

```text
initial 0.67361584091944859 1.2969799521460408 -0.0088652037425425712 -1.6511836482503293
legal 1
density -22.926964760357695 -22.926964760357695
finite_candidate -20.232726570194188 1 1
after 0.13840648352651719 -22.595052479759129
```

原 seed 下，起点合法，直接计算密度与缓存值相等且有限；有限候选按原准则被接受；20 次原更新后状态发生变化。

## 5. 实际测试与数值证据

最终在全新的 `cloud-phase2-test-env` 中，安装重建的 **carma_pack_modern-0.1.1-cp312-cp312-linux_x86_64.whl**，不是 editable 安装，也不是测试源目录的扩展。

**121 passed、2 warnings，2.16 秒**；完整 stdout、JUnit 和带退出码的实际命令分别在 `audit/phase2/pytest.log`、`pytest.xml`、`validation_command.json`。pip check 无损坏依赖。安装位置、Python 路径和版本在 installed_environment.json。

1. 上阶段 33 项覆盖全部重跑。原缺陷测试改为对**未修改的上游 native oracle** 复现，不能要求新版继续返回错误链；原“所有哈希相同”测试改为精确限制只允许列出的两处核心修复，并验证原始 oracle 快照的每个哈希。
2. 新增 88 项：包括直接检查真实初始状态、缓存、原 posterior 和独立 OU likelihood，显式初值与退化输入错误，以及有上限的失败路径。
3. 3 类代表性不规则时序 × 8 个固定种子：原问题短序列；带季节空隙、19 mag 偏置、异方差误差的 48 点序列；时间间隔跨多个数量级的序列。种子为 0、1、2、42、82、314159、271828、4294967295。每种组合均通过 native probe 在第一步之前确认合法且有限的起点；64 burnin、thin=2、64 保留样本均有实际移动。24 组中最少为 63 次相邻保留样本比较里的 **38 次变化**。这不是收敛证明。
4. 所有记录的密度与逐样本重算 posterior 的最大绝对差为 **0**。初始 posterior 与独立完整高斯协方差 OU oracle 加原 prior 比较，24 组最大绝对差约 **3.06×10^-13**。独立 covariance 包含原误差方差尺度，GetLogLikes 实际是含 prior 的未归一化 posterior；常数项 n log(2pi) 仍按上游省略。
5. 合法显式初值路径：5 个固定种子 × thin=1/3，与**另外编译、未修改上游核心**返回的样本和记录密度逐位一致（rtol=atol=0）。上阶段三类模型 reference-driver 比较的差也仍为 0。
6. OU 三组参数、CAR2、复共轭根 CARMA21、CARMA32 的均值/方差/插值/预测、完整高斯 likelihood 与条件模拟科学检查均通过。原数学 oracle 误差指标在 audit/phase2/numerical_metrics.json；其所记录滤波/预测最大误差约 **2.31×10^-14**。两类滤波器各 4000 个条件模拟的均值和联合协方差检查保持通过。
7. 明确覆盖：越界频率、负/过大 stdev、越界误差尺度、错误长度、NaN/Inf 显式参数；有限参数但非有限 density；常量、空/过短、重复或逆序时间、非正误差；极端尺度导致的原经验方差/proposal 下溢或溢出；采样计数溢出。C++ 原生入口同样验收，不能仅依赖 Python 包装的校验。
8. 使用 test-only 极窄正 stdev prior，强制固定种子的候选失败，确认恰当抛出“128 attempts”异常并在超时约束内退出；没有无界 while 重试。

初始化与链移动的详细逐组记录在 `audit/phase2/initialization_metrics.json`。未用修改数值阈值、删掉失败种子或 xfail 的方式换取通过。

## 6. 构建、依赖与交付

真实命令在 `audit/phase2/COMMANDS.md`；构建日志、独立 upstream/current probe 编译日志与干净环境依赖安装日志均保留。

继续使用上阶段云端专用原生依赖前缀：GCC 13.3、Armadillo 12.6.7、Boost 1.83、Python 3.12.14；没有进行 Mac 或系统级安装。原生 deb 的精确版本、SHA256 继续引用 audit/cloud_cpp_dependencies.json，构建/验收 Python 依赖快照在 phase2 目录。

新交付包包括 0.1.1 源包、wheel、相对 Phase 1 的增量补丁、相对上游的完整现代化补丁、只含两个核心文件修复的补丁、报告、原始复现输入与全部审计日志；旧版交付文件没有覆盖。MANIFEST 记录每个交付成员的 SHA256。

## 7. 剩余问题及 Mac 验证条件

- 原 CARp/CARMA 自己的自动初始化循环与无效显式初值 fallback 没有在这次 CAR1 修复中重写，尤其极端/退化数据仍需专门审计。ZCAR 数学路径未新增科学验收。
- 仍有两条历史 Python sqrt RuntimeWarning（np.where 会计算两分支）；本次验收目标输出有限，未修改其参数转换公式。acor 仍是可选诊断依赖，未测试其安装与估计质量。
- 原 RunCar1Sampler 的方差表达式在巨大均值/极小振幅下可能消减消去。本修复明确拒绝原公式无法产生有效 proposal 的输入，不宣称已支持任意动态范围。需要时应在另一个有独立数值证明的变更中处理稳定方差计算。
- 仅对当前 Linux/CPython 3.12/工具链完成验收。共享 RNG 保持 GIL，未引入并行安全性。位级随机序列一致性只针对记录的平台。
- wheel 依然依赖外部 Armadillo、Boost、BLAS/LAPACK、ARPACK/SuperLU 等。auditwheel 的符号版本约束仍至少为 manylinux_2_39 级别；该 wheel 未 repair/重新标记，**不是通用 Linux 或 Mac wheel**。
- 有限起点及短链移动不代表混合、收敛、参数恢复或 posterior coverage；这些需要每个科学问题的独立诊断和校准。

**结论：已具备开展下一步隔离 macOS 构建验证的代码与回归基线条件。** 原 CAR1 起点缺陷已有可重复修复和验收，但这不等于已证明 Apple Clang、macOS arm64 动态链接或 wheel 分发兼容。下一步应在独立 macOS 环境构建并重跑同套 oracle；本次没有在用户 Mac 执行或安装任何内容。
