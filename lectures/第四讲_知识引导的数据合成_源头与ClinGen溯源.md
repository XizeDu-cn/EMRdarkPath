# 知识引导的数据合成：源头线与 ClinGen 溯源

2026-09-29 初稿 · 2026-10-01 修订 · 国外原创线学习讲义 · 第四讲

修订内容：
- 补第零部分（ZeroGen 起点）；
- 补 CTRL、RAG、ProGen、SunGen 的训练过程、数据样例与玩具数字；
- 更正 ClinGen 与 AttrPrompt 的作者关系；
- 补"两张 5090 下各线用哪一档"。

---

## 读法说明

"知识图谱 + 大模型造数据"不是 ClinGen 首创，它有更早的源头；ClinGen 用上的是其中最弱的一种用法。

本讲分五部分：
0. **起点**：ZeroGen 让大模型造训练数据，暴露出三个问题；
1. **三条源头线**：分别回答这三个问题。每项工作写明谁被调用、谁被训练、数据长什么样、在什么真实数据上考试；
2. **ClinGen 逐步拆解**：提示词取自仓库代码原文（commit b94b14c）；
3. **ClinGen 溯源表**；
4. **对你的意义**：包括两张 5090 显卡能做到哪一档。

代码核对版本：ZeroGen `0581de7`，SunGen `4f51acf`，ClinGen `b94b14c`。

---

# 第零部分　起点：ZeroGen 和它暴露的三个问题

## 0.1　没有它的时候

2021 年以前，要训练一个情感分类器，就得找人标注几千条"影评 + 正面/负面"。换一个任务，再标一遍。

## 0.2　ZeroGen（Ye 等，EMNLP 2022）：让大模型直接造训练集

**被调用（参数不动）**：GPT2-XL，15 亿参数。

**提示词**：取自仓库 `tasks/imdb/imdb-x2.json`，原文：
```
The movie review in positive sentiment for movie "<C>" is: "
The movie review in negative sentiment for movie "<C>" is: "
```
分两步：先让模型编一个电影名 `<C>`，再把电影名填进去，让模型续写影评。最后一个引号是开着的，模型接着写的就是影评正文。**标签不是模型判的，是由用了哪条提示词决定的**：用"positive"那条，生成的就记为正面。

**产出数据**：README 给出的一条原样：
```json
{"C": "The Book of Mormon Musical", "X": "The Book of Mormon Musical brings all the drama and excitement of a real revival of the Broadway production to the big screen.", "Y": 0}
```
（这个仓库里标签 0 对应 positive 那条提示词。）

**被训练（改参数）**：DistilBERT（6600 万参数）或一层 LSTM。训练方式是普通的分类交叉熵：
- 输入一条合成影评，模型输出"正面"的概率 p；
- 标签是正面，损失 = −log p；标签是负面，损失 = −log(1−p)；
- 梯度下降改参数。

**考试**：IMDb、SST-2 等的**真实测试集**，即人写的影评、人标的标签。

## 0.3　暴露的三个问题

把造出来的数据拿出来看，会发现三类毛病：

| 问题 | 玩具例子（自拟） | 后来由哪条线回答 |
|---|---|---|
| **写什么**：千篇一律，覆盖面窄 | 一千条正面影评里，八百条都在夸"演技精湛、剧情感人" | 线一：条件生成 |
| **写得对不对**：事实错误 | "诺兰导演的《泰坦尼克号》……" | 线二：知识增强 |
| **哪些该留**：标签错、跑题 | 用"positive"提示词，模型却写出 "This movie was a total waste of time"，被记成正面 | 线三：去噪与降权 |

## 0.4　理想与现实

**理想**：生成器对任务的分布了如指掌，写什么、写多少、写得对，全都恰到好处，标签无误。

**现实**：
- 生成器有自己的偏好，只写它最常见的那几种；
- 它会编造；
- 提示词决定的标签不一定和它写出来的内容一致。

**三种转化**：
1. 从外部规定"写什么"（线一）；
2. 从外部提供"事实"（线二）；
3. 写完之后从外部判断"留不留"（线三）。

---

# 第一部分　三条源头线

## 线一：条件生成，控制"写什么"

### 1.1　CTRL（Keskar 等，Salesforce，2019）：把控制条件训练进模型

**为什么出现**：GPT-2 只能续写，没法指定"写一篇负面影评"或"写一篇维基百科风格的文章"。

**被训练（改参数）**：从头训练一个 16.3 亿参数的语言模型，训练文本约 140GB。

**关键做法**：每篇训练文本前面加一个"控制码"。控制码取自文本天然附带的信息，比如来源网站、Reddit 版块名、评分，不需要人工标注。训练数据样貌（示意）：
```
Reviews Rating: 1.0  这台吸尘器用了两周就坏了，客服也不回消息……
Wikipedia  光合作用是植物、藻类和某些细菌……
```

**训练目标从零讲**：语言模型的训练就是"预测下一个词"。
1. 对上面第一行，模型读到 `Reviews Rating: 1.0 这台吸尘器用了两周就`，要预测下一个词"坏"。
2. 如果模型给"坏"的概率是 0.2，这一步的损失是 −log 0.2 ≈ 1.61。
3. 训练后概率升到 0.6，损失降到 −log 0.6 ≈ 0.51。

因为控制码一直在前面，模型学会了"看到 Rating: 1.0 就更可能写负面词"。

**使用**：
```
输入：Reviews Rating: 5.0
输出：这是我买过最好用的吸尘器，吸力强、噪音小……
```

**要点**：条件是**训练进模型的**，换一种条件就得重新训练。

### 1.2　AttrPrompt（Yu 等，佐治亚理工，NeurIPS 2023 数据集与基准赛道）：条件写进提示词，不训练

**为什么出现**：大模型足够强以后，不用再训练控制码，直接在提示词里写明条件即可。问题变成：条件怎么选，才能让造出的数据多样、不偏？

**被调用（参数不动）**：ChatGPT（gpt-3.5-turbo）。先让它列出各数据集的属性维度和取值，由人检查；再按属性生成数据。

**提示词原文**（纽约时报新闻分类）：
```
Suppose you are a news writer. Please generate a {topic-class} news in NYT following the
requirements below:
1. Should focus on {subtopic};
2. Should be in length between {length:min-words} and {length:max-words} words;
3. The writing style should be {style};
4. The location should be in {location}.
```
每次生成前，从每个维度**各自随机**抽一个取值填进去。

**被训练（改参数）**：BERT-base，用合成数据微调成分类器，交叉熵损失，与 ZeroGen 相同。

**考试**：纽约时报（26 类）、亚马逊（23 类）、Reddit（45 类）、StackExchange（50 类）的**真实测试集**。

**结果**：只给类别的简单提示有明显的地域偏差；属性提示只用约 5% 的调用成本，就能达到同样效果。

**两个要点**：
1. "子话题"这一栏的取值是大模型自己列出来的。**ClinGen 做的，就是把这一栏的来源换成知识图谱。**
2. 各属性**独立抽样**。对新闻无所谓；对病历就是问题：年龄、性别、合并症、用药在真实病人身上是相关的（70 岁男性高血压病人用氨氯地平很常见，8 岁儿童不会），独立抽会造出不存在的组合。这就是你此前关心的"联合分布"难点。

### 1.3　GLAN（Li 等，微软，2024）：用知识分类体系系统地铺满"写什么"

**为什么出现**：属性是临时列出来的，覆盖面没有保证。能不能按人类知识的完整分类体系，一层层往下铺？

**被调用（参数不动）**：大模型逐层生成，每一层都以上一层为输入：
```
学科领域 → 子领域 → 具体学科 → 课程列表 → 每门课的教学大纲 → 每节课的关键概念 → 围绕关键概念出的作业题与答案
```

**被训练（改参数）**：用这些合成题目对 Mistral 等开源模型做监督微调（预测下一个词的交叉熵，同 1.1）。

**考试**：数学推理、编程、学术考试、逻辑推理等公开基准；训练时不用任何任务专属数据。

**要点**：新增一个领域，只需在分类树上加一个节点。**"写什么"由一棵结构化的知识树控制，而不是一张扁平的词表。** 搬到病历上，对应的是按 ICD 疾病分类树逐层铺开，而不是从一张病名表里随机抽。

---

## 线二：知识增强生成，保证"写得对"

### 2.1　RAG（Lewis 等，Facebook，NeurIPS 2020）：先检索，再生成

**为什么出现**：语言模型会一本正经地编造事实，而且知识一更新就要重新训练。

**组件**：
- 检索器 DPR：把问题和维基百科段落各编码成一个向量，向量越近越相关；
- 生成器 BART。

**数据样貌**：
```
问题：谁写了《物种起源》？
检索到的段落：[1]《物种起源》是查尔斯·达尔文于 1859 年出版的…… [2] ……
答案：查尔斯·达尔文
```

**训练（改参数）从零讲**：训练数据只有（问题，答案），**没有告诉模型哪一段才是该检索的**。RAG 的办法是把检索也当成一个概率：

p(答案 | 问题) = Σ 每段的检索概率 × 用这一段写出该答案的概率

玩具数字（自拟）：检索器取回两段。

| 段落 | 检索概率 | 用它写出"查尔斯·达尔文"的概率 | 乘积 |
|---|---|---|---|
| [1] 达尔文那段 | 0.4 | 0.9 | 0.36 |
| [2] 不相关的一段 | 0.6 | 0.05 | 0.03 |
| 合计 | | | 0.39 |

损失 = −log 0.39。梯度会做两件事：
- 让生成器在给了段落 [1] 时更确定地写出答案；
- **让检索器把段落 [1] 的概率调高**，因为调高它最能提升总概率。

这样，检索器在没有任何"该检索哪段"标注的情况下，学会了检索有用的段落。

原文只训练检索器的问题编码器和生成器，段落编码器固定不动，省得重建索引。

**考试**：Natural Questions 等开放域问答的真实测试集。

**要点**：事实来自**检索到的原文**，生成器负责组织语言。

### 2.2　KAPING（Baek 等，2023）：把知识图谱三元组直接塞进提示词

**为什么出现**：大模型时代不想再训练；知识图谱里的事实是结构化的三元组，能不能直接给大模型看？

**被调用（参数不动）**：大模型。

**步骤**：
1. 找出问题里提到的实体；
2. 从知识图谱中取出与这些实体相连的三元组；
3. 把三元组转成文字，按与问题的相似度排序，取最相关的若干条；
4. 放在问题前面，交给大模型回答。

**提示词格式**（示意）：
```
Below are the facts that might be relevant to answer the question:
(阿司匹林, 适应症, 缺血性脑卒中二级预防)
(阿司匹林, 禁忌症, 活动性消化道出血)
(阿司匹林, 药物类别, 抗血小板药)
Question: 一位有活动性胃溃疡出血的脑梗死患者，能否使用阿司匹林？
Answer:
```

**训练**：无。

**考试**：知识图谱问答基准的真实测试集，零样本。

**要点**：给模型的是**带关系的事实**，不只是实体名字。"阿司匹林"这个名字本身不告诉模型任何禁忌；"(阿司匹林, 禁忌症, 活动性消化道出血)"才告诉它。

### 2.3　MedReason（Wu 等，2025）：用知识图谱路径构造推理数据

**为什么出现**：医学推理题需要多步推理；大模型直接写的推理链常常跳步或出错。能不能让知识图谱提供推理的骨架？

**步骤（原文）**：
1. 从 7 个医学问答数据集取题；
2. 在医学知识图谱中，找出从"题目中的实体"到"答案实体"的**路径**；
3. 以这些路径为骨架，由大模型写出逐步推理（调用，参数不动）；
4. 检查推理"与临床逻辑和循证医学一致"，并请多个专科的医生评估质量。

**一条数据的样子**（示意）：
```
题目：65 岁男性，突发右侧肢体无力、言语不清 2 小时……最可能的诊断？
知识图谱路径：右侧偏瘫 —[提示]→ 左侧大脑半球病变 —[见于]→ 缺血性脑卒中；
              言语不清 —[提示]→ 优势半球受累
推理：①右侧肢体无力提示左侧大脑半球病变；②言语不清支持优势半球受累；③急性起病……
答案：急性缺血性脑卒中
```

**规模**：32,682 道题，每题配逐步推理。

**被训练（改参数）**：对多个开源模型做监督微调。输入是题目，目标输出是"推理 + 答案"整段文字，损失是逐词交叉熵（同 1.1）。
- DeepSeek-Distill-8B 最高提升 7.7%；
- 最好的 MedReason-8B 在 MedBullets 上比当时的 Huatuo-o1-8B 高 4.2%。

**考试**：公开医学基准的真实题目。

**要点**：知识图谱在这里提供的是**推理路径**，也就是事实之间的关系链；而且生成后有验证。

---

## 线三：合成数据去噪，决定"哪些该留"

### 3.1　ProGen（Ye, Gao, Feng, Wu, Yu, Kong，2022）：让小模型反馈哪些样本好，再拿好样本当示范

作者与 ZeroGen 是同一组（香港大学、上海人工智能实验室），是 ZeroGen 的直接续作。

**为什么出现**：ZeroGen 造的数据里大量低质量、重复样本，全拿来训练效果差，而且要造很多才够用。

**循环（每一轮）**：
1. **生成**（调用，参数不动）：大模型按提示词造一批数据。从第二轮起，提示词里附上上一轮挑出的好样本作为示范；
2. **训练**（改参数）：用这批数据训练一个小任务模型；
3. **估计每个样本的质量**：用"影响函数"；
4. **挑选**：分数最高的几条，作为下一轮的示范。

**影响函数从零讲**：它回答一个问题：**"如果把这条训练样本拿掉，验证集上的损失会变大还是变小？"**
- 拿掉后验证损失变大，说明这条样本有帮助，是好样本；
- 拿掉后验证损失变小，说明它在帮倒忙，可能是错标或跑题的样本。

真的逐条拿掉再重训太贵，影响函数用一阶近似直接算出这个变化量，不用重训。

玩具数字（自拟）：

| 样本 | 文本 | 合成标签 | 拿掉后验证损失的变化 | 判断 |
|---|---|---|---|---|
| a | "A heartfelt, beautifully acted film." | 正面 | +0.012 | 好 |
| b | "Great movie, great movie, great movie." | 正面 | +0.001 | 几乎无用（重复） |
| c | "This movie was a total waste of time." | 正面 | −0.009 | 有害（标签错） |

**一个难点**：零样本设定下**没有真实的验证集**，验证集本身也是合成的，里面同样有错标。所以原文在计算影响时用**对噪声稳健的损失**，避免被验证集里的错标带偏。"稳健损失"在 3.2 里具体讲。

**考试**：IMDb、SST-2、Rotten Tomatoes、Electronics、Yelp 五个情感分类数据集的真实测试集。

**结果**：只用 1% 规模的合成数据，就能达到或超过不带反馈的基线。

### 3.2　SunGen（Gao, Pi, Lin 等，ICLR 2023）：给每个合成样本学一个权重

与 ProGen 也是同一个圈子（作者有重叠）。

**为什么出现**：ProGen 是硬筛（留或不留），会丢信息；而且靠人设阈值。能不能让每个样本有一个 0 到 1 之间的权重，**自动学出来**，不用任何人工标注？

**两层优化，从零讲**：
- **待学的东西**：每个合成样本 i 一个权重 wᵢ。原文所有样本初始权重为 0.5；代码里写成 wᵢ = sigmoid(θᵢ)。
- **内层**（改小模型参数）：用加权交叉熵训练小模型。权重越大的样本，对训练影响越大。
- **外层**（改权重）：在另一批合成样本上算一个**对噪声稳健的损失**，看小模型表现如何；再把这个损失对每个 wᵢ 求梯度，即"调高 wᵢ，外层损失会变好还是变坏"，据此更新权重。
- 内外两层交替进行。

**为什么外层要用"稳健损失"**：
- 外层的评判数据也是合成的，也有错标。如果用普通交叉熵，模型会被要求把错标样本也判"对"，错标样本的权重就压不下去。
- 稳健损失（原文用反向交叉熵 RCE；代码里还提供平均绝对误差 MAE 等选项）的特点是：**单个样本能贡献的损失有上限**。少数错标样本喊得再响也盖不过多数正确样本。
- 于是"让模型符合大多数"这个方向胜出，错标样本的权重被自动压低。

玩具演示（自拟，4 条正面样本中有 1 条错标）：

| 样本 | 合成标签 | 实际内容 | 初始权重 | 若干轮后 |
|---|---|---|---|---|
| 1 | 正面 | 正面 | 0.5 | 0.86 |
| 2 | 正面 | 正面 | 0.5 | 0.81 |
| 3 | 正面 | 跑题（讲天气） | 0.5 | 0.32 |
| 4 | 正面 | **负面** | 0.5 | 0.07 |

**被调用（参数不动）**：生成用的预训练语言模型。

**被训练（改参数）**：小任务模型（一层双向 LSTM 或 DistilBERT），以及每个样本的权重。

**考试**：8 个文本分类基准的真实测试集。

**结果**：SunGen-LSTM 平均准确率相对提升 9.8%。

**要点**：到 2023 年，**"合成数据必须去噪或降权"已经是这条线的标准配置**。

### 3.3　这条线的现代形态

ProGen、SunGen 用一个小模型间接判断样本好坏。到 2024—2025 年，主流做法换成了**直接判对错**：
- 数学、代码：用程序验证答案，错的丢掉，即"拒绝采样"。据 DeepSeek-R1 论文，它的第二轮监督数据就是这样筛出约 60 万条推理样本。
- 不能程序验证的：用训练过的评委或批评器打分再筛，即第三讲阶段 4 和"病历查错器"一节。

**所以第三讲讲的查错器，就是线三在今天的形态。** 两讲在这里接上了。

---

# 第二部分　ClinGen 逐步拆解

## 基本信息

- Xu 等，ACL 2024 Findings（arXiv 2023 年 11 月）；
- 作者：Ran Xu, Hejie Cui, **Yue Yu**, Xuan Kan, Wenqi Shi, **Yuchen Zhuang**, Wei Jin, Joyce Ho, Carl Yang。第一作者和通讯作者来自埃默里大学（Emory）Carl Yang 组；**AttrPrompt 的第一、第二作者 Yue Yu 和 Yuchen Zhuang 都参与了**。
- 任务：**临床 NLP 的少样本学习**。每类只有 5 条真实样本，用大模型合成训练数据，训练小模型；
- 规模：8 类任务、18 个数据集（文本分类、关系抽取、自然语言推理、事实核查、问答、句子相似度、命名实体识别、属性抽取）。

## 第 1 步　准备"子话题"：两个来源二选一

**来源 A：知识图谱。** 用 iBKH（Integrative Biomedical Knowledge Hub），按类型取实体。例如疾病识别任务取所有"疾病"类实体；关系抽取任务取"药物—疾病"等三元组。

**来源 B：大模型。** 提示词原文：
```
Suppose you are a clinician and want to collect a set of <Entity Type>.
Could you list 300 entities about <Entity Type>?
```

得到的是一个个**实体名字**，每个类别存成一个文本文件，比如 `data/litcovid/kg/treatment.txt`，一行一个词。

**代码观察**：这些词表文件、从 iBKH 查询实体的代码，**都不在仓库里**；仓库的 `data/` 目录下每个数据集只有一个 `label.txt`。

## 第 2 步　准备"写作风格"

让大模型根据任务名和几条示范，列出可能的写作风格，比如"医学文献""医患对话"，存成 `styles.txt`（同样不在仓库里）。

## 第 3 步　拼提示词，生成（代码原文）

`src/text_class/clingen.py` 中的 `gen_one_prompt` 函数，每生成一条数据都这样拼：
```python
style   = random.sample(styles, 1)[0]                        # 随机抽一个风格
topic_i = random.sample(keyword_dict[class_name], 1)[0]      # 从该类词表随机抽一个子话题
prompt  = f"""Suppose you need to create a dataset for {args.domain}. Your task is to:
1. generate a sentence about {args.domain}.
2. the sentence should mimic the style of {style},
3. the sentence should be relevant to the subtopic of {topic_i} for {class_name}.
Some examples for {class_name} are:
Label: {class_name}
Text: {示范1的前三句}
Label: {class_name}
Text: {示范2的前三句}
Label: {class_name}
Text: {示范3的前三句}
Label: {class_name}
Text:"""
```

拼出来的一条（示意，COVID 文献分类的"治疗"类）：
```
Suppose you need to create a dataset for COVID-19 Literature. Your task is to:
1. generate a sentence about COVID-19 Literature.
2. the sentence should mimic the style of randomized controlled trial abstract,
3. the sentence should be relevant to the subtopic of remdesivir for treatment.
Some examples for treatment are:
Label: treatment
Text: ……（真实少样本示范的前三句）
……
Label: treatment
Text:
```

- **被调用（参数不动）**：gpt-3.5-turbo-0301，温度 1.0。消融实验另用了 text-curie-001 和 GPT-4（GPT-4 因预算只生成了 500 条）。
- **产出**：每个数据集约 5000 条合成样本。

**对照 ZeroGen 和 AttrPrompt 的提示词**：三者一脉相承。
- ZeroGen：标签写进提示词，外加一个模型自编的电影名；
- AttrPrompt：标签 + 多个属性；
- ClinGen：标签 + 两个属性（风格、子话题）+ 示范，子话题的取值来自知识图谱里的实体名。

## 第 4 步　训练小模型

- **被训练（改参数）**：PubMedBERT（Base 和 Large）；
- **两阶段微调**：先用每类 5 条真实样本微调，再用约 5000 条合成样本继续微调；交叉熵损失，6 个轮次；
- **合成数据的过滤、去噪、降权**：**原文没有报告，代码里也没有。**

## 第 5 步　考试

- 18 个数据集的**真实测试集**；
- **对照组**：
  - 传统增强：同义词替换、回译、Mixup、MELM、LightNER、KGPC；
  - 大模型造数据：ZeroGen、DemoGen、ProGen、S3。
- **结果**：
  - 平均提升 8.7%（PubMedBERT-Base）、7.7%（Large）；
  - 合成数据与真实数据的分布距离（CMD）从基线的 0.463—0.512 降到 0.432；
  - 人工抽查 600 条，未发现事实错误。
- **仓库的另一处注意**：README 说上传到 Hugging Face 的数据里有 `test.jsonl`，"stands for data from the test set"，即测试集数据也一并打包公开了。

---

# 第三部分　ClinGen 溯源表

| 编号 | 部件 | 本文叫法 | 来源 | 来路 | 发表时代际 | 当时该用什么 | 披露 |
|---|---|---|---|---|---|---|---|
| 件1 | 属性条件的提示生成 | knowledge-infused prompting | AttrPrompt（NeurIPS 2023），**其第一、第二作者均为本文合作者** | 国外原创线，**原作者参与转手** | 现役 | — | 正文引用 |
| 件2 | 属性取值来源：知识图谱 | clinical knowledge extraction from KG | 以 iBKH 为**实体词表** | 线二的最弱用法 | **名义现役，实现退化**：只取实体名，不用关系 | KAPING 式三元组进提示词（2023）；按图谱路径组织内容（后来的 MedReason 那一路） | 正文 |
| 件3 | 属性取值来源：大模型列举 | clinical knowledge extraction from LLM | 与 AttrPrompt 让 ChatGPT 列子话题相同 | 原作者转手 | 现役 | — | 正文 |
| 件4 | 写作风格属性 | writing style | AttrPrompt 的 style 维度 | 原作者转手 | 现役 | — | 正文 |
| 件5 | 生成器 | — | gpt-3.5-turbo-0301 | 国外 | 现役（GPT-4 已有，因预算只做了消融） | — | 正文 |
| 件6 | 学生模型 | — | PubMedBERT（2020） | 国外 | 现役 | — | 正文 |
| 件7 | 合成数据去噪 / 降权 | **缺席** | — | — | **落后一代** | ProGen（2022）、SunGen（ICLR 2023）；ProGen 甚至就在它的对照组里 | — |

**关于"知识图谱"这四个字**：在文本分类、实体识别这类任务里，知识图谱在 ClinGen 中的实际作用是**一张实体名单**：随机抽一个名字，写进 "the sentence should be relevant to the subtopic of {名字}"。图谱里的关系、路径、属性都没有用上（关系抽取任务取了三元组，是例外）。而且原文自己的结果显示，**用大模型列举的词表（件3），效果与用知识图谱（件2）相当甚至更好**，说明图谱在这里并不承重。

**结论**：
- **对的部分**：知识图谱只被当成词表用，是这条线最弱的用法；该有的去噪一环缺席，而 ProGen 已在它的对照组里，SunGen 已在 ICLR 2023 发表。
- **需要修正的部分**：它的核心部件 AttrPrompt 不是抄别人的旧组件，而是**原作者参与、半年前的现役工作**；生成器也是当时的现役模型。更准确的说法是：**ClinGen 是埃默里团队与 AttrPrompt 原作者合作完成的医学版 AttrPrompt；"knowledge-infused"这个名字，比它实际做的事说得满。**
- **它仍比 RareSyn、LLM-CARe 规矩的地方**：在 18 个真实测试集上考；对照组覆盖传统增强和同类造数据方法；生成代码公开。

---

# 第四部分　这对你意味着什么

## 回看 RareSyn 和 LLM-CARe 在这三条线上的位置

| | 线一：写什么 | 线二：写得对不对 | 线三：哪些该留 |
|---|---|---|---|
| ClinGen | AttrPrompt 属性 | 实体名单 | 无 |
| RareSyn | 借别的病种的真实病历当模板 | 知识图谱实体名单 + 手工加权 | 无 |
| LLM-CARe | 生成后按配额插词 | 无（同一模型自评） | 同一模型按清单自改 |

三篇在线二上都停在"实体名单"这一档；在线三上，要么缺席，要么用同一模型自评代替。

## 如果你自己做病历生成，三条线各该用到哪一档

| 线 | 该用的做法 | 两张 5090 的代价 |
|---|---|---|
| 线一 | 按**真实数据的联合分布**采样属性组合（年龄 × 性别 × 合并症 × 用药），而不是各属性独立抽；或者像 GLAN 一样按疾病分类树逐层铺开 | 不需要显卡：从真实数据统计联合频数，或用一个小的表格生成模型采样 |
| 线二 | 把**诊断相关的知识图谱子图（三元组或路径）**放进提示词，让病程、检查、用药有事实依据 | 不需要训练；如果本地跑 7—8B 生成器，单卡推理即可 |
| 线二（反查） | 生成后用图谱和结构化记录反查：写出的药是否适用于该诊断、是否撞禁忌 | 规则程序，CPU 即可 |
| 线三 | 生成后用**规则 + 训练过的查错器**过滤或打分（第三讲第五节） | 查错器训练单卡几小时到一天 |
| 线三（补充） | 按下游小模型的反馈给样本加权（SunGen 式） | BERT 级模型，单卡几小时 |

三条线各用到现役档，再配上第三讲的评测合同，就是一篇方法和评测都站得住的病历生成工作。**最贵的一环是查错器的训练，而两张 5090 完全够用。**

---

## 参考文献

- Keskar et al. 2019. CTRL: A Conditional Transformer Language Model for Controllable Generation. arXiv:1909.05858
- Lewis et al. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020
- Ye et al. 2022. ZeroGen: Efficient Zero-shot Learning via Dataset Generation. EMNLP 2022. arXiv:2202.07922 ；代码 https://github.com/jiacheng-ye/ZeroGen
- Ye et al. 2022. ProGen: Progressive Zero-shot Dataset Generation via In-context Feedback. Findings of EMNLP 2022. arXiv:2210.12329
- Gao et al. 2023. Self-Guided Noise-Free Data Generation for Efficient Zero-Shot Learning (SunGen). ICLR 2023. arXiv:2205.12679 ；代码 https://github.com/SumilerGAO/SunGen
- Baek et al. 2023. Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering (KAPING). arXiv:2306.04136
- Yu et al. 2023. Large Language Model as Attributed Training Data Generator: A Tale of Diversity and Bias (AttrPrompt). NeurIPS 2023 Datasets and Benchmarks. arXiv:2306.15895
- Xu et al. 2024. Knowledge-Infused Prompting: Assessing and Advancing Clinical Text Data Generation with Large Language Models (ClinGen). Findings of ACL 2024. arXiv:2311.00287
- Li et al. 2024. Synthetic Data (Almost) from Scratch: Generalized Instruction Tuning for Language Models (GLAN). arXiv:2402.13064
- Wu et al. 2025. MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs. arXiv:2504.00993
- DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948
